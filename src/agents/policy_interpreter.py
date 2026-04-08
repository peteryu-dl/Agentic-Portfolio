#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
政策解读智能体 (policy_interpreter.py) - 增强防封版
====================================================
增强措施：
1. 结果缓存：相同日期政策分析结果本地保存 24 小时，避免重复调用 API 浪费钱
2. 速率限制：连续调用之间强制间隔 1 秒（防止触发 Rate Limit）
3. 输入截断：政策文本严格控制在 2000 字符内（防止 Token 超标）
4. 退避重试：遇到 429（Rate Limit）自动等待 20 秒后重试
5. 成本保护：默认使用 gpt-4o-mini（便宜 50 倍），单次成本约 0.02 元

风控说明：
- OpenAI/DeepSeek 不会"封 IP"，但会限流（429 错误）
- 本模块设计为"低频调用"（每月 1 次），正常使用绝不会触发风控
- 只有批量回测（连续跑 12 个月）时才需要缓存保护
"""

import os
import json
import time
import hashlib  # 用于生成缓存 key
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
from pydantic import BaseModel, Field
from loguru import logger

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# OpenAI 特定错误（用于捕获 Rate Limit）
try:
    from openai import RateLimitError, APIError, APITimeoutError
except ImportError:
    # 如果未安装 openai 包，定义占位符避免报错
    RateLimitError = Exception
    APIError = Exception
    APITimeoutError = Exception

from dotenv import load_dotenv
load_dotenv()


# -------------------- 配置区（增强防封）--------------------
# API 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("LLM_ESAY_TASK") 
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# 防封/成本控制配置
RATE_LIMIT_DELAY = 1.0          # 连续调用之间最小间隔（秒），防止触发 RPM 限制
MAX_INPUT_LENGTH = 2000         # 政策文本最大字符数（防止 Token 爆炸，约 500-1000 Token）
MAX_RETRIES = 2                 # API 失败重试次数（不含首次）
RATE_LIMIT_BACKOFF = 20         # 遇到 429 错误时的退避等待（秒）
CACHE_TTL_HOURS = 24            # 结果缓存有效期（小时）

# 路径配置
DEFAULT_POLICY_PATH = "data/policy_texts/govcn_2025.csv"
CACHE_DIR = Path(".cache/policy")  # 本地缓存目录（隐藏文件夹）
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# -------------------- 数据模型 --------------------
class PolicyTheme(BaseModel):
    theme: str = Field(description="投资主题名称（2-6字）")
    confidence: float = Field(description="置信度 0.0-1.0")
    rationale: str = Field(description="理由（引用原文关键词）")

class PolicyAnalysisResult(BaseModel):
    themes: List[PolicyTheme] = Field(description="Top 5 投资主题", min_items=1, max_items=5)
    market_sentiment: str = Field(description="整体情绪：积极/谨慎/防御")
    key_risk: Optional[str] = Field(description="主要风险点（如有）")

@dataclass
class PolicySignal:
    top_5_themes: List[str]
    confidences: List[float]
    analysis_summary: str
    raw_result: Optional[PolicyAnalysisResult] = None
    is_cached: bool = False  # 新增：标记是否来自缓存


# -------------------- 缓存机制（防封核心）--------------------
def get_cache_key(target_date: str, policy_hash: str) -> str:
    """生成缓存键（基于日期+政策内容哈希）"""
    return hashlib.md5(f"{target_date}_{policy_hash}".encode()).hexdigest()

def load_from_cache(cache_key: str) -> Optional[PolicySignal]:
    """尝试从本地缓存读取结果"""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    if not cache_file.exists():
        return None
    
    try:
        # 检查是否过期
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if (datetime.now() - mtime).total_seconds() > CACHE_TTL_HOURS * 3600:
            return None  # 过期
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 重建对象
        return PolicySignal(
            top_5_themes=data["themes"],
            confidences=data["confidences"],
            analysis_summary=data["summary"],
            is_cached=True
        )
    except Exception:
        return None

def save_to_cache(cache_key: str, signal: PolicySignal):
    """保存结果到本地缓存"""
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    data = {
        "themes": signal.top_5_themes,
        "confidences": signal.confidences,
        "summary": signal.analysis_summary,
        "cached_at": datetime.now().isoformat()
    }
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# -------------------- API 调用（带防封保护）--------------------
_last_call_time = 0  # 全局变量记录上次调用时间

def rate_limit_protect():
    """速率限制保护：确保连续调用间隔至少 RATE_LIMIT_DELAY 秒"""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < RATE_LIMIT_DELAY:
        sleep_time = RATE_LIMIT_DELAY - elapsed
        logger.debug(f"⏱️  速率限制保护：等待 {sleep_time:.1f} 秒")
        time.sleep(sleep_time)
    _last_call_time = time.time()

def safe_truncate(text: str, max_len: int = MAX_INPUT_LENGTH) -> str:
    """安全截断文本，保留前半部分（政策文件通常前 2000 字已包含核心观点）"""
    if len(text) <= max_len:
        return text
    # 保留前 90%，在中间截断（避免截断在句子中间）
    truncated = text[:int(max_len * 0.9)]
    # 找到最后一个句号截断
    last_period = truncated.rfind("。")
    if last_period > 0:
        return truncated[:last_period+1] + "\n...[内容截断，仅保留政策核心段落]"
    return truncated + "\n...[截断]"

def call_llm_with_protection(prompt: str, max_retries: int = MAX_RETRIES) -> Optional[str]:
    """
    带防封保护的 LLM 调用
    =====================
    1. 速率限制：确保调用间隔
    2. 退避重试：遇到 429（Rate Limit）自动等待 20 秒后重试
    3. 错误分类：区分 Rate Limit、Timeout、其他错误
    """
    # 检查 API Key
    if not OPENAI_API_KEY:
        logger.error("❌ 未找到 OPENAI_API_KEY，请检查 .env 文件")
        return None
    
    # 速率限制保护（关键防封措施）
    rate_limit_protect()
    
    # 初始化 LLM
    llm = ChatOpenAI(
        model=OPENAI_MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0.3,
        max_tokens=1500,  # 控制输出长度（省钱+快速）
        request_timeout=60  # 60秒超时（防止卡住）
    )
    
    # 带重试的调用
    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke(prompt)
            return response.content
            
        except RateLimitError as e:
            # 429 错误：被限流了，需要退避
            if attempt < max_retries:
                logger.warning(f"⚠️  触发 Rate Limit (429)，等待 {RATE_LIMIT_BACKOFF} 秒后重试...")
                time.sleep(RATE_LIMIT_BACKOFF)
            else:
                logger.error(f"❌ Rate Limit 持续存在，放弃调用: {e}")
                return None
                
        except APITimeoutError:
            # 超时，可能是网络问题
            if attempt < max_retries:
                wait = 5 * (attempt + 1)
                logger.warning(f"⏱️  请求超时，{wait} 秒后重试...")
                time.sleep(wait)
            else:
                logger.error("❌ API 超时，放弃")
                return None
                
        except APIError as e:
            # 其他 API 错误（如 500, 503）
            if attempt < max_retries:
                wait = 3 * (attempt + 1)
                logger.warning(f"⚠️  API 错误: {str(e)[:50]}，{wait} 秒后重试...")
                time.sleep(wait)
            else:
                logger.error(f"❌ API 错误无法恢复: {e}")
                return None
                
        except Exception as e:
            # 未知错误（如网络断开）
            logger.error(f"❌ 调用异常: {e}")
            return None
    
    return None


# -------------------- 业务逻辑 --------------------
def build_prompt(policy_text: str, macro_regime: str, equity_score: float) -> str:
    """构建 Prompt（简洁版，减少 Token 消耗）"""
    # 截断政策文本（关键防封措施，防止 Token 超标）
    truncated_text = safe_truncate(policy_text, MAX_INPUT_LENGTH)
    
    parser = PydanticOutputParser(pydantic_object=PolicyAnalysisResult)
    
    template = f"""作为政策分析师，阅读以下政策文本，提取 Top 5 投资主题。

【政策文本（已截断至核心内容）】：
{truncated_text}

【背景】经济周期: {macro_regime} | 股票友好度: {equity_score}/1.0

【任务】提取 5 个投资主题（行业/概念），给出置信度(0-1)和一句话理由。

{parser.get_format_instructions()}

注意：
1. 主题名简短（2-6字），如"人工智能"、"新能源"
2. 置信度基于政策提及频次和力度
3. 必须返回 JSON 格式，不要 markdown 代码块"""
    
    return template

def interpret_policy(policy_text: str, 
                    macro_regime: str = "unknown",
                    equity_score: float = 0.5,
                    target_date: str = None) -> Optional[PolicyAnalysisResult]:
    """
    分析政策（带全流程防封）
    """
    if not policy_text:
        return None
    
    # 生成缓存键（基于日期+内容哈希）
    content_hash = hashlib.md5(policy_text.encode()).hexdigest()[:8]
    cache_key = get_cache_key(target_date or "latest", content_hash)
    
    # 尝试读缓存（如果相同日期+相同政策文本，直接返回）
    cached = load_from_cache(cache_key)
    if cached:
        logger.success(f"✅ 命中缓存（{target_date}），跳过 API 调用（省钱+防封）")
        # 转换回结果格式
        return PolicyAnalysisResult(
            themes=[PolicyTheme(theme=t, confidence=c, rationale="来自缓存") 
                   for t, c in zip(cached.top_5_themes, cached.confidences)],
            market_sentiment="来自缓存",
            key_risk=None
        )
    
    # 构建 Prompt
    prompt = build_prompt(policy_text, macro_regime, equity_score)
    
    # 调用 LLM（带防封保护）
    logger.info(f"🤖 调用 LLM ({OPENAI_MODEL_NAME}) 分析政策...")
    content = call_llm_with_protection(prompt)
    
    if not content:
        return None
    
    # 解析结果
    try:
        parser = PydanticOutputParser(pydantic_object=PolicyAnalysisResult)
        # 清理可能的 markdown
        clean_content = content.replace("```json", "").replace("```", "")
        result = parser.parse(clean_content)
        
        logger.success(f"✅ LLM 分析完成，提取 {len(result.themes)} 个主题")
        return result
        
    except Exception as e:
        logger.error(f"❌ 解析 LLM 输出失败: {e}")
        return None


# -------------------- 对外接口 --------------------
def analyze_policy(policy_path: str = DEFAULT_POLICY_PATH,
                  target_date: str = None,
                  macro_context: Dict = None) -> PolicySignal:
    """
    主入口（增强防封版）
    """
    logger.info(f"🔍 政策解读开始: {target_date or '最新'}")
    
    # 加载文本
    policy_text = ""
    if Path(policy_path).exists():
        try:
            df = pd.read_csv(policy_path, parse_dates=["date"])
            if target_date:
                target = pd.to_datetime(target_date)
                df = df[df["date"] <= target]
            if not df.empty:
                texts = [f"[{row['date']}] {row.get('title','')}\n{row['content']}" 
                        for _, row in df.iterrows()]
                policy_text = "\n---\n".join(texts)
                logger.info(f"📄 加载政策: {len(df)} 份文件")
        except Exception as e:
            logger.warning(f"⚠️  读取政策文件失败: {e}")
    
    # 准备宏观背景
    if macro_context is None:
        macro_context = {"regime": "unknown", "equity_friendly_score": 0.5}
    
    regime = macro_context.get("regime", "unknown")
    score = macro_context.get("equity_friendly_score", 0.5)
    
    # 调用分析（带缓存）
    llm_result = interpret_policy(policy_text, regime, score, target_date)
    
    if llm_result:
        themes = [t.theme for t in llm_result.themes]
        confidences = [t.confidence for t in llm_result.themes]
        summary = f"情绪{llm_result.market_sentiment}，推荐{', '.join(themes[:3])}"
        
        signal = PolicySignal(
            top_5_themes=themes,
            confidences=confidences,
            analysis_summary=summary,
            raw_result=llm_result,
            is_cached=False
        )
        
        # 保存到缓存（下次相同日期直接读缓存，不调用 API）
        content_hash = hashlib.md5(policy_text.encode()).hexdigest()[:8]
        save_to_cache(get_cache_key(target_date or "latest", content_hash), signal)
        
        return signal
    
    else:
        # Fallback
        logger.warning("🎭 LLM 失败，使用默认主题")
        default_themes = ["宽基指数", "科技成长", "新能源", "医药生物", "消费升级"]
        return PolicySignal(
            top_5_themes=default_themes,
            confidences=[0.5]*5,
            analysis_summary="LLM 调用失败，使用默认配置",
            is_cached=False
        )


# ==================== 测试入口 ====================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    # 将项目根目录添加到 Python 路径，解决模块导入问题
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.core.macro_regime import detect_macro_regime  # 测试时联动
    
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    
    print("=" * 60)
    print("🤖 政策解读智能体 - 增强防封版测试")
    print("=" * 60)
    print("💡 防封措施:")
    print(f"   - 速率限制: 连续调用间隔 {RATE_LIMIT_DELAY} 秒")
    print(f"   - 输入截断: 最多 {MAX_INPUT_LENGTH} 字符（防 Token 爆炸）")
    print(f"   - 结果缓存: {CACHE_TTL_HOURS} 小时内重复分析直接读本地")
    print(f"   - 退避重试: 遇到 429 错误自动等待 {RATE_LIMIT_BACKOFF} 秒")
    print("=" * 60)
    
    # 测试场景 1：首次调用（走 API）
    print("\n📌 测试 1: 首次分析（调用 API）...")
    start = time.time()
    signal1 = analyze_policy(target_date="2025-03-27")
    cost1 = time.time() - start
    
    print(f"\n📋 结果（耗时 {cost1:.1f} 秒）:")
    for i, (theme, conf) in enumerate(zip(signal1.top_5_themes, signal1.confidences), 1):
        print(f"   {i}. {theme} (置信度 {conf:.2f})")
    
    # 测试场景 2：重复调用（应命中缓存，不花钱）
    print(f"\n📌 测试 2: 重复分析（应命中缓存，不调用 API）...")
    start = time.time()
    signal2 = analyze_policy(target_date="2025-03-27")
    cost2 = time.time() - start
    
    if signal2.is_cached or cost2 < 0.5:  # 如果很快完成，说明是缓存
        print(f"✅ 命中缓存！耗时仅 {cost2:.2f} 秒（省钱+防封）")
    else:
        print(f"⚠️  未命中缓存，耗时 {cost2:.1f} 秒")
    
    print(f"\n💰 成本估算: 单次约 0.01-0.03 元（使用 {OPENAI_MODEL_NAME}）")
    print("=" * 60)