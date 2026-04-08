#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF 主题映射模块 (theme_mapper.py) - AI 自动打标
================================================
功能：让 LLM 自动为 ETF 打上三级标签，替代人工硬编码。

工作流程：
1. 读取 etf_basic.csv（包含 ETF 名称、代码、规模）
2. 使用 LLM 分析 ETF 名称（如"华夏上证科创板50成份ETF"）
3. 自动提取三级标签：
   - L1 资产大类: 股票/债券/商品/货币
   - L2 主题: 宽基/行业/策略/利率/信用/贵金属/能源
   - L3 子主题: 芯片/新能源/科创50/国债/黄金/沪深300
4. 保存为 processed_etf_basic.csv（带标签的档案）

为什么用 LLM 而非硬编码？
- 中国 ETF 命名混乱：同一只指数有多个 ETF（如沪深300有10只不同公司的）
- 新 ETF 上市频繁（每月新增5-10只），硬编码维护困难
- LLM 能理解语义："科创板50"→"科技-芯片-科创50"，"中证煤炭"→"能源-煤炭"

防封/成本控制：
- 批量处理：一次处理 10 只 ETF（减少 API 调用次数）
- 结果缓存：已打标的 ETF 24 小时内不再重复调用
- 使用 kimi-k2-turbo-preview（便宜且速度快）
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, Field
from loguru import logger

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from dotenv import load_dotenv
load_dotenv()


# -------------------- 配置区 --------------------
# API 配置（复用环境变量）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("LLM_ESAY_TASK")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# 批处理配置（成本控制）
BATCH_SIZE = 10              # 一次处理 10 只 ETF（平衡速度和 Token 消耗）
RATE_LIMIT_DELAY = 1.0       # 批次间间隔（秒）
MAX_ETF_NAME_LENGTH = 50     # ETF 名称截断（防止超长名称消耗 Token）

# 缓存配置
CACHE_DIR = Path(".cache/theme_mapping")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_HOURS = 24

# 路径配置
INPUT_PATH = Path("data/etf/etf_basic.csv")
OUTPUT_PATH = Path("data/etf/processed_etf_basic.csv")


# -------------------- 数据模型 --------------------
class ETFTags(BaseModel):
    """
    ETF 三级标签结构（Pydantic 模型）
    ================================
    强制 LLM 必须返回这三个字段，确保数据结构统一。
    
    示例：
    - 输入："易方达沪深300ETF"
    - 输出：L1=股票, L2=宽基, L3=沪深300
    
    - 输入："国泰中证煤炭ETF"
    - 输出：L1=股票, L2=行业, L3=能源-煤炭
    """
    L1_asset_class: str = Field(
        description="资产大类：股票、债券、商品、货币、跨境",
        enum=["股票", "债券", "商品", "货币", "跨境"]
    )
    L2_theme: str = Field(
        description="主题分类：宽基、行业、策略、利率债、信用债、贵金属、能源、农产品",
        examples=["宽基", "行业", "策略", "利率债", "贵金属"]
    )
    L3_sub_theme: str = Field(
        description="子主题：具体指数或细分行业，如沪深300、科创50、芯片、新能源、黄金、国债",
        examples=["沪深300", "科创50", "芯片", "新能源", "黄金", "国债"]
    )
    confidence: float = Field(
        description="打标置信度 0.0-1.0（名称越规范置信度越高）",
        ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="打标理由（一句话解释，如'名称含沪深300，判定为宽基指数'）"
    )


class BatchTagResult(BaseModel):
    """批量打标结果容器"""
    results: List[ETFTags] = Field(description="ETF 标签列表，顺序与输入一致")


# -------------------- 缓存机制 --------------------
def get_cache_key(ticker: str, name: str) -> str:
    """基于 ticker 和名称生成缓存键"""
    content = f"{ticker}_{name}"
    return hashlib.md5(content.encode()).hexdigest()

def load_cache(ticker: str, name: str) -> Optional[ETFTags]:
    """尝试从缓存读取标签"""
    cache_key = get_cache_key(ticker, name)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    if not cache_file.exists():
        return None
    
    try:
        # 检查有效期
        mtime = pd.Timestamp.fromtimestamp(cache_file.stat().st_mtime)
        if (pd.Timestamp.now() - mtime).total_seconds() > CACHE_TTL_HOURS * 3600:
            return None
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return ETFTags(**data)
    except Exception:
        return None

def save_cache(ticker: str, name: str, tags: ETFTags):
    """保存标签到缓存"""
    cache_key = get_cache_key(ticker, name)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(tags.dict(), f, ensure_ascii=False, indent=2)


# -------------------- LLM 打标逻辑 --------------------
def build_batch_prompt(etf_list: List[Dict]) -> str:
    """
    构建批量打标 Prompt
    ==================
    一次处理多只 ETF，减少 API 调用次数（省钱）。
    
    参数：
        etf_list: 字典列表，每个包含 ticker 和 name
    
    返回：
        格式化后的 Prompt 字符串
    """
    # 格式化 ETF 列表
    etf_texts = []
    for i, etf in enumerate(etf_list, 1):
        name = etf['name'][:MAX_ETF_NAME_LENGTH]  # 截断防 Token 爆炸
        ticker = etf['ticker']
        etf_texts.append(f"{i}. {ticker} | {name}")
    
    etf_block = "\n".join(etf_texts)
    
    parser = PydanticOutputParser(pydantic_object=BatchTagResult)
    
    prompt = f"""作为 ETF 分类专家，请为以下 {len(etf_list)} 只 ETF 打上三级标签。

【输入 ETF 列表】：
{etf_block}

【标签规则】：
- L1 资产大类：股票（权益类）、债券（固收类）、商品（黄金/原油等）、货币（货币基金）、跨境（港股/美股）
- L2 主题：宽基（大盘/中小盘指数）、行业（科技/消费/医药等）、策略（红利/低波动/价值等）、利率债、信用债、贵金属、能源
- L3 子主题：具体指数名称（沪深300、科创50）或细分行业（芯片、新能源、煤炭、银行）

【示例】：
- 510300.SH 沪深300ETF → L1:股票, L2:宽基, L3:沪深300
- 512480.SH 半导体ETF → L1:股票, L2:行业, L3:芯片
- 518880.SH 黄金ETF → L1:商品, L2:贵金属, L3:黄金
- 511010.SH 国债ETF → L1:债券, L2:利率债, L3:国债

【输出要求】：
严格返回 JSON 格式，包含 {len(etf_list)} 个结果，顺序与输入一致：
{parser.get_format_instructions()}

注意：
1. 必须严格按照输入顺序返回 {len(etf_list)} 个结果
2. 如果名称不明确（如"某某价值ETF"），根据常见命名规则推测
3. 跨境 ETF（含"恒生"、"纳指"、"标普"）L1 必须为"跨境"
4. 货币基金（含"货币"、"现金"）L1 必须为"货币"
"""
    return prompt

def tag_batch(etf_list: List[Dict]) -> List[Optional[ETFTags]]:
    """
    批量打标（核心函数）
    ===================
    使用 LLM 一次处理多只 ETF。
    
    返回：
        ETFTags 列表（与输入顺序一致），失败则为 None
    """
    if not OPENAI_API_KEY:
        logger.error("❌ 未找到 OPENAI_API_KEY")
        return [None] * len(etf_list)
    
    # 检查缓存（如果有缓存的直接返回，不打标）
    results = []
    uncached_etfs = []
    uncached_indices = []
    
    for i, etf in enumerate(etf_list):
        cached = load_cache(etf['ticker'], etf['name'])
        if cached:
            results.append(cached)
            logger.debug(f"💾 缓存命中: {etf['ticker']}")
        else:
            results.append(None)  # 占位
            uncached_etfs.append(etf)
            uncached_indices.append(i)
    
    # 如果全部命中缓存，直接返回
    if not uncached_etfs:
        return results
    
    # 对未缓存的调用 LLM
    try:
        llm = ChatOpenAI(
            model=OPENAI_MODEL_NAME,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0.1,  # 低温度（分类任务确定性高）
            max_tokens=2000,
            request_timeout=60
        )
        
        prompt = build_batch_prompt(uncached_etfs)
        parser = PydanticOutputParser(pydantic_object=BatchTagResult)
        
        logger.info(f"🤖 批量打标: {len(uncached_etfs)} 只 ETF (批次大小: {BATCH_SIZE})")
        
        # 调用 LLM
        response = llm.invoke(prompt)
        
        # 解析结果
        try:
            clean_content = response.content.replace("```json", "").replace("```", "")
            batch_result = parser.parse(clean_content)
            
            # 填充结果到对应位置
            for idx, tag_result in zip(uncached_indices, batch_result.results):
                results[idx] = tag_result
                # 保存缓存
                etf = etf_list[idx]
                save_cache(etf['ticker'], etf['name'], tag_result)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 解析 LLM 输出失败: {e}")
            # 失败时保持 None（后续会填充 Fallback）
            return results
            
    except Exception as e:
        logger.error(f"❌ LLM 调用失败: {e}")
        return results


def fallback_tag(etf_name: str) -> ETFTags:
    """
    Fallback 硬编码规则（LLM 失败时的安全网）
    ==========================================
    基于关键词的简单规则，确保系统不崩溃。
    这是最后的保险，正常情况下不应触发。
    """
    name = etf_name.lower()
    
    # 简单关键词匹配
    if any(x in name for x in ["沪深300", "上证50", "中证500", "创业板", "科创"]):
        return ETFTags(L1_asset_class="股票", L2_theme="宽基", L3_sub_theme="宽基指数", confidence=0.6, reasoning="基于关键词规则的Fallback")
    elif any(x in name for x in ["芯片", "半导体", "科技", "5g", "ai", "人工智能"]):
        return ETFTags(L1_asset_class="股票", L2_theme="行业", L3_sub_theme="科技", confidence=0.6, reasoning="基于关键词规则的Fallback")
    elif any(x in name for x in ["医药", "医疗", "生物"]):
        return ETFTags(L1_asset_class="股票", L2_theme="行业", L3_sub_theme="医药", confidence=0.6, reasoning="基于关键词规则的Fallback")
    elif any(x in name for x in ["新能源", "光伏", "碳中和"]):
        return ETFTags(L1_asset_class="股票", L2_theme="行业", L3_sub_theme="新能源", confidence=0.6, reasoning="基于关键词规则的Fallback")
    elif any(x in name for x in ["黄金", "贵金属"]):
        return ETFTags(L1_asset_class="商品", L2_theme="贵金属", L3_sub_theme="黄金", confidence=0.8, reasoning="基于关键词规则的Fallback")
    elif any(x in name for x in ["国债", "国开债"]):
        return ETFTags(L1_asset_class="债券", L2_theme="利率债", L3_sub_theme="国债", confidence=0.8, reasoning="基于关键词规则的Fallback")
    elif any(x in name for x in ["恒生", "纳指", "标普", "中概", "港股"]):
        return ETFTags(L1_asset_class="跨境", L2_theme="跨境权益", L3_sub_theme="港股/美股", confidence=0.7, reasoning="基于关键词规则的Fallback")
    else:
        return ETFTags(L1_asset_class="股票", L2_theme="其他", L3_sub_theme="其他", confidence=0.3, reasoning="无法识别，Fallback默认股票")


# -------------------- 主流程 --------------------
def process_etfs(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH):
    """
    ETF 打标主流程
    ==============
    读取原始 ETF 档案，批量打标，保存带标签的档案。
    """
    logger.info(f"🏷️  开始 ETF 主题映射...")
    
    # 读取输入
    if not input_path.exists():
        logger.error(f"❌ 未找到输入文件: {input_path}")
        return None
    
    df = pd.read_csv(input_path)
    logger.info(f"📊 读取 {len(df)} 只 ETF 基础信息")
    
    # 准备结果容器
    tags_list = []
    
    # 分批处理（控制成本和速率）
    total = len(df)
    for start_idx in range(0, total, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, total)
        batch = df.iloc[start_idx:end_idx]
        
        # 构建批次数据
        etf_list = [
            {"ticker": row["ticker"], "name": row["name"]}
            for _, row in batch.iterrows()
        ]
        
        # 调用 LLM 批量打标
        batch_tags = tag_batch(etf_list)
        
        # 检查失败并使用 Fallback 填充
        for i, (idx, row) in enumerate(batch.iterrows()):
            tag = batch_tags[i]
            if tag is None:
                logger.warning(f"⚠️  {row['ticker']} 打标失败，使用 Fallback")
                tag = fallback_tag(row["name"])
            tags_list.append(tag)
        
        # 进度显示
        logger.info(f"⏳ 进度: {end_idx}/{total} ({end_idx/total*100:.0f}%)")
        
        # 速率限制（批次间等待）
        if end_idx < total:
            time.sleep(RATE_LIMIT_DELAY)
    
    # 组装结果
    df["L1_asset_class"] = [t.L1_asset_class for t in tags_list]
    df["L2_theme"] = [t.L2_theme for t in tags_list]
    df["L3_sub_theme"] = [t.L3_sub_theme for t in tags_list]
    df["tag_confidence"] = [t.confidence for t in tags_list]
    df["tag_reasoning"] = [t.reasoning for t in tags_list]
    
    # 保存
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    # 统计报告
    stats = {
        "total": len(df),
        "by_l1": df["L1_asset_class"].value_counts().to_dict(),
        "by_l2": df["L2_theme"].value_counts().to_dict(),
        "high_confidence": (df["tag_confidence"] >= 0.8).sum(),
        "low_confidence": (df["tag_confidence"] < 0.6).sum()
    }
    
    logger.success(f"✅ 打标完成，保存至: {output_path}")
    logger.info(f"📊 统计:")
    logger.info(f"   总数: {stats['total']} 只")
    logger.info(f"   高置信度(≥0.8): {stats['high_confidence']} 只")
    logger.info(f"   资产分布: {stats['by_l1']}")
    
    # 展示样本
    print(f"\n📋 打标样本（前 5 只）:")
    print(df[["ticker", "name", "L1_asset_class", "L2_theme", "L3_sub_theme", "tag_confidence"]].head().to_string())
    
    return df


# ==================== 测试入口 ====================
if __name__ == "__main__":
    import sys
    
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    
    print("=" * 60)
    print("🏷️  ETF 主题映射模块 - AI 自动打标")
    print("=" * 60)
    print("💡 工作流程:")
    print("   1. 读取 etf_basic.csv（100 只 ETF）")
    print(f"   2. 分批打标（每批 {BATCH_SIZE} 只，避免 Token 爆炸）")
    print("   3. 使用 LLM 分析名称语义（如'科创板50'→科技-芯片-科创50）")
    print("   4. 保存 processed_etf_basic.csv（带三级标签）")
    print("=" * 60)
    
    # 检查输入文件
    if not INPUT_PATH.exists():
        print(f"\n❌ 未找到输入文件: {INPUT_PATH}")
        print("💡 请先运行: python scripts/generate_mock_etf.py")
        sys.exit(1)
    
    # 执行打标
    result_df = process_etfs()
    
    if result_df is not None:
        print("\n" + "=" * 60)
        print("✅ ETF 主题映射完成！")
        print("📁 输出文件: data/etf/processed_etf_basic.csv")
        print("\n🎯 标签体系验证:")
        print("   L1 资产大类:", result_df["L1_asset_class"].unique().tolist())
        print("   L2 主题:", result_df["L2_theme"].unique().tolist()[:10], "...")
        print("\n💡 下一步: 投资组合构建模块")
        print("   将根据这些标签匹配政策主题（如'人工智能'→'L3=芯片'的 ETF）")
        print("=" * 60)