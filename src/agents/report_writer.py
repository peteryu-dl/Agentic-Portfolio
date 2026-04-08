#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告生成模块 (report_writer.py) - 投资报告撰写员
=================================================
功能：自动生成投资策略报告（Markdown 格式），解释"为什么买这些 ETF"。

工作流程（像写研报一样）：
1. 【摘要】执行摘要：当前周期 + 核心观点 + 收益率预期
2. 【宏观】分析经济周期（美林时钟位置、增长/通胀趋势）
3. 【政策】解读政策信号（Top 5 主题、市场情绪、风险提示）
4. 【持仓】展示投资组合（大类配比、具体标的、选股逻辑）
5. 【归因】解释配置逻辑（为什么买这只 ETF，风险在哪）

输入：
- macro_state: 宏观状态（周期、评分、动量）
- policy_signal: 政策信号（Top 5 主题、置信度）
- portfolio: 投资组合（持仓列表、权重、逻辑）

输出：
- 月度投资报告（Markdown 格式，保存到 reports/ 目录）
- 包含文字分析 + 数据表格 + 风险提示

技术特点：
- 使用 LLM 生成文字部分（宏观分析、政策解读），确保可读性
- 使用模板生成数据表格（确保格式统一）
- Fallback 机制：LLM 失败时生成标准模板报告（确保不崩盘）
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field
from loguru import logger

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

# 尝试导入上游模块类型
try:
    from src.core.macro_regime import MacroState
    from src.agents.policy_interpreter import PolicySignal
    from src.core.portfolio_builder import Portfolio, PortfolioItem
    IMPORT_SUCCESS = True
except ImportError:
    IMPORT_SUCCESS = False
    # 占位符类型
    MacroState = dict
    PolicySignal = dict
    Portfolio = dict


# -------------------- 配置区 --------------------
# API 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

# 报告配置
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 成本控制（报告生成只需一次调用，但限制长度）
MAX_REPORT_TOKENS = 2000  # 控制生成长度（约 1500 汉字）
TEMPERATURE = 1           # 略高于分析模块（需要一定文采）


# -------------------- 数据模型 --------------------
class ReportSection(BaseModel):
    """报告章节"""
    title: str = Field(description="章节标题")
    content: str = Field(description="章节正文（Markdown 格式）")


class InvestmentReport(BaseModel):
    """
    结构化投资报告
    ==============
    用于 LLM 生成时的结构约束（确保输出完整）。
    """
    executive_summary: str = Field(description="执行摘要（3-4 句话概括核心观点）")
    macro_analysis: str = Field(description="宏观分析：当前周期位置、增长通胀趋势")
    policy_interpretation: str = Field(description="政策解读：推荐主题及理由")
    portfolio_strategy: str = Field(description="组合策略：配比逻辑、标的选择理由")
    risk_warning: str = Field(description="风险提示：潜在风险及应对建议")
    market_outlook: str = Field(description="市场展望：短期（1个月）和中期（3个月）看法")


# -------------------- 报告构建逻辑 --------------------
class ReportWriter:
    """
    投资报告撰写器
    =============
    整合所有信号，生成专业 Markdown 报告。
    """
    
    def __init__(self):
        self.llm = None
        if OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                model=OPENAI_MODEL_NAME,
                api_key=OPENAI_API_KEY,
                base_url=OPENAI_BASE_URL,
                temperature=TEMPERATURE,
                max_tokens=MAX_REPORT_TOKENS
            )
    
    def _build_prompt(self, 
                     macro_state: MacroState, 
                     policy_signal: PolicySignal, 
                     portfolio: Portfolio,
                     date: str) -> str:
        """
        构建报告生成 Prompt
        ==================
        提供所有必要上下文，让 LLM 写出专业报告。
        """
        # 提取宏观信息
        if isinstance(macro_state, dict):
            regime = macro_state.get("regime", "unknown")
            score = macro_state.get("equity_friendly_score", 0.5)
            growth_mom = macro_state.get("growth_momentum", 0)
            infl_mom = macro_state.get("inflation_momentum", 0)
        else:
            regime = macro_state.regime
            score = macro_state.equity_friendly_score
            growth_mom = macro_state.growth_momentum
            infl_mom = macro_state.inflation_momentum
        
        # 提取政策信息
        if isinstance(policy_signal, dict):
            themes = policy_signal.get("top_5_themes", [])
            confidences = policy_signal.get("confidences", [])
            policy_summary = policy_signal.get("analysis_summary", "")
        else:
            themes = policy_signal.top_5_themes
            confidences = policy_signal.confidences
            policy_summary = policy_signal.analysis_summary
        
        # 提取持仓信息
        if isinstance(portfolio, dict):
            allocation = portfolio.get("allocation", {})
            items = portfolio.get("items", [])
            total_etfs = portfolio.get("total_etfs", 0)
        else:
            allocation = portfolio.allocation
            items = portfolio.items
            total_etfs = portfolio.total_etfs
        
        # 格式化持仓为表格文本
        holdings_text = "| 代码 | 名称 | 权重 | 资产类别 | 逻辑 |\n"
        holdings_text += "|------|------|------|----------|------|\n"
        
        # 只展示前 10 只（避免 Prompt 过长）
        for item in items[:10]:
            if isinstance(item, dict):
                ticker = item.get("ticker", "")
                name = item.get("name", "")
                weight = item.get("weight", 0)
                asset = item.get("asset_class", "")
                logic = item.get("selection_logic", "")
            else:
                ticker = item.ticker
                name = item.name
                weight = item.weight
                asset = item.asset_class
                logic = item.selection_logic
            
            holdings_text += f"| {ticker} | {name} | {weight:.2%} | {asset} | {logic[:20]} |\n"
        
        # 格式化主题
        themes_text = ", ".join([f"{t}({c:.0%})" for t, c in zip(themes[:5], confidences[:5])])
        
        prompt = f"""作为资深投资策略师，请基于以下数据撰写一份专业的月度投资报告。

【报告日期】{date}

【宏观数据】
- 经济周期：{regime}（复苏/过热/滞胀/衰退）
- 权益友好度：{score:.0%}
- 增长动量：{growth_mom:+.1f}%
- 通胀动量：{infl_mom:+.1f}%

【政策信号】
- 推荐主题：{themes_text}
- 政策摘要：{policy_summary}

【投资组合】
- 大类配置：股票{allocation.get('股票', 0):.0%} / 债券{allocation.get('债券', 0):.0%} / 商品{allocation.get('商品', 0):.0%}
- 持仓数量：{total_etfs} 只 ETF
- 前十大持仓：
{holdings_text}

【写作要求】
1. 执行摘要：3-4 句话概括核心观点（周期判断 + 配置建议）
2. 宏观分析：解释当前处于美林时钟哪个象限，为什么
3. 政策解读：分析推荐主题的内在逻辑，引用政策关键词
4. 组合策略：解释股债商配比依据，以及 ETF 选择逻辑
5. 风险提示：列出 2-3 个主要风险（如政策转向、流动性收紧）
6. 市场展望：短期（1 个月）和中期（3 个月）看法

【格式要求】
- 使用 Markdown 格式
- 章节使用 ## 二级标题
- 关键数据使用 **粗体** 强调
- 总字数控制在 800-1200 字（专业简洁）
- 语气专业、客观、有说服力（像券商研报）

请直接输出完整的 Markdown 格式报告（包含标题和各个章节）：
"""
        return prompt
    
    def generate_with_llm(self, 
                         macro_state: MacroState, 
                         policy_signal: PolicySignal, 
                         portfolio: Portfolio,
                         date: str) -> Optional[str]:
        """
        使用 LLM 生成报告正文
        ====================
        """
        if not self.llm:
            logger.warning("⚠️  LLM 未配置，无法生成 AI 报告")
            return None
        
        try:
            prompt = self._build_prompt(macro_state, policy_signal, portfolio, date)
            
            logger.info(f"🤖 调用 LLM 生成投资报告...")
            response = self.llm.invoke(prompt)
            
            content = response.content
            
            # 清理可能的 markdown 代码块标记
            content = content.replace("```markdown", "").replace("```", "").strip()
            
            logger.success(f"✅ LLM 报告生成完成，长度 {len(content)} 字符")
            return content
            
        except Exception as e:
            logger.error(f"❌ LLM 报告生成失败: {e}")
            return None
    
    def generate_template_report(self,
                                macro_state: MacroState, 
                                policy_signal: PolicySignal, 
                                portfolio: Portfolio,
                                date: str) -> str:
        """
        模板报告（Fallback）
        ===================
        LLM 失败时使用，确保总有输出。
        基于预定义模板填空，保证结构完整。
        """
        # 提取数据（兼容 dict 和 dataclass）
        if isinstance(macro_state, dict):
            regime = macro_state.get("regime", "unknown")
            score = macro_state.get("equity_friendly_score", 0.5)
        else:
            regime = macro_state.regime
            score = macro_state.equity_friendly_score
        
        if isinstance(policy_signal, dict):
            themes = policy_signal.get("top_5_themes", [])
        else:
            themes = policy_signal.top_5_themes
        
        if isinstance(portfolio, dict):
            allocation = portfolio.get("allocation", {})
            items = portfolio.get("items", [])
            date_str = portfolio.get("date", date)
        else:
            allocation = portfolio.allocation
            items = portfolio.items
            date_str = portfolio.date
        
        regime_cn = {
            "recovery": "复苏期", "overheat": "过热期",
            "stagflation": "滞胀期", "recession": "衰退期"
        }.get(regime, "调整期")
        
        # 构建模板
        report = f"""# 月度投资策略报告

**报告日期：** {date_str}  
**策略状态：** {regime_cn} | 权益友好度 **{score:.0%}**

---

## 执行摘要

当前经济处于**{regime_cn}**，权益资产吸引力{score:.0%}。基于宏观周期模型，建议配置**股票{allocation.get('股票', 0):.0%}、债券{allocation.get('债券', 0):.0%}、商品{allocation.get('商品', 0):.0%}**。

政策层面重点关注**{', '.join(themes[:3])}**等主题。组合通过分散配置宽基指数与行业主题，力争在控制回撤的前提下获取Beta与Alpha收益。

---

## 宏观分析

### 周期定位
根据 PMI、CPI、PPI 等先行指标，当前经济呈现以下特征：
- **增长维度：** {"扩张" if score > 0.5 else "收缩"}动能{"增强" if regime == "recovery" else "趋缓"}
- **通胀维度：** 物价水平{"温和" if regime in ["recovery", "recession"] else "偏高"}
- **美林时钟位置：** {regime_cn}（{"股债双牛" if regime == "recovery" else "股熊债牛" if regime == "recession" else "股牛商品牛" if regime == "overheat" else "现金为王"}）

### 关键指标
- 权益友好度评分：**{score:.2f}**（0-1 区间）
- 建议超配/低配：股票 **{"+" if score > 0.6 else "-"}{abs(score-0.5)*20:.0f}%**

---

## 政策解读

本月政策信号聚焦以下方向：

{chr(10).join([f"{i+1}. **{t}**：政策支持力度较高，建议重点关注相关产业链配置机会" for i, t in enumerate(themes[:5])])}

---

## 投资组合

### 大类资产配置
- **股票：** {allocation.get('股票', 0):.1%}（宽基{STOCK_STRUCTURE.get('broad_based', 0.6):.0%} + 主题{STOCK_STRUCTURE.get('thematic', 0.4):.0%}）
- **债券：** {allocation.get('债券', 0):.1%}（利率债为主，防御配置）
- **商品：** {allocation.get('商品', 0):.1%}（黄金等贵金属，对冲通胀）

### 重点持仓

{chr(10).join([f"- **{item.get('ticker', item.ticker)}** {item.get('weight', item.weight):.2%}：{item.get('name', item.name)}（{item.get('selection_logic', item.selection_logic)[:30]}）" for item in items[:5]])}

*完整持仓清单见附录*

---

## 风险提示

1. **政策风险：** 宏观政策转向可能导致主题投资逻辑失效
2. **流动性风险：** 部分行业 ETF 成交清淡，大额进出可能冲击成本较高
3. **模型风险：** 宏观周期判断基于历史数据，无法预测黑天鹅事件

---

## 市场展望

**短期（1 个月）：** 维持{regime_cn}判断，关注{"PMI 能否站稳 50 荣枯线" if regime in ["recovery", "recession"] else "CPI 是否突破 3% 警戒线"}。

**中期（3 个月）：** 若{"增长动能持续修复" if regime == "recovery" else "通胀有效回落"}，建议逐步{"增配权益" if regime in ["recovery", "recession"] else "减配商品"}。

---

*本报告由 Q-Macro 策略系统自动生成，仅供参考，不构成投资建议。*  
*数据截止日期：{date_str}*
"""
        logger.info("📝 生成模板报告（LLM Fallback）")
        return report
    
    def write(self,
             macro_state: MacroState, 
             policy_signal: PolicySignal, 
             portfolio: Portfolio,
             date: str = None) -> Path:
        """
        主入口：生成并保存报告
        ======================
        优先尝试 LLM 生成，失败则使用模板。
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"📝 开始生成投资报告: {date}")
        
        # 尝试 LLM 生成
        content = self.generate_with_llm(macro_state, policy_signal, portfolio, date)
        
        # 如果 LLM 失败，使用模板
        if content is None:
            content = self.generate_template_report(macro_state, policy_signal, portfolio, date)
            report_type = "template"
        else:
            # 为 LLM 生成内容添加标准头部和尾部
            header = f"# 月度投资策略报告\n\n**报告日期：** {date}  \n**生成方式：** AI 智能生成\n\n---\n\n"
            footer = f"\n\n---\n\n*本报告由 Q-Macro AI 策略系统自动生成*  \n*数据截止日期：{date}*"
            content = header + content + footer
            report_type = "ai"
        
        # 保存文件
        filename = f"{date}-report.md"
        filepath = REPORTS_DIR / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.success(f"✅ 报告已保存: {filepath} ({report_type} 模式)")
        
        # 同时保存一份最新的（方便查看）
        latest_path = REPORTS_DIR / "latest-report.md"
        with open(latest_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filepath


# ==================== 测试入口 ====================
if __name__ == "__main__":
    import sys
    
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    
    print("=" * 60)
    print("📝 报告生成模块 - 投资月报撰写")
    print("=" * 60)
    print("💡 功能：将宏观状态、政策信号、持仓组合转化为可读性强的 Markdown 报告")
    print("=" * 60)
    
    # Mock 数据（如果上游模块未导入）
    if not IMPORT_SUCCESS:
        print("\n⚠️  使用 Mock 数据进行测试...")
        
        class MockMacro:
            regime = "recovery"
            equity_friendly_score = 0.75
            growth_momentum = 2.5
            inflation_momentum = -0.5
        
        class MockPolicy:
            top_5_themes = ["人工智能", "新能源", "消费升级"]
            confidences = [0.92, 0.85, 0.78]
            analysis_summary = "积极；推荐人工智能、新能源"
        
        class MockItem:
            def __init__(self, ticker, name, weight, asset, logic):
                self.ticker = ticker
                self.name = name
                self.weight = weight
                self.asset_class = asset
                self.selection_logic = logic
        
        class MockPortfolio:
            date = "2025-03-27"
            allocation = {"股票": 0.75, "债券": 0.15, "商品": 0.10}
            total_etfs = 8
            items = [
                MockItem("510300.SH", "沪深300ETF", 0.15, "股票", "宽基Beta"),
                MockItem("510500.SH", "中证500ETF", 0.12, "股票", "宽基Beta"),
                MockItem("512480.SH", "半导体ETF", 0.10, "股票", "政策主题：人工智能"),
                MockItem("515030.SH", "新能源车ETF", 0.08, "股票", "政策主题：新能源"),
                MockItem("518880.SH", "黄金ETF", 0.10, "商品", "通胀对冲"),
                MockItem("511010.SH", "国债ETF", 0.15, "债券", "防御配置"),
            ]
        
        macro = MockMacro()
        policy = MockPolicy()
        portfolio = MockPortfolio()
    else:
        # 尝试从上游模块加载真实数据（如果文件存在）
        try:
            from src.core.macro_regime import detect_macro_regime
            from src.agents.policy_interpreter import analyze_policy
            from src.core.portfolio_builder import PortfolioBuilder, save_portfolio
            
            print("\n📥 尝试加载真实数据...")
            # 这里简化处理，实际应调用各模块主函数
            macro = detect_macro_regime("2025-03-27")
            policy = analyze_policy(target_date="2025-03-27")
            # portfolio 需要 builder，这里用 mock
            portfolio = None
        except Exception as e:
            print(f"⚠️  加载真实数据失败: {e}")
            print("   使用 Mock 数据继续测试")
            portfolio = None
        
        if portfolio is None:
            # 创建简单 mock
            class MockPortfolio:
                date = "2025-03-27"
                allocation = {"股票": 0.70, "债券": 0.20, "商品": 0.10}
                total_etfs = 5
                items = []
    
    # 执行报告生成
    writer = ReportWriter()
    
    try:
        report_path = writer.write(macro, policy, portfolio, date="2025-03-27")
        
        print(f"\n✅ 报告生成成功！")
        print(f"📄 文件路径: {report_path}")
        print(f"📂 查看命令: cat {report_path}" if os.name != 'nt' else f"type {report_path}")
        
        # 打印报告预览（前 20 行）
        print(f"\n📋 报告预览（前 20 行）:")
        with open(report_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:20]
            print(''.join(lines))
            print("... [后续内容省略] ...")
        
    except Exception as e:
        logger.error(f"❌ 报告生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎯 下一步：运行完整流水线")
    print("   python Q-Macro.py --date 2025-03-27 --full")
    print("=" * 60)