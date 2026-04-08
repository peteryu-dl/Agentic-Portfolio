#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
投资组合构建模块 (portfolio_builder.py) - 资产配置大厨
=======================================================
功能：整合宏观状态、政策信号、市场校准结果，构建最终 ETF 投资组合。

工作流程（像做菜一样）：
1. 【备菜】读取带标签的 ETF 池（processed_etf_basic.csv）
2. 【调味】根据宏观状态确定基础配比（复苏期股70债20商10）
3. 【微调】根据 equity_friendly_score 调整股票权重（±10%）
4. 【配菜】股票内部：宽基 ETF 60% + 政策主题 ETF 40%
5. 【筛选】应用市场校准结果（剔除僵尸，拥挤 ETF 权重×0.75）
6. 【装盘】归一化权重，生成最终投资组合

输入：
- macro_state: 来自宏观状态识别（regime, equity_friendly_score）
- policy_signal: 来自政策解读（top_5_themes）
- market_condition: 来自市场校准（liquid_etfs, crowded_adjustments）

输出：
- portfolio: 最终持仓（ticker, weight, logic）
- allocation: 大类资产配比（股票xx%, 债券xx%, 商品xx%）
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from loguru import logger

# 导入上游模块（用于类型提示和联动）
try:
    from src.core.macro_regime import MacroState
    from src.agents.policy_interpreter import PolicySignal
    from src.core.market_calibration import MarketCondition
    IMPORT_SUCCESS = True
except ImportError:
    IMPORT_SUCCESS = False
    # 如果导入失败，定义占位符类型避免报错
    MacroState = dict
    PolicySignal = dict
    MarketCondition = dict


# -------------------- 配置区 --------------------
# 四象限基础配比（股/债/商）
REGIME_BASE_WEIGHTS = {
    "recovery":   {"stock": 0.70, "bond": 0.20, "commodity": 0.10},  # 复苏：猛买股票
    "overheat":   {"stock": 0.60, "bond": 0.10, "commodity": 0.30},  # 过热：减股加商品
    "stagflation":{"stock": 0.30, "bond": 0.30, "commodity": 0.40},  # 滞胀：均衡防守
    "recession":  {"stock": 0.40, "bond": 0.50, "commodity": 0.10},  # 衰退：重仓债券
}

# 股票内部配置比例
STOCK_STRUCTURE = {
    "broad_based": 0.60,    # 宽基 ETF（沪深300等）提供 Beta
    "thematic": 0.40        # 主题 ETF（芯片/新能源等）提供 Alpha
}

# 最小权重阈值（低于此值的持仓被剔除，避免碎单）
MIN_WEIGHT_THRESHOLD = 0.01  # 1%


# -------------------- 数据模型 --------------------
class PortfolioItem(BaseModel):
    """
    单个持仓项
    ==========
    包含标的、权重、配置逻辑（为什么选它）。
    """
    ticker: str = Field(description="ETF 代码（如 510300.SH）")
    name: str = Field(description="ETF 名称")
    weight: float = Field(description="最终权重（0-1，如 0.15 表示 15%）", ge=0, le=1)
    asset_class: str = Field(description="资产大类：股票/债券/商品/跨境")
    theme: str = Field(description="主题分类：宽基/行业/策略/利率债/贵金属等")
    sub_theme: str = Field(description="子主题：沪深300/芯片/黄金等")
    selection_logic: str = Field(description="选股逻辑说明")
    crowded_adjustment: float = Field(default=1.0, description="拥挤度调整因子（1.0 正常，0.75 拥挤）")

class Portfolio(BaseModel):
    """
    完整投资组合
    ============
    包含持仓列表、大类资产配置、元数据。
    """
    date: str = Field(description="组合日期")
    items: List[PortfolioItem] = Field(description="持仓列表")
    allocation: Dict[str, float] = Field(description="大类资产配置比例（股票/债券/商品）")
    macro_regime: str = Field(description="当前宏观周期")
    policy_themes: List[str] = Field(description="政策推荐主题")
    total_etfs: int = Field(description="持仓 ETF 数量")
    total_weight: float = Field(description="总权重（应为 1.0）")
    
    def to_json(self, path: Path):
        """保存为 JSON"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.dict(), f, ensure_ascii=False, indent=2)
    
    def to_dataframe(self) -> pd.DataFrame:
        """转为 DataFrame 方便查看"""
        return pd.DataFrame([item.dict() for item in self.items])


# -------------------- 核心构建逻辑 --------------------
class PortfolioBuilder:
    """
    投资组合构建器
    =============
    整合所有信号，生成最终持仓。
    """
    
    def __init__(self, 
                 etf_data_path: str = "data/etf/processed_etf_basic.csv"):
        """
        初始化
        参数：
            etf_data_path: 带标签的 ETF 数据路径
        """
        self.etf_df = self._load_etfs(etf_data_path)
        self.selected_etfs = []  # 中间结果存储
        
    def _load_etfs(self, path: str) -> pd.DataFrame:
        """加载带标签的 ETF 数据"""
        file_path = Path(path)
        if not file_path.exists():
            logger.error(f"❌ 未找到 ETF 数据: {file_path}")
            raise FileNotFoundError(f"请先运行 theme_mapper.py 生成 processed_etf_basic.csv")
        
        df = pd.read_csv(file_path)
        logger.info(f"📊 加载 ETF 池: {len(df)} 只（已带标签）")
        return df
    
    def _get_base_allocation(self, macro_state: MacroState) -> Dict[str, float]:
        """
        步骤 1：确定大类资产基础配比
        ============================
        根据宏观状态四象限，获取股债商基础权重。
        """
        # 处理两种输入类型（dataclass 或 dict）
        if isinstance(macro_state, dict):
            regime = macro_state.get("regime", "recession")
            score = macro_state.get("equity_friendly_score", 0.5)
        else:
            regime = macro_state.regime
            score = macro_state.equity_friendly_score
        
        base = REGIME_BASE_WEIGHTS.get(regime, REGIME_BASE_WEIGHTS["recession"]).copy()
        
        # 根据 equity_friendly_score 动态微调股票权重（±10%）
        # 原理：如果宏观状态很强（score=0.9），股票再加 10%；如果很弱（score=0.2），股票减 10%
        adjustment = (score - 0.5) * 0.2  # -0.1 到 +0.1
        
        base["stock"] = np.clip(base["stock"] + adjustment, 0.2, 0.8)
        # 债券和商品相应调整（保持总和为 1）
        remaining = 1.0 - base["stock"]
        base["bond"] = remaining * (base["bond"] / (base["bond"] + base["commodity"]))
        base["commodity"] = remaining - base["bond"]
        
        logger.info(f"🎯 基础配比（{regime}）: 股{base['stock']:.0%} 债{base['bond']:.0%} 商{base['commodity']:.0%}")
        logger.info(f"   动态调整: {'+' if adjustment>0 else ''}{adjustment:.1%} (基于评分 {score:.2f})")
        
        return base
    
    def _select_stock_etfs(self, 
                          stock_weight: float,
                          policy_signal: PolicySignal,
                          market_condition: MarketCondition) -> List[PortfolioItem]:
        """
        步骤 2：选择股票 ETF（分层配置）
        =================================
        股票内部：60% 宽基 + 40% 政策主题
        
        逻辑：
        1. 从 liquid_etfs 中筛选出股票类（L1=股票或跨境）
        2. 宽基部分（60%）：选择规模最大、流动性最好的 2-3 只宽基（如沪深300、中证500）
        3. 主题部分（40%）：匹配政策推荐的 Top 5 主题，各选 1-2 只代表性 ETF
        4. 应用拥挤度调整（拥挤的 ETF 权重 ×0.75）
        """
        items = []
        
        # 处理输入类型
        if isinstance(policy_signal, dict):
            themes = policy_signal.get("top_5_themes", ["宽基指数", "科技", "消费"])
        else:
            themes = policy_signal.top_5_themes
        
        if isinstance(market_condition, dict):
            liquid_etfs = market_condition.get("liquid_etfs", [])
            crowded_adj = market_condition.get("crowded_adjustments", {})
        else:
            liquid_etfs = market_condition.liquid_etfs
            crowded_adj = market_condition.crowded_adjustments
        
        # 筛选合格的股票类 ETF（在 liquid_etfs 列表中且是股票或跨境）
        eligible = self.etf_df[self.etf_df["ticker"].isin(liquid_etfs)]
        stock_etfs = eligible[eligible["L1_asset_class"].isin(["股票", "跨境"])]
        
        if stock_etfs.empty:
            logger.warning("⚠️  无合格股票 ETF，使用宽基 Fallback")
            stock_etfs = eligible  # 放宽条件
        
        # 2.1 宽基部分（60% 的股票权重）
        broad_weight_total = stock_weight * STOCK_STRUCTURE["broad_based"]
        broad_etfs = stock_etfs[stock_etfs["L2_theme"] == "宽基"].head(3)  # 选前3只最大的宽基
        
        if not broad_etfs.empty:
            # 等权分配（或按规模加权，这里简化用等权）
            weight_per_broad = broad_weight_total / len(broad_etfs)
            
            for _, etf in broad_etfs.iterrows():
                ticker = etf["ticker"]
                # 应用拥挤度调整
                adj_factor = crowded_adj.get(ticker, 1.0)
                final_weight = weight_per_broad * adj_factor
                
                items.append(PortfolioItem(
                    ticker=ticker,
                    name=etf["name"],
                    weight=final_weight,
                    asset_class=etf["L1_asset_class"],
                    theme="宽基",
                    sub_theme=etf.get("L3_sub_theme", "宽基指数"),
                    selection_logic=f"宽基Beta（{etf['L3_sub_theme']}）提供市场暴露",
                    crowded_adjustment=adj_factor
                ))
        
        # 2.2 主题部分（40% 的股票权重）- 匹配政策主题
        thematic_weight_total = stock_weight * STOCK_STRUCTURE["thematic"]
        
        # 为每个政策主题寻找匹配的 ETF
        matched_etfs = []
        for theme in themes[:5]:  # Top 5 主题
            # 模糊匹配：L3_sub_theme 或 name 包含主题关键词
            # 简单实现：查找 L2_theme=行业 且 name 包含主题关键词的 ETF
            # 实际可用更复杂的语义匹配，这里用关键词示例
            matches = stock_etfs[
                (stock_etfs["L2_theme"] == "行业") & 
                (stock_etfs["name"].str.contains(theme[:2], na=False))  # 简化的关键词匹配
            ]
            
            if matches.empty:
                # 如果没精确匹配，选该主题下的任意一只
                matches = stock_etfs[stock_etfs["L2_theme"] == "行业"].head(1)
            
            if not matches.empty and matches.iloc[0]["ticker"] not in [e.ticker for e in matched_etfs]:
                matched_etfs.append(matches.iloc[0])
        
        # 去重后分配权重
        if matched_etfs:
            weight_per_theme = thematic_weight_total / len(matched_etfs)
            
            for etf in matched_etfs:
                ticker = etf["ticker"]
                adj_factor = crowded_adj.get(ticker, 1.0)
                
                items.append(PortfolioItem(
                    ticker=ticker,
                    name=etf["name"],
                    weight=weight_per_theme * adj_factor,
                    asset_class=etf["L1_asset_class"],
                    theme="行业",
                    sub_theme=etf.get("L3_sub_theme", theme),
                    selection_logic=f"政策主题匹配：{theme}",
                    crowded_adjustment=adj_factor
                ))
        
        return items
    
    def _select_bond_and_commodity(self, 
                                   bond_weight: float, 
                                   commodity_weight: float,
                                   market_condition: MarketCondition) -> List[PortfolioItem]:
        """
        步骤 3：选择债券和商品 ETF
        ==========================
        简单策略：各选 1-2 只代表性的。
        """
        items = []
        
        if isinstance(market_condition, dict):
            liquid_etfs = market_condition.get("liquid_etfs", [])
            crowded_adj = market_condition.get("crowded_adjustments", {})
        else:
            liquid_etfs = market_condition.liquid_etfs
            crowded_adj = market_condition.crowded_adjustments
        
        eligible = self.etf_df[self.etf_df["ticker"].isin(liquid_etfs)]
        
        # 债券部分
        if bond_weight > 0:
            bond_etfs = eligible[eligible["L1_asset_class"] == "债券"].head(2)
            if not bond_etfs.empty:
                weight_per_bond = bond_weight / len(bond_etfs)
                
                for _, etf in bond_etfs.iterrows():
                    ticker = etf["ticker"]
                    adj = crowded_adj.get(ticker, 1.0)
                    
                    items.append(PortfolioItem(
                        ticker=ticker,
                        name=etf["name"],
                        weight=weight_per_bond * adj,
                        asset_class="债券",
                        theme=etf.get("L2_theme", "利率债"),
                        sub_theme=etf.get("L3_sub_theme", "国债"),
                        selection_logic="防御性配置（债券）",
                        crowded_adjustment=adj
                    ))
        
        # 商品部分
        if commodity_weight > 0:
            comm_etfs = eligible[eligible["L1_asset_class"] == "商品"].head(2)
            if comm_etfs.empty:
                # 如果没有商品 ETF，权重转移给债券（保守处理）
                logger.warning("⚠️  无商品 ETF，权重转移至债券")
                # 实际实现中需要重新平衡，这里简化
            else:
                weight_per_comm = commodity_weight / len(comm_etfs)
                
                for _, etf in comm_etfs.iterrows():
                    ticker = etf["ticker"]
                    adj = crowded_adj.get(ticker, 1.0)
                    
                    items.append(PortfolioItem(
                        ticker=ticker,
                        name=etf["name"],
                        weight=weight_per_comm * adj,
                        asset_class="商品",
                        theme=etf.get("L2_theme", "贵金属"),
                        sub_theme=etf.get("L3_sub_theme", "黄金"),
                        selection_logic="通胀对冲（商品）",
                        crowded_adjustment=adj
                    ))
        
        return items
    
    def build(self,
             macro_state: MacroState,
             policy_signal: PolicySignal,
             market_condition: MarketCondition,
             target_date: str = None) -> Portfolio:
        """
        主构建函数（大厨炒菜）
        ======================
        整合所有步骤，生成最终投资组合。
        """
        if target_date is None:
            target_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"🚀 开始构建投资组合，日期: {target_date}")
        
        # 步骤 1：确定大类配比
        allocation = self._get_base_allocation(macro_state)
        
        # 步骤 2：选择股票 ETF（宽基60%+主题40%）
        stock_items = self._select_stock_etfs(
            allocation["stock"], policy_signal, market_condition
        )
        
        # 步骤 3：选择债券和商品 ETF
        other_items = self._select_bond_and_commodity(
            allocation["bond"], allocation["commodity"], market_condition
        )
        
        # 合并所有持仓
        all_items = stock_items + other_items
        
        # 步骤 4：归一化（确保权重和为 1）
        total_raw_weight = sum(item.weight for item in all_items)
        
        if total_raw_weight == 0:
            logger.error("❌ 总权重为 0，构建失败")
            raise ValueError("无法构建组合：无合格 ETF")
        
        # 归一化并剔除碎单（<1%）
        normalized_items = []
        for item in all_items:
            normalized_weight = item.weight / total_raw_weight
            
            if normalized_weight >= MIN_WEIGHT_THRESHOLD:
                item.weight = round(normalized_weight, 4)  # 保留4位小数
                normalized_items.append(item)
        
        # 最终检查（因剔除碎单可能导致总权重略≠1，再次归一化）
        final_total = sum(item.weight for item in normalized_items)
        for item in normalized_items:
            item.weight = round(item.weight / final_total, 4)
        
        # 获取政策主题（用于记录）
        if isinstance(policy_signal, dict):
            themes = policy_signal.get("top_5_themes", [])
        else:
            themes = policy_signal.top_5_themes
        
        if isinstance(macro_state, dict):
            regime = macro_state.get("regime", "unknown")
        else:
            regime = macro_state.regime
        
        portfolio = Portfolio(
            date=target_date,
            items=normalized_items,
            allocation={
                "股票": round(allocation["stock"], 2),
                "债券": round(allocation["bond"], 2),
                "商品": round(allocation["commodity"], 2)
            },
            macro_regime=regime,
            policy_themes=themes,
            total_etfs=len(normalized_items),
            total_weight=round(sum(item.weight for item in normalized_items), 2)
        )
        
        logger.success(f"✅ 组合构建完成: {portfolio.total_etfs} 只 ETF，总权重 {portfolio.total_weight}")
        return portfolio


# -------------------- 输出保存 --------------------
def save_portfolio(portfolio: Portfolio, output_dir: str = "portfolios"):
    """
    保存投资组合到文件
    ==================
    保存为 JSON（机器可读）和 CSV（人可读）。
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # JSON 格式（完整数据）
    json_file = output_path / f"{portfolio.date}.json"
    portfolio.to_json(json_file)
    
    # CSV 格式（简洁表格）
    csv_file = output_path / f"{portfolio.date}.csv"
    df = portfolio.to_dataframe()
    df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    
    # 文本报告（给人看的摘要）
    report_file = output_path / f"{portfolio.date}_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"投资组合报告 - {portfolio.date}\n")
        f.write("=" * 50 + "\n")
        f.write(f"宏观周期: {portfolio.macro_regime}\n")
        f.write(f"政策主题: {', '.join(portfolio.policy_themes)}\n")
        f.write(f"资产配置: 股票{portfolio.allocation['股票']:.0%} 债券{portfolio.allocation['债券']:.0%} 商品{portfolio.allocation['商品']:.0%}\n")
        f.write("-" * 50 + "\n")
        f.write("持仓明细:\n")
        for item in portfolio.items:
            f.write(f"{item.ticker:12s} {item.weight:6.2%} {item.name:20s} ({item.selection_logic})\n")
    
    logger.info(f"💾 组合已保存:")
    logger.info(f"   JSON: {json_file}")
    logger.info(f"   CSV: {csv_file}")
    logger.info(f"   报告: {report_file}")


# ==================== 测试入口 ====================
if __name__ == "__main__":
    import sys
    
    # 尝试导入上游模块用于测试
    try:
        from src.core.macro_regime import MacroState, REGIME_BASE_WEIGHTS
        from src.agents.policy_interpreter import PolicySignal
        from src.core.market_calibration import MarketCondition
    except ImportError:
        # 如果导入失败，创建 Mock 数据测试
        print("⚠️  未找到上游模块，使用 Mock 数据进行测试")
        
        MacroState = lambda **kwargs: type('obj', (object,), kwargs)()
        PolicySignal = lambda **kwargs: type('obj', (object,), kwargs)()
        MarketCondition = lambda **kwargs: type('obj', (object,), kwargs)()
    
    # 配置日志
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    
    print("=" * 60)
    print("🍳 投资组合构建模块 - 大厨炒菜测试")
    print("=" * 60)
    print("💡 工作流程:")
    print("   1. 读取带标签的 ETF 池（processed_etf_basic.csv）")
    print("   2. 根据宏观状态确定股债商配比（复苏70/20/10）")
    print("   3. 股票内部：宽基60% + 政策主题40%")
    print("   4. 应用市场校准（剔除僵尸，拥挤×0.75）")
    print("   5. 归一化权重，生成最终持仓")
    print("=" * 60)
    
    # 检查输入文件
    if not Path("data/etf/processed_etf_basic.csv").exists():
        print("\n❌ 未找到 processed_etf_basic.csv")
        print("💡 请先运行: python src/agents/theme_mapper.py")
        sys.exit(1)
    
    # 创建 Builder
    try:
        builder = PortfolioBuilder()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    
    # Mock 输入数据（如果上游模块未生成真实数据）
    print("\n📥 准备输入信号...")
    
    # Mock 宏观状态（复苏期，评分 0.75）
    mock_macro = MacroState(
        regime="recovery",
        equity_friendly_score=0.75,
        growth_momentum=2.5,
        inflation_momentum=-0.5,
        raw_data={"pmi": 51.5, "cpi_yoy": 1.8}
    )
    
    # Mock 政策信号（人工智能、新能源等）
    mock_policy = PolicySignal(
        top_5_themes=["人工智能", "新能源", "消费升级", "医药生物", "高端制造"],
        confidences=[0.92, 0.88, 0.75, 0.71, 0.68],
        analysis_summary="积极；推荐人工智能、新能源",
        is_cached=False
    )
    
    # Mock 市场校准（假设 70 只合格，其中 5 只拥挤）
    mock_market = MarketCondition(
        liquid_etfs=[f"510{i:03d}.SH" for i in range(300, 370)],  # 模拟 70 只
        crowded_adjustments={f"510330.SH": 0.75, f"588000.SH": 0.75},  # 2 只拥挤
        stats={"liquid_passed": 70, "crowded_detected": 2}
    )
    
    # 执行构建
    print("\n🔨 开始构建...")
    try:
        portfolio = builder.build(
            macro_state=mock_macro,
            policy_signal=mock_policy,
            market_condition=mock_market,
            target_date="2025-03-27"
        )
        
        # 保存结果
        save_portfolio(portfolio)
        
        # 打印摘要
        print(f"\n📋 投资组合摘要:")
        print(f"   日期: {portfolio.date}")
        print(f"   周期: {portfolio.macro_regime}（复苏期）")
        print(f"   配比: 股{portfolio.allocation['股票']:.0%} 债{portfolio.allocation['债券']:.0%} 商{portfolio.allocation['商品']:.0%}")
        print(f"   持仓: {portfolio.total_etfs} 只 ETF")
        print(f"\n🏆 Top 5 持仓:")
        for i, item in enumerate(sorted(portfolio.items, key=lambda x: x.weight, reverse=True)[:5], 1):
            crowded_tag = "⚠️拥挤" if item.crowded_adjustment < 1.0 else "  "
            print(f"   {i}. {item.ticker} {item.weight:6.2%} {crowded_tag} {item.name}")
            print(f"      └─ 逻辑: {item.selection_logic}")
        
        print(f"\n✅ 测试完成！组合文件保存在 portfolios/ 目录")
        
    except Exception as e:
        logger.error(f"❌ 构建失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎯 下一步: 运行完整流水线")
    print("   python Q-Macro.py --date 2025-03-27 --full")
    print("=" * 60)