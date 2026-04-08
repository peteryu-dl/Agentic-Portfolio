#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单月策略执行流水线 (run_monthly_pipeline.py) - 最终整合版
===========================================================
整合所有 6 大模块的完整工作流：
1. 宏观状态识别 (Macro Regime)
2. 政策解读 (Policy Interpreter) 
3. ETF 主题映射 (Theme Mapper) - 如未打标则自动执行
4. 市场校准 (Market Calibration)
5. 投资组合构建 (Portfolio Builder)
6. 报告生成 (Report Writer)

使用方式：
    python scripts/run_monthly_pipeline.py --date 2025-03-31
    python scripts/run_monthly_pipeline.py              # 默认今天
    python scripts/run_monthly_pipeline.py --fast-debug # 跳过 LLM 调用（用缓存/默认值）
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from loguru import logger
import pandas as pd

# 导入所有模块（与之前提供的文件对应）
from src.core.macro_regime import detect_macro_regime
from src.agents.policy_interpreter import analyze_policy
from src.agents.theme_mapper import process_etfs  # 新增：ETF 打标
from src.agents.report_writer import ReportWriter  # 新增：报告生成
from src.core.market_calibration import calibrate_market
from src.core.portfolio_builder import PortfolioBuilder, save_portfolio, Portfolio
from src.core.portfolio_builder import REGIME_BASE_WEIGHTS  # 用于显示


def check_data_ready(target_date: str) -> bool:
    """
    数据准备检查（增强版）
    =====================
    检查所有必要数据文件是否存在。
    """
    logger.info("🔍 检查数据准备情况...")
    
    required_files = [
        ("data/etf/etf_basic.csv", "ETF 基础档案"),
        ("data/etf/etf_2025_ohlcva.csv", "ETF 历史行情"),
        ("data/etf/processed_etf_basic.csv", "ETF 主题标签（如缺失将自动打标）")
    ]
    
    missing_required = []
    missing_optional = []
    
    for filepath, desc in required_files:
        full_path = Path(filepath)
        if not full_path.exists():
            if "processed" in filepath:
                missing_optional.append((filepath, desc))
            else:
                missing_required.append((filepath, desc))
    
    if missing_required:
        logger.error("🚨 缺少必要数据文件：")
        for fp, desc in missing_required:
            logger.error(f"  ❌ {desc}: {fp}")
        logger.info("💡 解决方案：")
        logger.info("   1. 真实数据：等待 AKShare IP 解封后运行 python scripts/fetch_etf_data.py")
        logger.info("   2. Mock 数据：python scripts/generate_mock_etf.py")
        return False
    
    # 如果缺少 processed（带标签的），提示将自动打标
    if missing_optional:
        logger.warning("⚠️  未找到 ETF 主题标签，将在流水线中自动执行 LLM 打标（消耗 API Token）")
        logger.info("   提示：如想跳过打标，可先运行 python src/agents/theme_mapper.py")
    
    # 检查政策文本（可选）
    policy_file = Path("data/policy_texts/govcn_2025.csv")
    if policy_file.exists():
        logger.success(f"✅ 发现政策文本: {policy_file}")
    else:
        logger.warning(f"⚠️  未找到政策文本，将使用 Fallback 默认主题")
    
    logger.success("✅ 数据检查通过")
    return True


def run_pipeline(target_date: str, save_results: bool = True, fast_debug: bool = False) -> Portfolio:
    """
    执行完整策略流水线（6 大模块整合）
    ==================================
    
    参数：
        target_date: 目标日期（如 "2025-03-31"）
        save_results: 是否保存结果到文件
        fast_debug: 快速调试模式（跳过 LLM 调用，使用缓存/默认值）
    
    返回：
        Portfolio 对象（包含完整持仓信息）
    """
    print("=" * 70)
    print("🚀 Q-Macro 智能策略执行流水线")
    print(f"📅 目标日期: {target_date}")
    print(f"⚙️  模式: {'快速调试' if fast_debug else '完整流程（含 LLM）'}")
    print("=" * 70)
    
    # 配置日志
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    
    # ========== 第 1 步：数据检查 ==========
    if not check_data_ready(target_date):
        return None
    
    # ========== 第 2 步：宏观状态识别 ==========
    logger.info("\n📊 [1/6] 宏观状态识别（判断经济周期）...")
    try:
        macro_state = detect_macro_regime(target_date)
        regime_cn = {
            "recovery": "复苏期", "overheat": "过热期",
            "stagflation": "滞胀期", "recession": "衰退期"
        }.get(macro_state.regime, "调整期")
        
        logger.success(f"✅ 周期判断: {regime_cn} ({macro_state.regime})")
        logger.info(f"   权益友好度: {macro_state.equity_friendly_score:.2f} | "
                    f"增长动量: {macro_state.growth_momentum:+.1f}% | "
                    f"通胀动量: {macro_state.inflation_momentum:+.1f}%")
    except Exception as e:
        logger.error(f"❌ 宏观状态识别失败: {e}")
        return None
    
    # ========== 第 3 步：政策解读 ==========
    logger.info("\n🏛️  [2/6] 政策解读（提取投资主题）...")
    try:
        # 构造宏观背景字典供政策模块使用
        macro_context = {
            "regime": macro_state.regime,
            "equity_friendly_score": macro_state.equity_friendly_score
        }
        
        policy_signal = analyze_policy(
            target_date=target_date,
            macro_context=macro_context
        )
        
        # 格式化显示 Top 5 主题
        themes_display = " | ".join([
            f"{t}({c:.0%})" for t, c in 
            zip(policy_signal.top_5_themes[:5], policy_signal.confidences[:5])
        ])
        logger.success(f"✅ 政策主题: {themes_display}")
        
        if policy_signal.is_cached:
            logger.info("   💾 来自缓存（未消耗 API Token）")
            
    except Exception as e:
        logger.error(f"❌ 政策解读失败: {e}")
        return None
    
    # ========== 第 4 步：ETF 主题映射（如需要）==========
    # 检查是否已有打标数据，没有则自动执行
    processed_file = Path("data/etf/processed_etf_basic.csv")
    if not processed_file.exists():
        logger.info("\n🏷️  [3/6] ETF 主题映射（AI 自动打标）...")
        try:
            # 调用 theme_mapper 生成 processed_etf_basic.csv
            process_etfs(
                input_path="data/etf/etf_basic.csv",
                output_path="data/etf/processed_etf_basic.csv"
            )
            logger.success("✅ ETF 打标完成")
        except Exception as e:
            logger.error(f"❌ ETF 打标失败: {e}")
            return None
    else:
        logger.info("\n🏷️  [3/6] ETF 主题映射（使用已有标签）...")
        logger.success("✅ 发现已有标签文件，跳过打标")
    
    # ========== 第 5 步：市场校准 ==========
    logger.info("\n🔍 [4/6] 市场校准（筛选合格 ETF）...")
    try:
        market_cond = calibrate_market(target_date)
        
        liquid_count = len(market_cond.liquid_etfs)
        crowded_count = market_cond.stats.get('crowded_detected', 0)
        
        logger.success(f"✅ 通过筛选: {liquid_count} 只 ETF 合格")
        if crowded_count > 0:
            logger.warning(f"   ⚠️  其中 {crowded_count} 只交易拥挤（权重将降 25%）")
            
    except Exception as e:
        logger.error(f"❌ 市场校准失败: {e}")
        return None
    
    # ========== 第 6 步：投资组合构建 ==========
    logger.info("\n🍳 [5/6] 投资组合构建（计算最优权重）...")
    try:
        builder = PortfolioBuilder()
        
        portfolio = builder.build(
            macro_state=macro_state,
            policy_signal=policy_signal,
            market_condition=market_cond,
            target_date=target_date
        )
        
        # 验证组合完整性
        if portfolio.total_weight < 0.99 or portfolio.total_weight > 1.01:
            logger.warning(f"⚠️  权重总和异常: {portfolio.total_weight:.2f}（应接近 1.0）")
        
        logger.success(f"✅ 组合构建完成: {portfolio.total_etfs} 只 ETF | "
                      f"总权重: {portfolio.total_weight:.2f}")
        
    except Exception as e:
        logger.error(f"❌ 组合构建失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ========== 第 7 步：保存结果 ==========
    if save_results:
        logger.info("\n💾 [6/6] 保存结果 & 生成报告...")
        
        # 7.1 保存 JSON + CSV
        save_portfolio(portfolio)
        
        # 7.2 额外保存 CSV（方便 Excel）
        save_portfolio_csv(portfolio)
        
        # 7.3 生成 Markdown 报告（AI 撰写）
        try:
            writer = ReportWriter()
            report_path = writer.write(
                macro_state=macro_state,
                policy_signal=policy_signal,
                portfolio=portfolio,
                date=target_date
            )
            logger.success(f"✅ 投资报告: {report_path.name}")
        except Exception as e:
            logger.warning(f"⚠️  报告生成失败（非致命）: {e}")
    
    # ========== 第 8 步：打印可视化摘要 ==========
    print_portfolio_summary(portfolio, macro_state, policy_signal)
    
    logger.success("\n🎉 策略流水线执行完成！")
    return portfolio


def save_portfolio_csv(portfolio: Portfolio):
    """
    额外保存 CSV 格式（方便 Excel 打开查看）
    注意：适配之前定义的 Portfolio Pydantic 模型（items 而非 positions）
    """
    output_dir = Path("portfolios")
    output_dir.mkdir(exist_ok=True)
    
    # 从 Pydantic 模型提取数据
    data = []
    for item in portfolio.items:
        data.append({
            "date": portfolio.date,
            "ticker": item.ticker,
            "name": item.name,
            "weight": f"{item.weight:.2%}",
            "asset_class": item.asset_class,
            "theme": item.theme,
            "sub_theme": item.sub_theme,
            "adjustment": item.crowded_adjustment,
            "logic": item.selection_logic
        })
    
    df = pd.DataFrame(data)
    csv_path = output_dir / f"{portfolio.date}_portfolio.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"💾 CSV 持仓表: {csv_path}")


def print_portfolio_summary(portfolio: Portfolio, macro_state, policy_signal):
    """
    打印可视化的投资组合摘要（给用户看的"成绩单"）
    """
    print("\n" + "=" * 70)
    print("📋 投资组合摘要")
    print("=" * 70)
    
    # 基本信息
    regime_cn = {
        "recovery": "复苏期", "overheat": "过热期",
        "stagflation": "滞胀期", "recession": "衰退期"
    }.get(portfolio.macro_regime, "调整期")
    
    print(f"📅 日期: {portfolio.date}")
    print(f"🏷️  经济周期: {regime_cn} ({portfolio.macro_regime})")
    print(f"📈 权益评分: {macro_state.equity_friendly_score:.0%}")
    print(f"🎯 政策主题: {', '.join(portfolio.policy_themes[:3])}")
    
    # 大类资产配置（可视化进度条）
    print(f"\n💼 资产配置:")
    emojis = {"stock": "📈 股票", "bond": "📊 债券", "commodity": "🥇 商品"}
    for asset, weight in portfolio.allocation.items():
        emoji = emojis.get(asset, asset)
        bar = "█" * int(weight * 20) + "░" * (20 - int(weight * 20))
        print(f"   {emoji:8s}: {weight:.1%} | {bar}")
    
    # 持仓明细（分组显示）
    print(f"\n🏷️  持仓明细（共 {portfolio.total_etfs} 只）:")
    
    # 按资产类别分组
    items_by_class = {}
    for item in portfolio.items:
        ac = item.asset_class
        if ac not in items_by_class:
            items_by_class[ac] = []
        items_by_class[ac].append(item)
    
    for asset_class in ["股票", "债券", "商品", "跨境", "货币"]:
        if asset_class not in items_by_class:
            continue
        emoji = {"股票": "📈", "债券": "📊", "商品": "🥇", "跨境": "🌏", "货币": "💵"}.get(asset_class, "•")
        
        for i, item in enumerate(items_by_class[asset_class], 1):
            crowded_marker = " ⚠️拥挤" if item.crowded_adjustment < 1.0 else ""
            theme_tag = f"[{item.sub_theme}]" if item.sub_theme else ""
            
            print(f"   {emoji} {item.ticker:12s} | {item.weight:>6.2%} | {theme_tag:12s} {item.name:20s}{crowded_marker}")
    
    # 风险提示
    crowded_total = sum(1 for p in portfolio.items if p.crowded_adjustment < 1.0)
    if crowded_total > 0:
        print(f"\n⚠️  风险提示: {crowded_total} 只 ETF 交易拥挤（已自动降权 25%）")
    
    # 文件位置提示
    print("\n📁 输出文件:")
    print(f"   • JSON: portfolios/{portfolio.date}.json")
    print(f"   • CSV:  portfolios/{portfolio.date}_portfolio.csv")
    print(f"   • 报告: reports/{portfolio.date}-report.md")
    print("=" * 70)


def main():
    """
    命令行入口
    """
    parser = argparse.ArgumentParser(description="Q-Macro 智能策略执行流水线")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="目标日期（如 2025-03-31），默认今天"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存结果到文件（仅打印）"
    )
    parser.add_argument(
        "--fast-debug",
        action="store_true",
        help="快速调试模式（跳过 LLM 调用，使用缓存/默认值）"
    )
    
    args = parser.parse_args()
    
    # 执行流水线
    portfolio = run_pipeline(
        target_date=args.date,
        save_results=not args.no_save,
        fast_debug=args.fast_debug
    )
    
    if portfolio:
        print(f"\n✅ 成功！查看详细报告:")
        print(f"   cat reports/{args.date}-report.md")
        return 0
    else:
        print(f"\n❌ 执行失败，请检查日志和配置文件（.env）")
        return 1


if __name__ == "__main__":
    sys.exit(main())