#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q-Macro 智能投研系统 - 中央指挥塔 (Q-Macro.py)
==============================================
一站式 ETF 配置策略执行入口。

功能：
  1. 单月策略生成：宏观研判 → 政策解读 → 组合构建 → 报告输出
  2. 历史回测验证：基于历史持仓计算净值曲线与风险指标
  3. 批量历史模拟：生成多个月份的组合用于回测
  4. 定时任务模式：模拟每月自动运行（演示用）

使用方式：
  # 生成今日策略 + 自动回测（如果历史数据足够）
  python Q-Macro.py
  
  # 指定日期生成策略
  python Q-Macro.py --date 2025-03-31
  
  # 仅运行回测（分析历史表现）
  python Q-Macro.py --backtest-only
  
  # 批量生成 3 个月历史数据（用于填充回测）
  python Q-Macro.py --batch-history 2025-01-01 2025-03-31
  
  # 强制重新打标 ETF（忽略缓存）
  python Q-Macro.py --date 2025-03-31 --force

  # 调试模式（详细日志）
  python Q-Macro.py --debug
"""

import sys
import argparse
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
from loguru import logger

# 确保项目根目录在路径中
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入所有业务模块（与之前创建的模块对应）
try:
    from src.core.macro_regime import detect_macro_regime
    from src.agents.policy_interpreter import analyze_policy
    from src.agents.theme_mapper import process_etfs
    from src.agents.report_writer import ReportWriter
    from src.core.market_calibration import calibrate_market  # 修正：之前报错处
    from src.core.portfolio_builder import PortfolioBuilder, save_portfolio
    from src.core.backtester import Backtester
    MODULES_READY = True
except ImportError as e:
    MODULES_READY = False
    logger.error(f"❌ 模块导入失败: {e}")
    logger.info("💡 请确保 src/ 目录下所有模块文件已创建")


# -------------------- 配置区 --------------------
class Config:
    """运行配置"""
    VERSION = "1.0.0"
    DEFAULT_DATE = datetime.now().strftime("%Y-%m-%d")
    PORTFOLIOS_DIR = Path("portfolios")
    REPORTS_DIR = Path("reports")
    DATA_DIR = Path("data/etf")
    MIN_HISTORY_FOR_BACKTEST = 2  # 最少需要2个月数据才自动回测


# -------------------- 指挥塔核心类 --------------------
class QMacroOrchestrator:
    """
    Q-Macro 指挥塔
    ==============
    协调所有模块的执行流程，像交响乐团的指挥。
    """
    
    def __init__(self, force_refresh: bool = False, debug: bool = False):
        """
        初始化
        参数：
            force_refresh: 是否强制重新生成 ETF 标签（忽略缓存）
            debug: 是否开启调试模式（详细日志）
        """
        self.force_refresh = force_refresh
        self.debug = debug
        self._setup_logging()
        
        # 确保目录存在
        Config.PORTFOLIOS_DIR.mkdir(parents=True, exist_ok=True)
        Config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """配置日志格式"""
        logger.remove()
        log_format = "<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}"
        if self.debug:
            logger.add(sys.stdout, format=log_format, level="DEBUG")
            logger.add("logs/debug.log", rotation="10 MB", level="DEBUG")
        else:
            logger.add(sys.stdout, format=log_format, level="INFO")
    
    def check_environment(self) -> bool:
        """
        环境检查
        检查必要的文件和配置是否存在。
        """
        logger.info("🔍 检查运行环境...")
        
        # 检查 .env 文件（API Key）
        if not Path(".env").exists():
            logger.warning("⚠️  未找到 .env 文件")
            logger.info("   提示：创建 .env 文件并配置 OPENAI_API_KEY")
        
        # 检查基础数据
        required_files = [
            Config.DATA_DIR / "etf_basic.csv",
            Config.DATA_DIR / "etf_2025_ohlcva.csv"
        ]
        
        missing = [f for f in required_files if not f.exists()]
        if missing:
            logger.error("❌ 缺少必要数据文件：")
            for f in missing:
                logger.error(f"   • {f}")
            logger.info("💡 解决方案：")
            logger.info("   1. 真实数据：等待 AKShare IP 解封后运行数据获取脚本")
            logger.info("   2. Mock 数据：运行 python scripts/generate_mock_etf.py")
            return False
        
        # 检查处理后的 ETF 标签（如果不强制刷新）
        processed_file = Config.DATA_DIR / "processed_etf_basic.csv"
        if not processed_file.exists():
            logger.warning("⚠️  未找到 ETF 主题标签文件")
            logger.info("   将在首次运行时自动执行 LLM 打标（消耗 API Token）")
        
        logger.success("✅ 环境检查通过")
        return True
    
    def run_single_month(self, target_date: str, skip_backtest: bool = False) -> Optional[dict]:
        """
        执行单月完整流程（6大模块串联）
        =================================
        这是核心工作流，生成单月的投资组合。
        
        返回：
            包含 portfolio、metrics 的字典，或 None（如果失败）
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 启动单月策略生成 | 日期: {target_date}")
        logger.info(f"{'='*60}")
        
        try:
            # ---- 模块 1: 宏观状态识别 ----
            logger.info("\n📊 [1/6] 宏观状态识别...")
            macro_state = detect_macro_regime(target_date)
            regime_cn = {
                "recovery": "复苏期", "overheat": "过热期",
                "stagflation": "滞胀期", "recession": "衰退期"
            }.get(macro_state.regime, "调整期")
            logger.success(f"✅ 周期定位: {regime_cn} | 权益友好度: {macro_state.equity_friendly_score:.0%}")
            
            # ---- 模块 2: 政策解读 ----
            logger.info("\n🏛️  [2/6] 政策解读...")
            macro_context = {
                "regime": macro_state.regime,
                "equity_friendly_score": macro_state.equity_friendly_score
            }
            policy_signal = analyze_policy(target_date, macro_context)
            themes_str = " | ".join([f"{t}({c:.0%})" for t, c in 
                                   zip(policy_signal.top_5_themes[:3], policy_signal.confidences[:3])])
            logger.success(f"✅ 政策主题: {themes_str}")
            
            # ---- 模块 3: ETF 主题映射（按需）----
            processed_file = Config.DATA_DIR / "processed_etf_basic.csv"
            if not processed_file.exists() or self.force_refresh:
                logger.info("\n🏷️  [3/6] ETF 主题映射（AI 打标）...")
                process_etfs(
                    input_path=str(Config.DATA_DIR / "etf_basic.csv"),
                    output_path=str(processed_file)
                )
                logger.success("✅ ETF 打标完成")
            else:
                logger.info("\n🏷️  [3/6] ETF 主题映射（使用缓存）...")
                logger.success("✅ 使用已有标签（跳过 LLM 调用）")
            
            # ---- 模块 4: 市场校准 ----
            logger.info("\n🔍 [4/6] 市场校准...")
            market_cond = calibrate_market(target_date)
            logger.success(f"✅ 通过筛选: {len(market_cond.liquid_etfs)} 只 ETF | "
                          f"拥挤检测: {market_cond.stats.get('crowded_detected', 0)} 只")
            
            # ---- 模块 5: 投资组合构建 ----
            logger.info("\n🍳 [5/6] 投资组合构建...")
            builder = PortfolioBuilder()
            portfolio = builder.build(
                macro_state=macro_state,
                policy_signal=policy_signal,
                market_condition=market_cond,
                target_date=target_date
            )
            save_portfolio(portfolio)
            logger.success(f"✅ 组合构建完成: {portfolio.total_etfs} 只 ETF | "
                          f"配比: 股{portfolio.allocation['股票']:.0%} 债{portfolio.allocation['债券']:.0%}")
            
            # ---- 模块 6: 报告生成 ----
            logger.info("\n📝 [6/6] 生成投资报告...")
            writer = ReportWriter()
            report_path = writer.write(macro_state, policy_signal, portfolio, target_date)
            logger.success(f"✅ 报告已生成: {report_path}")
            
            # ---- 自动回测（如果历史数据足够）----
            if not skip_backtest:
                self._auto_backtest()
            
            logger.info(f"\n{'='*60}")
            logger.success(f"🎉 {target_date} 策略生成完成！")
            logger.info(f"{'='*60}")
            logger.info(f"📁 输出文件:")
            logger.info(f"   • 组合 JSON: portfolios/{target_date}.json")
            logger.info(f"   • 组合 CSV:  portfolios/{target_date}_portfolio.csv")
            logger.info(f"   • 投研报告:  reports/{target_date}-report.md")
            
            return {
                "date": target_date,
                "portfolio": portfolio,
                "macro": macro_state,
                "policy": policy_signal
            }
            
        except Exception as e:
            logger.error(f"\n❌ 执行失败: {e}")
            if self.debug:
                traceback.print_exc()
            return None
    
    def _auto_backtest(self):
        """自动回测（如果历史数据足够）"""
        try:
            json_files = list(Config.PORTFOLIOS_DIR.glob("*.json"))
            json_files = [f for f in json_files if "-" in f.stem and f.stem != "latest"]
            
            if len(json_files) >= Config.MIN_HISTORY_FOR_BACKTEST:
                logger.info(f"\n📈 [Bonus] 自动执行回测验证（发现 {len(json_files)} 个月历史数据）...")
                backtester = Backtester()
                backtester.run_full_backtest()
                logger.success("✅ 回测完成，查看 reports/backtest_report.md")
            else:
                logger.info(f"\n⏭️  [跳过回测] 历史数据不足（仅 {len(json_files)} 个月，需 ≥{Config.MIN_HISTORY_FOR_BACKTEST}）")
                logger.info("   提示：使用 --batch-history 生成多个月份数据")
        except Exception as e:
            logger.warning(f"⚠️  自动回测失败（非致命）: {e}")
    
    def run_backtest_only(self):
        """仅运行回测模式"""
        logger.info(f"\n{'='*60}")
        logger.info("📊 独立回测模式")
        logger.info(f"{'='*60}")
        
        try:
            backtester = Backtester()
            returns, metrics = backtester.run_full_backtest()
            
            if metrics:
                print(f"\n{'='*60}")
                print("🎯 回测绩效速览")
                print(f"{'='*60}")
                print(f"📈 累计收益: {metrics.total_return:+.2%}")
                print(f"📊 夏普比率: {metrics.sharpe_ratio:.2f}")
                print(f"😰 最大回撤: {metrics.max_drawdown:.2%}")
                print(f"🎯 月度胜率: {metrics.win_rate:.1%}")
                print(f"{'='*60}")
                return True
            return False
        except Exception as e:
            logger.error(f"❌ 回测失败: {e}")
            if self.debug:
                traceback.print_exc()
            return False
    
    def run_batch_history(self, start_date: str, end_date: str):
        """
        批量生成历史数据（用于填充回测）
        按月生成策略（实际使用时请确保有对应历史数据）
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"📚 批量历史生成: {start_date} 至 {end_date}")
        logger.info(f"{'='*60}")
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        current = start
        count = 0
        
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            logger.info(f"\n🔄 处理 {date_str}...")
            
            # 检查是否已存在
            if (Config.PORTFOLIOS_DIR / f"{date_str}.json").exists():
                logger.info("   已存在，跳过")
            else:
                result = self.run_single_month(date_str, skip_backtest=True)
                if result:
                    count += 1
                    logger.success(f"   生成成功（{count}）")
                else:
                    logger.error(f"   生成失败")
            
            # 下一个月
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1, day=1)
            else:
                current = current.replace(month=current.month + 1, day=1)
        
        logger.info(f"\n{'='*60}")
        logger.success(f"🎉 批量生成完成，共 {count} 个月")
        logger.info("💡 现在可以运行回测：python Q-Macro.py --backtest-only")
        logger.info(f"{'='*60}")
        
        # 批量生成后自动回测
        self.run_backtest_only()


# -------------------- 命令行入口 --------------------
def main():
    parser = argparse.ArgumentParser(
        description="Q-Macro 智能投研系统 - 一键生成 ETF 配置策略",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成今日策略（默认）
  python Q-Macro.py
  
  # 指定日期
  python Q-Macro.py --date 2025-03-31
  
  # 仅回测历史表现
  python Q-Macro.py --backtest-only
  
  # 生成 3 个月历史数据（用于回测）
  python Q-Macro.py --batch-history 2025-01-01 2025-03-31
  
  # 强制重新打标 ETF（消耗 API Token）
  python Q-Macro.py --force
  
  # 调试模式（详细错误信息）
  python Q-Macro.py --debug
        """
    )
    
    parser.add_argument("--date", type=str, default=Config.DEFAULT_DATE,
                       help=f"目标日期 (YYYY-MM-DD)，默认今天 ({Config.DEFAULT_DATE})")
    parser.add_argument("--backtest-only", action="store_true",
                       help="仅运行回测，不生成新策略")
    parser.add_argument("--batch-history", nargs=2, metavar=("START", "END"),
                       help="批量生成历史组合，参数：开始日期 结束日期 (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true",
                       help="强制重新执行 ETF 主题映射（忽略缓存）")
    parser.add_argument("--debug", action="store_true",
                       help="开启调试模式（显示详细日志和错误堆栈）")
    parser.add_argument("--version", action="version", version=f"Q-Macro v{Config.VERSION}")
    
    args = parser.parse_args()
    
    # 欢迎信息
    print(f"\n{'='*70}")
    print(f"🤖 Q-Macro 智能投研系统 v{Config.VERSION}")
    print(f"📅 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    if not MODULES_READY:
        print("\n❌ 环境未就绪，请检查模块导入错误")
        return 1
    
    # 初始化指挥塔
    orchestrator = QMacroOrchestrator(force_refresh=args.force, debug=args.debug)
    
    # 环境检查
    if not orchestrator.check_environment():
        return 1
    
    # 根据模式执行
    if args.backtest_only:
        # 仅回测模式
        success = orchestrator.run_backtest_only()
        return 0 if success else 1
        
    elif args.batch_history:
        # 批量历史生成
        start, end = args.batch_history
        orchestrator.run_batch_history(start, end)
        return 0
        
    else:
        # 标准模式：生成单月策略
        result = orchestrator.run_single_month(args.date)
        return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())