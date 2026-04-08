#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测模块 (backtester.py) 
==========================================
功能：验证策略的历史表现，计算收益率、风险指标，生成净值曲线。

工作流程：
1. 【调取档案】读取历史投资组合记录（portfolios/ 目录下的 JSON 文件）
2. 【调取行情】读取 ETF 历史价格（计算每只 ETF 的月度收益率）
3. 【模拟交易】假设每月初按策略买入，月末卖出/再平衡
4. 【计算收益】加权计算组合月度收益，链式计算累计净值
5. 【风险体检】计算最大回撤（最多亏多少）、夏普比率（性价比）
6. 【出图报告】绘制净值曲线、回撤图，生成回测报告

核心概念解释：
- 累计收益率：从起点到现在总共赚了百分之几（如 +15%）
- 最大回撤：从最高点跌下去最多的百分比（如 -8%，表示曾浮亏8%）
- 夏普比率：每冒1份风险能赚多少超额收益（>1 合格，>2 优秀）
- 月度胜率：赚钱的月份占比（如 8/12 表示12个月里8个月盈利）

输入：
- portfolios/*.json：历史持仓记录（由 portfolio_builder 生成）
- data/etf/etf_2025_ohlcva.csv：ETF 历史行情（OHLCV）

输出：
- backtest_results.json：详细回测数据（每个月的收益率）
- backtest_report.md：回测报告（文字分析 + 绩效指标）
- backtest_chart.png：净值曲线图（可视化）
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
from loguru import logger

# 尝试导入绘图库（如果未安装会友好提示）
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("⚠️  未安装 matplotlib，将跳过图表生成")
    logger.info("   安装命令: pip install matplotlib")

# 导入我们的 Portfolio 类型（用于解析 JSON）
try:
    from src.core.portfolio_builder import Portfolio, PortfolioItem
    HAS_PORTFOLIO = True
except ImportError:
    HAS_PORTFOLIO = False
    logger.warning("⚠️  未找到 Portfolio 类型定义，将使用字典解析")


# -------------------- 配置区 --------------------
# 回测参数设置（可调整）

RISK_FREE_RATE = 0.03  # 年化无风险利率（假设 3%，用于计算夏普比率）
INITIAL_CAPITAL = 1000000  # 初始资金（100万，用于计算绝对收益金额）
MIN_HISTORY_MONTHS = 2  # 最小历史月份数（至少要有2个月数据才能回测）


# -------------------- 数据模型 --------------------
@dataclass
class MonthlyReturn:
    """单月收益记录"""
    date: str  # 月末日期（如 "2025-01-31"）
    portfolio_return: float  # 该月组合收益率（如 0.05 表示 +5%）
    benchmark_return: float  # 基准收益率（如沪深300，可选）
    cumulative_return: float  # 累计收益率（从起点到现在）
    nav: float  # 净值（从1开始，如 1.15 表示累计涨15%）
    max_drawdown: float  # 截至该月的历史最大回撤（负数，如 -0.08）


@dataclass
class BacktestMetrics:
    """回测绩效指标"""
    total_return: float  # 总收益率（如 0.20 表示 20%）
    annualized_return: float  # 年化收益率
    annualized_volatility: float  # 年化波动率（标准差）
    sharpe_ratio: float  # 夏普比率
    max_drawdown: float  # 最大回撤（历史最差情况）
    max_drawdown_period: str  # 最大回撤发生时间段
    win_rate: float  # 月度胜率（赚钱月份比例）
    profit_factor: float  # 盈亏比（总盈利/总亏损）
    num_months: int  # 回测月份数


# -------------------- 核心回测引擎 --------------------
class Backtester:
    """
    策略回测引擎
    ============
    """
    
    def __init__(self, 
                 portfolios_dir: str = "portfolios",
                 prices_path: str = "data/etf/etf_2025_ohlcva.csv"):
        """
        初始化回测引擎
        
        参数：
            portfolios_dir: 历史持仓 JSON 文件存放目录
            prices_path: ETF 历史价格数据路径（OHLCV 格式）
        """
        self.portfolios_dir = Path(portfolios_dir)
        self.prices_path = Path(prices_path)
        
        # 加载的数据缓存
        self.portfolios: List[Dict] = []  # 历史持仓列表
        self.prices_df: Optional[pd.DataFrame] = None  # 价格数据
        self.returns_df: Optional[pd.DataFrame] = None  # 计算出的 ETF 收益率
        
    def load_historical_portfolios(self) -> bool:
        """
        步骤 1：加载历史投资组合记录
        ===========================
        从 portfolios/ 目录读取所有 JSON 文件，按日期排序。
        """
        logger.info("📂 加载历史投资组合...")
        
        if not self.portfolios_dir.exists():
            logger.error(f"❌ 目录不存在: {self.portfolios_dir}")
            logger.info("💡 请先运行策略生成组合: python scripts/run_monthly_pipeline.py")
            return False
        
        # 查找所有 JSON 文件（排除 latest.json 等非日期文件）
        json_files = [
            f for f in self.portfolios_dir.glob("*.json") 
            if f.stem != "latest" and "-" in f.stem  # 过滤如 2025-03-27.json
        ]
        
        if len(json_files) < MIN_HISTORY_MONTHS:
            logger.error(f"❌ 历史组合数量不足: 找到 {len(json_files)} 个，需要至少 {MIN_HISTORY_MONTHS} 个")
            logger.info("💡 建议：")
            logger.info("   1. 运行多次策略生成不同月份的组合")
            logger.info("   2. 或使用 Mock 数据生成多个月份的历史记录")
            return False
        
        # 读取并解析每个 JSON
        self.portfolios = []
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 提取日期和持仓
                date_str = data.get("date", json_file.stem)
                self.portfolios.append({
                    "date": date_str,
                    "file": json_file.name,
                    "data": data,
                    "items": data.get("items", data.get("positions", []))  # 兼容两种命名
                })
            except Exception as e:
                logger.warning(f"⚠️  读取 {json_file.name} 失败: {e}")
                continue
        
        logger.success(f"✅ 加载了 {len(self.portfolios)} 个月的历史组合")
        for p in self.portfolios[:3]:  # 展示前3个
            logger.info(f"   📅 {p['date']}: {len(p['items'])} 只 ETF")
            
        return True
    
    def load_price_data(self) -> bool:
        """
        步骤 2：加载 ETF 价格数据并计算收益率
        ======================================
        读取 OHLCVA 数据，计算每只 ETF 的月度收益率。
        """
        logger.info("📊 加载 ETF 价格数据...")
        
        if not self.prices_path.exists():
            logger.error(f"❌ 价格数据不存在: {self.prices_path}")
            return False
        
        try:
            # 读取 CSV（假设标准格式：ticker, date, open, high, low, close, volume, amount）
            df = pd.read_csv(self.prices_path)
            
            # 确保列名正确（兼容大小写）
            df.columns = [col.lower() for col in df.columns]
            
            # 解析日期
            df['date'] = pd.to_datetime(df['date'])
            
            # 按 ticker 和日期排序
            df = df.sort_values(['ticker', 'date'])
            
            self.prices_df = df
            
            # 计算月度收益率（简化版：用月度收盘价变化）
            # 实际策略：按每月初买入，月末卖出的逻辑
            logger.info("🧮 计算 ETF 月度收益率...")
            
            # 为每个 ticker 计算月度收益
            monthly_returns = []
            
            for ticker in df['ticker'].unique():
                ticker_df = df[df['ticker'] == ticker].copy()
                
                # 设置日期索引以便计算月度频率
                ticker_df.set_index('date', inplace=True)
                
                # 取每月最后一个交易日的收盘价（月末价格）
                monthly_close = ticker_df['close'].resample('ME').last()
                
                # 计算月度收益率（本月末 / 上月末 - 1）
                monthly_return = monthly_close.pct_change().dropna()
                
                # 格式化数据
                for date, ret in monthly_return.items():
                    monthly_returns.append({
                        'ticker': ticker,
                        'month_end': date.strftime('%Y-%m-%d'),
                        'monthly_return': float(ret)
                    })
            
            self.returns_df = pd.DataFrame(monthly_returns)
            
            logger.success(f"✅ 价格数据加载完成: {len(df)} 条记录，{df['ticker'].nunique()} 只 ETF")
            logger.info(f"   📈 计算出 {len(self.returns_df)} 个月度收益记录")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 价格数据处理失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def calculate_portfolio_returns(self) -> List[MonthlyReturn]:
        """
        步骤 3：计算策略的月度收益序列
        ==============================
        核心逻辑：根据每月持仓权重，加权计算组合收益。
        
        公式：
        组合月收益 = Σ(ETF权重 × ETF月收益率)
        
        举例：
        - 持仓：沪深300(50%)涨5%，债券(30%)涨1%，黄金(20%)涨3%
        - 组合收益 = 0.5×0.05 + 0.3×0.01 + 0.2×0.03 = 3.4%
        """
        logger.info("🧮 计算组合历史收益...")
        
        if not self.portfolios or self.returns_df is None:
            logger.error("❌ 缺少必要数据（先执行前两个步骤）")
            return []
        
        monthly_returns = []
        cumulative_nav = 1.0  # 初始净值设为1
        max_nav = 1.0  # 历史最高净值（用于计算回撤）
        current_max_dd = 0.0  # 当前最大回撤
        
        # 按时间顺序遍历每个月的组合
        for i, portfolio in enumerate(self.portfolios):
            date_str = portfolio['date']
            
            # 找到该月或下月的收益率数据（因为组合通常是月初生成，持有到月末）
            # 简化处理：假设组合 date 是月初，计算到下个月末的收益
            try:
                # 解析日期
                year, month, day = map(int, date_str.split('-'))
                
                # 查找该月的 ETF 收益率（使用当月数据或下月数据，取决于再平衡频率）
                # 假设月度再平衡：每月初生成组合，持有到月底
                target_month = f"{year}-{month:02d}"
                
                month_return = 0.0
                valid_weights = 0.0
                
                # 遍历该组合的每个持仓
                for item in portfolio['items']:
                    ticker = item.get('ticker', item.get('code', ''))
                    weight = item.get('weight', 0)
                    
                    if not ticker or weight <= 0:
                        continue
                    
                    # 查找该 ETF 在该月的收益率
                    etf_return_row = self.returns_df[
                        (self.returns_df['ticker'] == ticker) & 
                        (self.returns_df['month_end'].str.startswith(target_month))
                    ]
                    
                    if not etf_return_row.empty:
                        etf_return = etf_return_row.iloc[0]['monthly_return']
                        month_return += weight * etf_return
                        valid_weights += weight
                    else:
                        # 如果找不到该月数据，假设收益为0（现金化）
                        logger.debug(f"   未找到 {ticker} 在 {target_month} 的收益数据")
                
                # 归一化（如果部分 ETF 数据缺失，重新加权）
                if valid_weights > 0 and valid_weights < 0.99:
                    month_return = month_return / valid_weights  # 按比例放大
                    logger.debug(f"   {date_str} 数据完整度 {valid_weights:.1%}，已归一化")
                
                # 更新净值
                cumulative_nav *= (1 + month_return)
                
                # 计算回撤
                if cumulative_nav > max_nav:
                    max_nav = cumulative_nav
                    current_dd = 0.0
                else:
                    current_dd = (cumulative_nav - max_nav) / max_nav
                
                # 更新历史最大回撤
                if current_dd < current_max_dd:
                    current_max_dd = current_dd
                
                # 记录该月数据
                monthly_returns.append(MonthlyReturn(
                    date=date_str,
                    portfolio_return=month_return,
                    benchmark_return=0.0,  # 暂时无基准，可扩展加入沪深300
                    cumulative_return=cumulative_nav - 1,
                    nav=cumulative_nav,
                    max_drawdown=current_max_dd
                ))
                
                logger.info(f"   {date_str}: 当月 {month_return:+.2%} | 累计 {cumulative_nav-1:+.2%} | 回撤 {current_max_dd:.2%}")
                
            except Exception as e:
                logger.warning(f"⚠️  处理 {date_str} 时出错: {e}")
                continue
        
        logger.success(f"✅ 完成收益计算: 共 {len(monthly_returns)} 个月")
        return monthly_returns
    
    def calculate_metrics(self, returns: List[MonthlyReturn]) -> BacktestMetrics:
        """
        步骤 4：计算绩效指标（专业体检报告）
        ====================================
        计算各种金融指标，评估策略好坏。
        """
        logger.info("📊 计算绩效指标...")
        
        if len(returns) < 2:
            logger.error("❌ 数据不足，无法计算指标")
            return None
        
        # 提取收益率序列
        rets = [r.portfolio_return for r in returns]
        navs = [r.nav for r in returns]
        
        # 1. 总收益率（期末净值 - 1）
        total_return = navs[-1] - 1
        
        # 2. 年化收益率（几何平均）
        n_months = len(returns)
        n_years = n_months / 12.0
        if n_years > 0:
            annualized_return = (1 + total_return) ** (1 / n_years) - 1
        else:
            annualized_return = 0
        
        # 3. 年化波动率（标准差年化）
        monthly_vol = np.std(rets, ddof=1)  # 样本标准差
        annualized_vol = monthly_vol * math.sqrt(12)  # 月波动转年波动（乘根号12）
        
        # 4. 夏普比率（超额收益 / 波动率）
        if annualized_vol > 0:
            sharpe = (annualized_return - RISK_FREE_RATE) / annualized_vol
        else:
            sharpe = 0
        
        # 5. 最大回撤（从历史数据中提取最差值）
        max_dd = min(r.max_drawdown for r in returns)
        max_dd_idx = next(i for i, r in enumerate(returns) if r.max_drawdown == max_dd)
        max_dd_period = returns[max_dd_idx].date if max_dd_idx < len(returns) else "未知"
        
        # 6. 月度胜率（赚钱月份占比）
        winning_months = sum(1 for r in rets if r > 0)
        win_rate = winning_months / len(rets)
        
        # 7. 盈亏比（平均盈利 / 平均亏损）
        gains = [r for r in rets if r > 0]
        losses = [r for r in rets if r < 0]
        if losses:
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = abs(np.mean(losses))
            profit_factor = avg_gain / avg_loss if avg_loss > 0 else 0
        else:
            profit_factor = float('inf') if gains else 0
        
        metrics = BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            max_drawdown_period=max_dd_period,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_months=n_months
        )
        
        # 打印摘要
        logger.success("✅ 绩效指标计算完成:")
        logger.info(f"   📈 总收益: {total_return:+.2%}")
        logger.info(f"   📊 年化收益: {annualized_return:+.2%}")
        logger.info(f"   📉 年化波动: {annualized_vol:.2%}")
        logger.info(f"   ⚖️  夏普比率: {sharpe:.2f} (>1 合格, >2 优秀)")
        logger.info(f"   😰 最大回撤: {max_dd:.2%} (发生在 {max_dd_period})")
        logger.info(f"   🎯 月度胜率: {win_rate:.1%} ({winning_months}/{n_months})")
        
        return metrics
    
    def plot_results(self, returns: List[MonthlyReturn], metrics: BacktestMetrics, 
                     output_path: str = "reports/backtest_chart.png"):
        """
        步骤 5：生成可视化图表（净值曲线 + 回撤图）
        ===========================================
        绘制专业的回测图表，直观展示策略表现。
        """
        if not HAS_MATPLOTLIB:
            logger.warning("⚠️  跳过图表生成（matplotlib 未安装）")
            return
        
        logger.info("📈 生成回测图表...")
        
        # 准备数据
        dates = [datetime.strptime(r.date, '%Y-%m-%d') for r in returns]
        navs = [r.nav for r in returns]
        cummulative_rets = [r.cumulative_return for r in returns]
        drawdowns = [r.max_drawdown * 100 for r in returns]  # 转为百分比显示
        
        # 创建图表（2行1列：上为净值，下为回撤）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                         gridspec_kw={'height_ratios': [3, 1]},
                                         sharex=True)
        
        # 上图：净值曲线
        ax1.plot(dates, navs, 'b-', linewidth=2, label='Strategy NAV')
        ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        ax1.fill_between(dates, 1, navs, where=[n >= 1 for n in navs], alpha=0.3, color='green')
        ax1.fill_between(dates, 1, navs, where=[n < 1 for n in navs], alpha=0.3, color='red')
        
        # 标注关键数据点（最高点和当前点）
        max_idx = np.argmax(navs)
        ax1.scatter([dates[max_idx]], [navs[max_idx]], color='gold', s=100, zorder=5, label=f'Peak: {navs[max_idx]:.2f}')
        
        ax1.set_ylabel('Net Asset Value', fontsize=12)
        ax1.set_title(f'Backtest Results: {metrics.num_months} Months | Return: {metrics.total_return:+.1%} | Sharpe: {metrics.sharpe_ratio:.2f}', 
                      fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 下图：回撤图（水下曲线）
        ax2.fill_between(dates, drawdowns, 0, color='red', alpha=0.3)
        ax2.plot(dates, drawdowns, 'r-', linewidth=1)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_title('Drawdown History (Maximum Pain)', fontsize=12)
        
        # 格式化日期显示
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # 保存
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        logger.success(f"✅ 图表已保存: {output_file}")
        
        plt.close()
    
    def generate_report(self, returns: List[MonthlyReturn], metrics: BacktestMetrics,
                        output_path: str = "reports/backtest_report.md"):
        """
        步骤 6：生成回测报告（Markdown 格式）
        =====================================
        撰写专业的回测分析报告，包含指标解释和投资建议。
        """
        logger.info("📝 生成回测报告...")
        
        # 转换为 DataFrame 以便分析
        df = pd.DataFrame([asdict(r) for r in returns])
        
        # 构建报告内容
        report = f"""# 策略回测报告

**生成时间：** {datetime.now().strftime('%Y-%m-%d %H:%M')}  
**回测周期：** {metrics.num_months} 个月  
**初始资金：** ¥{INITIAL_CAPITAL:,}

---

## 1. 绩效摘要（Executive Summary）

| 指标 | 数值 | 评价 |
|------|------|------|
| **累计收益率** | {metrics.total_return:+.2%} | {'✅ 盈利' if metrics.total_return > 0 else '❌ 亏损'} |
| **年化收益率** | {metrics.annualized_return:+.2%} | {'优秀' if metrics.annualized_return > 0.15 else '良好' if metrics.annualized_return > 0.08 else '一般'} |
| **最大回撤** | {metrics.max_drawdown:.2%} | {'可控' if metrics.max_drawdown > -0.1 else '需注意' if metrics.max_drawdown > -0.2 else '风险较高'} |
| **夏普比率** | {metrics.sharpe_ratio:.2f} | {'优秀' if metrics.sharpe_ratio > 2 else '良好' if metrics.sharpe_ratio > 1 else '需改进'} |
| **月度胜率** | {metrics.win_rate:.1%} | {metrics.win_rate*100:.0f}个月盈利，{(1-metrics.win_rate)*metrics.num_months:.0f}个月亏损 |
| **盈亏比** | {metrics.profit_factor:.2f} | {'赚多亏少' if metrics.profit_factor > 1.5 else '盈亏平衡' if metrics.profit_factor > 0.8 else '赚少亏多'} |

### 综合评价
> 该策略在回测期内**{('表现优异，显著跑赢市场基准' if metrics.sharpe_ratio > 1.5 else '表现稳健，风险收益比合理' if metrics.sharpe_ratio > 0.8 else '表现一般，需优化改进')}**。
> 年化收益率为 **{metrics.annualized_return:+.1%}**，期间最大回撤为 **{metrics.max_drawdown:.1%}**（发生在 {metrics.max_drawdown_period}）。
> 策略夏普比率 **{metrics.sharpe_ratio:.2f}**，说明每承担 1 单位风险可获得 {metrics.sharpe_ratio:.2f} 单位超额收益。

---

## 2. 收益分析（Returns Analysis）

### 2.1 月度收益明细

| 月份 | 月度收益 | 累计净值 | 回撤 | 评价 |
|------|----------|----------|------|------|
"""
        
        # 添加每个月的数据行
        for r in returns:
            emoji = "🟢" if r.portfolio_return > 0 else "🔴" if r.portfolio_return < 0 else "⚪"
            report += f"| {r.date} | {r.portfolio_return:+.2%} | {r.nav:.4f} | {r.max_drawdown:.2%} | {emoji} |\n"
        
        report += f"""
### 2.2 收益统计
- **最佳月份：** {max(returns, key=lambda x: x.portfolio_return).date} ({max(returns, key=lambda x: x.portfolio_return).portfolio_return:+.2%})
- **最差月份：** {min(returns, key=lambda x: x.portfolio_return).date} ({min(returns, key=lambda x: x.portfolio_return).portfolio_return:+.2%})
- **平均月收益：** {np.mean([r.portfolio_return for r in returns]):.2%}
- **收益波动率（月）：** {np.std([r.portfolio_return for r in returns]):.2%}

---

## 3. 风险分析（Risk Analysis）

### 3.1 回撤记录
- **历史最大回撤：** {metrics.max_drawdown:.2%}
- **回撤发生时间：** {metrics.max_drawdown_period}
- **当前回撤状态：** {returns[-1].max_drawdown:.2%} {'（已从峰值修复）' if returns[-1].max_drawdown == 0 else '（仍处于回撤中）'}

### 3.2 风险指标说明
1. **最大回撤（Max Drawdown）：** 从前期高点到最低点的最大跌幅，反映最坏情况下的亏损。
   - 本策略最大回撤 **{abs(metrics.max_drawdown):.1%}**，意味着如果在最高点买入，最多浮亏 {abs(metrics.max_drawdown):.1%}。
   - {'控制优秀' if metrics.max_drawdown > -0.1 else '需警惕' if metrics.max_drawdown > -0.2 else '风险较高'}。

2. **夏普比率（Sharpe Ratio）：** 衡量风险调整后收益，数值越高说明"性价比"越好。
   - 本策略夏普比率 **{metrics.sharpe_ratio:.2f}**。
   - **解读：** {'>2 优秀，>1 合格，<0.5 较差' if metrics.sharpe_ratio < 2 else '处于优秀水平，风险收益匹配度极佳'}。

3. **胜率与盈亏比：**
   - 月度胜率 **{metrics.win_rate:.1%}**：{'超过半数月份盈利，体验较好' if metrics.win_rate > 0.5 else '盈利月份略少于亏损，需优化入场时机'}。
   - 盈亏比 **{metrics.profit_factor:.2f}**：{'平均盈利大于亏损，赚的时候赚得多' if metrics.profit_factor > 1 else '需提高止盈或严格止损'}。

---

## 4. 策略建议（Recommendations）

基于回测结果，建议：

1. **仓位管理：** 当前策略风险水平{'适中，可维持现有仓位' if metrics.max_drawdown > -0.15 else '偏高，建议降低杠杆或增加对冲'}。
2. **优化方向：** {'可考虑增加择时模块，降低高波动时期仓位' if metrics.annualized_volatility > 0.2 else '波动率控制良好，重点优化收益端'}。
3. **心理预期：** 投资者需接受期间可能出现的 **{abs(metrics.max_drawdown):.0%}** 回撤，避免在此阶段恐慌赎回。

---

## 5. 数据说明（Disclaimer）

- **回测局限性：** 历史表现不代表未来收益，实际交易存在滑点、冲击成本等摩擦。
- **数据来源：** ETF 价格数据来自 akshare，组合权重由策略引擎生成。
- **计算方法：** 月度收益按月初调仓、月末再平衡计算，未考虑交易成本（约 0.1%/次）和税费。

---

*报告由 Q-Macro 回测系统自动生成*  
*回测区间：{returns[0].date} 至 {returns[-1].date}*
"""
        
        # 保存报告
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.success(f"✅ 回测报告已保存: {output_file}")
        return output_file
    
    def run_full_backtest(self) -> Tuple[Optional[List[MonthlyReturn]], Optional[BacktestMetrics]]:
        """
        执行完整回测流程（一键运行）
        ==========================
        """
        logger.info("🚀 启动完整回测流程...")
        
        # 1. 加载数据
        if not self.load_historical_portfolios():
            return None, None
        
        if not self.load_price_data():
            return None, None
        
        # 2. 计算收益
        returns = self.calculate_portfolio_returns()
        if not returns:
            logger.error("❌ 收益计算失败")
            return None, None
        
        # 3. 计算指标
        metrics = self.calculate_metrics(returns)
        if not metrics:
            return None, None
        
        # 4. 生成图表
        self.plot_results(returns, metrics)
        
        # 5. 生成报告
        self.generate_report(returns, metrics)
        
        # 6. 保存原始数据（JSON）
        results_data = {
            "monthly_returns": [asdict(r) for r in returns],
            "metrics": asdict(metrics),
            "generated_at": datetime.now().isoformat()
        }
        
        results_file = Path("reports/backtest_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 详细数据已保存: {results_file}")
        
        logger.success("🎉 回测完成！查看报告: reports/backtest_report.md")
        
        return returns, metrics


# ==================== 测试入口 ====================
if __name__ == "__main__":
    import sys
    
    # 配置日志显示
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    
    print("=" * 70)
    print("📊 策略回测模块 - 历史表现验证")
    print("=" * 70)
    print("💡 工作流程：")
    print("   1. 读取 portfolios/ 目录的历史持仓记录")
    print("   2. 读取 ETF 价格数据计算月度收益")
    print("   3. 链式计算累计净值和最大回撤")
    print("   4. 计算夏普比率、胜率等专业指标")
    print("   5. 生成净值曲线图和 Markdown 回测报告")
    print("=" * 70)
    
    # 检查必要目录
    if not Path("portfolios").exists():
        print("\n❌ 未找到 portfolios/ 目录")
        print("💡 请先运行策略生成历史组合：")
        print("   python scripts/run_monthly_pipeline.py --date 2025-01-31")
        print("   python scripts/run_monthly_pipeline.py --date 2025-02-28")
        print("   python scripts/run_monthly_pipeline.py --date 2025-03-27")
        print("   （至少需要 2 个月的数据才能回测）")
        sys.exit(1)
    
    if not Path("data/etf/etf_2025_ohlcva.csv").exists():
        print("\n❌ 未找到 ETF 价格数据")
        print("💡 请确保数据文件存在：data/etf/etf_2025_ohlcva.csv")
        sys.exit(1)
    
    # 执行回测
    backtester = Backtester()
    returns, metrics = backtester.run_full_backtest()
    
    if returns and metrics:
        print("\n" + "=" * 70)
        print("🎯 回测绩效速览")
        print("=" * 70)
        print(f"📈 累计收益：{(metrics.total_return)*100:+.1f}%")
        print(f"📊 夏普比率：{metrics.sharpe_ratio:.2f} ({'优秀' if metrics.sharpe_ratio > 2 else '良好' if metrics.sharpe_ratio > 1 else '一般'})")
        print(f"😰 最大回撤：{metrics.max_drawdown*100:.1f}%")
        print(f"🎯 月度胜率：{metrics.win_rate*100:.0f}%")
        print("=" * 70)
        print("\n📁 查看完整报告：")
        print("   Markdown: reports/backtest_report.md")
        print("   图表:     reports/backtest_chart.png")
        print("   数据:     reports/backtest_results.json")
    else:
        print("\n❌ 回测执行失败，请检查日志")
        sys.exit(1)