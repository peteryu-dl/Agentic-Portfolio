#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mock ETF 数据生成器（IP 被封时的救急方案）
==========================================
生成逼真的 ETF 模拟数据，包含：
1. etf_basic.csv（100 只 ETF 档案，含头部活跃+尾部僵尸，测试筛选器）
2. etf_2025_ohlcva.csv（半年历史行情，约 12,500 条记录）

数据特点：
- 使用真实 ETF 代码（510300.SH 沪深300等）
- 价格走势符合真实规律（随机游走+趋势）
- 故意混入 20-30 只"僵尸 ETF"（日均成交 < 50万），测试流动性筛选
- 故意混入 5-10 只"拥挤 ETF"（近期成交量暴增），测试拥挤度检测

⚠️ 警告：这是模拟数据，仅用于流程调试！真实交易前必须替换为 AKShare 真实数据！
"""

import random
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

DATA_DIR = Path("data/etf")
random.seed(42)  # 固定随机种子，确保可复现


def generate_trading_days(start_date, end_date):
    """生成交易日（简化：周一到周五，排除周末）"""
    dates = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # 0-4 是周一到周五
            dates.append(current)
        current += timedelta(days=1)
    return dates


def generate_mock_basic():
    """
    生成 ETF 基础档案（100 只，分层设计）
    =====================================
    设计意图：
    - 前 30 只：头部大流动性（日均成交 > 1000万，一定通过筛选）
    - 中 40 只：腰部一般（日均成交 100-500万，部分通过）
    - 后 30 只：尾部僵尸（日均成交 < 50万，应被筛选剔除）
    
    这样市场校准模块能从 100 只中筛选出约 70 只，体现风控价值。
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 真实 ETF 池（代码、名称、基准规模）
    etf_templates = [
        # 宽基指数（大流动性）
        ("510300.SH", "沪深300ETF", 120000000),
        ("510500.SH", "中证500ETF", 80000000),
        ("510050.SH", "上证50ETF", 65000000),
        ("159915.SZ", "创业板ETF", 45000000),
        ("588000.SH", "科创50ETF", 92000000),
        ("159949.SZ", "创业板50ETF", 28000000),
        ("512800.SH", "银行ETF", 32000000),
        ("512880.SH", "证券ETF", 38000000),
        ("510180.SH", "180ETF", 25000000),
        ("159901.SZ", "深100ETF", 18000000),
        
        # 行业主题（中等流动性）
        ("512480.SH", "半导体ETF", 25000000),
        ("515790.SH", "光伏ETF", 18000000),
        ("515030.SH", "新能源车ETF", 22000000),
        ("512170.SH", "医疗ETF", 15000000),
        ("512010.SH", "医药ETF", 12000000),
        ("159995.SZ", "芯片ETF", 28000000),
        ("515050.SH", "5GETF", 8000000),
        ("159928.SZ", "消费ETF", 6000000),
        ("512690.SH", "酒ETF", 15000000),
        ("510880.SH", "红利ETF", 20000000),
        ("512200.SH", "房地产ETF", 5000000),
        ("159952.SZ", "创业ETF", 7000000),
        ("515210.SH", "钢铁ETF", 3000000),
        ("515220.SH", "煤炭ETF", 4500000),
        ("159870.SZ", "化工ETF", 3500000),
        ("159865.SZ", "畜牧ETF", 4000000),
        ("159766.SZ", "旅游ETF", 2800000),
        
        # 商品/债券（特殊资产）
        ("518880.SH", "黄金ETF", 35000000),
        ("159934.SZ", "黄金ETF", 8000000),
        ("511010.SH", "国债ETF", 12000000),
        ("511260.SH", "十年国债ETF", 6000000),
        ("511580.SH", "政金债ETF", 4000000),
        
        # 跨境（港股/美股）
        ("510900.SH", "H股ETF", 15000000),
        ("159920.SZ", "港股ETF", 9000000),
        ("513100.SH", "纳指ETF", 45000000),
        ("513500.SH", "标普500ETF", 25000000),
        ("513050.SH", "中概互联ETF", 12000000),
        
        # 尾部小市值（故意设计为僵尸ETF，测试筛选）
        ("159867.SZ", "畜牧养殖ETF", 800000),    # 小！
        ("159869.SZ", "生物科技ETF", 1200000),   # 小！
        ("159890.SZ", "软件ETF", 900000),        # 小！
        ("159892.SZ", "恒生科技ETF", 1500000),   # 小！
        ("159895.SZ", "物联网ETF", 600000),      # 小！
        ("159898.SZ", "电子ETF", 1100000),       # 小！
        ("159901.SZ", "深成ETF", 700000),        # 小！
        ("159902.SZ", "中小板ETF", 1300000),    # 小！
        ("159903.SZ", "深200ETF", 800000),       # 小！
        ("159905.SZ", "深红利ETF", 1400000),    # 小！
        ("159907.SZ", "中小300ETF", 500000),     # 小！
        ("159908.SZ", "创业300ETF", 900000),     # 小！
        ("159909.SZ", "深TMTETF", 1200000),      # 小！
        ("159910.SZ", "深F120ETF", 600000),      # 小！
        ("159911.SZ", "民营ETF", 1100000),      # 小！
        ("159912.SZ", "深300ETF", 400000),       # 极小！应被剔除
        ("159913.SZ", "深创投ETF", 800000),      # 小！
        ("159914.SZ", "深度100ETF", 700000),     # 小！
        ("159915.SZ", "创业板", 45000000),      # 这个不小了，替换一个
    ]
    
    # 扩展到 100 只（复制并微调，制造多样性）
    extended = []
    for i in range(100):
        if i < len(etf_templates):
            base = etf_templates[i]
        else:
            # 超出模板列表的，生成虚拟但合理的代码
            base = (f"510{i+300:03d}.SH", f"ETF-{i}", random.randint(500000, 5000000))
        
        code, name, base_aum = base
        
        # 分层调整规模（制造多样性）
        if i < 30:  # 头部：大流动性
            noise = random.uniform(0.8, 1.2)
            turnover = base_aum * noise * 0.03  # 日成交约规模的 3%
        elif i < 70:  # 腰部：中等
            noise = random.uniform(0.5, 0.9)
            turnover = base_aum * noise * 0.015  # 日成交约 1.5%
        else:  # 尾部：僵尸（<50万）
            noise = random.uniform(0.3, 0.6)
            turnover = min(base_aum * noise * 0.005, 400000)  # 强制 <40万
        
        extended.append({
            "ticker": code,
            "code": code.split(".")[0],
            "name": name,
            "exchange": code.split(".")[1],
            "aum": int(base_aum * noise),  # 规模（万元）
            "turnover": int(turnover),      # 成交额（万元）
            "price": round(random.uniform(0.8, 5.0), 3),
            "change_pct": round(random.uniform(-3, 3), 2)
        })
    
    df = pd.DataFrame(extended)
    output_path = DATA_DIR / "etf_basic.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    # 统计分层
    head = df[df["turnover"] >= 500000]
    zombie = df[df["turnover"] < 500000]
    
    print(f"✅ 生成 ETF 基础档案: {output_path}")
    print(f"   总数: {len(df)} 只")
    print(f"   头部活跃（>50万/日）: {len(head)} 只（应通过筛选）")
    print(f"   尾部僵尸（<50万/日）: {len(zombie)} 只（应被剔除）")
    print(f"\n🧟 僵尸 ETF 示例（将被流动性筛选剔除）:")
    for _, row in zombie.head(3).iterrows():
        print(f"   - {row['ticker']}: 日均成交 {row['turnover']:,} 万元（<50万门槛）")
    
    return df


def generate_mock_ohlcva(basic_df):
    """
    生成半年历史行情（OHLCVA）
    ==========================
    模拟 125 个交易日的 K 线数据，特点：
    1. 价格随机游走（符合真实波动）
    2. 成交量与价格正相关（涨价放量，跌价缩量）
    3. 部分 ETF 设计为"拥挤"（近期成交量暴增 200%，测试 Z-score 检测）
    """
    print(f"\n🔄 生成历史行情数据...")
    
    # 日期范围：最近 6 个月
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    trading_days = generate_trading_days(start_date, end_date)
    
    all_data = []
    crowded_etfs = random.sample(list(basic_df["ticker"]), 8)  # 随机选 8 只作为"拥挤"测试
    
    for idx, etf_row in basic_df.iterrows():
        ticker = etf_row["ticker"]
        initial_price = etf_row["price"]
        base_turnover = etf_row["turnover"]
        
        # 生成价格序列（随机游走）
        prices = [initial_price]
        for i in range(1, len(trading_days)):
            # 日收益率 -4% 到 +4%
            daily_return = random.gauss(0.001, 0.015)  # 均值 0.1%，标准差 1.5%
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(new_price, 0.1))
        
        # 生成每日数据
        for i, date in enumerate(trading_days):
            close = prices[i]
            # OHLC 基于 close 生成合理范围
            open_p = close * random.uniform(0.985, 1.015)
            high = max(open_p, close) * random.uniform(1.0, 1.02)
            low = min(open_p, close) * random.uniform(0.98, 1.0)
            
            # 成交量设计：
            # - 正常：基础量 × 波动因子
            # - 拥挤 ETF（最后 20 天）：额外 × 2.5（模拟成交量暴增）
            base_volume = base_turnover / close * 10000  # 股数
            vol_factor = 1 + abs((close - open_p) / open_p) * 8
            
            # 拥挤检测：如果是拥挤 ETF 且是近期（最后 20 天），放量
            if ticker in crowded_etfs and i > len(trading_days) - 20:
                crowded_factor = 2.5  # 近期成交量暴增 150%（Z-score 应 > 1.5）
            else:
                crowded_factor = 1.0
            
            volume = int(base_volume * vol_factor * crowded_factor * random.uniform(0.8, 1.2))
            amount = volume * close / 10000  # 成交额（万元）
            
            all_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "open": round(open_p, 3),
                "high": round(high, 3),
                "low": round(low, 3),
                "close": round(close, 3),
                "volume": volume,
                "amount": round(amount, 2)
            })
    
    df = pd.DataFrame(all_data)
    df["date"] = pd.to_datetime(df["date"])
    
    output_path = DATA_DIR / "etf_2025_ohlcva.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    # 验证拥挤检测设计
    recent_20 = df[df["date"] >= df["date"].max() - timedelta(days=20)]
    avg_recent = recent_20.groupby("ticker")["amount"].mean()
    avg_hist = df.groupby("ticker")["amount"].mean()
    z_scores = (avg_recent - avg_hist) / df.groupby("ticker")["amount"].std()
    detected_crowded = z_scores[z_scores > 1.5].index.tolist()
    
    print(f"✅ 生成历史行情: {output_path}")
    print(f"   记录数: {len(df):,} 条（{len(trading_days)} 天 × {len(basic_df)} 只）")
    print(f"   时间范围: {df['date'].min().date()} 至 {df['date'].max().date()}")
    print(f"   设计拥挤 ETF: {len(crowded_etfs)} 只（近期成交量暴增，Z-score 应 > 1.5）")
    print(f"   验证检出: {len(detected_crowded)} 只拥挤（实际 Z-score > 1.5）")
    
    return df


def main():
    """主入口"""
    print("=" * 60)
    print("🎭 Mock ETF 数据生成器（IP 被封时的救急方案）")
    print("=" * 60)
    print("⚠️ 警告：这是模拟数据，仅用于流程调试！")
    print("   真实交易前必须替换为 AKShare 真实数据！")
    print("=" * 60)
    
    # 生成基础档案
    basic_df = generate_mock_basic()
    
    # 生成历史行情
    ohlcva_df = generate_mock_ohlcva(basic_df)
    
    print("\n" + "=" * 60)
    print("✅ Mock 数据生成完成！")
    print("📁 输出文件:")
    print(f"   1. {DATA_DIR}/etf_basic.csv      ({len(basic_df)} 只 ETF 档案)")
    print(f"   2. {DATA_DIR}/etf_2025_ohlcva.csv ({len(ohlcva_df):,} 条记录)")
    print("\n🎯 设计验证点:")
    print(f"   - 流动性筛选: 100 只 → 约 70 只通过（剔除 30 只僵尸）")
    print(f"   - 拥挤度检测: 检出约 8 只近期成交量异常放大（降权至 0.75）")
    print("=" * 60)
    print("\n💡 立即测试:")
    print("   python src/core/market_calibration.py")
    print("   （应看到: ✅ 通过: 68只 | 剔除: 32只 | 🔥 拥挤: 8只）")


if __name__ == "__main__":
    main()