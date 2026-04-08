#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场校准模块 (market_calibration.py)
======================================

核心功能：
1. 流动性筛选：剔除日均成交额 < 1万元的"僵尸 ETF"（没人交易，买卖价差大）
2. 拥挤度检测：识别成交量突然暴增的 ETF（Z-score > 1.5），避免追高

"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass
from loguru import logger


# -------------------- 配置区（安检标准）--------------------
# 这些阈值可调，但默认经验值如下：

MIN_AVG_AMOUNT = 10000  # 最小日均成交额（万元），低于此值视为僵尸ETF
CROWDED_Z_THRESHOLD = 1.5  # 拥挤度Z-score阈值，超过此值视为过热
LOOKBACK_SHORT = 5         # 短期窗口（5日），用来计算近期成交量
LOOKBACK_LONG = 120        # 长期窗口（120日，约半年），用来计算基准成交量


@dataclass
class MarketCondition:
    """
    市场条件数据包
    ==============
    保存筛选后的结果，传给下游（投资组合构建模块）。
    
    属性：
        liquid_etfs: 合格ETF代码列表（通过流动性筛选的）
        crowded_adjustments: 拥挤度调整因子字典 {etf_code: 因子}
                            1.0 = 正常，0.75 = 拥挤（降低权重）
        stats: 统计信息（用于调试和报告）
    """
    liquid_etfs: List[str]
    crowded_adjustments: Dict[str, float]
    stats: Dict


def load_etf_data(target_date: str, data_path: str = "data/etf/etf_2025_ohlcva.csv") -> pd.DataFrame:
    """
    加载 ETF 量价数据
    ==================
    读取包含 OHLCVA 数据的 CSV 文件（Open, High, Low, Close, Volume, Amount）。
    
    预期列名：
    - date: 日期
    - ticker: ETF代码（如 510300.SH）
    - close: 收盘价
    - volume: 成交量（股数）
    - amount: 成交额（万元）
    - 其他...
    """
    file_path = Path(data_path)
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"❌ 找不到ETF数据文件: {file_path}\n"
            f"💡 请确保数据文件已放置在正确位置，或先运行数据获取脚本"
        )
    
    # 读取CSV，解析日期
    df = pd.read_csv(file_path, parse_dates=["date"])
    
    # 过滤：只保留目标日期之前的数据（避免用未来数据）
    target = pd.to_datetime(target_date)
    df = df[df["date"] <= target]
    
    logger.info(f"📊 加载ETF数据: {len(df)} 条记录，截至 {target.strftime('%Y-%m-%d')}")
    return df


def calculate_liquidity(df: pd.DataFrame, min_amount: float = MIN_AVG_AMOUNT) -> List[str]:
    """
    流动性筛选（剔除僵尸 ETF）
    ==========================
    计算过去 20 日平均成交额，剔除低于阈值的 ETF。
    
    逻辑：
    - 如果日均成交 < 1万元，说明几乎没人交易
    - 这种 ETF 买卖价差大（盘口差），且容易受大单影响价格
    
    参数：
        df: ETF 原始数据
        min_amount: 最小日均成交额（万元）
    
    返回：
        合格ETF代码列表（字符串列表）
    """
    logger.info(f"💧 开始流动性筛选（阈值: {min_amount:,} 万元）...")
    
    # 按ETF代码分组，计算每个ETF的20日平均成交额
    # groupby 就像把一堆扑克牌按花色分组
    liquidity_stats = []
    
    for ticker, group in df.groupby("ticker"):
        # 取最近20个交易日
        recent = group.tail(20)
        
        if len(recent) < 10:  # 数据太少（新上市或停牌），跳过
            continue
            
        avg_amount = recent["amount"].mean()  # 平均成交额（万元）
        
        liquidity_stats.append({
            "ticker": ticker,
            "avg_amount": avg_amount,
            "data_points": len(recent)
        })
    
    # 转为DataFrame方便查看
    stats_df = pd.DataFrame(liquidity_stats)
    
    # 筛选：平均成交额 >= 阈值
    liquid_etfs = stats_df[stats_df["avg_amount"] >= min_amount]["ticker"].tolist()
    
    # 记录被剔除的（用于调试）
    zombies = stats_df[stats_df["avg_amount"] < min_amount]
    if not zombies.empty:
        logger.warning(f"🧟 剔除僵尸ETF: {len(zombies)} 只")
        for _, row in zombies.head(3).iterrows():  # 只显示前3个
            logger.warning(f"   - {row['ticker']}: 日均成交 {row['avg_amount']:.0f} 万元")
    
    logger.success(f"✅ 流动性筛选通过: {len(liquid_etfs)} 只 ETF（剔除 {len(zombies)} 只）")
    return liquid_etfs


def detect_crowdedness(df: pd.DataFrame, liquid_etfs: List[str], 
                       z_threshold: float = CROWDED_Z_THRESHOLD) -> Dict[str, float]:
    """
    拥挤度检测（避免追高）
    ======================
    计算成交量 Z-score，识别近期交易异常活跃的 ETF。
    
    Z-score 计算公式：
    Z = (短期平均成交量 - 长期平均成交量) / 长期标准差
    
    解读：
    - Z > 1.5: 近期成交量比过去半年均值高 1.5 个标准差（异常活跃，可能过热）
    - Z < -1.5: 异常冷清（但这种情况较少，通常不处理）
    
    参数：
        df: 原始数据（过滤后的）
        liquid_etfs: 通过流动性筛选的ETF列表（只检测这些）
        z_threshold: Z-score阈值
    
    返回：
        调整因子字典 {ticker: factor}，拥挤的设为 0.75，正常为 1.0
    """
    logger.info(f"🔥 开始拥挤度检测（Z-score 阈值: {z_threshold}）...")
    
    adjustments = {}  # 结果字典
    
    # 只检查合格的ETF
    df_filtered = df[df["ticker"].isin(liquid_etfs)]
    
    crowded_count = 0
    
    for ticker, group in df_filtered.groupby("ticker"):
        # 确保数据按日期排序
        group = group.sort_values("date")
        
        # 需要至少 120+5 天数据才能计算
        if len(group) < LOOKBACK_LONG + LOOKBACK_SHORT:
            # 数据不足，默认不拥挤（给 1.0）
            adjustments[ticker] = 1.0
            continue
        
        # 计算成交量（这里用 amount 成交额更准确，也可改用 volume 股数）
        recent_vol = group["amount"].tail(LOOKBACK_SHORT).mean()  # 近5日平均
        long_vol = group["amount"].tail(LOOKBACK_LONG).mean()     # 近120日平均
        long_std = group["amount"].tail(LOOKBACK_LONG).std()      # 近120日标准差
        
        # 避免除以0（标准差为0意味着成交量完全没变化，极少见）
        if long_std == 0:
            z_score = 0
        else:
            z_score = (recent_vol - long_vol) / long_std
        
        # 判断拥挤度
        if z_score > z_threshold:
            adjustments[ticker] = 0.75  # 拥挤：降低权重至75%
            crowded_count += 1
            if crowded_count <= 3:  # 只记录前3个，避免日志太长
                logger.info(f"   ⚠️  {ticker} 拥挤 (Z={z_score:.2f}) -> 权重调整 0.75")
        else:
            adjustments[ticker] = 1.0  # 正常：保持100%
    
    logger.success(f"✅ 拥挤度检测完成: {crowded_count} 只拥挤, {len(liquid_etfs)-crowded_count} 只正常")
    return adjustments


def calibrate_market(target_date: str, 
                     data_path: str = "data/etf/etf_2025_ohlcva.csv") -> MarketCondition:
    """
    市场校准主函数
    ==============
    整合流动性筛选和拥挤度检测，输出最终可用的 ETF 池。
    
    流程：
    1. 加载数据 -> 2. 流动性筛选 -> 3. 拥挤度检测 -> 4. 打包结果
    
    参数：
        target_date: 目标日期（如 "2025-12-31"）
        data_path: ETF数据文件路径
    
    返回：
        MarketCondition 对象（包含合格ETF列表和调整因子）
    """
    logger.info(f"🔍 开始市场校准，目标日期: {target_date}")
    
    # 第1步：加载数据
    try:
        df = load_etf_data(target_date, data_path)
    except FileNotFoundError:
        logger.warning("⚠️ 未找到ETF数据，进入演示模式")
        return _demo_market_condition()
    
    if df.empty:
        logger.error("❌ ETF数据为空")
        return _demo_market_condition()
    
    # 第2步：流动性筛选（剔除僵尸）
    liquid_etfs = calculate_liquidity(df, MIN_AVG_AMOUNT)
    
    if not liquid_etfs:
        logger.error("❌ 没有ETF通过流动性筛选")
        return _demo_market_condition()
    
    # 第3步：拥挤度检测（识别过热）
    adjustments = detect_crowdedness(df, liquid_etfs, CROWDED_Z_THRESHOLD)
    
    # 第4步：组装结果
    stats = {
        "total_etfs": df["ticker"].nunique(),
        "liquid_etfs": len(liquid_etfs),
        "crowded_etfs": sum(1 for v in adjustments.values() if v < 1.0),
        "min_amount_threshold": MIN_AVG_AMOUNT,
        "z_threshold": CROWDED_Z_THRESHOLD
    }
    
    result = MarketCondition(
        liquid_etfs=liquid_etfs,
        crowded_adjustments=adjustments,
        stats=stats
    )
    
    logger.success(f"✅ 市场校准完成: {len(liquid_etfs)} 只合格")
    return result


def _demo_market_condition() -> MarketCondition:
    """
    演示模式（当没有真实数据时使用）
    """
    logger.info("🎭 使用演示数据（模拟市场条件）")
    
    # 模拟几只常见的宽基ETF
    demo_etfs = ["510300.SH", "510500.SH", "510050.SH", "159915.SZ", "588000.SH"]
    
    # 模拟调整因子（假设有一只拥挤）
    adjustments = {etf: 1.0 for etf in demo_etfs}
    adjustments["159915.SZ"] = 0.75  # 假设创业板ETF拥挤
    
    return MarketCondition(
        liquid_etfs=demo_etfs,
        crowded_adjustments=adjustments,
        stats={"mode": "demo", "total_etfs": len(demo_etfs), "crowded": 1}
    )


# ==================== 测试入口 ====================
if __name__ == "__main__":
    """
    直接运行此文件时的测试代码
    """
    import sys
    
    # 配置日志
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    
    print("=" * 60)
    print("🧪 市场校准模块 - 功能测试")
    print("=" * 60)
    
    test_date = "2025-12-31"
    
    try:
        # 执行校准
        condition = calibrate_market(test_date)
        
        print(f"\n📋 市场校准结果 ({test_date}):")
        print(f"  合格ETF数量: {len(condition.liquid_etfs)}")
        print(f"  拥挤ETF数量: {condition.stats.get('crowded_etfs', 0)}")
        
        print(f"\n🏷️  合格ETF列表（前5只）:")
        for etf in condition.liquid_etfs[:5]:
            adj = condition.crowded_adjustments.get(etf, 1.0)
            status = "拥挤(0.75)" if adj < 1.0 else "正常(1.0)"
            print(f"   - {etf}: {status}")
        
        print(f"\n📊 统计信息:")
        for key, val in condition.stats.items():
            print(f"   {key}: {val}")
            
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("💡 注意: 如果显示'演示模式'，说明尚未准备ETF数据文件")
    print("   数据格式要求: data/etf/etf_2025_ohlcva.csv")
    print("   列名: date, ticker, close, volume, amount")
    print("=" * 60)