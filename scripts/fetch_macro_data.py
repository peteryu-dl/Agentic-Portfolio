#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据获取脚本 (fetch_macro_data.py)
==================================
从 AKShare 下载中国宏观经济指标（PMI、CPI、PPI）。

数据来源：
- PMI: 国家统计局制造业采购经理指数（月度，每月第一个工作日发布）
- CPI: 居民消费价格指数（月度，每月9-10号发布）
- PPI: 工业生产者出厂价格指数（月度，与CPI同期发布）

AKShare 实际返回列名映射：
- PMI: 月份 -> date, 制造业-指数 -> pmi
- CPI: 月份 -> date, 全国-同比增长 -> cpi_yoy
- PPI: 月份 -> date, 当月同比增长 -> ppi_yoy
"""

from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import akshare as ak
from loguru import logger


# -------------------- 配置区 --------------------
DATA_DIR = Path("data/macro_data")          # 原始数据存放处
PROCESSED_DIR = Path("data/processed_macro_data")  # 处理后数据存放处
YEARS_HISTORY = 2                           # 获取最近2年历史数据


def ensure_dirs():
    """创建必要的文件夹（如果不存在）"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def parse_date(date_series):
    """
    解析中文日期格式
    =================
    将 "2026年02月份" 转换为标准 datetime 对象。
    
    处理步骤：
    1. 转为字符串
    2. 去掉"份"字（Python日期格式不支持"月份"）
    3. 使用中文格式解析（%Y年%m月）
    """
    date_series = date_series.astype(str).str.replace("份", "", regex=False)
    return pd.to_datetime(date_series, format="%Y年%m月", errors="coerce")


def fetch_pmi() -> pd.DataFrame:
    """
    获取 PMI 数据（制造业采购经理指数）
    ===================================
    PMI 是经济的"体检表"，50是荣枯分界线。
    
    关键指标：
    - 制造业-指数: 制造业综合 PMI（我们用的核心指标）
    - 非制造业-指数: 服务业 PMI（暂不使用）
    """
    logger.info("🔍 获取 PMI...")
    
    try:
        df = ak.macro_china_pmi()
        
        # 选择并重命名列（根据实际列名精确映射）
        df = df[["月份", "制造业-指数"]].copy()           # 只取需要的两列
        df.columns = ["date", "pmi"]                       # 重命名为英文方便处理
        
        # 数据类型转换
        df["date"] = parse_date(df["date"])                # 中文日期转标准日期
        df["pmi"] = pd.to_numeric(df["pmi"], errors="coerce")  # 确保是数字
        
        # 清洗：删除无效行
        df = df.dropna(subset=["date", "pmi"])
        
        # 只保留最近N年（策略不需要太久远的数据）
        start_date = datetime.now() - timedelta(days=365 * YEARS_HISTORY)
        df = df[df["date"] >= start_date].sort_values("date")
        
        # 记录日志：显示最新数据点
        latest = df.iloc[-1]
        logger.success(f"✅ PMI: {len(df)}条 | 最新 {latest['date'].strftime('%Y-%m')}: {latest['pmi']}")
        return df
        
    except Exception as e:
        logger.error(f"❌ PMI 获取失败: {e}")
        return pd.DataFrame(columns=["date", "pmi"])


def fetch_cpi() -> pd.DataFrame:
    """
    获取 CPI 数据（居民消费价格指数同比）
    =====================================
    CPI 衡量通胀水平，同比是和去年同期比（消除季节性）。
    
    关键指标：
    - 全国-同比增长: 全国CPI同比涨幅（如1.3表示1.3%）
    """
    logger.info("🔍 获取 CPI...")
    
    try:
        df = ak.macro_china_cpi()
        
        # 精确列名映射（根据实际返回格式）
        df = df[["月份", "全国-同比增长"]].copy()
        df.columns = ["date", "cpi_yoy"]
        
        df["date"] = parse_date(df["date"])
        df["cpi_yoy"] = pd.to_numeric(df["cpi_yoy"], errors="coerce")
        df = df.dropna(subset=["date", "cpi_yoy"])
        
        start_date = datetime.now() - timedelta(days=365 * YEARS_HISTORY)
        df = df[df["date"] >= start_date].sort_values("date")
        
        latest = df.iloc[-1]
        logger.success(f"✅ CPI: {len(df)}条 | 最新 {latest['date'].strftime('%Y-%m')}: {latest['cpi_yoy']}%")
        return df
        
    except Exception as e:
        logger.error(f"❌ CPI 获取失败: {e}")
        return pd.DataFrame(columns=["date", "cpi_yoy"])


def fetch_ppi() -> pd.DataFrame:
    """
    获取 PPI 数据（工业生产者出厂价格指数同比）
    ==========================================
    PPI 衡量工业企业产品出厂价格，反映上游通胀压力。
    
    关键指标：
    - 当月同比增长: PPI同比（负值表示工业品降价）
    """
    logger.info("🔍 获取 PPI...")
    
    try:
        df = ak.macro_china_ppi()
        
        # 精确列名映射（注意PPI用的是"当月"和"当月同比增长"）
        df = df[["月份", "当月同比增长"]].copy()
        df.columns = ["date", "ppi_yoy"]
        
        df["date"] = parse_date(df["date"])
        df["ppi_yoy"] = pd.to_numeric(df["ppi_yoy"], errors="coerce")
        df = df.dropna(subset=["date", "ppi_yoy"])
        
        start_date = datetime.now() - timedelta(days=365 * YEARS_HISTORY)
        df = df[df["date"] >= start_date].sort_values("date")
        
        latest = df.iloc[-1]
        logger.success(f"✅ PPI: {len(df)}条 | 最新 {latest['date'].strftime('%Y-%m')}: {latest['ppi_yoy']}%")
        return df
        
    except Exception as e:
        logger.error(f"❌ PPI 获取失败: {e}")
        return pd.DataFrame(columns=["date", "ppi_yoy"])


def merge_data(datasets: dict) -> pd.DataFrame:
    """
    合并各指标数据
    ==============
    将 PMI、CPI、PPI 按日期对齐，合并成一张宽表（每月一行，多列）。
    
    合并策略：
    - 以 CPI 为基准（CPI发布日期最稳定，每月9-10号）
    - 左连接其他指标（保留CPI所有月份，其他指标缺失则留空）
    - 实际运行中三个指标日期基本对齐（都是月度数据）
    """
    logger.info("🔄 合并数据...")
    
    # 以 CPI 为基准（它最稳定）
    if "cpi" not in datasets or datasets["cpi"].empty:
        logger.error("❌ CPI 数据缺失，无法合并")
        return pd.DataFrame()
    
    merged = datasets["cpi"].copy()
    
    # 左连接 PMI（日期对齐，保留所有月份）
    if "pmi" in datasets and not datasets["pmi"].empty:
        merged = merged.merge(datasets["pmi"], on="date", how="outer")
        logger.info(f"   合并 PMI: {len(datasets['pmi'])}条")
    
    # 左连接 PPI
    if "ppi" in datasets and not datasets["ppi"].empty:
        merged = merged.merge(datasets["ppi"], on="date", how="outer")
        logger.info(f"   合并 PPI: {len(datasets['ppi'])}条")
    
    # 按日期排序（从早到晚）
    merged = merged.sort_values("date").reset_index(drop=True)
    
    return merged


def save_data(datasets: dict, merged: pd.DataFrame):
    """
    保存数据到文件
    ==============
    保存两份：
    1. 分开的原始数据（便于单独查看某个指标）
    2. 合并后的宽表（策略直接读取这个）
    
    编码使用 utf-8-sig，让 Excel 能正确识别中文（不会乱码）。
    """
    # 保存原始分开的数据
    for name, df in datasets.items():
        if not df.empty:
            path = DATA_DIR / f"{name}_data.csv"
            df.to_csv(path, index=False, encoding="utf-8-sig")
            logger.info(f"💾 已保存: {path}")
    
    # 保存合并后的主数据（策略将读取这个文件）
    merged_path = PROCESSED_DIR / "merged_macro_data.csv"
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    logger.success(f"💾 合并数据: {merged_path} ({len(merged)}条)")


def print_report(df: pd.DataFrame):
    """
    打印数据质量报告
    ================
    显示：
    - 时间范围（最早到最晚）
    - 每个指标的有效数据条数（缺失值统计）
    - 最新一期的数据快照（快速验证）
    """
    if df.empty:
        return
    
    print("\n📊 数据质量报告:")
    print(f"   时间范围: {df['date'].min().strftime('%Y-%m')} 至 {df['date'].max().strftime('%Y-%m')}")
    print(f"   总记录数: {len(df)} 个月")
    
    # 各指标数据完整度统计
    for col in ["cpi_yoy", "pmi", "ppi_yoy"]:
        if col in df.columns:
            valid = df[col].notna().sum()
            pct = valid / len(df) * 100
            print(f"   {col}: {valid}/{len(df)} 条有效 ({pct:.0f}%)")
    
    # 显示最新数据（快速验证）
    print(f"\n📈 最新数据 ({df['date'].iloc[-1].strftime('%Y-%m')}):")
    latest = df.iloc[-1]
    if "pmi" in latest and pd.notna(latest["pmi"]):
        print(f"   PMI: {latest['pmi']:.1f} ({'扩张' if latest['pmi'] > 50 else '收缩'})")
    if "cpi_yoy" in latest and pd.notna(latest["cpi_yoy"]):
        print(f"   CPI: {latest['cpi_yoy']:.1f}%")
    if "ppi_yoy" in latest and pd.notna(latest["ppi_yoy"]):
        print(f"   PPI: {latest['ppi_yoy']:.1f}%")


def main():
    """
    主入口
    ======
    执行流程：
    1. 创建目录
    2. 获取三个指标（PMI、CPI、PPI）
    3. 合并对齐
    4. 保存文件
    5. 打印报告
    """
    print("=" * 60)
    print("🚀 宏观数据获取工具")
    print("=" * 60)
    
    # 配置日志格式（带颜色、带时间）
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), 
               format="<green>{time:HH:mm:ss}</green> | {message}")
    
    # 第1步：准备文件夹
    ensure_dirs()
    
    # 第2步：获取数据（像去三个不同部门收报表）
    datasets = {
        "pmi": fetch_pmi(),
        "cpi": fetch_cpi(),
        "ppi": fetch_ppi()
    }
    
    # 第3步：检查核心数据（至少要有CPI才能判断通胀）
    if datasets["cpi"].empty:
        logger.error("❌ 核心指标 CPI 获取失败，无法继续")
        return
    
    # 第4步：合并与保存
    merged = merge_data(datasets)
    save_data(datasets, merged)
    
    # 第5步：报告
    print_report(merged)
    
    print("\n" + "=" * 60)
    print("✅ 数据获取完成！")
    print("💡 下一步: python src/core/macro_regime.py")
    print("   这次将使用真实数据判断经济周期！")
    print("=" * 60)


if __name__ == "__main__":
    main()