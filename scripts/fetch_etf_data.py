#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF 数据获取脚本 (fetch_etf_data.py) - 最终生产版
===================================================
功能：
1. 从 AKShare 下载 Top 100 ETF 的半年历史行情（OHLCVA）
2. 保存全市场 ETF 基础信息（etf_basic.csv，供 LLM 打标使用）
3. 智能缓存：24 小时内重复运行自动提示使用缓存，避免重复请求被封 IP
4. 防封机制：随机间隔 1.5-2.0 秒、失败重试、连续失败冷却

数据范围：
- 半年数据（最近 6 个月，约 125 个交易日）
- 轻量快速，降低被封风险

使用方式：
    python scripts/fetch_etf_data.py              # 首次下载或自动使用缓存
    python scripts/fetch_etf_data.py --force      # 强制刷新（24 小时内慎用）
"""

import time
import random
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import akshare as ak
from loguru import logger


# -------------------- 配置区 --------------------
DATA_DIR = Path("data/etf")          # 数据保存目录
TOP_N_ETF = 100                      # 下载历史行情的 ETF 数量

# 动态计算半年日期范围（最近 6 个月）
END_DATE = datetime.now().strftime("%Y%m%d")                               # 今天，如 20250327
START_DATE = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")     # 6 个月前

# 防封配置：降低请求频率，避免被封
MIN_SLEEP = 1.5                      # 最小间隔（秒），原 0.3 秒太快被封
MAX_SLEEP = 2.0                      # 最大间隔（秒），随机抖动避免规律
MAX_RETRIES = 2                      # 单只 ETF 失败重试次数（共 3 次尝试）
FAIL_THRESHOLD = 5                   # 连续失败多少次触发冷却
COOLDOWN_TIME = 15                   # 冷却时间（秒），给服务器喘息

# 缓存配置
CACHE_HOURS = 24                     # 缓存有效期（小时），24 小时内不重复下载


def check_cache(force_refresh: bool = False) -> bool:
    """
    智能缓存检查
    ============
    检查本地是否已有当天数据，避免重复请求被封 IP。
    
    逻辑：
    - 检查 etf_basic.csv 和 etf_2025_ohlcva.csv 是否存在
    - 检查文件修改时间是否在 24 小时内
    - 如果 force_refresh=True，跳过检查强制重新下载
    
    返回：
        True: 使用缓存，跳过下载
        False: 需要重新下载
    """
    if force_refresh:
        logger.info("🔄 强制刷新模式（--force），跳过缓存检查")
        return False
    
    basic_file = DATA_DIR / "etf_basic.csv"
    ohlcva_file = DATA_DIR / "etf_2025_ohlcva.csv"
    
    # 文件不存在，必须下载
    if not basic_file.exists() or not ohlcva_file.exists():
        logger.info("📭 未找到本地缓存，需要重新下载")
        return False
    
    # 检查文件修改时间
    try:
        mtime = datetime.fromtimestamp(ohlcva_file.stat().st_mtime)
        age_hours = (datetime.now() - mtime).total_seconds() / 3600
        
        # 如果在 24 小时内，使用缓存
        if age_hours < CACHE_HOURS:
            logger.success(f"✅ 发现 {int(age_hours)} 小时前下载的缓存数据（仍在有效期内）")
            logger.info(f"   文件: {ohlcva_file} ({ohlcva_file.stat().st_size / 1024:.1f} KB)")
            
            # 提示用户如何避免被封
            print(f"\n💡 提示: 24 小时内已下载过数据")
            print(f"   使用缓存可避免重复请求被封 IP")
            print(f"   如需强制更新，请运行: python scripts/fetch_etf_data.py --force")
            
            return True
        else:
            logger.info(f"📅 缓存已过期（{int(age_hours)} 小时前），重新下载")
            return False
            
    except Exception as e:
        logger.warning(f"⚠️ 缓存检查失败: {e}，重新下载")
        return False


def ensure_dir():
    """确保数据目录存在（如果不存在则创建）"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 数据目录就绪: {DATA_DIR}")


def save_etf_basic(df: pd.DataFrame):
    """
    保存 ETF 基础信息表（全市场档案）
    ==================================
    供后续 LLM 主题打标模块使用。
    
    包含字段：
    - ticker: 标准化代码（如 510300.SH），唯一标识
    - code: 数字代码（510300）
    - name: ETF 简称（如 沪深300ETF）← 核心字段，LLM 通过名称识别主题
    - exchange: 交易所（SH/SZ）
    - aum: 规模（万元），判断是否为迷你基金（<5000 万有风险）
    - turnover: 成交额（万元），判断流动性
    
    💡 为什么这些字段足够支撑打标？
    中国 ETF 命名极度规范："指数名+ETF"（如"芯片ETF"、"沪深300ETF"）
    LLM 通过 name 字段即可准确推断：
    - 资产类别（股票/债券/商品）
    - 主题（科技/消费/医药/宽基）
    - 风格（大盘/小盘/价值/成长）
    """
    if df.empty:
        return
    
    logger.info("💾 保存 ETF 基础信息表...")
    
    # 标准化列名（AKShare 返回中文，转为英文方便后续模块读取）
    column_map = {
        "代码": "code",
        "名称": "name",
        "成交额": "turnover",
        "成交金额": "turnover",  # 备选列名
        "流通市值": "aum",       # Assets Under Management，规模
        "最新价": "price",
        "涨跌幅": "change_pct"
    }
    
    for cn, en in column_map.items():
        if cn in df.columns:
            df = df.rename(columns={cn: en})
    
    # 构建标准化 ticker（加 .SH/.SZ 后缀）
    def format_ticker(row):
        code = str(row["code"]).strip()
        # 判断交易所：51/58/56 开头是上海，其他是深圳
        exchange = "SH" if code.startswith(("51", "58", "56")) else "SZ"
        return f"{code}.{exchange}"
    
    df["ticker"] = df.apply(format_ticker, axis=1)
    df["exchange"] = df["ticker"].apply(lambda x: x.split(".")[1])
    
    # 选择需要的列（确保核心字段存在）
    core_cols = ["ticker", "code", "name", "exchange"]
    optional_cols = ["turnover", "aum", "price", "change_pct"]
    final_cols = [c for c in core_cols + optional_cols if c in df.columns]
    
    basic_df = df[final_cols].copy()
    
    # 数据清洗：确保数值列为数字类型（不是字符串）
    for col in ["turnover", "aum", "price"]:
        if col in basic_df.columns:
            basic_df[col] = pd.to_numeric(basic_df[col], errors="coerce")
    
    # 保存为 CSV（utf-8-sig 编码让 Excel 正确显示中文）
    output_path = DATA_DIR / "etf_basic.csv"
    basic_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    logger.success(f"✅ 已保存 ETF 基础信息: {output_path} ({len(basic_df)} 只)")
    
    # 展示样本，确认可用于后续打标
    print(f"\n📋 基础信息样本（供 LLM 打标使用）:")
    print(basic_df.head(3)[["ticker", "name", "aum"]].to_string())
    print(f"\n💡 后续 theme_mapper.py 将读取此文件，通过 'name' 字段推断主题")


def get_etf_list() -> pd.DataFrame:
    """
    获取 ETF 实时列表（全市场概况）
    ================================
    使用 AKShare 的 fund_etf_spot_em() 接口获取所有 ETF 的实时快照。
    
    失败处理：
    - 如果提示 Connection aborted / RemoteDisconnected，说明 IP 被封
    - 给出友好提示，建议等待 15 分钟后重试
    """
    logger.info("🔍 获取 ETF 基础列表...")
    
    try:
        df = ak.fund_etf_spot_em()
        logger.info(f"📊 全市场共 {len(df)} 只 ETF")
        return df
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ 获取列表失败: {error_msg}")
        
        # 检测是否被封 IP（连接被重置）
        if "Connection aborted" in error_msg or "RemoteDisconnected" in error_msg:
            print(f"\n⚠️  你的 IP 可能被东方财富临时封禁（反爬机制）")
            print(f"   通常封 5-15 分钟，建议:")
            print(f"   1. 立即停止运行，等待 15 分钟")
            print(f"   2. 不要频繁重试（会加重封禁）")
            print(f"   3. 15 分钟后重新运行: python scripts/fetch_etf_data.py")
            print(f"   4. 脚本会自动使用缓存，避免重复请求")
        
        # 返回空 DataFrame，让上层逻辑处理
        return pd.DataFrame()


def select_top_etfs(df: pd.DataFrame, top_n: int = TOP_N_ETF) -> List[dict]:
    """
    筛选 Top N ETF（按成交额排序）
    ================================
    从全市场 ETF 中挑选成交额最大的前 N 只作为候选池。
    
    为什么选择成交额而非市值？
    - 成交额反映真实交易活跃度（流动性）
    - 市值大的 ETF 可能是机构长期持有，交易不一定活跃
    
    返回：
        字典列表，每个包含 code, name, market, amount
    """
    if df.empty:
        return []
    
    logger.info(f"🏆 选择 Top {top_n} 流动性最好的 ETF...")
    
    # 确定列名（AKShare 不同版本列名可能略有差异）
    code_col = "代码" if "代码" in df.columns else df.columns[0]
    name_col = "名称" if "名称" in df.columns else df.columns[1]
    
    # 寻找成交额列
    amount_col = None
    for col in df.columns:
        if "成交" in col or "金额" in col:
            amount_col = col
            break
    if not amount_col:
        amount_col = df.columns[3]  # 默认第 4 列通常是成交额
    
    # 按成交额降序排序，取前 N
    df_sorted = df.sort_values(by=amount_col, ascending=False)
    top_df = df_sorted.head(top_n)
    
    # 构建结果列表
    selected = []
    for _, row in top_df.iterrows():
        code = str(row[code_col]).strip()
        name = str(row[name_col]).strip()
        # 判断交易所：51/58/56 开头是上海，其他（通常是 15/16 开头）是深圳
        market = "SH" if code.startswith(("51", "58", "56")) else "SZ"
        
        selected.append({
            "code": code,
            "name": name,
            "market": market,
            "amount": row[amount_col]
        })
    
    # 日志输出头部和尾部，展示候选池的多样性
    logger.success(f"✅ 候选池构建完成: {len(selected)} 只 ETF")
    logger.info(f"   第 1 名: {selected[0]['code']}.{selected[0]['market']} - {selected[0]['name']}")
    logger.info(f"   第 {len(selected)} 名: {selected[-1]['code']}.{selected[-1]['market']} - {selected[-1]['name']}")
    
    return selected


def download_etf_history(etf_code: str, market: str, name: str) -> Optional[pd.DataFrame]:
    """
    下载单只 ETF 的历史行情（单次尝试）
    =====================================
    使用 AKShare 的 fund_etf_hist_em() 接口获取日频数据。
    
    参数：
        etf_code: ETF 数字代码（如 510300）
        market: 交易所（SH/SZ，AKShare 用 "1" 表示上海，"0" 表示深圳）
        name: ETF 名称（仅用于日志）
    
    返回：
        DataFrame 包含 date, ticker, open, high, low, close, volume, amount
        失败返回 None（触发外层重试）
    """
    # AKShare 市场代码转换：SH -> "1", SZ -> "0"
    market_code = "1" if market == "SH" else "0"
    
    try:
        # 调用 AKShare 获取历史 K 线（日频）
        df = ak.fund_etf_hist_em(
            symbol=etf_code,
            period="daily",           # 日频
            start_date=START_DATE,    # 半年前的日期
            end_date=END_DATE,        # 今天
            adjust=""                 # 不复权（如需前复权可改为 "qfq"）
        )
        
        # 如果返回空表，说明该 ETF 可能已退市或数据缺失
        if df.empty:
            return None
        
        # 标准化列名（中文 -> 英文，方便后续模块统一处理）
        df = df.rename(columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount"
        })
        
        # 添加标准化 ticker 列（如 510300.SH）
        df["ticker"] = f"{etf_code}.{market}"
        
        # 确保日期格式正确（转为 datetime 对象）
        df["date"] = pd.to_datetime(df["date"])
        
        # 确保数值列为数字类型（防止 AKShare 返回字符串导致计算错误）
        numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # 返回需要的列（按标准顺序）
        return df[["date", "ticker", "open", "high", "low", "close", "volume", "amount"]]
        
    except Exception as e:
        # 捕获所有异常（包括网络错误、RemoteDisconnected 等），返回 None 触发重试
        return None


def download_with_retry(etf: dict) -> Optional[pd.DataFrame]:
    """
    带重试机制的 ETF 数据下载（防封核心逻辑）
    =========================================
    外层包装：失败自动重试，带指数退避等待。
    
    策略：
    - 第 1 次失败：等待 3-4 秒（随机抖动）
    - 第 2 次失败：等待 6-7 秒（随机抖动）
    - 第 3 次仍失败：放弃，记录为失败
    
    参数：
        etf: 字典，包含 code, name, market
    
    返回：
        DataFrame 或 None（重试后仍失败）
    """
    for attempt in range(MAX_RETRIES + 1):  # 0, 1, 2（共 3 次尝试）
        # 尝试下载
        df = download_etf_history(etf["code"], etf["market"], etf["name"])
        
        # 成功，立即返回
        if df is not None:
            return df
        
        # 失败，且不是最后一次尝试，则等待后重试
        if attempt < MAX_RETRIES:
            # 退避策略：第 1 次等 3 秒，第 2 次等 6 秒（给服务器喘息）
            # 加随机抖动（0-1 秒），避免请求规律化被识别为爬虫
            wait_time = 3 * (attempt + 1) + random.uniform(0, 1)
            
            logger.warning(f"   ⚠️  第 {attempt + 1} 次失败，等待 {wait_time:.1f} 秒后重试...")
            time.sleep(wait_time)
    
    # 所有重试用完仍失败，返回 None
    return None


def main(force_refresh: bool = False):
    """
    主流程（整合缓存检查、数据获取、防封机制）
    ===========================================
    执行步骤：
    1. 检查缓存（24 小时内避免重复请求）
    2. 获取 ETF 列表并保存基础信息
    3. 筛选 Top 100 下载历史行情（带防封逻辑）
    4. 合并保存
    
    参数：
        force_refresh: 是否强制刷新（True 则忽略缓存）
    """
    # 显示日期范围（友好提示）
    start_display = datetime.strptime(START_DATE, "%Y%m%d").strftime("%Y-%m-%d")
    end_display = datetime.strptime(END_DATE, "%Y%m%d").strftime("%Y-%m-%d")
    
    print("=" * 60)
    print("🚀 ETF 数据下载（半年数据 + 智能缓存 + 防封机制）")
    print(f"📅 行情范围: {start_display} 至 {end_display}（半年约 125 个交易日）")
    print(f"💾 输出文件: etf_basic.csv（全市场档案）+ etf_2025_ohlcva.csv（历史行情）")
    print("=" * 60)
    
    # 配置日志（简化格式，带时间戳和颜色）
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), format="<green>{time:HH:mm:ss}</green> | {message}")
    
    # 确保目录存在
    ensure_dir()
    
    # 🔥 步骤 1：缓存检查（关键防封措施）
    if check_cache(force_refresh):
        print(f"\n✅ 使用本地缓存数据，跳过下载")
        print(f"💡 如需更新数据，请明天再试或添加 --force 参数")
        return  # 直接退出，不执行下载
    
    # 步骤 2：获取全市场 ETF 列表
    etf_list_df = get_etf_list()
    if etf_list_df.empty:
        logger.error("❌ 无法获取 ETF 列表，退出")
        return  # 上层函数已打印被封提示
    
    # 步骤 3：保存基础信息（全市场，供后续 LLM 打标）
    save_etf_basic(etf_list_df)
    
    # 步骤 4：筛选 Top 100 下载历史行情
    selected_etfs = select_top_etfs(etf_list_df, TOP_N_ETF)
    if not selected_etfs:
        logger.error("❌ 未选中任何 ETF，退出")
        return
    
    # 步骤 5：循环下载（带防封逻辑）
    all_data = []        # 存储成功的 DataFrame
    fail_count = 0       # 连续失败计数器（用于冷却检测）
    total_failed = 0     # 总失败计数（用于最终报告）
    
    for i, etf in enumerate(selected_etfs, 1):
        # 进度显示：如 [010/100] 510300.SH (沪深300ETF)
        logger.info(f"⬇️  [{i:3d}/{len(selected_etfs)}] {etf['code']}.{etf['market']} ({etf['name'][:8]})...")
        
        # 使用带重试的下载函数
        df = download_with_retry(etf)
        
        if df is not None:
            # 成功：保存数据，重置连续失败计数
            all_data.append(df)
            logger.success(f"   ✅ 成功，{len(df)} 条记录")
            fail_count = 0
        else:
            # 失败：增加计数，记录日志
            logger.error(f"   ❌ 放弃（重试 {MAX_RETRIES} 次后仍失败）")
            fail_count += 1
            total_failed += 1
        
        # 🔥 防封策略 1：连续失败检测（IP 可能被封，需要冷却）
        if fail_count >= FAIL_THRESHOLD:
            logger.warning(f"⏸️  连续 {FAIL_THRESHOLD} 次失败，触发防封冷却（暂停 {COOLDOWN_TIME} 秒）...")
            time.sleep(COOLDOWN_TIME)
            fail_count = 0  # 重置计数器，继续尝试
        
        # 🔥 防封策略 2：请求间隔（最后一只不需要等待）
        if i < len(selected_etfs):
            # 随机间隔 1.5-2.0 秒（带抖动，避免规律化请求）
            sleep_time = random.uniform(MIN_SLEEP, MAX_SLEEP)
            time.sleep(sleep_time)
    
    # 步骤 6：保存合并后的历史数据
    if not all_data:
        logger.error("❌ 全部下载失败，建议等待 15 分钟后重试")
        return
    
    # 合并所有 ETF 的数据
    combined = pd.concat(all_data, ignore_index=True)
    # 按日期和 ticker 排序（方便后续查询）
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
    
    # 保存到 CSV
    output_file = DATA_DIR / "etf_2025_ohlcva.csv"
    combined.to_csv(output_file, index=False, encoding="utf-8-sig")
    
    # 步骤 7：统计报告
    success_count = len(all_data)
    success_rate = success_count / len(selected_etfs) * 100
    
    print(f"\n📊 下载完成报告:")
    print(f"   目标数量: {len(selected_etfs)} 只 ETF")
    print(f"   成功下载: {success_count} 只 ({success_rate:.1f}%)")
    print(f"   失败数量: {total_failed} 只")
    print(f"   总记录数: {len(combined):,} 条")
    print(f"   交易日数: {combined['date'].nunique()} 天")
    print(f"   时间范围: {combined['date'].min().date()} 至 {combined['date'].max().date()}")
    print(f"   文件大小: {output_file.stat().st_size / 1024:.1f} KB")
    
    # 成功率提示
    if success_rate >= 90:
        print(f"\n✅ 优秀！成功率 {success_rate:.0f}%，数据质量高")
    elif success_rate >= 70:
        print(f"\n✅ 良好，数据可用（成功率 {success_rate:.0f}%）")
    else:
        print(f"\n⚠️  成功率仅 {success_rate:.0f}%，样本偏少，建议稍后重试")
    
    print(f"\n📁 输出文件清单:")
    print(f"   1. {DATA_DIR}/etf_basic.csv      (ETF 基础档案，{len(etf_list_df)} 只，供 LLM 打标)")
    print(f"   2. {DATA_DIR}/etf_2025_ohlcva.csv (历史行情，{success_count} 只半年数据，供策略计算)")
    
    print(f"\n💡 后续步骤:")
    print(f"   python src/core/market_calibration.py  （流动性筛选，剔除僵尸 ETF）")
    print(f"   python src/agents/theme_mapper.py      （读取 etf_basic.csv 进行 LLM 主题打标）")


if __name__ == "__main__":
    # 命令行参数解析：支持 --force 强制刷新
    parser = argparse.ArgumentParser(description="下载 ETF 历史数据（半年数据 + 智能缓存）")
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="强制刷新，忽略 24 小时缓存（慎用，可能再次被封 IP）"
    )
    args = parser.parse_args()
    
    # 执行主函数
    main(force_refresh=args.force)