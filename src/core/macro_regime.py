#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
宏观状态识别模块 (macro_regime.py)
===================================

核心逻辑（简化版美林时钟）：
想象经济是一台车：
- PMI（制造业指数）是车速表，>50 表示踩油门（扩张），<50 表示踩刹车（收缩）
- CPI（物价指数）是水温表，>2.5% 表示发动机过热（高通胀）

根据这两个表，我们把经济分为四季：
1. 复苏（春天）：车速加快 + 温度正常  → 猛买股票
2. 过热（夏天）：车速很快 + 温度很高  → 买点商品（黄金、原油）
3. 滞胀（秋天）：车速变慢 + 温度很高  → 防守为主
4. 衰退（冬天）：车速很慢 + 温度很低  → 持有债券
"""

# -------------------- 导入工具包 --------------------
# 这些是从"工具箱"里拿出的特定工具，每个都有专门用途

import json  # JSON处理器：把Python字典转换成字符串保存，或从字符串恢复成字典
from pathlib import Path  # 路径工具：处理文件路径，自动兼容Windows(\)和Mac(/)
from typing import Dict, Any, Optional  # 类型提示：给变量贴标签，帮助IDE自动补全和检查错误
from dataclasses import dataclass  # 数据类装饰器：自动帮你生成__init__、__repr__等方法，省代码

import pandas as pd  # 数据分析神器：类似Excel的表格处理，但比Excel快100倍
import numpy as np  # 数值计算库：处理数学运算，特别是数组和矩阵计算
from loguru import logger  # 日志工具：打印带时间戳、带颜色的运行信息，比print更专业


# -------------------- 数据容器定义 --------------------
# @dataclass 是Python的语法糖，帮你自动生成"初始化、打印、比较"等方法
# 就像自动填写表格，你只需要定义"有哪些字段"

@dataclass
class MacroState:
    """
    宏观状态数据包
    ==============
    用类(class)而不是字典(dict)，是为了防止拼写错误（比如把regime写成regin会立即报错）。
    
    属性说明：
        regime: 经济周期，只能是四个字符串之一：
                "recovery", "overheat", 
                "stagflation", "recession"
        equity_friendly_score: 股票友好度评分（0.0-1.0之间）
                              0.8表示"非常适合买股票"，0.2表示"股票风险大"
        growth_momentum: 增长动量（PMI最近3个月是上升还是下降，单位：%）
        inflation_momentum: 通胀动量（CPI最近3个月是上升还是下降）
        raw_data: 原始数据快照（记录当时用的具体数字，方便事后查证）
    """
    regime: str                          # 周期状态（字符串类型）
    equity_friendly_score: float         # 评分（小数类型）
    growth_momentum: float               # 增长动量（小数类型）
    inflation_momentum: float            # 通胀动量（小数类型）
    raw_data: Dict[str, Any]             # 原始数据（字典类型，键是字符串，值可以是任何类型）


# -------------------- 配置常量（判断阈值）--------------------
# 全部用大写，表示这些是"固定不变的配置"

THRESHOLDS = {
    "pmi": 50.0,              # PMI荣枯线：50以上是经济扩张，以下是收缩
    "industrial_growth": 5.0, # 工业增加值同比增速阈值（%）
    "cpi": 2.5,               # CPI温和通胀上限：超过就算通胀偏高
    "ppi": 0.0                # PPI零线：正数表示工业品涨价，负数表示跌价
}

# 四个经济周期的"基础配方"（股/债/商品的配置比例）
REGIME_BASE_WEIGHTS = {
    "recovery": {"stock": 0.7, "bond": 0.2, "commodity": 0.1},    # 复苏：70%股票，20%债券，10%商品
    "overheat": {"stock": 0.6, "bond": 0.1, "commodity": 0.3},    # 过热：减股票加仓商品（抗通胀）
    "stagflation": {"stock": 0.3, "bond": 0.3, "commodity": 0.4}, # 滞胀：均衡防守，多配商品
    "recession": {"stock": 0.4, "bond": 0.5, "commodity": 0.1}    # 衰退：重仓债券（50%），股票少配
}


# -------------------- 辅助函数：加载数据 --------------------

def load_macro_data(target_date: str, data_dir: str = "data/processed_macro_data") -> pd.DataFrame:
    """
    从CSV文件读取宏观数据
    =====================
    
    参数：
        target_date: 你想查询的日期，格式"2025-03-31"（字符串）
        data_dir: 数据存放的文件夹路径（字符串）
    
    返回：
        一行数据（DataFrame的Series类型），包含PMI、CPI等指标
    
    举个例子：
        如果输入"2025-03-31"，但数据只有3月20日的，就返回3月20日的数据
        （因为那是"不晚于目标日期的最新数据"）
    """
    # 拼接完整的文件路径：data/processed_macro_data/merged_macro_data.csv
    file_path = Path(data_dir) / "merged_macro_data.csv"
    
    # 检查文件是否存在，如果不存在就抛出错误
    if not file_path.exists():
        raise FileNotFoundError(
            f"❌ 找不到数据文件: {file_path}\n"
            f"💡 请先运行数据获取脚本，或者确认数据文件已放在正确位置"
        )
    
    # pd.read_csv是pandas的读文件函数，parse_dates=["date"]表示把date列转为真正的日期格式
    df = pd.read_csv(file_path, parse_dates=["date"])
    
    # 把用户输入的字符串（如"2025-03-31"）转成Python能计算的日期对象
    target = pd.to_datetime(target_date)
    
    # 筛选：只保留日期不晚于target的行（即过去的数据，不能用未来数据判断过去）
    # df["date"] <= target 会生成一个True/False列表，df[...]只保留True对应的行
    available_data = df[df["date"] <= target]
    
    # 如果没有数据（比如目标日期太早，数据库里还没有），报错
    if available_data.empty:
        raise ValueError(f"目标日期 {target_date} 之前没有可用数据")
    
    # iloc[-1]表示取最后一行（即最新的那条记录）
    # iloc是"integer location"的缩写，按数字位置取行
    return available_data.iloc[-1]


# -------------------- 辅助函数：计算动量 --------------------

def calculate_momentum(series: pd.Series, periods: int = 3) -> float:
    """
    计算变化趋势（动量）
    ==================
    
    参数：
        series: 时间序列数据（比如过去12个月的PMI数值，pandas的Series类型）
        periods: 回看多少期（默认3，即对比3个月前的数据）
    
    返回：
        变化率百分比（float类型），正数表示上升，负数表示下降
    
    计算公式：
        (最新值 - N期前的值) / |N期前的值| × 100%
    
    举个例子：
        如果3个月前PMI是48，现在是51，动量就是 (51-48)/48*100% = 6.25%
        表示经济在加速改善
    """
    # 如果数据不够N期，就调整期数（避免报错）
    if len(series) < periods + 1:
        periods = len(series) - 1
    
    # 如果只有1条数据，无法计算趋势，返回0
    if periods < 1:
        return 0.0
    
    # iloc[-1]是最后一个元素（最新），iloc[-(periods+1)]是N期前的元素
    latest = series.iloc[-1]           # 最新值
    previous = series.iloc[-(periods + 1)]  # N期前的值
    
    # 避免除以0的错误（虽然宏观数据一般不会有0，但 defensive programming）
    if previous == 0:
        return 0.0
    
    # 计算变化率
    momentum = (latest - previous) / abs(previous) * 100
    return float(momentum)


# -------------------- 核心函数：判断经济周期 --------------------

def detect_macro_regime(target_date: str, data_dir: str = "data/processed_macro_data") -> MacroState:
    """
    识别宏观状态（主入口函数）
    =========================
    
    参数：
        target_date: 分析目标日期，如"2025-03-31"
        data_dir: 数据目录路径
    
    返回：
        MacroState对象（打包好的分析结果）
    """
    # logger.info会在控制台打印带时间的信息，方便追踪程序运行到哪了
    logger.info(f"🔍 开始分析宏观状态，目标日期: {target_date}")
    
    # ---------- 步骤1：加载数据 ----------
    try:
        # 尝试加载数据，如果文件不存在会跳到except块
        data = load_macro_data(target_date, data_dir)
        actual_date = data['date'].strftime('%Y-%m-%d')  # 把日期对象格式化成字符串
        logger.info(f"📊 使用数据日期: {actual_date}（目标日期{target_date}前的最新数据）")
    except FileNotFoundError:
        # 如果数据文件不存在，进入"演示模式"（用假数据展示逻辑）
        logger.warning("⚠️ 未找到数据文件，进入演示模式（使用模拟数据）")
        return _demo_mode(target_date)
    
    # ---------- 步骤2：提取关键指标 ----------
    # 使用.get()安全获取数据，如果不存在返回NaN（Not a Number，表示缺失值）
    pmi = data.get("pmi", np.nan)              # 制造业PMI指数
    cpi_yoy = data.get("cpi_yoy", np.nan)      # CPI同比（与去年同期比）
    ppi_yoy = data.get("ppi_yoy", np.nan)      # PPI同比（工业品出厂价格）
    industrial_yoy = data.get("industrial_yoy", np.nan)  # 工业增加值同比
    
    # 记录原始数据，用于返回和调试
    raw_snapshot = {
        "data_date": actual_date,
        "pmi": float(pmi) if not np.isnan(pmi) else None,
        "cpi_yoy": float(cpi_yoy) if not np.isnan(cpi_yoy) else None,
        "ppi_yoy": float(ppi_yoy) if not np.isnan(ppi_yoy) else None,
        "industrial_yoy": float(industrial_yoy) if not np.isnan(industrial_yoy) else None
    }
    
    logger.info(f"📈 指标快照: PMI={pmi}, CPI={cpi_yoy}%, PPI={ppi_yoy}%, 工业={industrial_yoy}%")
    
    # ---------- 步骤3：判断增长状态----------
    # 默认假设经济不好（保守原则，证据不足时偏悲观）
    growth_positive = False
    
    if not np.isnan(pmi) and pmi > THRESHOLDS["pmi"]:
        growth_positive = True
        growth_reason = f"PMI({pmi})>荣枯线({THRESHOLDS['pmi']})"
    elif not np.isnan(industrial_yoy) and industrial_yoy > THRESHOLDS["industrial_growth"]:
        growth_positive = True
        growth_reason = f"工业增速({industrial_yoy}%)>阈值({THRESHOLDS['industrial_growth']}%)"
    else:
        growth_reason = "增长指标未达扩张标准"
    
    # ---------- 步骤4：判断通胀状态----------
    inflation_high = False  # 默认通胀不高
    
    if not np.isnan(cpi_yoy) and cpi_yoy > THRESHOLDS["cpi"]:
        inflation_high = True
        inflation_reason = f"CPI({cpi_yoy}%)>阈值({THRESHOLDS['cpi']}%)"
    elif not np.isnan(ppi_yoy) and ppi_yoy > THRESHOLDS["ppi"]:
        inflation_high = True
        inflation_reason = f"PPI({ppi_yoy}%)>零值，工业品涨价"
    else:
        inflation_reason = "通胀指标温和"
    
    logger.info(f"🎯 初步判断: 增长{'向好' if growth_positive else '承压'}({growth_reason}), "
                f"通胀{'高位' if inflation_high else '温和'}({inflation_reason})")
    
    # ---------- 步骤5：四象限判断（核心逻辑）----------
    
    if growth_positive and inflation_high:
        regime = "overheat"        # 过热：经济好但物价涨
    elif growth_positive and not inflation_high:
        regime = "recovery"        # 复苏：经济好且物价稳
    elif not growth_positive and inflation_high:
        regime = "stagflation"     # 滞胀：经济差但物价涨
    else:
        regime = "recession"       # 衰退：经济差且物价跌
    
    # 中文翻译，方便看日志
    regime_names = {
        "overheat": "过热期(扩张+高通胀)",
        "recovery": "复苏期(扩张+低通胀)", 
        "stagflation": "滞胀期(收缩+高通胀)",
        "recession": "衰退期(收缩+低通胀)"
    }
    logger.info(f"🏷️ 周期定位: {regime_names[regime]}")
    
    # ---------- 步骤6：计算动量（趋势分析）----------
    # 为了计算动量，我们需要加载历史数据（而不仅是单点数据）
    try:
        # 重新加载完整数据集（这次要时间序列）
        df = pd.read_csv(Path(data_dir) / "merged_macro_data.csv", parse_dates=["date"])
        target = pd.to_datetime(target_date)
        hist = df[df["date"] <= target].tail(6)  # 取最近6个月数据
        
        if len(hist) >= 3:
            # 计算PMI的3个月动量
            if "pmi" in hist.columns and not hist["pmi"].isna().all():
                growth_mom = calculate_momentum(hist["pmi"], 3)
            else:
                growth_mom = 0.0
            
            # 计算CPI的3个月动量
            if "cpi_yoy" in hist.columns and not hist["cpi_yoy"].isna().all():
                inflation_mom = calculate_momentum(hist["cpi_yoy"], 3)
            else:
                inflation_mom = 0.0
        else:
            growth_mom = 0.0
            inflation_mom = 0.0
    except:
        # 如果历史数据加载失败，动量设为0（不影响主逻辑）
        growth_mom = 0.0
        inflation_mom = 0.0
    
    logger.info(f"📊 动量分析: 增长动量={growth_mom:+.1f}%, 通胀动量={inflation_mom:+.1f}%")
    
    # ---------- 步骤7：生成股票友好度评分 ----------
    # 基础分：根据周期给基础分（复苏0.8分，过热0.6分，衰退0.4分，滞胀0.3分）
    base_scores = {"recovery": 0.8, "overheat": 0.6, "stagflation": 0.3, "recession": 0.4}
    base_score = base_scores[regime]
    
    # 调整项：根据动量微调（增长动量好加分，通胀动量高减分）
    # np.clip把数值限制在-0.1到+0.1之间，避免调整过大
    momentum_adj = np.clip((growth_mom - inflation_mom) / 100, -0.1, 0.1)
    
    # 最终分：限制在0-1之间（用np.clip裁剪）
    final_score = float(np.clip(base_score + momentum_adj, 0.0, 1.0))
    
    logger.info(f"💯 权益友好度: 基础={base_score}, 动量调整={momentum_adj:+.2f}, 最终={final_score:.2f}")
    
    # ---------- 步骤8：打包结果 ----------
    result = MacroState(
        regime=regime,
        equity_friendly_score=final_score,
        growth_momentum=growth_mom,
        inflation_momentum=inflation_mom,
        raw_data=raw_snapshot
    )
    
    logger.success(f"✅ 分析完成: {regime} | 评分: {final_score:.2f}")
    return result


# -------------------- 演示模式（没数据时用的假数据）--------------------

def _demo_mode(target_date: str) -> MacroState:
    """
    演示模式
    ========
    当真实数据不存在时，返回假设数据，用于展示代码逻辑。
    """
    logger.info("🎭 使用模拟数据运行演示...")
    
    # 模拟一个"复苏期"的数据
    return MacroState(
        regime="recovery",
        equity_friendly_score=0.75,  # 75%的推荐度
        growth_momentum=2.5,         # 增长在改善
        inflation_momentum=-0.5,     # 通胀在下降（好事）
        raw_data={
            "mode": "demo",
            "date": target_date,
            "pmi": 51.5,              # 略高于50，扩张
            "cpi_yoy": 1.8,           # 低于2.5，通胀温和
            "ppi_yoy": -1.2,          # 负数，工业品降价（利好企业成本）
            "industrial_yoy": 6.2     # 高于5%，工业增长强劲
        }
    )
# ==================== 脚本入口 ====================
if __name__ == "__main__":
    """
    当直接运行这个文件时执行的测试代码。
    这个判断句的意思是："如果我是被直接运行（不是被导入），才执行下面的代码"。
    """
    
    # 配置日志输出到控制台（带颜色）
    import sys
    logger.remove()  # 移除默认配置
    logger.add(
        sys.stdout,  # 输出到标准输出（控制台）
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
        level="INFO",
        colorize=True
    )
    
    print("=" * 60)
    print("🧪 宏观状态识别模块 - 功能测试")
    print("=" * 60)
    
    # 测试用的目标日期
    test_date = "2025-12-31"  # 你可以改成实际有数据的日期，比如 "2026-02-28"
    
    try:
        # 调用核心函数进行宏观状态分析
        state = detect_macro_regime(test_date)
        
        # 打印结果（使用 f-string 格式化输出）
        print(f"\n📋 分析结果 ({test_date}):")
        print(f"  经济周期: {state.regime}")
        print(f"  股票推荐度: {state.equity_friendly_score:.2%}")
        print(f"  增长动量: {state.growth_momentum:+.2f}%")
        print(f"  通胀动量: {state.inflation_momentum:+.2f}%")
        
        # 显示该周期下的建议配置
        weights = REGIME_BASE_WEIGHTS.get(state.regime, {})
        print(f"\n💼 建议资产配置:")
        print(f"  股票: {weights.get('stock', 0):.0%}")
        print(f"  债券: {weights.get('bond', 0):.0%}")
        print(f"  商品: {weights.get('commodity', 0):.0%}")
        
        print(f"\n📊 原始数据快照:")
        for key, value in state.raw_data.items():
            print(f"    {key}: {value}")
        
        print("\n✅ 测试完成！数据运行正常。")
        
    except FileNotFoundError:
        print(f"\n⚠️  找不到数据文件")
        print("💡 请先运行: python scripts/fetch_macro_data.py")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()  # 打印详细错误堆栈