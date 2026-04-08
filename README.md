# Q-Macro: 宏观驱动 ETF 配置策略系统

基于宏观视角的大类资产配置策略，结合传统量化规则与大语言模型（LLM）的混合架构智能体系统。通过美林时钟四象限模型识别经济周期，利用 LLM 解析政策文本提取投资主题，动态构建 ETF 投资组合并生成专业投资报告。

## 核心特性

- **宏观周期驱动**：基于 PMI/CPI/PPI 等指标识别复苏/过热/滞胀/衰退四象限，动态调整股/债/商品配置比例
- **AI 政策解读**：使用 LangChain + LLM 分析政府文件，自动提取 Top-5 投资主题与置信度评分
- **智能 ETF 映射**：三级标签体系（资产大类→主题→子主题）+ LLM 辅助打标，实现政策主题到具体 ETF 的精准匹配
- **市场校准机制**：流动性筛选（避免僵尸 ETF）+ 拥挤度检测（成交量 Z-score，防止高位接盘）
- **分层资产配置**：股票内部采用宽基 ETF（60%）+ 主题 ETF（40%）的结构化配置
- **全自动报告生成**：基于 LLM 生成包含宏观状态、政策解读、持仓逻辑的 Markdown 投资月报
- **完整回测框架**：支持净值计算、基准对比（全天候组合）、夏普比率/最大回撤等多指标评估

## 技术栈

- **包管理**: [UV](https://docs.astral.sh/uv/)（`uv pip install` / `uv venv`）
- **智能体框架**: LangChain（支持 LLM 路由、工具调用、结构化输出）
- **数据源**: [AKShare](https://www.akshare.xyz/)（宏观指标、市场指数），本地 CSV（政策文本、ETF 量价）
- **数据验证**: Pydantic（ETF 标签、报告数据结构强类型校验）
- **语言**: Python 3.10+
- **配置**: 环境变量管理（`.env` 文件配置 LLM API 参数）

## 项目架构

```text
Q-Macro/
├── data/                          # 数据层
│   ├── etf/                       # ETF 量价与基本信息
│   │   ├── etf_2025_ohlcva.csv    # 日频价格/成交量
│   │   ├── etf_basic.csv          # 基本信息
│   │   └── processed_etf_basic.csv# LLM 三级标签结果
│   ├── macro_data/                # 原始宏观指标
│   ├── processed_macro_data/      # 标准化数据
│   └── policy_texts/              # 非结构化政策文本
│       └── govcn_2025.csv         # 政府文件/部委公告
├── src/
│   ├── core/                      # 硬编码规则模块（确定性逻辑）
│   │   ├── macro_regime.py        # 宏观状态识别（四象限模型）
│   │   ├── market_calibration.py  # ETF 流动性筛选与拥挤度计算
│   │   └── portfolio_builder.py   # 动态组合构建与权重分配
│   └── agents/                    # LLM 智能体模块（语义理解）
│       ├── policy_interpreter.py  # 政策翻译官：文本→投资主题
│       ├── theme_mapper.py        # ETF 标签工程师：三级标签打标
│       └── report_writer.py       # 归因分析师：生成投资报告
├── scripts/                       # 可执行脚本
│   ├── fetch_macro_data.py        # 从 AKShare 拉取宏观数据
│   ├── fetch_etf_data.py          # 从 AKShare 拉取ETF数据
│   ├── generate_mock_etf.py       # Mock ETF 数据生成器
│   ├── generate_sample_policy.py  # 政策文本示例生成器
│   ├── process_macro_data.py      # 数据清洗与时间格式统一
│   └── run_monthly_pipeline.py    # 单月策略执行（核心流水线）
├── portfolios/                    # 输出：每月 ETF 组合权重（JSON）
├── reports/                       # 输出：月度投资报告（Markdown）
├── results/
│   └── backtest_results/          # 回测结果
│       ├── nav_series.csv         # 策略与基准净值序列
│       ├── metrics.txt            # 年化收益/夏普/最大回撤等指标
│       ├── positions.csv          # 持仓记录
│       └── charts/                # 回测可视化图表
├── Q-Macro.py                     # 一键运行入口（CLI）
├── pyproject.toml                 # 项目依赖配置
└── .env.example                   # 环境变量模板（LLM API Key 等）
```

## 模块详解

### 1. 宏观状态识别（`macro_regime.py`）
- **输入**: 目标日期、processed_macro_data（PMI、CPI、PPI、工业增加值）
- **逻辑**: 简化版美林时钟
  - 增长判断: PMI > 50 或 工业增加值同比 > 5%
  - 通胀判断: CPI > 2.5% 或 PPI > 0%
  - 四象限划分: 复苏(growth↑inflation↓)/过热(↑↑)/滞胀(↓↑)/衰退(↓↓)
- **输出**: `regime`（周期状态）、`equity_friendly_score`（0-1 权益友好度评分）、`growth_momentum`（增长动量）

### 2. 市场校准（`market_calibration.py`）
- **流动性筛选**: 20 日平均成交额 > 阈值（默认 1万元），剔除僵尸 ETF
- **拥挤度检测**: 成交量 Z-score = (5 日均量 - 120 日均量) / 120 日标准差
  - Z > 1.5 判定为拥挤，应用 0.75 权重调整因子
- **输出**: `liquid_etfs`（合格标的池）、`crowded_adjustments`（权重调整字典）

### 3. 政策解读智能体（`policy_interpreter.py`）
- **输入**: 目标日期前 30 天政策文本、当前宏观状态
- **Prompt 工程**: 要求 LLM 提取推荐行业/主题、情绪评分、政策力度
- **聚合逻辑**: 统计多条政策的主题分布，计算置信度，输出 Top-5 主题
- **容错机制**: LLM 失败时回退到基于宏观状态的默认主题推荐

### 4. ETF 主题映射（`theme_mapper.py`）
- **三级标签体系**（Pydantic 模型强校验）:
  - L1: 资产大类（股票/债券/商品/货币）
  - L2: 主题（宽基/行业/策略/利率/信用/贵金属/能源等）
  - L3: 子主题（芯片/新能源/科创 50/国债/黄金等）
- **实现**: 基于 ETF 名称与跟踪指数名称，使用 LLM 批量打标，支持增量更新

### 5. 投资组合构建（`portfolio_builder.py`）
- **基础权重表**（四象限预设）:
  - 复苏: 股 70%、债 20%、商 10%
  - 过热: 股 60%、债 10%、商 30%
  - 滞胀: 股 30%、债 30%、商 40%
  - 衰退: 股 40%、债 50%、商 10%
- **动态调整**: 基于 `equity_friendly_score` 对股票权重进行 ±10% 浮动
- **分层配置**: 股票部分内部，宽基 ETF 占 60%，政策主题 ETF 占 40%
- **权重分配**: 同类 ETF 等权分配，应用拥挤度调整因子，最终归一化

### 6. 报告生成（`report_writer.py`）
- **结构化生成**: 分别生成宏观状态、政策解读、市场条件、组合描述、决策总结五部分
- **输入数据模型**: MacroState（Pydantic）、PolicySignal、MarketCondition、Portfolio
- **输出**: 完整 Markdown 格式月度投资报告，保存在 `reports/`

### 7. 回测引擎（`run_backtest.py`）
- **计算逻辑**: 基于每月持仓 JSON 与 ETF 日频价格数据，计算每日持仓权重与收益率
- **基准**: 全天候组合（股债商固定权重，如 50/30/20）
- **评估指标**: 年化收益率、年化波动率、夏普比率、最大回撤、Calmar 比率
- **可视化**: 生成策略 vs 基准的净值曲线对比图

## 快速开始

### 环境配置

```bash
# 克隆项目后，使用 UV 创建环境并安装依赖
uv venv
uv pip install -e .

# 配置环境变量（LLM API）
cp .env.example .env
# 编辑 .env 文件，填入 API Key 与模型名称（支持 OpenAI/Claude/DeepSeek 等）
```

### 数据准备
- **获取宏观数据（AKShare）**
python scripts/fetch_macro_data.py

- **数据清洗（时间格式统一、缺失值处理）**
python scripts/process_macro_data.py

- **ETF 标签初始化（仅需执行一次，LLM 批量打标）**
python -c "from src.agents.theme_mapper import batch_tag_etfs; batch_tag_etfs()"

### 运行策略

- **运行单月完整流程（数据获取→状态识别→组合构建→报告生成）**
python Q-Macro.py --date 2025-03-31 --full

- **仅生成指定月份的投资组合（跳过数据获取）**
python Q-Macro.py --date 2025-03-31 --build-only

- **批量回测全年数据**
python Q-Macro.py --backtest --start 2025-01-01 --end 2025-12-31

- **快速调试模式（跳过 LLM 调用，使用缓存/默认值）**
python Q-Macro.py --date 2025-03-31 --fast-debug

### 结果查看 
- **投资组合**: portfolios/2025-03.json（标的权重、逻辑说明）
- **投资报告**: reports/2025-03-report.md（可读性强的月度分析）
- **回测结果**: results/backtest_results/metrics.txt 与 charts/

## 策略逻辑简述

- **周期定位**: 通过 PMI（增长）与 CPI/PPI（通胀）判断当前处于美林时钟的哪个象限，确定股债商的基础配比
- **政策增强**: 使用 LLM 分析近 30 天政策文件，提取 5 个高置信度投资主题（如"新质生产力"、"设备更新"）
- **标的筛选**: 基于流动性（成交额）和拥挤度（成交量 Z-score）过滤 ETF，确保可交易且不追高
- **精准映射**: 通过三级标签体系，将政策主题映射到具体行业/主题 ETF，宽基 ETF 提供 Beta 暴露
- **动态权重**: 结合宏观状态评分与拥挤度调整，生成最终 ETF 权重，确保风险可控
- **归因报告**: LLM 自动生成报告，解释"为什么当前配置这些 ETF"，提升策略可解释性

## 架构设计原则

- **简单优先**: 使用 UV 管理依赖，AKShare 作为唯一外部数据源，降低环境复杂度
- **模块解耦**: Core（硬编码）与 Agents（LLM）分离，通过清晰的 Pydantic 接口传递数据
- **数据可得**: 仅依赖 AKShare 可获取的宏观指标 + 本地已有数据，确保策略可完全本地复现
- **人机结合**: LLM 负责非结构化文本（政策解读、报告生成），硬编码负责结构化计算（指标、权重、回测）
- **容错设计**: 每个 LLM 模块均配备 fallback 机制，确保在 API 失效或限流时策略仍可运行

## 开发提示

- **调试技巧**: 使用 --fast-debug 参数跳过 LLM 调用，使用预设主题和模板报告加速开发
- **数据检查**: 若某月份策略表现异常，优先检查 data/processed_macro_data/ 该月期的宏观数据是否缺失
- **LLM 成本**: ETF 标签仅需初始化时打标一次；政策解读与报告生成每月执行，建议使用性价比高的模型（如 GPT-4o-mini/DeepSeek-V3）
- **扩展方向**: 可通过修改 REGIME_BASE_WEIGHTS 调整四象限配比，或在 theme_mapper.py 中扩展四级标签

## License

MIT License
