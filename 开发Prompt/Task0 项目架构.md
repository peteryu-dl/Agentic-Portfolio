Q-Macro/
├── data/                  # 宏观数据 + ETF数据
│   ├── etf/               # ETF量价与基本信息
│   │   ├── etf_2025_ohlcva.csv        # ETF量价数据
│   │   ├── etf_basic.csv              # ETF基本信息
│   │   ├── processed_etf_basic.csv     # LLM打标签结果
│   │   └── tag_analysis.txt           # 标签分析结果
│   ├── macro_data/        # 原始宏观指标（CPI、PMI等）
│   ├── processed_macro_data/ # 处理后的时间对齐数据
│   └── policy_texts/      # 政策文本数据
│       └── govcn_2025.csv              # 2025年政策文件
├── src/                   # 核心源码
│   ├── core/              # 策略核心逻辑（纯Python）
│   │   ├── macro_regime.py        # 宏观状态识别
│   │   ├── market_calibration.py  # ETF流动性与拥挤度校准
│   │   └── portfolio_builder.py   # 动态组合构建
│   └── agents/            # LLM智能体
│       ├── policy_interpreter.py  # 政策翻译官
│       ├── theme_mapper.py        # ETF标签工程师
│       └── report_writer.py       # 归因分析师
├── scripts/               # 可执行脚本
│   ├── fetch_macro_data.py        # 从AKShare拉取宏观数据
│   ├── process_macro_data.py      # 处理宏观数据时间格式
│   ├── run_monthly_pipeline.py    # 单月策略执行
│   ├── run_all_pipline.py         # 批量跑全年
│   ├── parse_portfolios_to_csv.py # JSON → CSV 转换
│   └── run_backtest.py            # 回测入口
├── portfolios/            # 输出：每月ETF组合（JSON）
├── reports/               # 输出：每月投资报告（Markdown）
├── results/               # 回测结果：净值/图表/指标
│   └── backtest_results/
│       ├── charts/        # 回测图表
│       ├── metrics.txt    # 回测指标
│       ├── nav_series.csv # 净值序列
│       ├── positions.csv  # 持仓记录
│       └── trade_records.csv # 交易记录
├── 开发Prompt/            # 开发过程记录
├── Q-Macro.py             # 一键运行入口
├── README.md              # 项目说明
├── .env.example           # 环境变量示例文件
├── .gitignore             # Git忽略文件
├── position.csv           # 持仓数据
├── pyproject.toml         # 项目配置文件
└── uv.lock                # 依赖锁定文件