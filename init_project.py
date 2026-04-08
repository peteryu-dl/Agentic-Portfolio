#!/usr/bin/env python3
"""
脚手架搭建脚本 - 创建所有需要的文件夹
"""

from pathlib import Path

def main():
    root = Path.cwd()
    print(f"🏗️  在 {root} 创建项目结构...")
    
    # 创建所有需要的目录
    dirs = [
        "src/core",           # 硬编码核心模块
        "src/agents",         # LLM智能体
        "scripts",            # 工具脚本
        "data/etf",           # ETF数据
        "data/macro_data",    # 原始宏观数据
        "data/processed_macro_data",
        "data/policy_texts",  # 政策文本
        "portfolios",         # 输出
        "reports",            # 输出
        "results/backtest_results/charts",
        "logs",               # 日志
    ]
    
    for d in dirs:
        (root / d).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}")
    
    # 创建 Python 包标记文件（至关重要！）
    inits = ["src/__init__.py", "src/core/__init__.py", "src/agents/__init__.py"]
    for f in inits:
        (root / f).touch()
        print(f"  ✓ {f}")
    
    print("\n✅ 目录结构创建完成！")
    print("下一步：在 pyproject.toml 底部添加 [build-system] 配置，然后 uv pip install -e .")

if __name__ == "__main__":
    main()