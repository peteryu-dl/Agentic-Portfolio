# 在 Python 里运行这个调试代码（可以直接在终端输入 python 进入交互模式）
import akshare as ak
import pandas as pd

# 看看 PMI 返回什么
df_pmi = ak.macro_china_pmi()
print("PMI 列名:", df_pmi.columns.tolist())
print("PMI 前3行:\n", df_pmi.head(3))
print()

# 看看 CPI 返回什么  
df_cpi = ak.macro_china_cpi()
print("CPI 列名:", df_cpi.columns.tolist())
print("CPI 前3行:\n", df_cpi.head(3))

# 看看 PPI 返回什么  
df_ppi = ak.macro_china_ppi()
print("PPI 列名:", df_ppi.columns.tolist())
print("PPI 前3行:\n", df_ppi.head(3))