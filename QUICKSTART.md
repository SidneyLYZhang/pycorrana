# PyCorrAna 快速入门指南

## 安装

```bash
# 从源码安装
cd pycorrana
pip install -e .

# 或安装依赖后直接使用
pip install numpy pandas scipy matplotlib seaborn statsmodels scikit-learn openpyxl
```

## 5分钟快速上手

### 1. 一行代码完成分析

```python
from pycorrana import quick_corr
import pandas as pd

# 加载数据
df = pd.read_csv('your_data.csv')

# 一键分析
result = quick_corr(df)

# 查看相关系数矩阵
print(result['correlation_matrix'])

# 查看显著相关对
for pair in result['significant_pairs'][:5]:
    print(f"{pair['var1']} vs {pair['var2']}: {pair['correlation']:.3f}")
```

### 2. 指定目标变量

```python
# 只分析与目标变量的相关性
result = quick_corr(df, target='sales', plot=True, export='results.xlsx')
```

### 3. 使用交互式工具

```bash
# 启动交互式分析
python -m pycorrana.cli.interactive
```

### 4. 使用命令行工具

```bash
# 完整分析
python -m pycorrana.cli.main_cli analyze data.csv --target sales --export results.xlsx

# 数据信息
python -m pycorrana.cli.main_cli info data.csv

# 偏相关分析
python -m pycorrana.cli.main_cli partial data.csv -x income -y happiness -c age
```

## 核心功能速查

| 功能 | 代码示例 |
|------|---------|
| 基础分析 | `quick_corr(df)` |
| 指定目标 | `quick_corr(df, target='Y')` |
| 指定方法 | `quick_corr(df, method='spearman')` |
| 缺失值处理 | `quick_corr(df, missing_strategy='drop')` |
| 导出结果 | `quick_corr(df, export='results.xlsx')` |
| 偏相关 | `partial_corr(df, x='A', y='B', covars='Z')` |
| 距离相关 | `distance_correlation(df['A'], df['B'])` |
| 互信息 | `mutual_info_score(df['A'], df['B'])` |

## 示例数据集

```python
from pycorrana.datasets import load_iris, load_titanic, make_correlated_data

# 鸢尾花数据集
df = load_iris()

# 泰坦尼克号数据集
df = load_titanic()

# 生成相关数据
df = make_correlated_data(n_samples=500, correlation=0.7)
```

## 运行演示

```bash
python demo.py
```

## 更多示例

查看 `examples/basic_usage.py` 获取完整示例代码。
