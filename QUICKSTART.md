# PyCorrAna 快速入门指南

## 安装

```bash
# 从 PyPI 安装
pip install pycorrana

# 从源码安装
git clone https://github.com/sidneylyzhang/pycorrana.git
cd pycorrana
pip install -e .

# 安装开发依赖
pip install -e ".[dev]"

# 安装文档依赖
pip install -e ".[docs]"
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
pycorrana-interactive
```

### 4. 使用命令行工具

```bash
# 完整分析
pycorrana analyze data.csv --target sales --export results.xlsx

# 数据信息
pycorrana info data.csv

# 偏相关分析
pycorrana partial data.csv -x income -y happiness -c age

# 非线性检测
pycorrana nonlinear data.csv --top 20
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
| 半偏相关 | `semipartial_corr(df, x='A', y='B', covars='Z')` |
| 距离相关 | `distance_correlation(df['A'], df['B'])` |
| 互信息 | `mutual_info_score(df['A'], df['B'])` |
| 典型相关 | `cca(df[X_vars], df[Y_vars])` |
| 大数据优化 | `CorrAnalyzer(df, large_data_config=config)` |

## 示例数据集

```python
from pycorrana.datasets import load_iris, load_titanic, load_wine, make_correlated_data, list_datasets

# 查看可用数据集
print(list_datasets())

# 鸢尾花数据集
df = load_iris()

# 泰坦尼克号数据集
df = load_titanic()

# 葡萄酒数据集
df = load_wine()

# 生成相关数据
df = make_correlated_data(n_samples=500, correlation=0.7)
```

## 大数据优化

```python
from pycorrana import CorrAnalyzer
from pycorrana.utils import LargeDataConfig

# 配置大数据优化
config = LargeDataConfig(
    sample_size=100000,      # 采样大小
    auto_sample=True,        # 自动采样
    auto_optimize=True,      # 自动优化内存
    verbose=True
)

# 使用配置分析大数据集
analyzer = CorrAnalyzer(large_df, large_data_config=config)
analyzer.fit()
```

## 偏相关分析

```python
from pycorrana import partial_corr, partial_corr_matrix, semipartial_corr

# 偏相关：控制协变量后的净相关
result = partial_corr(
    df,
    x='income',
    y='happiness',
    covars=['age', 'education']
)
print(f"偏相关系数: {result['partial_correlation']:.3f}")

# 半偏相关（部分相关）
result = semipartial_corr(df, x='income', y='happiness', covars='age')

# 偏相关矩阵
pcorr_matrix = partial_corr_matrix(df, covars='age')
```

## 非线性检测

```python
from pycorrana import distance_correlation, mutual_info_score
from pycorrana.core.nonlinear import nonlinear_dependency_report

# 距离相关（检测非线性关系）
result = distance_correlation(df['X'], df['Y'], return_pvalue=True)
print(f"dCor: {result['dcor']:.3f}, p-value: {result['p_value']:.4f}")

# 互信息
result = mutual_info_score(df['X'], df['Y'])
print(f"MI: {result['mi_normalized']:.3f}")

# 非线性依赖报告
report = nonlinear_dependency_report(df, top_n=10)
print(report)
```

## 典型相关分析

```python
from pycorrana import cca, cca_permutation_test

# 典型相关分析：研究两组变量之间的关系
result = cca(df[['x1', 'x2']], df[['y1', 'y2']])
print(f"典型相关系数: {result['canonical_correlations']}")

# 查看显著性检验
for test in result['significance_tests']:
    print(f"第{test['canonical_index']}典型相关: r={test['canonical_correlation']:.3f}, p={test['p_value']:.4f}")

# 置换检验
perm_result = cca_permutation_test(df[['x1', 'x2']], df[['y1', 'y2']], n_permutations=1000)
print(f"置换检验 p 值: {perm_result['p_value']:.4f}")
```

## 数据清洗

```python
from pycorrana.utils.data_utils import load_data, handle_missing, detect_outliers

# 加载数据（自动识别格式）
df = load_data('data.csv')

# 缺失值处理
df_clean = handle_missing(
    df,
    strategy='fill',      # 'drop', 'fill', 'warn'
    fill_method='knn',    # 'mean', 'median', 'mode', 'knn'
    verbose=True
)

# 异常值检测
outliers = detect_outliers(
    df,
    method='iqr',         # 'iqr', 'zscore'
    visualize=True
)
```

## 运行演示

```bash
python demo.py
```

## 更多示例

查看 `examples/basic_usage.py` 获取完整示例代码。

## Python 版本要求

- Python >= 3.10
- 支持 Python 3.10, 3.11, 3.12, 3.13
