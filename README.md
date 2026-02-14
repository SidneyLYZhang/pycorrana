# PyCorrAna - Python Correlation Analysis Toolkit

<p align="center">
  <img src="docs/logo.png" alt="PyCorrAna Logo" width="200">
</p>

<p align="center">
  <strong>自动化相关性分析工具 - 降低决策成本，一键输出关键结果</strong>
</p>

<p align="center">
  <a href="#特性">特性</a> •
  <a href="#安装">安装</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="https://pycorrana.readthedocs.io/zh-cn/latest/">文档</a> •
  <a href="#详细示例">示例</a>
</p>

---

## 初心

相关性的分析，在实际分析工作中，并不是最复杂的部分，却常常在入门阶段挡住很多人的学习/工作进度，
这是我自己常用的一些工具的重新包装，希望能让更多人可以快速上手相关性分析。


## 特性

PyCorrAna 是一个**方便快速入手**的 Python 相关性分析工具，核心设计理念：

- **自动化常规操作** - 智能识别数据类型，自动选择最优相关系数方法
- **降低决策成本** - 无需纠结用 Pearson 还是 Spearman，工具自动帮你选择
- **一键输出关键结果** - 从数据加载到结果导出，一行代码搞定

### 主要功能

| 模块  | 功能  |
| --- | --- |
| **数据接入** | 支持 CSV/Excel/pandas/polars，自动类型推断 |
| **缺失值处理** | 删除/填充（均值/中位数/众数/KNN预测） |
| **相关性计算** | 自动方法选择（Pearson/Spearman/Kendall/Cramér's V/Eta等） |
| **显著性检验** | 自动 p 值计算，支持多重比较校正 |
| **可视化** | 热力图、散点图矩阵、箱线图、相关网络图 |
| **结果导出** | Excel/CSV/HTML/Markdown 结果 |
| **偏相关分析** | 控制协变量后的净相关分析 |
| **非线性检测** | 距离相关、互信息、MIC |
| **大数据优化** | 智能采样、分块计算、内存优化 |

---

## 安装

```bash
# 基础安装
pip install pycorrana

# 包含所有可选依赖
pip install pycorrana[all]

# 开发模式安装
git clone https://github.com/sidneylyzhang/pycorrana.git
cd pycorrana
pip install -e .
```

另外，我个人建议还是使用 `uv` 进行安装，可以获得更好的使用体验，也不会过多影响当前的python环境，也可以更方便的使用工具自带的两个命令行工具。

---

## 快速开始

### 1. 一行代码完成分析

```python
from pycorrana import quick_corr

# 从文件加载并分析
result = quick_corr('data.csv')

# 或者使用 DataFrame
import pandas as pd
df = pd.read_csv('data.csv')
result = quick_corr(df, target='sales')
```

### 2. 交互式分析

```bash
# 启动交互式命令行工具
pycorrana-interactive
```

### 3. 命令行工具

```bash
# 完整分析
pycorrana analyze data.csv --target sales --export results.xlsx

# 数据清洗
pycorrana clean data.csv --dropna --output cleaned.csv

# 偏相关分析
pycorrana partial data.csv -x income -y happiness -c age,education

# 非线性依赖检测
pycorrana nonlinear data.csv --top 20
```

---

## 详细示例

### 示例 1: 基础相关性分析

```python
from pycorrana import quick_corr
from pycorrana.datasets import load_iris

# 加载示例数据
df = load_iris()

# 一键分析
result = quick_corr(
    df,
    method='auto',           # 自动选择方法
    missing_strategy='warn', # 缺失值警告但不处理
    plot=True,               # 自动生成图表
    export='results.xlsx',   # 导出结果
    verbose=True             # 输出详细信息
)

# 查看结果
corr_matrix = result['correlation_matrix']
pvalue_matrix = result['pvalue_matrix']
significant_pairs = result['significant_pairs']
```

### 示例 2: 指定目标变量

```python
from pycorrana import quick_corr

# 只分析与目标变量的相关性
result = quick_corr(
    df,
    target='survived',  # 目标变量
    method='auto',
    plot=True
)

# 查看与目标变量最相关的特征
for pair in result['significant_pairs'][:5]:
    print(f"{pair['var1']} vs {pair['var2']}: {pair['correlation']:.3f}")
```

### 示例 3: 使用分析器类（更灵活）

```python
from pycorrana import CorrAnalyzer

# 创建分析器
analyzer = CorrAnalyzer(df, method='auto', verbose=True)

# 执行分析
analyzer.fit()

# 自定义可视化
analyzer.plot_heatmap(cluster=True, savefig='heatmap.png')
analyzer.plot_pairplot(columns=['A', 'B', 'C', 'D'])

# 导出结果
analyzer.export_results('results.xlsx', format='excel')

# 查看文本摘要
print(analyzer.summary())
```

### 示例 4: 大数据集优化

```python
from pycorrana import CorrAnalyzer
from pycorrana.utils import LargeDataConfig

# 配置大数据优化参数
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

### 示例 5: 偏相关分析

```python
from pycorrana import partial_corr, partial_corr_matrix, semipartial_corr

# 控制年龄后，计算收入与幸福感的偏相关
result = partial_corr(
    df,
    x='income',
    y='happiness',
    covars=['age', 'education'],  # 控制变量
    method='pearson'
)

print(f"偏相关系数: {result['partial_correlation']:.3f}")
print(f"p值: {result['p_value']:.4e}")
print(f"95% CI: [{result['ci_95'][0]:.3f}, {result['ci_95'][1]:.3f}]")

# 半偏相关（部分相关）
result = semipartial_corr(df, x='income', y='happiness', covars='age')

# 偏相关矩阵
pcorr_matrix = partial_corr_matrix(df, covars='age')
print(pcorr_matrix)
```

### 示例 6: 非线性依赖检测

```python
from pycorrana import distance_correlation, mutual_info_score
from pycorrana.core.nonlinear import nonlinear_dependency_report

# 距离相关（可检测非线性关系）
result = distance_correlation(df['X'], df['Y'], return_pvalue=True)
print(f"dCor: {result['dcor']:.3f}, p-value: {result['p_value']:.4f}")

# 互信息
result = mutual_info_score(df['X'], df['Y'])
print(f"MI: {result['mi_normalized']:.3f}")

# 生成非线性检测报告
report = nonlinear_dependency_report(df, top_n=10)
print(report)
```

### 示例 7: 数据清洗和预处理

```python
from pycorrana.utils.data_utils import load_data, handle_missing, detect_outliers

# 加载数据
df = load_data('data.csv')  # 自动识别格式

# 缺失值处理
df_clean = handle_missing(
    df,
    strategy='fill',      # 填充策略
    fill_method='knn',    # KNN预测填充
    verbose=True
)

# 异常值检测
outliers = detect_outliers(
    df,
    method='iqr',         # IQR方法
    visualize=True        # 显示箱线图
)
```

---

## 自动方法选择规则

PyCorrAna 根据变量类型自动选择最优的相关系数方法：

| 变量类型 | 自动选用方法 | 备注  |
| --- | --- | --- |
| 数值 + 数值 | Pearson（若正态）/ Spearman | 默认用 Spearman 更稳健 |
| 数值 + 二分类 | 点双列相关 | 自动将二分类映射为 0/1 |
| 数值 + 多分类无序 | Eta 系数 / ANOVA | 输出相关比 |
| 分类 + 分类 | Cramér's V | 基于卡方检验 |
| 有序 + 有序 | Kendall's Tau / Spearman | 默认 Kendall 处理重复秩 |

---

## 项目结构

```
pycorrana/
├── src/pycorrana/            # 源代码
│   ├── __init__.py           # 包入口
│   ├── core/                 # 核心分析模块
│   │   ├── analyzer.py       # 主分析器
│   │   ├── visualizer.py     # 可视化
│   │   ├── reporter.py       # 报告生成
│   │   ├── partial_corr.py   # 偏相关分析
│   │   └── nonlinear.py      # 非线性检测
│   ├── utils/                # 工具函数
│   │   ├── data_utils.py     # 数据处理
│   │   ├── stats_utils.py    # 统计工具
│   │   └── large_data.py     # 大数据优化
│   ├── cli/                  # 命令行工具
│   │   ├── main_cli.py       # 主CLI
│   │   └── interactive.py    # 交互式CLI
│   └── datasets.py           # 示例数据集
├── tests/                    # 测试
├── examples/                 # 示例代码
├── docs/                     # 文档
├── pyproject.toml            # 项目配置
└── README.md                 # 本文件
```

---

## 依赖

- Python >= 3.10
- numpy >= 1.21.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- statsmodels >= 0.13.0
- scikit-learn >= 1.0.0
- polars >= 0.15.0
- typer >= 0.9.0
- rich >= 13.0.0
- openpyxl >= 3.0.0

---

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 贡献

欢迎提交 Issue 和 Pull Request！
