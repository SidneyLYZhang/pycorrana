# PyCorrAna 框架总结

## 项目概述

PyCorrAna 是一个完整的 Python 相关性分析工具包，实现了用户要求的所有功能，并提供了简洁易用的 API 和 CLI 工具。

## 已实现功能清单

### 一、数据接入与基础清洗 ✅

| 功能 | 实现位置 | 说明 |
|------|---------|------|
| 读取常见格式 | `utils/data_utils.py:load_data()` | 支持 CSV/Excel/pandas/polars |
| 缺失值处理 | `utils/data_utils.py:handle_missing()` | dropna + 填充（均值/中位数/众数/KNN） |
| 缺失比例预警 | `utils/data_utils.py:handle_missing()` | 自动输出缺失值报告 |
| 自动类型推断 | `utils/data_utils.py:infer_types()` | 数值/二分类/多分类/有序/日期时间 |
| 异常值可视化 | `utils/data_utils.py:detect_outliers()` | IQR/Z-Score 方法 + 箱线图 |
| 大数据检测 | `utils/data_utils.py:is_large_data()` | 自动检测大数据集 |

### 二、相关性计算引擎 ✅

| 功能 | 实现位置 | 说明 |
|------|---------|------|
| 自动系数选择 | `core/analyzer.py:_compute_pair()` | 根据变量类型自动选择最优方法 |
| 数值+数值 | `utils/stats_utils.py` | Pearson（正态）/ Spearman |
| 数值+二分类 | `utils/stats_utils.py:point_biserial()` | 点双列相关 |
| 数值+多分类 | `utils/stats_utils.py:eta_coefficient()` | Eta 系数 / ANOVA |
| 分类+分类 | `utils/stats_utils.py:cramers_v()` | Cramér's V |
| 有序+有序 | `core/analyzer.py:_compute_pair()` | Kendall's Tau |
| 显著性检验 | `core/analyzer.py:compute_correlation()` | 自动输出 p 值 |
| 多重比较校正 | `utils/stats_utils.py:correct_pvalues()` | Bonferroni/FDR/Holm |
| 批量计算 | `core/analyzer.py:compute_correlation()` | 相关系数矩阵 + p值矩阵 |

### 三、可视化核心 ✅

| 功能 | 实现位置 | 说明 |
|------|---------|------|
| 相关性热力图 | `core/visualizer.py:plot_heatmap()` | 带系数标注、聚类可选 |
| 散点图矩阵 | `core/visualizer.py:plot_pairplot()` | 自动为数值变量绘制 |
| 箱线图/小提琴图 | `core/visualizer.py:plot_boxplot()` | 数值×分类分组展示 |
| 相关网络图 | `core/visualizer.py:plot_correlation_network()` | 基于 networkx |
| 一键保存 | 各 plot 方法 | 支持 PNG/SVG/PDF |

### 四、结果导出与简易报告 ✅

| 功能 | 实现位置 | 说明 |
|------|---------|------|
| 导出表格 | `core/reporter.py:export_results()` | CSV/Excel 格式 |
| 文本摘要 | `core/reporter.py:generate_summary()` | 显著相关对列表 + 解释指南 |
| HTML 报告 | `core/reporter.py:to_html()` | 完整的 HTML 格式报告 |
| Markdown 报告 | `core/reporter.py:to_markdown()` | Markdown 格式 |
| 控制台友好 | `core/analyzer.py:summary()` | Jupyter 中直接渲染 |

### 五、包功能 ✅

| 功能 | 实现位置 | 说明 |
|------|---------|------|
| 单函数入口 | `core/analyzer.py:quick_corr()` | 一键完成全套分析 |
| 精简参数 | `quick_corr()` | method/plot/export 等 |
| 日志提示 | `CorrAnalyzer.__init__()` | 自动告知使用的方法 |
| 示例数据集 | `datasets.py` | iris/titanic/wine/生成数据 |

### 六、偏相关分析 ✅

| 功能 | 实现位置 | 说明 |
|------|---------|------|
| 偏相关分析 | `core/partial_corr.py:partial_corr()` | 控制协变量后的净相关 |
| 偏相关矩阵 | `core/partial_corr.py:partial_corr_matrix()` | 批量计算 |
| 半偏相关 | `core/partial_corr.py:semipartial_corr()` | 部分相关 |
| 分析器类 | `core/partial_corr.py:PartialCorrAnalyzer` | 完整分析流程 |

### 七、非线性依赖检测 ✅

| 功能 | 实现位置 | 说明 |
|------|---------|------|
| 距离相关 | `core/nonlinear.py:distance_correlation()` | 检测任意依赖 |
| 互信息 | `core/nonlinear.py:mutual_info_score()` | 非线性关联 |
| MIC | `core/nonlinear.py:maximal_information_coefficient()` | 最大信息系数 |
| 分析器类 | `core/nonlinear.py:NonlinearAnalyzer` | 完整分析流程 |

### 八、典型相关分析 ✅

| 功能 | 实现位置 | 说明 |
|------|---------|------|
| 典型相关分析 | `core/cca.py:cca()` | 两组变量间的典型相关 |
| 置换检验 | `core/cca.py:cca_permutation_test()` | 显著性验证 |
| Wilks' Lambda | `core/cca.py:_wilks_lambda_test()` | 显著性检验 |
| 冗余指数 | `core/cca.py:_compute_redundancy()` | 方差解释比例 |
| 分析器类 | `core/cca.py:CCAAnalyzer` | 完整分析流程 |

### 九、大数据优化 ✅

| 功能 | 实现位置 | 说明 |
|------|---------|------|
| 智能采样 | `utils/large_data.py:smart_sample()` | 随机/分层采样 |
| 分块计算 | `utils/large_data.py:chunked_correlation()` | 大矩阵分块处理 |
| 内存优化 | `utils/large_data.py:optimize_dataframe()` | 减少内存占用 |
| 配置类 | `utils/large_data.py:LargeDataConfig` | 灵活配置参数 |

### 十、CLI 工具 ✅

| 功能 | 实现位置 | 说明 |
|------|---------|------|
| 分模块 CLI | `cli/main_cli.py` | analyze/clean/partial/nonlinear/info |
| 交互式 CLI | `cli/interactive.py` | 问答式完整流程 |

## 项目结构

```
pycorrana/
├── src/pycorrana/
│   ├── __init__.py              # 包入口
│   ├── datasets.py              # 示例数据集
│   ├── core/                    # 核心分析模块
│   │   ├── analyzer.py          # 主分析器 (quick_corr, CorrAnalyzer)
│   │   ├── visualizer.py        # 可视化 (热力图、散点图等)
│   │   ├── reporter.py          # 报告生成
│   │   ├── partial_corr.py      # 偏相关分析
│   │   ├── nonlinear.py         # 非线性检测
│   │   └── cca.py               # 典型相关分析
│   ├── utils/                   # 工具函数
│   │   ├── data_utils.py        # 数据处理
│   │   ├── stats_utils.py       # 统计工具
│   │   └── large_data.py        # 大数据优化
│   └── cli/                     # 命令行工具
│       ├── main_cli.py          # 分模块CLI
│       └── interactive.py       # 交互式CLI
├── tests/                       # 测试
├── examples/                    # 示例代码
├── docs/                        # 文档
├── demo.py                      # 演示脚本
├── pyproject.toml               # 项目配置
├── README.md                    # 项目说明
├── ARCHITECTURE.md              # 架构文档
├── QUICKSTART.md                # 快速入门
└── FRAMEWORK_SUMMARY.md         # 本文件
```

## 使用示例

### Python API

```python
from pycorrana import quick_corr, CorrAnalyzer
from pycorrana.core.partial_corr import partial_corr
from pycorrana.core.nonlinear import distance_correlation
from pycorrana.core.cca import cca, CCAAnalyzer
from pycorrana.utils import LargeDataConfig

# 一行代码分析
result = quick_corr('data.csv', target='sales', export='results.xlsx')

# 使用分析器类
analyzer = CorrAnalyzer(df)
analyzer.fit()
analyzer.plot_heatmap()
analyzer.export_results('results.xlsx')

# 大数据优化
config = LargeDataConfig(sample_size=100000, auto_sample=True)
analyzer = CorrAnalyzer(large_df, large_data_config=config)

# 偏相关
partial_corr(df, x='income', y='happiness', covars=['age', 'education'])

# 非线性检测
distance_correlation(df['X'], df['Y'], return_pvalue=True)

# 典型相关分析
result = cca(df[['x1', 'x2', 'x3']], df[['y1', 'y2']])
print(f"第一典型相关系数: {result['canonical_correlations'][0]:.3f}")
```

### CLI 工具

```bash
# 完整分析
pycorrana analyze data.csv --target sales --export results.xlsx

# 数据清洗
pycorrana clean data.csv --dropna --output cleaned.csv

# 偏相关
pycorrana partial data.csv -x income -y happiness -c age,education

# 非线性检测
pycorrana nonlinear data.csv --top 20

# 交互式
pycorrana-interactive
```

## 技术亮点

1. **智能方法选择**：根据变量类型自动选择最优相关系数方法
2. **完整的p值处理**：自动计算 + 多重比较校正
3. **丰富的可视化**：热力图、散点图矩阵、网络图等
4. **多种导出格式**：Excel/CSV/HTML/Markdown
5. **双重CLI**：分模块CLI + 交互式CLI
6. **大数据优化**：智能采样、分块计算、内存优化
7. **典型相关分析**：支持两组变量间的多元相关性分析
8. **完善的文档**：README + 架构文档 + 快速入门 + 示例代码
9. **现代Python**：支持 Python 3.10-3.13

## 测试验证

```bash
# 运行演示脚本
python demo.py

# 运行测试
python -m pytest tests/

# CLI 测试
pycorrana analyze test_data.csv --export results.xlsx
```

## 扩展性

框架设计考虑了良好的扩展性：
- 添加新方法：修改 `_compute_pair()`
- 添加新可视化：在 `CorrVisualizer` 添加方法
- 添加新导出格式：在 `CorrReporter` 添加方法
- 添加新数据优化：在 `large_data.py` 添加函数

## 总结

PyCorrAna 完整实现了用户的所有需求，提供了：
- ✅ 数据接入与清洗
- ✅ 智能相关性计算
- ✅ 丰富的可视化
- ✅ 多种结果导出
- ✅ 简洁的API
- ✅ 完善的CLI工具
- ✅ 偏相关分析
- ✅ 非线性依赖检测
- ✅ 典型相关分析
- ✅ 交互式界面
- ✅ 示例数据集
- ✅ 大数据优化
