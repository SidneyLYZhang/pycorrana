# PyCorrAna 架构文档

## 项目概述

PyCorrAna 是一个 Python 相关性分析工具包，设计理念是**自动化常规操作、降低决策成本、一键输出关键结果**。

## 架构设计

### 1. 整体架构

```
pycorrana/
├── src/pycorrana/              # 主包
│   ├── __init__.py             # 包入口，导出主要API
│   ├── datasets.py             # 示例数据集
│   ├── core/                   # 核心分析模块
│   │   ├── analyzer.py         # 主分析器（quick_corr, CorrAnalyzer）
│   │   ├── visualizer.py       # 可视化（热力图、散点图等）
│   │   ├── reporter.py         # 报告生成（Excel/HTML/Markdown）
│   │   ├── partial_corr.py     # 偏相关分析
│   │   ├── nonlinear.py        # 非线性依赖检测
│   │   └── cca.py              # 典型相关分析
│   ├── utils/                  # 工具函数
│   │   ├── data_utils.py       # 数据处理（加载、清洗、类型推断）
│   │   ├── stats_utils.py      # 统计工具（p值校正、系数解释）
│   │   └── large_data.py       # 大数据优化（采样、分块计算）
│   └── cli/                    # 命令行工具
│       ├── main_cli.py         # 分模块CLI
│       └── interactive.py      # 交互式CLI
├── tests/                      # 测试
├── examples/                   # 示例代码
└── docs/                       # 文档
```

### 2. 数据流水线

```
输入数据
    ↓
[数据接入与基础清洗]
    - load_data(): 支持 CSV/Excel/pandas/polars
    - infer_types(): 自动类型推断
    - handle_missing(): 缺失值处理
    - detect_outliers(): 异常值检测
    - is_large_data(): 大数据检测
    ↓
[大数据优化] (可选)
    - smart_sample(): 智能采样
    - optimize_dataframe(): 内存优化
    - chunked_correlation(): 分块计算
    ↓
[相关性计算引擎]
    - 自动方法选择（根据变量类型）
    - 批量计算相关系数矩阵
    - 显著性检验 + p值校正
    ↓
[可视化核心]
    - 热力图（支持聚类）
    - 散点图矩阵
    - 箱线图/小提琴图
    - 相关网络图
    ↓
[结果导出与报告]
    - Excel/CSV 导出
    - HTML/Markdown 报告
    - 文本摘要
```

### 3. 核心模块详解

#### 3.1 analyzer.py - 主分析器

**主要类/函数：**
- `quick_corr()`: 一键分析入口函数
- `CorrAnalyzer`: 分析器类，提供完整分析流程

**关键方法：**
- `preprocess()`: 数据预处理
- `compute_correlation()`: 计算相关性
- `fit()`: 执行完整分析
- `plot_heatmap()`: 热力图
- `export_results()`: 导出结果
- `summary()`: 文本摘要

**大数据支持：**
- 自动检测大数据集
- 支持 `LargeDataConfig` 配置
- 自动采样和内存优化

#### 3.2 visualizer.py - 可视化

**主要类：**
- `CorrVisualizer`: 可视化器

**关键方法：**
- `plot_heatmap()`: 相关性热力图
- `plot_pairplot()`: 散点图矩阵
- `plot_boxplot()`: 箱线图/小提琴图
- `plot_correlation_network()`: 相关网络图
- `plot_significant_pairs()`: 显著相关对条形图

#### 3.3 reporter.py - 报告生成

**主要类：**
- `CorrReporter`: 报告生成器

**关键方法：**
- `export_results()`: 导出为 Excel/CSV
- `generate_summary()`: 生成文本摘要
- `to_markdown()`: Markdown 报告
- `to_html()`: HTML 报告

#### 3.4 partial_corr.py - 偏相关分析

**主要函数/类：**
- `partial_corr()`: 计算偏相关系数
- `partial_corr_matrix()`: 偏相关矩阵
- `semipartial_corr()`: 半偏相关（部分相关）
- `PartialCorrAnalyzer`: 偏相关分析器类

#### 3.5 nonlinear.py - 非线性检测

**主要函数/类：**
- `distance_correlation()`: 距离相关
- `mutual_info_score()`: 互信息
- `maximal_information_coefficient()`: MIC
- `nonlinear_dependency_report()`: 非线性依赖报告
- `NonlinearAnalyzer`: 非线性分析器类

#### 3.6 cca.py - 典型相关分析

**主要函数/类：**
- `cca()`: 执行典型相关分析
- `cca_permutation_test()`: 置换检验验证显著性
- `CCAAnalyzer`: 典型相关分析器类

**关键功能：**
- 计算两组变量之间的典型相关系数
- Wilks' Lambda 显著性检验
- Fisher Z 变换置信区间
- 冗余指数（Redundancy Index）计算
- 典型变量系数和得分

#### 3.7 large_data.py - 大数据优化

**主要函数/类：**
- `LargeDataConfig`: 大数据配置类
- `smart_sample()`: 智能采样（支持分层采样）
- `chunked_correlation()`: 分块计算相关性
- `chunked_apply()`: 分块应用函数
- `optimize_dataframe()`: 内存优化

### 4. 自动方法选择规则

| 变量类型 | 自动选用方法 | 备注 |
|----------|-------------|------|
| 数值 + 数值 | Pearson（若正态）/ Spearman | 默认 Spearman 更稳健 |
| 数值 + 二分类 | 点双列相关 | 自动映射为 0/1 |
| 数值 + 多分类 | Eta 系数 / ANOVA | 输出相关比 |
| 分类 + 分类 | Cramér's V | 基于卡方检验 |
| 有序 + 有序 | Kendall's Tau | 处理重复秩 |

### 5. CLI 工具架构

#### 5.1 分模块 CLI (main_cli.py)

```bash
pycorrana <command> [options]

Commands:
    analyze     执行完整的相关性分析
    clean       数据清洗和预处理
    partial     偏相关分析
    nonlinear   非线性依赖检测
    info        查看数据信息
```

#### 5.2 交互式 CLI (interactive.py)

```
启动: pycorrana-interactive

流程:
1. 数据加载（文件/示例数据集）
2. 主菜单选择
   - 执行完整分析
   - 数据探索
   - 数据清洗
   - 相关性分析
   - 偏相关分析
   - 非线性依赖检测
   - 可视化
   - 导出结果
```

### 6. 扩展性设计

#### 6.1 添加新的相关系数方法

在 `analyzer.py` 的 `_compute_pair()` 方法中添加：

```python
elif pair_type == 'new_type':
    corr_val, p_val = new_correlation_function(x, y)
    return corr_val, p_val, "new_method"
```

#### 6.2 添加新的可视化类型

在 `visualizer.py` 的 `CorrVisualizer` 类中添加新方法：

```python
def plot_new_chart(self, ...):
    # 实现新的可视化
    pass
```

#### 6.3 添加新的导出格式

在 `reporter.py` 的 `CorrReporter` 类中添加新方法：

```python
def to_new_format(self, ...):
    # 实现新的导出格式
    pass
```

### 7. 依赖关系

```
pycorrana
├── numpy (数组计算)
├── pandas (数据处理)
├── polars (高性能数据处理)
├── scipy (统计检验)
├── matplotlib (基础绘图)
├── seaborn (高级可视化)
├── statsmodels (统计模型)
├── scikit-learn (机器学习工具)
├── typer (CLI框架)
├── rich (终端美化)
└── openpyxl (Excel支持)
```

### 8. 设计原则

1. **单一职责原则**：每个模块只负责一个功能领域
2. **开闭原则**：对扩展开放，对修改关闭
3. **依赖倒置**：高层模块不依赖低层模块，都依赖抽象
4. **DRY 原则**：避免重复代码
5. **KISS 原则**：保持简单，一键完成分析

### 9. 使用场景

| 场景 | 推荐用法 |
|------|---------|
| 快速探索数据 | `quick_corr(df)` |
| 指定目标变量 | `quick_corr(df, target='Y')` |
| 自定义参数 | `CorrAnalyzer(df, method='spearman').fit()` |
| 批量处理 | CLI 工具 `pycorrana analyze` |
| 交互式分析 | `pycorrana-interactive` |
| 控制混淆变量 | `partial_corr(df, x, y, covars)` |
| 检测非线性 | `nonlinear_dependency_report(df)` |
| 两组变量关系 | `cca(df[X_vars], df[Y_vars])` |
| 大数据集 | `CorrAnalyzer(df, large_data_config=config)` |

### 10. 性能考虑

- 大数据集：使用 `LargeDataConfig` 配置自动采样
- 内存优化：使用 `optimize_dataframe()` 减少内存占用
- 计算加速：使用 `method='spearman'` 避免正态性检验
- 分块计算：使用 `chunked_correlation()` 处理超大矩阵
- 可视化：限制散点图矩阵的列数（<=6）

### 11. Python 版本支持

- 最低版本：Python 3.10
- 支持版本：Python 3.10, 3.11, 3.12, 3.13
