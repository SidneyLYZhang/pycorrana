.. _user_guide:

============
用户指南
============

本指南详细介绍 PyCorrAna 的各项功能和使用方法。

数据输入与输出
==============

支持的数据格式
--------------

PyCorrAna 支持多种数据输入格式：

**文件路径**

.. code-block:: python

   from pycorrana import quick_corr

   result = quick_corr('data.csv')        # CSV 文件
   result = quick_corr('data.xlsx')       # Excel 文件

**pandas DataFrame**

.. code-block:: python

   import pandas as pd
   from pycorrana import quick_corr

   df = pd.read_csv('data.csv')
   result = quick_corr(df)

**polars DataFrame**

.. code-block:: python

   import polars as pl
   from pycorrana import quick_corr

   df = pl.read_csv('data.csv')
   result = quick_corr(df)

数据加载选项
------------

使用 :func:`pycorrana.utils.data_utils.load_data` 函数可以更灵活地加载数据：

.. code-block:: python

   from pycorrana.utils.data_utils import load_data

   df = load_data(
       'data.xlsx',
       sheet_name='Sheet1',  # Excel 工作表名
       header=0,             # 表头行
       index_col=None        # 索引列
   )

数据预处理
==========

类型推断
--------

PyCorrAna 会自动推断每列的数据类型：

.. code-block:: python

   from pycorrana.utils.data_utils import infer_types

   type_mapping = infer_types(df)
   print(type_mapping)
   # {'age': 'numeric', 'income': 'numeric', 'gender': 'binary', 'city': 'categorical'}

推断的类型包括：

- ``numeric`` - 数值型
- ``binary`` - 二分类
- ``categorical`` - 多分类
- ``ordinal`` - 有序分类
- ``datetime`` - 日期时间

缺失值处理
----------

支持多种缺失值处理策略：

**删除缺失值**

.. code-block:: python

   analyzer = CorrAnalyzer(df, missing_strategy='drop')

**填充缺失值**

.. code-block:: python

   analyzer = CorrAnalyzer(
       df,
       missing_strategy='fill',
       fill_method='mean'    # 'mean', 'median', 'mode', 'knn'
   )

**手动处理**

.. code-block:: python

   from pycorrana.utils.data_utils import handle_missing

   df_clean = handle_missing(
       df,
       strategy='fill',
       fill_method='knn',
       n_neighbors=5
   )

异常值检测
----------

.. code-block:: python

   from pycorrana.utils.data_utils import detect_outliers

   outliers = detect_outliers(df, method='iqr')
   print(outliers)

相关性计算
==========

自动方法选择
------------

PyCorrAna 的核心特性是自动选择最适合的相关系数方法：

.. list-table::
   :header-rows: 1

   * - 变量类型组合
     - 自动选用方法
     - 说明
   * - 数值 + 数值
     - Pearson / Spearman
     - 正态分布用 Pearson，否则用 Spearman
   * - 数值 + 二分类
     - 点双列相关
     - 自动映射为 0/1
   * - 数值 + 多分类
     - Eta 系数
     - 输出相关比
   * - 分类 + 分类
     - Cramér's V
     - 基于卡方检验
   * - 有序 + 有序
     - Kendall's Tau
     - 处理重复秩

指定计算方法
------------

你也可以手动指定计算方法：

.. code-block:: python

   analyzer = CorrAnalyzer(df, method='spearman')

支持的方法：

- ``'auto'`` - 自动选择（默认）
- ``'pearson'`` - Pearson 相关系数
- ``'spearman'`` - Spearman 秩相关
- ``'kendall'`` - Kendall's Tau

显著性检验
==========

p 值校正
--------

多重比较时需要进行 p 值校正：

.. code-block:: python

   analyzer = CorrAnalyzer(
       df,
       pvalue_correction='fdr_bh'  # Benjamini-Hochberg FDR
   )

支持的校正方法：

- ``'bonferroni'`` - Bonferroni 校正
- ``'fdr_bh'`` - Benjamini-Hochberg FDR（默认）
- ``'fdr_by'`` - Benjamini-Yekutieli FDR
- ``'holm'`` - Holm 校正
- ``None`` - 不进行校正

获取显著相关对
--------------

.. code-block:: python

   result = analyzer.fit()
   significant_pairs = result['significant_pairs']
   
   for pair in significant_pairs:
       print(f"{pair['var1']} - {pair['var2']}: r={pair['correlation']:.3f}, p={pair['p_value']:.4f}")

大数据优化
==========

PyCorrAna 提供了针对大数据集的优化策略。

自动检测大数据
--------------

PyCorrAna 会自动检测大数据集：

.. code-block:: python

   from pycorrana.utils.data_utils import is_large_data, estimate_memory_usage

   # 检测是否为大数据
   if is_large_data(df):
       print("检测到大数据集")

   # 查看内存使用
   mem_mb = estimate_memory_usage(df)
   print(f"内存使用: {mem_mb:.1f} MB")

配置大数据优化
--------------

使用 ``LargeDataConfig`` 配置大数据优化参数：

.. code-block:: python

   from pycorrana import CorrAnalyzer
   from pycorrana.utils import LargeDataConfig

   config = LargeDataConfig(
       sample_size=100000,      # 采样大小
       auto_sample=True,        # 自动采样
       auto_optimize=True,      # 自动优化内存
       stratify_col=None,       # 分层采样列
       random_state=42,         # 随机种子
       verbose=True
   )

   analyzer = CorrAnalyzer(large_df, large_data_config=config)
   analyzer.fit()

智能采样
--------

手动使用采样功能：

.. code-block:: python

   from pycorrana.utils import smart_sample

   # 随机采样
   sampled_df = smart_sample(
       df,
       sample_size=50000,
       random_state=42
   )

   # 分层采样
   sampled_df = smart_sample(
       df,
       sample_size=50000,
       stratify_col='category',  # 按此列分层
       random_state=42
   )

内存优化
--------

.. code-block:: python

   from pycorrana.utils import optimize_dataframe

   # 优化 DataFrame 内存使用
   optimized_df = optimize_dataframe(df, verbose=True)

分块计算
--------

对于超大矩阵，可以使用分块计算：

.. code-block:: python

   from pycorrana.utils import chunked_correlation

   # 分块计算相关性
   corr_matrix = chunked_correlation(
       df,
       chunk_size=10000,
       method='spearman'
   )

可视化
======

热力图
------

.. code-block:: python

   analyzer.plot_heatmap(
       figsize=(12, 10),      # 图表大小
       annot=True,            # 显示数值标注
       cmap='RdBu_r',         # 颜色映射
       center=0,              # 颜色中心
       vmin=-1, vmax=1,       # 颜色范围
       cluster=True,          # 层次聚类
       cluster_method='ward'  # 聚类方法
   )

散点图矩阵
----------

.. code-block:: python

   analyzer.plot_pairplot(
       columns=['var1', 'var2', 'var3'],
       hue='category',        # 按分类着色
       diag_kind='kde',       # 对角线图类型
       corner=True            # 只显示下三角
   )

箱线图和小提琴图
----------------

.. code-block:: python

   analyzer.plot_boxplot(
       numeric_col='price',
       categorical_col='category',
       kind='violin',         # 'box', 'violin', 'boxen'
       split=True             # 分割显示
   )

相关网络图
----------

.. code-block:: python

   analyzer.visualizer.plot_correlation_network(
       analyzer.corr_matrix,
       threshold=0.3,         # 只显示 |r| > 0.3 的边
       layout='spring'        # 布局算法
   )

结果导出
========

Excel 导出
----------

.. code-block:: python

   analyzer.export_results(
       'results.xlsx',
       format='excel'
   )

导出的 Excel 包含多个工作表：

- 相关性矩阵
- p 值矩阵
- 显著相关对列表
- 方法说明

CSV 导出
--------

.. code-block:: python

   analyzer.export_results(
       'correlation_matrix.csv',
       format='csv'
   )

文本摘要
--------

.. code-block:: python

   summary = analyzer.summary()
   print(summary)

偏相关分析
==========

基本用法
--------

控制一个或多个协变量：

.. code-block:: python

   from pycorrana import partial_corr

   result = partial_corr(
       df,
       x='income',
       y='happiness',
       covars=['age', 'education']
   )
   print(f"偏相关系数: {result['partial_correlation']:.3f}")
   print(f"p值: {result['p_value']:.4e}")

偏相关矩阵
----------

.. code-block:: python

   from pycorrana import partial_corr_matrix

   matrix = partial_corr_matrix(
       df,
       covars=['age'],
       columns=['income', 'happiness', 'health']
   )

半偏相关
--------

.. code-block:: python

   from pycorrana import semipartial_corr

   result = semipartial_corr(
       df,
       x='income',
       y='happiness',
       covars=['age']
   )

非线性分析
==========

距离相关
--------

距离相关可以检测任意形式的依赖关系：

.. code-block:: python

   from pycorrana import distance_correlation

   result = distance_correlation(df['x'], df['y'], return_pvalue=True)
   print(f"距离相关系数: {result['dcor']:.4f}")
   print(f"p值: {result['p_value']:.4f}")

互信息
------

.. code-block:: python

   from pycorrana import mutual_info_score

   result = mutual_info_score(df['x'], df['y'])
   print(f"互信息: {result['mi_normalized']:.4f}")

最大信息系数
------------

.. code-block:: python

   from pycorrana import maximal_information_coefficient

   result = maximal_information_coefficient(df['x'], df['y'])
   print(f"MIC: {result['mic']:.4f}")

非线性依赖报告
--------------

.. code-block:: python

   from pycorrana import nonlinear_dependency_report

   report = nonlinear_dependency_report(df, top_n=10)
   print(report)

使用示例数据集
==============

加载内置数据集
--------------

.. code-block:: python

   from pycorrana import load_iris, load_titanic, load_wine

   iris = load_iris()
   titanic = load_titanic()
   wine = load_wine()

生成模拟数据
------------

.. code-block:: python

   from pycorrana import make_correlated_data

   df = make_correlated_data(
       n_samples=1000,
       n_features=10,
       correlation=0.7
   )

查看可用数据集
--------------

.. code-block:: python

   from pycorrana import list_datasets

   datasets = list_datasets()
   for name, desc in datasets.items():
       print(f"{name}: {desc}")
