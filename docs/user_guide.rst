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

   r, p = partial_corr(
       df,
       x='income',
       y='happiness',
       covars=['age', 'education']
   )

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

   r, p = semipartial_corr(
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

   dcor = distance_correlation(df['x'], df['y'])
   print(f"距离相关系数: {dcor:.4f}")

互信息
------

.. code-block:: python

   from pycorrana import mutual_info_score

   mi = mutual_info_score(df['x'], df['y'])
   print(f"互信息: {mi:.4f}")

最大信息系数
------------

.. code-block:: python

   from pycorrana import maximal_information_coefficient

   mic = maximal_information_coefficient(df['x'], df['y'])
   print(f"MIC: {mic:.4f}")

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
       correlation_strength=0.7,
       noise_level=0.1
   )

查看可用数据集
--------------

.. code-block:: python

   from pycorrana import list_datasets

   datasets = list_datasets()
   for name, desc in datasets.items():
       print(f"{name}: {desc}")
