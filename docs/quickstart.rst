.. _quickstart:

============
快速开始
============

本指南将帮助你快速上手 PyCorrAna，了解基本用法和核心功能。

基本用法
========

一行代码完成分析
----------------

PyCorrAna 的核心设计理念是简化分析流程。使用 :func:`pycorrana.quick_corr` 函数可以一键完成完整的分析：

.. code-block:: python

   from pycorrana import quick_corr

   result = quick_corr('data.csv')

这一行代码会自动完成：

1. 加载数据文件
2. 自动识别数据类型
3. 选择合适的相关系数方法
4. 计算相关性矩阵
5. 进行显著性检验
6. 生成可视化图表

使用 DataFrame
--------------

如果你已经有 pandas DataFrame，可以直接传入：

.. code-block:: python

   import pandas as pd
   from pycorrana import quick_corr

   df = pd.read_csv('data.csv')
   result = quick_corr(df)

指定目标变量
------------

当你只关心某个目标变量与其他变量的相关性时：

.. code-block:: python

   result = quick_corr(df, target='sales')

这会计算所有变量与 ``sales`` 变量的相关性。

使用分析器类
============

对于更精细的控制，可以使用 :class:`pycorrana.CorrAnalyzer` 类：

.. code-block:: python

   from pycorrana import CorrAnalyzer

   analyzer = CorrAnalyzer(
       df,
       method='spearman',      # 指定方法
       missing_strategy='fill', # 缺失值填充
       fill_method='mean'       # 使用均值填充
   )
   
   result = analyzer.fit()
   
   analyzer.plot_heatmap()
   analyzer.export_results('results.xlsx')

分析器配置选项
--------------

.. list-table::
   :header-rows: 1

   * - 参数
     - 默认值
     - 说明
   * - method
     - 'auto'
     - 相关系数方法：'auto', 'pearson', 'spearman', 'kendall'
   * - missing_strategy
     - 'warn'
     - 缺失值处理：'warn', 'drop', 'fill'
   * - fill_method
     - None
     - 填充方法：'mean', 'median', 'mode', 'knn'
   * - pvalue_correction
     - 'fdr_bh'
     - p 值校正方法
   * - large_data_config
     - None
     - 大数据优化配置

大数据优化
==========

PyCorrAna 提供了针对大数据集的优化策略。

自动检测大数据
--------------

PyCorrAna 会自动检测大数据集（默认阈值：10万行或500MB）并提示优化建议。

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
       verbose=True
   )

   analyzer = CorrAnalyzer(large_df, large_data_config=config)
   analyzer.fit()

智能采样
--------

.. code-block:: python

   from pycorrana.utils import smart_sample

   # 随机采样
   sampled_df = smart_sample(df, sample_size=50000)

   # 分层采样
   sampled_df = smart_sample(df, sample_size=50000, stratify_col='category')

可视化
======

热力图
------

.. code-block:: python

   analyzer.plot_heatmap(
       figsize=(12, 10),
       annot=True,        # 显示数值
       cmap='RdBu_r',     # 颜色映射
       cluster=True       # 层次聚类
   )

散点图矩阵
----------

.. code-block:: python

   analyzer.plot_pairplot(
       columns=['var1', 'var2', 'var3'],
       hue='category'     # 按分类着色
   )

箱线图
------

.. code-block:: python

   analyzer.plot_boxplot(
       numeric_col='price',
       categorical_col='category',
       kind='violin'      # 'box', 'violin', 'boxen'
   )

导出结果
========

导出为 Excel
------------

.. code-block:: python

   analyzer.export_results('results.xlsx', format='excel')

导出为 CSV
----------

.. code-block:: python

   analyzer.export_results('results.csv', format='csv')

查看摘要
--------

.. code-block:: python

   print(analyzer.summary())

使用示例数据集
==============

PyCorrAna 提供了几个内置示例数据集：

.. code-block:: python

   from pycorrana import load_iris, load_titanic, load_wine

   iris = load_iris()
   titanic = load_titanic()
   wine = load_wine()

查看可用数据集：

.. code-block:: python

   from pycorrana import list_datasets

   print(list_datasets())

生成模拟数据：

.. code-block:: python

   from pycorrana import make_correlated_data

   df = make_correlated_data(
       n_samples=1000,
       n_features=10,
       correlation=0.7
   )

命令行工具
==========

PyCorrAna 提供了命令行工具，无需编写代码即可进行分析：

完整分析
--------

.. code-block:: bash

   pycorrana analyze data.csv --target sales --export results.xlsx

数据清洗
--------

.. code-block:: bash

   pycorrana clean data.csv --dropna --output cleaned.csv

偏相关分析
----------

.. code-block:: bash

   pycorrana partial data.csv -x income -y happiness -c age,education

非线性检测
----------

.. code-block:: bash

   pycorrana nonlinear data.csv --top 20

交互式模式
----------

.. code-block:: bash

   pycorrana-interactive

进阶功能
========

偏相关分析
----------

控制协变量后的净相关分析：

.. code-block:: python

   from pycorrana import partial_corr

   result = partial_corr(
       df,
       x='income',
       y='happiness',
       covars=['age', 'education']
   )
   print(f"偏相关系数: {result['partial_correlation']:.3f}")

半偏相关
--------

.. code-block:: python

   from pycorrana import semipartial_corr

   result = semipartial_corr(df, x='income', y='happiness', covars='age')

非线性依赖检测
--------------

检测变量间的非线性关系：

.. code-block:: python

   from pycorrana import (
       distance_correlation,
       mutual_info_score,
       nonlinear_dependency_report
   )

   result = distance_correlation(df['x'], df['y'], return_pvalue=True)
   print(f"dCor: {result['dcor']:.3f}")
   
   result = mutual_info_score(df['x'], df['y'])
   print(f"MI: {result['mi_normalized']:.3f}")
   
   report = nonlinear_dependency_report(df)

下一步
======

- 阅读 :doc:`user_guide` 了解更多详细用法
- 查看 :doc:`api` 了解完整的 API 参考
- 浏览 :doc:`examples` 获取更多示例代码
