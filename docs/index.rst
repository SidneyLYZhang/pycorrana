.. PyCorrAna Documentation

===================================
PyCorrAna - Python 相关性分析工具包
===================================

.. image:: logo.png
   :alt: PyCorrAna Logo
   :width: 200px
   :align: center

**自动化相关性分析工具 - 降低决策成本，一键输出关键结果**

PyCorrAna 是一个方便快速入手的 Python 相关性分析工具，核心设计理念：

- **自动化常规操作** - 智能识别数据类型，自动选择最优相关系数方法
- **降低决策成本** - 无需纠结用 Pearson 还是 Spearman，工具自动帮你选择
- **一键输出关键结果** - 从数据加载到结果导出，一行代码搞定

主要特性
========

- **数据接入** - 支持 CSV/Excel/pandas/polars，自动类型推断
- **缺失值处理** - 删除/填充（均值/中位数/众数/KNN预测）
- **相关性计算** - 自动方法选择（Pearson/Spearman/Kendall/Cramér's V/Eta等）
- **显著性检验** - 自动 p 值计算，支持多重比较校正
- **可视化** - 热力图、散点图矩阵、箱线图、相关网络图
- **结果导出** - Excel/CSV/HTML/Markdown 结果
- **偏相关分析** - 控制协变量后的净相关分析
- **非线性检测** - 距离相关、互信息、MIC
- **大数据优化** - 智能采样、分块计算、内存优化

目录
====

.. toctree::
   :maxdepth: 2
   :caption: 入门指南

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: 用户指南

   user_guide
   cli
   examples

.. toctree::
   :maxdepth: 2
   :caption: API 参考

   api

.. toctree::
   :maxdepth: 1
   :caption: 其他

   changelog
   contributing

快速示例
========

一行代码完成分析：

.. code-block:: python

   from pycorrana import quick_corr

   result = quick_corr('data.csv')

指定目标变量：

.. code-block:: python

   result = quick_corr(df, target='sales')

大数据优化：

.. code-block:: python

   from pycorrana import CorrAnalyzer
   from pycorrana.utils import LargeDataConfig

   config = LargeDataConfig(sample_size=100000, auto_sample=True)
   analyzer = CorrAnalyzer(large_df, large_data_config=config)
   analyzer.fit()

使用命令行工具：

.. code-block:: bash

   pycorrana analyze data.csv --target sales --export results.xlsx

索引与搜索
==========

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
