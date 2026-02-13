.. _api:

=================
API 参考
=================

本节提供 PyCorrAna 的完整 API 参考文档。

核心分析模块
============

.. module:: pycorrana.core.analyzer

quick_corr 函数
---------------

.. autofunction:: pycorrana.quick_corr

CorrAnalyzer 类
---------------

.. autoclass:: pycorrana.CorrAnalyzer
   :members:
   :inherited-members:

偏相关分析模块
==============

.. module:: pycorrana.core.partial_corr

partial_corr 函数
-----------------

.. autofunction:: pycorrana.partial_corr

partial_corr_matrix 函数
------------------------

.. autofunction:: pycorrana.partial_corr_matrix

semipartial_corr 函数
---------------------

.. autofunction:: pycorrana.semipartial_corr

PartialCorrAnalyzer 类
----------------------

.. autoclass:: pycorrana.PartialCorrAnalyzer
   :members:
   :inherited-members:

非线性分析模块
==============

.. module:: pycorrana.core.nonlinear

distance_correlation 函数
-------------------------

.. autofunction:: pycorrana.distance_correlation

mutual_info_score 函数
----------------------

.. autofunction:: pycorrana.mutual_info_score

maximal_information_coefficient 函数
------------------------------------

.. autofunction:: pycorrana.maximal_information_coefficient

nonlinear_dependency_report 函数
--------------------------------

.. autofunction:: pycorrana.nonlinear_dependency_report

NonlinearAnalyzer 类
--------------------

.. autoclass:: pycorrana.NonlinearAnalyzer
   :members:
   :inherited-members:

可视化模块
==========

.. module:: pycorrana.core.visualizer

CorrVisualizer 类
-----------------

.. autoclass:: pycorrana.core.visualizer.CorrVisualizer
   :members:
   :inherited-members:

报告生成模块
============

.. module:: pycorrana.core.reporter

CorrReporter 类
---------------

.. autoclass:: pycorrana.core.reporter.CorrReporter
   :members:
   :inherited-members:

数据工具模块
============

.. module:: pycorrana.utils.data_utils

数据加载
--------

.. autofunction:: pycorrana.utils.data_utils.load_data

类型推断
--------

.. autofunction:: pycorrana.utils.data_utils.infer_types

缺失值处理
----------

.. autofunction:: pycorrana.utils.data_utils.handle_missing

异常值检测
----------

.. autofunction:: pycorrana.utils.data_utils.detect_outliers

统计工具模块
============

.. module:: pycorrana.utils.stats_utils

正态性检验
----------

.. autofunction:: pycorrana.utils.stats_utils.check_normality

p 值校正
--------

.. autofunction:: pycorrana.utils.stats_utils.correct_pvalues

Cramér's V
----------

.. autofunction:: pycorrana.utils.stats_utils.cramers_v

Eta 系数
--------

.. autofunction:: pycorrana.utils.stats_utils.eta_coefficient

点双列相关
----------

.. autofunction:: pycorrana.utils.stats_utils.point_biserial

相关系数解释
------------

.. autofunction:: pycorrana.utils.stats_utils.interpret_correlation

数据集模块
==========

.. module:: pycorrana.datasets

示例数据集
----------

.. autofunction:: pycorrana.load_iris

.. autofunction:: pycorrana.load_titanic

.. autofunction:: pycorrana.load_wine

.. autofunction:: pycorrana.make_correlated_data

.. autofunction:: pycorrana.list_datasets

命令行模块
==========

.. module:: pycorrana.cli

主命令行工具
------------

.. automodule:: pycorrana.cli.main_cli
   :members:

交互式工具
----------

.. automodule:: pycorrana.cli.interactive
   :members:
