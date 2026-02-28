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

典型相关分析模块
================

.. module:: pycorrana.core.cca

典型相关分析（Canonical Correlation Analysis, CCA）是一种多元统计方法，
用于研究两组变量之间的线性关系。它寻找两组变量的线性组合，使得这些组合之间的相关性最大化。

cca 函数
--------

.. autofunction:: pycorrana.cca

cca_permutation_test 函数
-------------------------

.. autofunction:: pycorrana.cca_permutation_test

CCAAnalyzer 类
--------------

.. autoclass:: pycorrana.CCAAnalyzer
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

数据处理工具
============

.. module:: pycorrana.utils.data_utils

load_data 函数
--------------

.. autofunction:: pycorrana.utils.data_utils.load_data

infer_types 函数
----------------

.. autofunction:: pycorrana.utils.data_utils.infer_types

handle_missing 函数
-------------------

.. autofunction:: pycorrana.utils.data_utils.handle_missing

detect_outliers 函数
--------------------

.. autofunction:: pycorrana.utils.data_utils.detect_outliers

is_large_data 函数
------------------

.. autofunction:: pycorrana.utils.data_utils.is_large_data

estimate_memory_usage 函数
--------------------------

.. autofunction:: pycorrana.utils.data_utils.estimate_memory_usage

统计工具
========

.. module:: pycorrana.utils.stats_utils

check_normality 函数
--------------------

.. autofunction:: pycorrana.utils.stats_utils.check_normality

correct_pvalues 函数
--------------------

.. autofunction:: pycorrana.utils.stats_utils.correct_pvalues

cramers_v 函数
--------------

.. autofunction:: pycorrana.utils.stats_utils.cramers_v

eta_coefficient 函数
--------------------

.. autofunction:: pycorrana.utils.stats_utils.eta_coefficient

point_biserial 函数
-------------------

.. autofunction:: pycorrana.utils.stats_utils.point_biserial

interpret_correlation 函数
--------------------------

.. autofunction:: pycorrana.utils.stats_utils.interpret_correlation

大数据优化模块
==============

.. module:: pycorrana.utils.large_data

LargeDataConfig 类
------------------

.. autoclass:: pycorrana.utils.large_data.LargeDataConfig
   :members:
   :inherited-members:

smart_sample 函数
-----------------

.. autofunction:: pycorrana.utils.large_data.smart_sample

chunked_correlation 函数
------------------------

.. autofunction:: pycorrana.utils.large_data.chunked_correlation

chunked_apply 函数
------------------

.. autofunction:: pycorrana.utils.large_data.chunked_apply

optimize_dataframe 函数
-----------------------

.. autofunction:: pycorrana.utils.large_data.optimize_dataframe

示例数据集
==========

.. module:: pycorrana.datasets

load_iris 函数
--------------

.. autofunction:: pycorrana.datasets.load_iris

load_titanic 函数
-----------------

.. autofunction:: pycorrana.datasets.load_titanic

load_wine 函数
--------------

.. autofunction:: pycorrana.datasets.load_wine

make_correlated_data 函数
-------------------------

.. autofunction:: pycorrana.datasets.make_correlated_data

list_datasets 函数
------------------

.. autofunction:: pycorrana.datasets.list_datasets
