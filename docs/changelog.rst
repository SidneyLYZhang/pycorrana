.. _changelog:

============
更新日志
============

所有重要的变更都将记录在此文件中。

版本 0.1.0 (2026-02-13)
========================

首次发布
--------

这是 PyCorrAna 的首个正式版本。

新增功能
--------

核心分析功能
~~~~~~~~~~~~

- 添加 :func:`quick_corr` 一键分析函数
- 添加 :class:`CorrAnalyzer` 分析器类
- 支持自动方法选择（Pearson/Spearman/Kendall/Cramér's V/Eta等）
- 支持显著性检验和 p 值校正

数据预处理
~~~~~~~~~~

- 支持多种数据格式（CSV/Excel/pandas/polars）
- 自动类型推断
- 缺失值处理（删除/填充）
- 异常值检测

可视化
~~~~~~

- 相关性热力图（支持聚类）
- 散点图矩阵
- 箱线图/小提琴图
- 相关网络图

结果导出
~~~~~~~~

- Excel 导出
- CSV 导出
- 文本摘要

偏相关分析
~~~~~~~~~~

- :func:`partial_corr` 偏相关系数计算
- :func:`partial_corr_matrix` 偏相关矩阵
- :class:`PartialCorrAnalyzer` 分析器类

非线性分析
~~~~~~~~~~

- :func:`distance_correlation` 距离相关
- :func:`mutual_info_score` 互信息
- :func:`maximal_information_coefficient` MIC
- :class:`NonlinearAnalyzer` 分析器类

命令行工具
~~~~~~~~~~

- ``pycorrana analyze`` 完整分析
- ``pycorrana clean`` 数据清洗
- ``pycorrana partial`` 偏相关分析
- ``pycorrana nonlinear`` 非线性检测
- ``pycorrana-interactive`` 交互式工具

示例数据集
~~~~~~~~~~

- 鸢尾花数据集 (iris)
- 泰坦尼克数据集 (titanic)
- 葡萄酒数据集 (wine)
- 模拟数据生成器
