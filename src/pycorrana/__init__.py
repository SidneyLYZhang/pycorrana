"""
PyCorrAna - Python Correlation Analysis Toolkit
===============================================

一个方便快速入手的Python相关性分析工具，核心在于：
自动化常规操作、降低决策成本、一键输出关键结果。

主要功能
--------
- 数据接入与基础清洗：自动识别格式、缺失值处理、类型推断
- 相关性计算引擎：自动选择最优相关系数方法
- 可视化核心：热力图、散点图矩阵、箱线图
- 结果导出：CSV/Excel、文本摘要、控制台友好输出

快速开始
--------
>>> from pycorrana import quick_corr
>>> import pandas as pd
>>> df = pd.read_csv('data.csv')
>>> result = quick_corr(df, target='sales')

或者使用交互式CLI：
$ pycorrana-interactive
"""

__version__ = "0.1.5"
__author__ = "SidneyZhang<zly@lyzhang.me>"

from .core.analyzer import quick_corr, CorrAnalyzer
from .core.partial_corr import (
    partial_corr, 
    partial_corr_matrix,
    semipartial_corr,
    PartialCorrAnalyzer
)
from .core.nonlinear import (
    distance_correlation, 
    mutual_info_score,
    maximal_information_coefficient,
    nonlinear_dependency_report,
    NonlinearAnalyzer
)
from .datasets import load_iris, load_titanic, load_wine, make_correlated_data, list_datasets

__all__ = [
    'quick_corr',
    'CorrAnalyzer',
    'partial_corr',
    'partial_corr_matrix',
    'semipartial_corr',
    'PartialCorrAnalyzer',
    'distance_correlation',
    'mutual_info_score',
    'maximal_information_coefficient',
    'nonlinear_dependency_report',
    'NonlinearAnalyzer',
    'load_iris',
    'load_titanic',
    'load_wine',
    'make_correlated_data',
    'list_datasets',
]
