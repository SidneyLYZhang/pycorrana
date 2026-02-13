"""核心分析模块"""
from .analyzer import quick_corr, CorrAnalyzer
from .partial_corr import (
    partial_corr, 
    partial_corr_matrix, 
    semipartial_corr,
    PartialCorrAnalyzer
)
from .nonlinear import (
    distance_correlation, 
    mutual_info_score,
    maximal_information_coefficient,
    nonlinear_dependency_report,
    NonlinearAnalyzer
)

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
]
