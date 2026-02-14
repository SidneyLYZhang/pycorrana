"""
统计工具函数
===========
提供统计检验、p值校正等辅助功能。
"""

import warnings
from typing import List
import numpy as np
import pandas as pd
from scipy import stats


def check_normality(series: pd.Series, 
                   method: str = 'shapiro',
                   alpha: float = 0.05) -> bool:
    """
    正态性检验。
    
    Parameters
    ----------
    series : pd.Series
        输入数据
    method : str, default='shapiro'
        检验方法：'shapiro'（Shapiro-Wilk）或 'kstest'（Kolmogorov-Smirnov）
    alpha : float, default=0.05
        显著性水平
        
    Returns
    -------
    bool
        是否服从正态分布
    """
    data = series.dropna()
    
    if len(data) < 3:
        return False
    
    if method == 'shapiro':
        # Shapiro-Wilk检验，适用于小样本
        if len(data) > 5000:
            warnings.warn("Shapiro-Wilk检验不适用于大样本（>5000），使用K-S检验代替")
            stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        else:
            stat, p_value = stats.shapiro(data)
    
    elif method == 'kstest':
        stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    
    else:
        raise ValueError(f"未知的检验方法: {method}")
    
    return p_value > alpha


def correct_pvalues(p_values: List[float], 
                   method: str = 'fdr_bh') -> np.ndarray:
    """
    多重比较校正。
    
    Parameters
    ----------
    p_values : list
        p值列表
    method : str, default='fdr_bh'
        校正方法：
        - 'bonferroni': Bonferroni校正
        - 'fdr_bh': Benjamini-Hochberg FDR
        - 'fdr_by': Benjamini-Yekutieli FDR
        - 'holm': Holm逐步校正
        
    Returns
    -------
    np.ndarray
        校正后的p值
    """
    from statsmodels.stats.multitest import multipletests
    
    p_values = np.array(p_values)
    p_values = p_values[~np.isnan(p_values)]
    
    if len(p_values) == 0:
        return np.array([])
    
    method_map = {
        'bonferroni': 'bonferroni',
        'fdr_bh': 'fdr_bh',
        'fdr_by': 'fdr_by',
        'holm': 'holm',
        'sidak': 'sidak',
    }
    
    if method not in method_map:
        warnings.warn(f"未知的校正方法: {method}，使用fdr_bh")
        method = 'fdr_bh'
    
    reject, pvals_corrected, _, _ = multipletests(
        p_values, alpha=0.05, method=method_map[method]
    )
    
    return pvals_corrected


def cramers_v(x: pd.Series, y: pd.Series) -> tuple:
    """
    计算Cramér's V系数（分类变量间的相关性）。
    
    Parameters
    ----------
    x, y : pd.Series
        两个分类变量
        
    Returns
    -------
    tuple
        (Cramér's V值, p值)
    """
    # 创建列联表
    contingency_table = pd.crosstab(x, y)
    
    # 卡方检验
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    # 计算Cramér's V
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    
    if min(kcorr-1, rcorr-1) == 0:
        return 0.0, p_value
    
    cramers_v = np.sqrt(phi2corr / min(kcorr-1, rcorr-1))
    
    return cramers_v, p_value


def eta_coefficient(x: pd.Series, y: pd.Series) -> tuple:
    """
    计算Eta系数（数值变量与分类变量的相关比）。
    
    Parameters
    ----------
    x : pd.Series
        数值变量
    y : pd.Series
        分类变量
        
    Returns
    -------
    tuple
        (Eta值, p值)
    """
    # 确保x是数值，y是分类
    if not pd.api.types.is_numeric_dtype(x):
        x, y = y, x
    
    # 分组
    groups = [group.dropna().values for name, group in x.groupby(y)]
    
    if len(groups) < 2:
        return 0.0, 1.0
    
    # 单因素方差分析
    f_stat, p_value = stats.f_oneway(*groups)
    
    # 计算Eta平方
    ss_between = sum(len(g) * (np.mean(g) - np.mean(x))**2 for g in groups)
    ss_total = sum((x - np.mean(x))**2)
    
    if ss_total == 0:
        return 0.0, p_value
    
    eta_squared = ss_between / ss_total
    eta = np.sqrt(eta_squared)
    
    return eta, p_value


def point_biserial(x: pd.Series, y: pd.Series) -> tuple:
    """
    计算点双列相关系数（数值变量与二分类变量）。
    
    Parameters
    ----------
    x : pd.Series
        数值变量
    y : pd.Series
        二分类变量
        
    Returns
    -------
    tuple
        (相关系数, p值)
    """
    # 确保y是二分类
    if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_numeric_dtype(y):
        # 将二分类映射为0/1
        unique_vals = y.dropna().unique()
        y_numeric = y.map({unique_vals[0]: 0, unique_vals[1]: 1})
    elif not pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
        unique_vals = x.dropna().unique()
        x_numeric = x.map({unique_vals[0]: 0, unique_vals[1]: 1})
        y_numeric = y
        x, y = y_numeric, x_numeric
        y_numeric = y
    else:
        y_numeric = y
    
    # 使用scipy的pointbiserialr
    r, p_value = stats.pointbiserialr(y_numeric, x)
    
    return r, p_value


def interpret_correlation(r: float, method: str = 'general') -> str:
    """
    解释相关系数的强度。
    
    Parameters
    ----------
    r : float
        相关系数值
    method : str, default='general'
        解释标准：'general'（通用）或 'cohen'（Cohen标准）
        
    Returns
    -------
    str
        相关性强度描述
    """
    r_abs = abs(r)
    
    if method == 'cohen':
        if r_abs < 0.1:
            return "可忽略"
        elif r_abs < 0.3:
            return "小"
        elif r_abs < 0.5:
            return "中等"
        elif r_abs < 0.7:
            return "大"
        else:
            return "非常大"
    else:
        if r_abs < 0.1:
            return "极弱相关"
        elif r_abs < 0.3:
            return "弱相关"
        elif r_abs < 0.5:
            return "中等相关"
        elif r_abs < 0.7:
            return "强相关"
        elif r_abs < 0.9:
            return "很强相关"
        else:
            return "极强相关"
