"""
偏相关分析模块
=============
提供控制协变量后的净相关分析功能。
"""

import warnings
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import inv


def partial_corr(data: pd.DataFrame,
                x: str,
                y: str,
                covars: Union[str, List[str]],
                method: str = 'pearson') -> dict:
    """
    计算两个变量在控制协变量后的偏相关系数。
    
    偏相关用于衡量两个变量在排除其他变量影响后的净相关关系。
    
    Parameters
    ----------
    data : pd.DataFrame
        输入数据
    x : str
        第一个变量名
    y : str
        第二个变量名
    covars : str or list
        需要控制的协变量（一个或多个）
    method : str, default='pearson'
        相关方法：'pearson'、'spearman'
        
    Returns
    -------
    dict
        包含偏相关系数、p值、置信区间等信息的字典
        
    Examples
    --------
    >>> # 控制年龄后，计算身高与体重的偏相关
    >>> result = partial_corr(df, x='height', y='weight', covars='age')
    
    >>> # 控制多个变量
    >>> result = partial_corr(df, x='X1', y='X2', covars=['Z1', 'Z2', 'Z3'])
    """
    # 确保covars是列表
    if isinstance(covars, str):
        covars = [covars]
    
    # 检查变量是否存在
    all_vars = [x, y] + covars
    missing_vars = [v for v in all_vars if v not in data.columns]
    if missing_vars:
        raise ValueError(f"变量不存在: {missing_vars}")
    
    # 提取数据并删除缺失值
    sub_data = data[all_vars].dropna()
    
    if len(sub_data) < len(covars) + 3:
        raise ValueError(f"样本量不足（需要至少{len(covars) + 3}个观测）")
    
    n = len(sub_data)
    k = len(covars)
    
    # 使用Spearman时转换为秩
    if method == 'spearman':
        sub_data = sub_data.rank()
    
    # 计算偏相关系数（使用相关矩阵求逆法）
    corr_matrix = sub_data.corr().values
    
    try:
        # 求逆
        inv_corr = inv(corr_matrix)
        
        # 偏相关系数公式
        # r_xy.Z = -C_xy / sqrt(C_xx * C_yy)
        # 其中C是逆相关矩阵
        idx_x = 0
        idx_y = 1
        
        r_partial = -inv_corr[idx_x, idx_y] / np.sqrt(
            inv_corr[idx_x, idx_x] * inv_corr[idx_y, idx_y]
        )
        
        # 限制在[-1, 1]范围内
        r_partial = np.clip(r_partial, -1, 1)
        
    except np.linalg.LinAlgError:
        # 如果矩阵奇异，使用回归残差法
        from sklearn.linear_model import LinearRegression
        
        # X对协变量回归
        X_covars = sub_data[covars].values
        y_x = sub_data[x].values
        y_y = sub_data[y].values
        
        reg_x = LinearRegression().fit(X_covars, y_x)
        reg_y = LinearRegression().fit(X_covars, y_y)
        
        # 残差
        resid_x = y_x - reg_x.predict(X_covars)
        resid_y = y_y - reg_y.predict(X_covars)
        
        # 残差相关
        r_partial, _ = stats.pearsonr(resid_x, resid_y)
    
    # 计算p值
    # t统计量: t = r * sqrt((n-k-2)/(1-r^2))
    if abs(r_partial) < 1:
        t_stat = r_partial * np.sqrt((n - k - 2) / (1 - r_partial**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - k - 2))
    else:
        t_stat = np.inf
        p_value = 0.0
    
    # 计算置信区间（Fisher变换）
    z = np.arctanh(r_partial)
    se_z = 1 / np.sqrt(n - k - 3)
    z_crit = stats.norm.ppf(0.975)
    
    ci_lower = np.tanh(z - z_crit * se_z)
    ci_upper = np.tanh(z + z_crit * se_z)
    
    return {
        'x': x,
        'y': y,
        'covariates': covars,
        'partial_correlation': r_partial,
        'p_value': p_value,
        't_statistic': t_stat,
        'df': n - k - 2,
        'n': n,
        'ci_95': (ci_lower, ci_upper),
        'method': method
    }


def partial_corr_matrix(data: pd.DataFrame,
                       covars: Union[str, List[str]],
                       columns: Optional[List[str]] = None,
                       method: str = 'pearson') -> pd.DataFrame:
    """
    计算偏相关矩阵（控制协变量后）。
    
    Parameters
    ----------
    data : pd.DataFrame
        输入数据
    covars : str or list
        需要控制的协变量
    columns : list, optional
        要分析的列，默认为所有数值列
    method : str, default='pearson'
        相关方法
        
    Returns
    -------
    pd.DataFrame
        偏相关系数矩阵
        
    Examples
    --------
    >>> # 控制年龄后，计算所有变量的偏相关矩阵
    >>> pcorr_matrix = partial_corr_matrix(df, covars='age')
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # 确保协变量不在分析列中
    if isinstance(covars, str):
        covars = [covars]
    
    columns = [c for c in columns if c not in covars]
    
    n_cols = len(columns)
    pcorr_matrix = pd.DataFrame(np.eye(n_cols), index=columns, columns=columns)
    
    # 计算所有变量对的偏相关
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns[i+1:], i+1):
            try:
                result = partial_corr(data, col1, col2, covars, method)
                pcorr = result['partial_correlation']
                pcorr_matrix.iloc[i, j] = pcorr
                pcorr_matrix.iloc[j, i] = pcorr
            except Exception as e:
                warnings.warn(f"计算 {col1}-{col2} 偏相关失败: {e}")
                pcorr_matrix.iloc[i, j] = np.nan
                pcorr_matrix.iloc[j, i] = np.nan
    
    return pcorr_matrix


def semipartial_corr(data: pd.DataFrame,
                    x: str,
                    y: str,
                    covars: Union[str, List[str]],
                    method: str = 'pearson') -> dict:
    """
    计算半偏相关系数。
    
    半偏相关衡量X与Y（控制Z后的Y）之间的相关关系。
    与偏相关不同，半偏相关只控制一个变量的协变量影响。
    
    Parameters
    ----------
    data : pd.DataFrame
        输入数据
    x : str
        第一个变量名（不控制协变量）
    y : str
        第二个变量名（控制协变量）
    covars : str or list
        需要控制的协变量（应用于y）
    method : str, default='pearson'
        相关方法
        
    Returns
    -------
    dict
        包含半偏相关系数等信息的字典
    """
    # 确保covars是列表
    if isinstance(covars, str):
        covars = [covars]
    
    # 提取数据
    all_vars = [x, y] + covars
    sub_data = data[all_vars].dropna()
    
    n = len(sub_data)
    k = len(covars)
    
    # 使用Spearman时转换为秩
    if method == 'spearman':
        sub_data = sub_data.rank()
    
    # Y对协变量回归，获取残差
    from sklearn.linear_model import LinearRegression
    
    X_covars = sub_data[covars].values
    y_vals = sub_data[y].values
    x_vals = sub_data[x].values
    
    reg = LinearRegression().fit(X_covars, y_vals)
    resid_y = y_vals - reg.predict(X_covars)
    
    # X与Y残差的相关
    r_semipartial, p_value = stats.pearsonr(x_vals, resid_y)
    
    # 计算置信区间
    z = np.arctanh(r_semipartial)
    se_z = 1 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(0.975)
    
    ci_lower = np.tanh(z - z_crit * se_z)
    ci_upper = np.tanh(z + z_crit * se_z)
    
    return {
        'x': x,
        'y': y,
        'covariates': covars,
        'semipartial_correlation': r_semipartial,
        'p_value': p_value,
        'n': n,
        'ci_95': (ci_lower, ci_upper),
        'method': method
    }


class PartialCorrAnalyzer:
    """
    偏相关分析器类
    
    提供偏相关分析的完整流程。
    
    Examples
    --------
    >>> analyzer = PartialCorrAnalyzer(df)
    >>> result = analyzer.fit(x='income', y='happiness', covars=['age', 'education'])
    >>> matrix = analyzer.fit_matrix(covars='age')
    """
    
    def __init__(self, data: pd.DataFrame, verbose: bool = True):
        """
        Parameters
        ----------
        data : pd.DataFrame
            输入数据
        verbose : bool, default=True
            是否输出详细信息
        """
        self.data = data.copy()
        self.verbose = verbose
        self.results = []
    
    def fit(self,
           x: str,
           y: str,
           covars: Union[str, List[str]],
           method: str = 'pearson') -> dict:
        """
        执行偏相关分析。
        
        Parameters
        ----------
        x : str
            第一个变量
        y : str
            第二个变量
        covars : str or list
            协变量
        method : str, default='pearson'
            相关方法
            
        Returns
        -------
        dict
            分析结果
        """
        result = partial_corr(self.data, x, y, covars, method)
        self.results.append(result)
        
        if self.verbose:
            print(f"\n偏相关分析: {x} vs {y}")
            print(f"  控制变量: {covars}")
            print(f"  偏相关系数: {result['partial_correlation']:.4f}")
            print(f"  p值: {result['p_value']:.4e}")
            print(f"  95% CI: [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
        
        return result
    
    def fit_matrix(self,
                  covars: Union[str, List[str]],
                  columns: Optional[List[str]] = None,
                  method: str = 'pearson') -> pd.DataFrame:
        """
        计算偏相关矩阵。
        
        Parameters
        ----------
        covars : str or list
            协变量
        columns : list, optional
            要分析的列
        method : str, default='pearson'
            相关方法
            
        Returns
        -------
        pd.DataFrame
            偏相关系数矩阵
        """
        matrix = partial_corr_matrix(self.data, covars, columns, method)
        
        if self.verbose:
            print(f"\n偏相关矩阵（控制: {covars}）")
            print(matrix.round(3))
        
        return matrix
    
    def compare(self,
               x: str,
               y: str,
               covars_list: List[Union[str, List[str]]],
               method: str = 'pearson') -> pd.DataFrame:
        """
        比较不同协变量控制下的偏相关。
        
        Parameters
        ----------
        x : str
            第一个变量
        y : str
            第二个变量
        covars_list : list
            协变量组合列表
        method : str, default='pearson'
            相关方法
            
        Returns
        -------
        pd.DataFrame
            比较结果
        """
        results = []
        
        # 先计算简单相关
        simple_corr = self.data[[x, y]].corr().iloc[0, 1]
        results.append({
            '控制变量': '无（简单相关）',
            '相关系数': simple_corr,
            '变化': 0.0
        })
        
        for covars in covars_list:
            result = partial_corr(self.data, x, y, covars, method)
            results.append({
                '控制变量': str(covars),
                '相关系数': result['partial_correlation'],
                '变化': result['partial_correlation'] - simple_corr
            })
        
        df = pd.DataFrame(results)
        
        if self.verbose:
            print(f"\n{x} vs {y} 的偏相关比较:")
            print(df.to_string(index=False))
        
        return df
