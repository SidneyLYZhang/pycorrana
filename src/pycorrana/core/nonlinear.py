"""
非线性依赖检测模块
==================
提供距离相关、互信息等非线性依赖检测方法。
"""

import warnings
from typing import Optional, List
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from scipy.special import digamma
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


def distance_correlation(x: np.ndarray, 
                        y: np.ndarray,
                        return_pvalue: bool = False,
                        n_permutations: int = 1000) -> dict:
    """
    计算距离相关系数（Distance Correlation）。
    
    距离相关可以检测变量间的任意类型依赖关系（包括非线性），
    当且仅当变量独立时距离相关为0。
    
    Parameters
    ----------
    x, y : np.ndarray
        两个变量的观测值
    return_pvalue : bool, default=False
        是否计算p值（通过置换检验）
    n_permutations : int, default=1000
        置换检验次数
        
    Returns
    -------
    dict
        包含距离相关系数和p值的字典
        
    Examples
    --------
    >>> x = np.random.randn(100)
    >>> y = x**2 + np.random.randn(100) * 0.1
    >>> result = distance_correlation(x, y, return_pvalue=True)
    >>> print(f"dCor: {result['dcor']:.4f}, p-value: {result['p_value']:.4f}")
    
    References
    ----------
    Székely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007)
    "Measuring and testing dependence by correlation of distances"
    Annals of Statistics, Vol. 35 No. 6, pp. 2769-2794
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # 删除缺失值
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    n = len(x)
    
    if n < 3:
        return {'dcor': np.nan, 'p_value': np.nan}
    
    # 计算距离矩阵
    def distance_matrix(a):
        """计算欧氏距离矩阵"""
        return squareform(pdist(a.reshape(-1, 1), metric='euclidean'))
    
    # 中心化距离矩阵
    def center_distance_matrix(D):
        """双中心化距离矩阵"""
        row_mean = D.mean(axis=1, keepdims=True)
        col_mean = D.mean(axis=0, keepdims=True)
        grand_mean = D.mean()
        return D - row_mean - col_mean + grand_mean
    
    # 计算距离矩阵
    Dx = distance_matrix(x)
    Dy = distance_matrix(y)
    
    # 双中心化
    A = center_distance_matrix(Dx)
    B = center_distance_matrix(Dy)
    
    # 计算距离协方差和方差
    dcov_sq = np.mean(A * B)
    dvar_x = np.mean(A * A)
    dvar_y = np.mean(B * B)
    
    # 距离相关系数
    if dvar_x > 0 and dvar_y > 0:
        dcor = np.sqrt(dcov_sq) / np.sqrt(np.sqrt(dvar_x) * np.sqrt(dvar_y))
    else:
        dcor = 0.0
    
    result = {'dcor': dcor, 'n': n}
    
    # 置换检验
    if return_pvalue:
        count = 0
        dcor_obs = dcor
        
        for _ in range(n_permutations):
            y_perm = np.random.permutation(y)
            Dy_perm = distance_matrix(y_perm)
            B_perm = center_distance_matrix(Dy_perm)
            
            dcov_sq_perm = np.mean(A * B_perm)
            dvar_y_perm = np.mean(B_perm * B_perm)
            
            if dvar_x > 0 and dvar_y_perm > 0:
                dcor_perm = np.sqrt(dcov_sq_perm) / np.sqrt(np.sqrt(dvar_x) * np.sqrt(dvar_y_perm))
            else:
                dcor_perm = 0.0
            
            if dcor_perm >= dcor_obs:
                count += 1
        
        p_value = (count + 1) / (n_permutations + 1)
        result['p_value'] = p_value
    
    return result


def distance_correlation_matrix(data: pd.DataFrame,
                                columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    计算距离相关矩阵。
    
    Parameters
    ----------
    data : pd.DataFrame
        输入数据
    columns : list, optional
        要分析的列
        
    Returns
    -------
    pd.DataFrame
        距离相关系数矩阵
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = len(columns)
    dcor_matrix = pd.DataFrame(np.eye(n_cols), index=columns, columns=columns)
    
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns[i+1:], i+1):
            result = distance_correlation(
                data[col1].values, 
                data[col2].values
            )
            dcor = result['dcor']
            dcor_matrix.iloc[i, j] = dcor
            dcor_matrix.iloc[j, i] = dcor
    
    return dcor_matrix


def mutual_info_score(x: np.ndarray,
                     y: np.ndarray,
                     discrete_features: bool = False,
                     n_neighbors: int = 3) -> dict:
    """
    计算互信息分数。
    
    互信息衡量两个变量共享的信息量，可以检测非线性依赖关系。
    
    Parameters
    ----------
    x, y : np.ndarray
        两个变量的观测值
    discrete_features : bool, default=False
        x是否为离散变量
    n_neighbors : int, default=3
        KNN估计的邻居数
        
    Returns
    -------
    dict
        包含互信息分数的字典
        
    Examples
    --------
    >>> x = np.random.randn(100)
    >>> y = np.sin(x) + np.random.randn(100) * 0.1
    >>> result = mutual_info_score(x, y)
    >>> print(f"MI: {result['mi']:.4f}, Normalized: {result['mi_normalized']:.4f}")
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # 删除缺失值
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    n = len(x)
    
    if n < 3:
        return {'mi': 0.0, 'mi_normalized': 0.0, 'n': n}
    
    # 使用sklearn的互信息计算
    try:
        mi = mutual_info_regression(
            x.reshape(-1, 1), 
            y,
            discrete_features=discrete_features,
            n_neighbors=n_neighbors,
            random_state=42
        )[0]
    except Exception as e:
        warnings.warn(f"互信息计算失败: {e}")
        mi = 0.0
    
    # 归一化互信息（0-1范围）
    # 使用最小熵进行归一化
    from sklearn.metrics import mutual_info_score as sklearn_mi
    
    # 离散化用于计算熵
    x_discrete = pd.qcut(x, q=min(10, n//5), duplicates='drop', labels=False)
    y_discrete = pd.qcut(y, q=min(10, n//5), duplicates='drop', labels=False)
    
    # 计算熵
    def calc_entropy(a):
        probs = np.bincount(a) / len(a)
        return entropy(probs[probs > 0])
    
    h_x = calc_entropy(x_discrete)
    h_y = calc_entropy(y_discrete)
    
    # 归一化
    if h_x > 0 and h_y > 0:
        mi_normalized = mi / min(h_x, h_y)
    else:
        mi_normalized = 0.0
    
    return {
        'mi': mi,
        'mi_normalized': mi_normalized,
        'entropy_x': h_x,
        'entropy_y': h_y,
        'n': n
    }


def mutual_info_matrix(data: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      normalized: bool = True) -> pd.DataFrame:
    """
    计算互信息矩阵。
    
    Parameters
    ----------
    data : pd.DataFrame
        输入数据
    columns : list, optional
        要分析的列
    normalized : bool, default=True
        是否返回归一化互信息
        
    Returns
    -------
    pd.DataFrame
        互信息矩阵
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = len(columns)
    mi_matrix = pd.DataFrame(np.zeros((n_cols, n_cols)), index=columns, columns=columns)
    
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns[i:], i):
            result = mutual_info_score(
                data[col1].values,
                data[col2].values
            )
            
            if normalized:
                mi = result['mi_normalized']
            else:
                mi = result['mi']
            
            mi_matrix.iloc[i, j] = mi
            mi_matrix.iloc[j, i] = mi
    
    return mi_matrix


def maximal_information_coefficient(x: np.ndarray,
                                   y: np.ndarray,
                                   bins: int = 10) -> dict:
    """
    计算最大信息系数（Maximal Information Coefficient, MIC）。
    
    MIC可以检测各种函数关系的强度，范围在[0, 1]。
    
    Parameters
    ----------
    x, y : np.ndarray
        两个变量的观测值
    bins : int, default=10
        最大分箱数
        
    Returns
    -------
    dict
        包含MIC和MAS等统计量的字典
        
    References
    ----------
    Reshef et al. (2011) "Detecting Novel Associations in Large Data Sets"
    Science, 334(6062), 1518-1524
    """
    try:
        from minepy import MINE
        
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        
        # 删除缺失值
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        
        mine = MINE(alpha=0.6, c=15)
        mine.compute_score(x, y)
        
        return {
            'mic': mine.mic(),
            'mas': mine.mas(),
            'mev': mine.mev(),
            'mcn': mine.mcn(),
            'n': len(x)
        }
    except ImportError:
        warnings.warn("minepy未安装，无法计算MIC。使用互信息代替。")
        result = mutual_info_score(x, y)
        return {
            'mic': result['mi_normalized'],
            'mas': np.nan,
            'mev': np.nan,
            'mcn': np.nan,
            'n': result['n']
        }


def nonlinear_dependency_report(data: pd.DataFrame,
                               columns: Optional[List[str]] = None,
                               methods: List[str] = ['dcor', 'mi', 'mic'],
                               top_n: int = 10) -> pd.DataFrame:
    """
    生成非线性依赖检测报告。
    
    Parameters
    ----------
    data : pd.DataFrame
        输入数据
    columns : list, optional
        要分析的列
    methods : list, default=['dcor', 'mi', 'mic']
        要使用的检测方法
    top_n : int, default=10
        显示前N个结果
        
    Returns
    -------
    pd.DataFrame
        检测报告
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    results = []
    
    for i, col1 in enumerate(columns):
        for col2 in enumerate(columns[i+1:], i+1):
            x = data[col1].values
            y = data[col2].values
            
            row = {'var1': col1, 'var2': col2}
            
            if 'dcor' in methods:
                dcor_result = distance_correlation(x, y)
                row['dCor'] = dcor_result['dcor']
            
            if 'mi' in methods:
                mi_result = mutual_info_score(x, y)
                row['MI'] = mi_result['mi_normalized']
            
            if 'mic' in methods:
                mic_result = maximal_information_coefficient(x, y)
                row['MIC'] = mic_result['mic']
            
            results.append(row)
    
    df = pd.DataFrame(results)
    
    # 计算综合得分（各方法平均）
    score_cols = [c for c in df.columns if c not in ['var1', 'var2']]
    if score_cols:
        df['avg_score'] = df[score_cols].mean(axis=1)
        df = df.sort_values('avg_score', ascending=False)
    
    return df.head(top_n)


class NonlinearAnalyzer:
    """
    非线性依赖分析器类
    
    提供非线性依赖检测的完整流程。
    
    Examples
    --------
    >>> analyzer = NonlinearAnalyzer(df)
    >>> result = analyzer.analyze_pair('X', 'Y')
    >>> report = analyzer.generate_report()
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
    
    def analyze_pair(self,
                    x: str,
                    y: str,
                    methods: List[str] = ['dcor', 'mi', 'mic']) -> dict:
        """
        分析一对变量的非线性依赖。
        
        Parameters
        ----------
        x, y : str
            变量名
        methods : list
            要使用的检测方法
            
        Returns
        -------
        dict
            分析结果
        """
        x_vals = self.data[x].values
        y_vals = self.data[y].values
        
        result = {'var1': x, 'var2': y}
        
        if 'dcor' in methods:
            dcor_result = distance_correlation(x_vals, y_vals)
            result['dCor'] = dcor_result['dcor']
        
        if 'mi' in methods:
            mi_result = mutual_info_score(x_vals, y_vals)
            result['MI'] = mi_result['mi_normalized']
        
        if 'mic' in methods:
            mic_result = maximal_information_coefficient(x_vals, y_vals)
            result['MIC'] = mic_result['mic']
        
        self.results.append(result)
        
        if self.verbose:
            print(f"\n非线性依赖分析: {x} vs {y}")
            for key, val in result.items():
                if key not in ['var1', 'var2']:
                    print(f"  {key}: {val:.4f}")
        
        return result
    
    def generate_report(self,
                       columns: Optional[List[str]] = None,
                       top_n: int = 10) -> pd.DataFrame:
        """
        生成非线性依赖检测报告。
        
        Parameters
        ----------
        columns : list, optional
            要分析的列
        top_n : int, default=10
            显示前N个结果
            
        Returns
        -------
        pd.DataFrame
            检测报告
        """
        report = nonlinear_dependency_report(
            self.data, columns=columns, top_n=top_n
        )
        
        if self.verbose:
            print("\n非线性依赖检测报告（Top {}）:".format(top_n))
            print(report.to_string(index=False))
        
        return report
