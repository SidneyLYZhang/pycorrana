"""
非线性依赖检测模块
==================
提供距离相关、互信息等非线性依赖检测方法。
"""

import warnings
from typing import Optional, List, TYPE_CHECKING
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_regression

if TYPE_CHECKING:
    from ..utils.large_data import LargeDataConfig


SAMPLE_SIZE_NONLINEAR = 20_000


def distance_correlation(x: np.ndarray, 
                        y: np.ndarray,
                        return_pvalue: bool = False,
                        n_permutations: int = 1000,
                        confidence_level: float = 0.95,
                        n_bootstrap: int = 500) -> dict:
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
    confidence_level : float, default=0.95
        置信区间水平
    n_bootstrap : int, default=500
        Bootstrap抽样次数（用于计算置信区间）
        
    Returns
    -------
    dict
        包含距离相关系数、p值和置信区间的字典
        
    Examples
    --------
    >>> x = np.random.randn(100)
    >>> y = x**2 + np.random.randn(100) * 0.1
    >>> result = distance_correlation(x, y, return_pvalue=True)
    >>> print(f"dCor: {result['dcor']:.4f}, p-value: {result['p_value']:.4f}")
    >>> print(f"95% CI: {result['confidence_interval']}")
    
    References
    ----------
    Székely, G.J., Rizzo, M.L., and Bakirov, N.K. (2007)
    "Measuring and testing dependence by correlation of distances"
    Annals of Statistics, Vol. 35 No. 6, pp. 2769-2794
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    n = len(x)
    
    if n < 3:
        return {'dcor': np.nan, 'p_value': np.nan, 'confidence_interval': (np.nan, np.nan), 'n': n}
    
    def distance_matrix(a):
        return squareform(pdist(a.reshape(-1, 1), metric='euclidean'))
    
    def center_distance_matrix(D):
        row_mean = D.mean(axis=1, keepdims=True)
        col_mean = D.mean(axis=0, keepdims=True)
        grand_mean = D.mean()
        return D - row_mean - col_mean + grand_mean
    
    def _compute_dcor(x_vals, y_vals):
        Dx = distance_matrix(x_vals)
        Dy = distance_matrix(y_vals)
        A = center_distance_matrix(Dx)
        B = center_distance_matrix(Dy)
        dcov_sq = np.mean(A * B)
        dvar_x = np.mean(A * A)
        dvar_y = np.mean(B * B)
        if dvar_x > 0 and dvar_y > 0:
            return np.sqrt(dcov_sq) / np.sqrt(np.sqrt(dvar_x) * np.sqrt(dvar_y))
        return 0.0
    
    Dx = distance_matrix(x)
    Dy = distance_matrix(y)
    A = center_distance_matrix(Dx)
    B = center_distance_matrix(Dy)
    
    dcov_sq = np.mean(A * B)
    dvar_x = np.mean(A * A)
    dvar_y = np.mean(B * B)
    
    if dvar_x > 0 and dvar_y > 0:
        dcor = np.sqrt(dcov_sq) / np.sqrt(np.sqrt(dvar_x) * np.sqrt(dvar_y))
    else:
        dcor = 0.0
    
    result = {'dcor': dcor, 'n': n}
    
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
    
    if n >= 30:
        dcor_bootstrap = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            x_boot = x[indices]
            y_boot = y[indices]
            
            try:
                dcor_boot = _compute_dcor(x_boot, y_boot)
                dcor_bootstrap.append(dcor_boot)
            except Exception:
                pass
        
        if dcor_bootstrap:
            alpha = 1 - confidence_level
            lower = np.percentile(dcor_bootstrap, alpha / 2 * 100)
            upper = np.percentile(dcor_bootstrap, (1 - alpha / 2) * 100)
            result['confidence_interval'] = (lower, upper)
        else:
            result['confidence_interval'] = (np.nan, np.nan)
    else:
        result['confidence_interval'] = (np.nan, np.nan)
    
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
                     n_neighbors: int = 3,
                     return_pvalue: bool = False,
                     n_permutations: int = 1000) -> dict:
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
    return_pvalue : bool, default=False
        是否计算p值（通过置换检验）
    n_permutations : int, default=1000
        置换检验次数
        
    Returns
    -------
    dict
        包含互信息分数、p值和置信区间的字典
        
    Examples
    --------
    >>> x = np.random.randn(100)
    >>> y = np.sin(x) + np.random.randn(100) * 0.1
    >>> result = mutual_info_score(x, y, return_pvalue=True)
    >>> print(f"MI: {result['mi']:.4f}, Normalized: {result['mi_normalized']:.4f}, p-value: {result['p_value']:.4f}")
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # 删除缺失值
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    n = len(x)
    
    if n < 3:
        return {'mi': 0.0, 'mi_normalized': 0.0, 'n': n, 'p_value': np.nan, 'confidence_interval': (np.nan, np.nan)}
    
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
    
    result = {
        'mi': mi,
        'mi_normalized': mi_normalized,
        'entropy_x': h_x,
        'entropy_y': h_y,
        'n': n
    }
    
    # 计算p值（置换检验）
    if return_pvalue:
        count = 0
        mi_obs = mi
        
        for _ in range(n_permutations):
            y_perm = np.random.permutation(y)
            try:
                mi_perm = mutual_info_regression(
                    x.reshape(-1, 1), 
                    y_perm,
                    discrete_features=discrete_features,
                    n_neighbors=n_neighbors,
                    random_state=42
                )[0]
            except Exception:
                mi_perm = 0.0
            
            if mi_perm >= mi_obs:
                count += 1
        
        p_value = (count + 1) / (n_permutations + 1)
        result['p_value'] = p_value
    
    # 计算置信区间（基于Bootstrap）
    if n >= 30:
        n_bootstrap = 1000
        mi_bootstrap = []
        mi_normalized_bootstrap = []
        
        for _ in range(n_bootstrap):
            # 有放回抽样
            indices = np.random.choice(n, size=n, replace=True)
            x_boot = x[indices]
            y_boot = y[indices]
            
            try:
                mi_boot = mutual_info_regression(
                    x_boot.reshape(-1, 1), 
                    y_boot,
                    discrete_features=discrete_features,
                    n_neighbors=n_neighbors,
                    random_state=42
                )[0]
            except Exception:
                mi_boot = 0.0
            
            mi_bootstrap.append(mi_boot)
            
            # 计算归一化互信息
            x_boot_discrete = pd.qcut(x_boot, q=min(10, len(x_boot)//5), duplicates='drop', labels=False)
            y_boot_discrete = pd.qcut(y_boot, q=min(10, len(y_boot)//5), duplicates='drop', labels=False)
            
            h_x_boot = calc_entropy(x_boot_discrete)
            h_y_boot = calc_entropy(y_boot_discrete)
            
            if h_x_boot > 0 and h_y_boot > 0:
                mi_normalized_boot = mi_boot / min(h_x_boot, h_y_boot)
            else:
                mi_normalized_boot = 0.0
            
            mi_normalized_bootstrap.append(mi_normalized_boot)
        
        # 计算95%置信区间
        ci_mi = np.percentile(mi_bootstrap, [2.5, 97.5])
        ci_mi_normalized = np.percentile(mi_normalized_bootstrap, [2.5, 97.5])
        
        result['confidence_interval'] = (ci_mi[0], ci_mi[1])
        result['confidence_interval_normalized'] = (ci_mi_normalized[0], ci_mi_normalized[1])
    else:
        result['confidence_interval'] = (np.nan, np.nan)
        result['confidence_interval_normalized'] = (np.nan, np.nan)
    
    return result


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


def _optimal_partition_1d(sorted_values: np.ndarray, k: int) -> np.ndarray:
    """
    使用动态规划找到一维数据的最优划分边界。
    
    目标是找到k-1个边界点，使得划分后的熵最小（信息量最大）。
    
    Parameters
    ----------
    sorted_values : np.ndarray
        已排序的值
    k : int
        划分区间数
        
    Returns
    -------
    np.ndarray
        最优边界位置（索引）
    """
    n = len(sorted_values)
    if k <= 1 or n < k:
        return np.array([])
    
    unique_vals, counts = np.unique(sorted_values, return_counts=True)
    n_unique = len(unique_vals)
    
    if n_unique <= k:
        boundaries = np.arange(1, n_unique)
        return boundaries
    
    cumsum = np.cumsum(counts)
    total = cumsum[-1]
    
    def calc_entropy_range(start: int, end: int) -> float:
        if start >= end:
            return 0.0
        count = cumsum[end] - (cumsum[start - 1] if start > 0 else 0)
        if count == 0:
            return 0.0
        p = count / total
        return -p * np.log(p)
    
    dp = np.full((n_unique + 1, k + 1), -np.inf)
    split = np.zeros((n_unique + 1, k + 1), dtype=int)
    
    for i in range(1, n_unique + 1):
        dp[i, 1] = calc_entropy_range(0, i - 1)
    
    for j in range(2, k + 1):
        for i in range(j, n_unique + 1):
            for t in range(j - 1, i):
                val = dp[t, j - 1] + calc_entropy_range(t, i - 1)
                if val > dp[i, j]:
                    dp[i, j] = val
                    split[i, j] = t
    
    boundaries = []
    i, j = n_unique, k
    while j > 1:
        boundaries.append(split[i, j])
        i = split[i, j]
        j -= 1
    
    boundaries = sorted(boundaries)
    return np.array(boundaries)


def _compute_mutual_info_discrete(x_discrete: np.ndarray, y_discrete: np.ndarray) -> float:
    """
    计算离散变量的互信息。
    
    Parameters
    ----------
    x_discrete, y_discrete : np.ndarray
        离散化后的变量
        
    Returns
    -------
    float
        互信息值
    """
    n = len(x_discrete)
    
    joint_counts = {}
    for i in range(n):
        key = (x_discrete[i], y_discrete[i])
        joint_counts[key] = joint_counts.get(key, 0) + 1
    
    x_counts = {}
    y_counts = {}
    for (xi, yi), count in joint_counts.items():
        x_counts[xi] = x_counts.get(xi, 0) + count
        y_counts[yi] = y_counts.get(yi, 0) + count
    
    mi = 0.0
    for (xi, yi), count in joint_counts.items():
        p_xy = count / n
        p_x = x_counts[xi] / n
        p_y = y_counts[yi] / n
        mi += p_xy * np.log(p_xy / (p_x * p_y))
    
    return mi


def _find_optimal_grid(x: np.ndarray, y: np.ndarray, i: int, j: int) -> float:
    """
    在i×j网格下找到最优划分并计算最大互信息。
    
    Parameters
    ----------
    x, y : np.ndarray
        原始数据
    i, j : int
        网格分辨率
        
    Returns
    -------
    float
        最优划分下的互信息
    """
    n = len(x)
    
    sort_idx_x = np.argsort(x)
    x_sorted = x[sort_idx_x]
    y_by_x = y[sort_idx_x]
    
    x_boundaries = _optimal_partition_1d(x_sorted, i)
    
    if len(x_boundaries) == 0:
        x_discrete = np.zeros(n, dtype=int)
    else:
        x_discrete = np.searchsorted(x_boundaries, np.arange(n))
        x_discrete = np.searchsorted(x_boundaries, np.searchsorted(x_sorted, x))
    
    sort_idx_y = np.argsort(y)
    y_sorted = y[sort_idx_y]
    x_by_y = x[sort_idx_y]
    
    y_boundaries = _optimal_partition_1d(y_sorted, j)
    
    if len(y_boundaries) == 0:
        y_discrete = np.zeros(n, dtype=int)
    else:
        y_discrete = np.searchsorted(y_boundaries, np.searchsorted(y_sorted, y))
    
    mi = _compute_mutual_info_discrete(x_discrete, y_discrete)
    
    return mi


def _compute_mic_core(x: np.ndarray, y: np.ndarray, B: Optional[float] = None) -> dict:
    """
    计算MIC的核心函数。
    
    MIC = max_{i*j < B} I*(X,Y,i,j) / log(min(i,j))
    
    Parameters
    ----------
    x, y : np.ndarray
        输入数据
    B : float, optional
        网格复杂度上限，默认为n^0.6
        
    Returns
    -------
    dict
        包含MIC及相关统计量的字典
    """
    n = len(x)
    
    if B is None:
        B = n ** 0.6
    
    B = int(B)
    B = max(B, 4)
    
    mic = 0.0
    best_i, best_j = 2, 2
    mas = 0.0
    mev = 0.0
    
    max_grid = min(B, int(np.sqrt(n)))
    max_grid = max(max_grid, 2)
    
    scores_matrix = {}
    
    for i in range(2, max_grid + 1):
        for j in range(2, max_grid + 1):
            if i * j > B:
                continue
            
            mi_optimal = _find_optimal_grid(x, y, i, j)
            
            normalization = np.log(min(i, j))
            if normalization > 0:
                score = mi_optimal / normalization
            else:
                score = 0.0
            
            scores_matrix[(i, j)] = score
            
            if score > mic:
                mic = score
                best_i, best_j = i, j
            
            if i == j:
                if score > mas:
                    mas = score
    
    if mic > 0:
        for (i, j), score in scores_matrix.items():
            if score > 0:
                ratio = score / mic
                if ratio > mev:
                    mev = ratio
    
    mcn = best_i * best_j
    
    return {
        'mic': mic,
        'mas': mas,
        'mev': mev,
        'mcn': mcn,
        'best_grid': (best_i, best_j)
    }


def maximal_information_coefficient(x: np.ndarray,
                                   y: np.ndarray,
                                   B: Optional[float] = None,
                                   return_pvalue: bool = False,
                                   n_permutations: int = 1000,
                                   confidence_level: float = 0.95,
                                   n_bootstrap: int = 500) -> dict:
    """
    计算最大信息系数（Maximal Information Coefficient, MIC）。
    
    MIC可以检测各种函数关系的强度，范围在[0, 1]。
    使用完整的MIC算法实现，包括最优网格划分和归一化。
    
    核心公式：
    MIC(X,Y) = max_{i*j < B} I*(X,Y,i,j) / log(min(i,j))
    
    其中：
    - i, j：X轴和Y轴的网格划分数量
    - B = n^0.6：网格复杂度上限
    - I*：在i×j网格下的最大互信息
    - 分母log(min(i,j))：归一化因子
    
    .. warning::
       当前版本为纯 Python 实现，没有特殊优化，计算速度较慢。
       对于大数据集（n > 1000），建议先进行采样处理。
    
    Parameters
    ----------
    x, y : np.ndarray
        两个变量的观测值
    B : float, optional
        网格复杂度上限，默认为n^0.6
    return_pvalue : bool, default=False
        是否计算p值（通过置换检验）
    n_permutations : int, default=1000
        置换检验次数
    confidence_level : float, default=0.95
        置信区间水平
    n_bootstrap : int, default=500
        Bootstrap抽样次数
        
    Returns
    -------
    dict
        包含以下键的字典：
        - mic: MIC值，范围[0, 1]
        - mas: 最大对称得分（Maximum Asymmetry Score）
        - mev: 最大边缘值（Maximum Edge Value）
        - mcn: 最大网格数（Maximum Cell Number）
        - n: 样本量
        - p_value: p值（如果return_pvalue=True）
        - confidence_interval: 置信区间（如果样本量足够）
        
    Examples
    --------
    >>> x = np.random.randn(100)
    >>> y = x**2 + np.random.randn(100) * 0.1
    >>> result = maximal_information_coefficient(x, y, return_pvalue=True)
    >>> print(f"MIC: {result['mic']:.4f}, p-value: {result['p_value']:.4f}")
    
    对于大数据集，建议先采样：
    
    >>> from pycorrana.utils import smart_sample
    >>> sampled_df = smart_sample(df, sample_size=500)
    >>> result = maximal_information_coefficient(sampled_df['x'], sampled_df['y'])
    
    References
    ----------
    Reshef et al. (2011) "Detecting Novel Associations in Large Data Sets"
    Science, 334(6062), 1518-1524
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    
    n = len(x)
    
    if n < 10:
        base_result = {
            'mic': np.nan,
            'mas': np.nan,
            'mev': np.nan,
            'mcn': np.nan,
            'n': n,
            'confidence_interval': (np.nan, np.nan)
        }
        if return_pvalue:
            base_result['p_value'] = np.nan
        return base_result
    
    result = _compute_mic_core(x, y, B)
    result['n'] = n
    
    if return_pvalue:
        count = 0
        mic_obs = result['mic']
        
        for _ in range(n_permutations):
            y_perm = np.random.permutation(y)
            perm_result = _compute_mic_core(x, y_perm, B)
            if perm_result['mic'] >= mic_obs:
                count += 1
        
        p_value = (count + 1) / (n_permutations + 1)
        result['p_value'] = p_value
    
    if n >= 30:
        mic_bootstrap = []
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n, size=n, replace=True)
            x_boot = x[indices]
            y_boot = y[indices]
            
            boot_result = _compute_mic_core(x_boot, y_boot, B)
            mic_bootstrap.append(boot_result['mic'])
        
        alpha = 1 - confidence_level
        lower = np.percentile(mic_bootstrap, alpha / 2 * 100)
        upper = np.percentile(mic_bootstrap, (1 - alpha / 2) * 100)
        
        result['confidence_interval'] = (lower, upper)
    else:
        result['confidence_interval'] = (np.nan, np.nan)
    
    return result


def nonlinear_dependency_report(data: pd.DataFrame,
                               columns: Optional[List[str]] = None,
                               methods: List[str] = ['dcor', 'mi', 'mic'],
                               top_n: int = 10,
                               sample_size: int = SAMPLE_SIZE_NONLINEAR,
                               random_state: int = 42,
                               verbose: bool = True) -> pd.DataFrame:
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
    sample_size : int, default=20000
        大数据集采样大小，设为None或0禁用采样
    random_state : int, default=42
        随机种子
    verbose : bool, default=True
        是否输出信息
        
    Returns
    -------
    pd.DataFrame
        检测报告
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    n_rows = len(data)
    if sample_size and sample_size > 0 and n_rows > sample_size:
        data = data.sample(n=sample_size, random_state=random_state)
        if verbose:
            print(f"大数据集检测: 采样 {n_rows} -> {sample_size} 行")
    
    results = []
    
    for i, col1 in enumerate(columns):
        for col2 in columns[i+1:]:
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
    
    score_cols = [c for c in df.columns if c not in ['var1', 'var2']]
    if score_cols:
        df['avg_score'] = df[score_cols].mean(axis=1)
        df = df.sort_values('avg_score', ascending=False)
    
    return df.head(top_n)


class NonlinearAnalyzer:
    """
    非线性依赖分析器类
    
    提供非线性依赖检测的完整流程。
    
    Parameters
    ----------
    data : pd.DataFrame
        输入数据
    verbose : bool, default=True
        是否输出详细信息
    sample_size : int, default=20000
        大数据集采样大小，设为None或0禁用采样
        
    Examples
    --------
    >>> analyzer = NonlinearAnalyzer(df)
    >>> result = analyzer.analyze_pair('X', 'Y')
    >>> report = analyzer.generate_report()
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 verbose: bool = True,
                 sample_size: int = SAMPLE_SIZE_NONLINEAR):
        """
        Parameters
        ----------
        data : pd.DataFrame
            输入数据
        verbose : bool, default=True
            是否输出详细信息
        sample_size : int, default=20000
            大数据集采样大小
        """
        self.original_data = data.copy()
        self.verbose = verbose
        self.sample_size = sample_size
        self._sampled = False
        
        n_rows = len(data)
        if sample_size and sample_size > 0 and n_rows > sample_size:
            self.data = data.sample(n=sample_size, random_state=42)
            self._sampled = True
            if verbose:
                print(f"大数据集检测: 采样 {n_rows} -> {sample_size} 行")
        else:
            self.data = data.copy()
        
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
            self.data, columns=columns, top_n=top_n,
            sample_size=0, verbose=False
        )
        
        if self.verbose:
            print("\n非线性依赖检测报告（Top {}）:".format(top_n))
            print(report.to_string(index=False))
        
        return report
