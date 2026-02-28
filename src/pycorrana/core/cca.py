"""
典型相关分析模块 (Canonical Correlation Analysis, CCA)
=====================================================
提供两组变量之间的典型相关分析功能。

典型相关分析是一种多元统计方法，用于研究两组变量之间的线性关系。
它寻找两组变量的线性组合，使得这些组合之间的相关性最大化。
"""

import warnings
from typing import Optional, List, Union, Tuple, Dict, TYPE_CHECKING
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import inv, eig, sqrtm

if TYPE_CHECKING:
    from ..utils.large_data import LargeDataConfig


def _validate_cca_inputs(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    验证并准备CCA输入数据。
    
    Parameters
    ----------
    X, Y : np.ndarray
        输入数据矩阵
        
    Returns
    -------
    tuple
        (处理后的X, 处理后的Y, 样本量)
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X和Y的样本量不一致: X有{X.shape[0]}行, Y有{Y.shape[0]}行")
    
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1))
    X = X[mask]
    Y = Y[mask]
    
    n = X.shape[0]
    
    if n < max(X.shape[1], Y.shape[1]) + 1:
        raise ValueError(f"样本量不足: 需要至少{max(X.shape[1], Y.shape[1]) + 1}个观测")
    
    return X, Y, n


def _compute_cca_eigen(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算CCA的特征值和特征向量。
    
    使用标准的CCA算法：
    1. 计算协方差矩阵
    2. 求解广义特征值问题
    3. 得到典型相关系数和典型变量系数
    
    Parameters
    ----------
    X, Y : np.ndarray
        中心化后的数据矩阵
        
    Returns
    -------
    tuple
        (典型相关系数, X的典型变量系数, Y的典型变量系数, 特征值)
    """
    n = X.shape[0]
    p = X.shape[1]
    q = Y.shape[1]
    
    Sxx = np.cov(X, rowvar=False)
    Syy = np.cov(Y, rowvar=False)
    Sxy = np.cov(X, Y, rowvar=False)[:p, p:]
    Syx = Sxy.T
    
    Sxx_inv = inv(Sxx)
    Syy_inv = inv(Syy)
    
    M1 = Sxx_inv @ Sxy @ Syy_inv @ Syx
    M2 = Syy_inv @ Syx @ Sxx_inv @ Sxy
    
    eigenvalues1, eigenvectors1 = eig(M1)
    eigenvalues2, eigenvectors2 = eig(M2)
    
    idx1 = np.argsort(eigenvalues1.real)[::-1]
    eigenvalues1 = eigenvalues1.real[idx1]
    eigenvectors1 = eigenvectors1.real[:, idx1]
    
    idx2 = np.argsort(eigenvalues2.real)[::-1]
    eigenvalues2 = eigenvalues2.real[idx2]
    eigenvectors2 = eigenvectors2.real[:, idx2]
    
    eigenvalues = eigenvalues1
    canonical_corrs = np.sqrt(np.clip(eigenvalues, 0, 1))
    
    k = min(p, q)
    canonical_corrs = canonical_corrs[:k]
    
    A = eigenvectors1[:, :k]
    B = eigenvectors2[:, :k]
    
    for i in range(k):
        norm_a = np.sqrt(A[:, i].T @ Sxx @ A[:, i])
        if norm_a > 0:
            A[:, i] = A[:, i] / norm_a
        
        norm_b = np.sqrt(B[:, i].T @ Syy @ B[:, i])
        if norm_b > 0:
            B[:, i] = B[:, i] / norm_b
    
    return canonical_corrs, A, B, eigenvalues[:k]


def _wilks_lambda_test(canonical_corrs: np.ndarray, n: int, p: int, q: int) -> List[Dict]:
    """
    使用Wilks' Lambda进行典型相关显著性检验。
    
    Wilks' Lambda检验用于检验典型相关系数是否显著不为零。
    
    Parameters
    ----------
    canonical_corrs : np.ndarray
        典型相关系数
    n : int
        样本量
    p : int
        X的变量数
    q : int
        Y的变量数
        
    Returns
    -------
    list
        每个典型相关系数的检验结果
    """
    k = len(canonical_corrs)
    results = []
    
    for i in range(k):
        remaining_corrs = canonical_corrs[i:]
        remaining_eigenvalues = remaining_corrs ** 2
        
        wilks_lambda = 1.0
        for ev in remaining_eigenvalues:
            wilks_lambda *= (1 - ev)
        
        df1 = (p - i) * (q - i)
        df2 = (n - i - 1.5) * (p + q - 2 * i) - 0.5 * ((p - i) * (q - i) - 1)
        
        if df2 > 0 and wilks_lambda > 0:
            chi_sq = -(n - i - 0.5 * (p + q + 3)) * np.log(wilks_lambda)
            p_value = 1 - stats.chi2.cdf(chi_sq, df1)
        else:
            chi_sq = np.nan
            p_value = np.nan
        
        results.append({
            'canonical_index': i + 1,
            'canonical_correlation': canonical_corrs[i],
            'wilks_lambda': wilks_lambda,
            'chi_square': chi_sq,
            'df': df1,
            'p_value': p_value
        })
    
    return results


def _compute_confidence_intervals(canonical_corrs: np.ndarray, n: int, 
                                   confidence_level: float = 0.95) -> List[Tuple[float, float]]:
    """
    计算典型相关系数的置信区间。
    
    使用Fisher Z变换计算置信区间。
    
    Parameters
    ----------
    canonical_corrs : np.ndarray
        典型相关系数
    n : int
        样本量
    confidence_level : float
        置信水平
        
    Returns
    -------
    list
        每个典型相关系数的置信区间
    """
    alpha = 1 - confidence_level
    z_crit = stats.norm.ppf(1 - alpha / 2)
    
    intervals = []
    for r in canonical_corrs:
        if r >= 1.0:
            intervals.append((1.0, 1.0))
        elif r <= 0.0:
            intervals.append((0.0, 0.0))
        else:
            z = np.arctanh(r)
            se = 1 / np.sqrt(n - 3)
            
            z_lower = z - z_crit * se
            z_upper = z + z_crit * se
            
            r_lower = np.tanh(z_lower)
            r_upper = np.tanh(z_upper)
            
            r_lower = max(0.0, min(1.0, r_lower))
            r_upper = max(0.0, min(1.0, r_upper))
            
            intervals.append((r_lower, r_upper))
    
    return intervals


def _compute_redundancy(X: np.ndarray, Y: np.ndarray, 
                        A: np.ndarray, B: np.ndarray,
                        canonical_corrs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    计算冗余指数（Redundancy Index）。
    
    冗余指数衡量一组变量的方差能被另一组变量的典型变量解释的比例。
    
    Parameters
    ----------
    X, Y : np.ndarray
        原始数据矩阵
    A, B : np.ndarray
        典型变量系数
    canonical_corrs : np.ndarray
        典型相关系数
        
    Returns
    -------
    tuple
        (X被Y解释的冗余, Y被X解释的冗余, X的方差解释比例, Y的方差解释比例)
    """
    k = len(canonical_corrs)
    
    X_scores = X @ A
    Y_scores = Y @ B
    
    var_X_scores = np.var(X_scores, axis=0)
    var_Y_scores = np.var(Y_scores, axis=0)
    
    total_var_X = np.var(X, axis=0).sum()
    total_var_Y = np.var(Y, axis=0).sum()
    
    if total_var_X > 0:
        X_var_explained = var_X_scores / total_var_X
    else:
        X_var_explained = np.zeros(k)
    
    if total_var_Y > 0:
        Y_var_explained = var_Y_scores / total_var_Y
    else:
        Y_var_explained = np.zeros(k)
    
    redundancy_X_given_Y = X_var_explained * (canonical_corrs ** 2)
    redundancy_Y_given_X = Y_var_explained * (canonical_corrs ** 2)
    
    return redundancy_X_given_Y, redundancy_Y_given_X, X_var_explained, Y_var_explained


def cca(X: Union[np.ndarray, pd.DataFrame],
        Y: Union[np.ndarray, pd.DataFrame],
        compute_significance: bool = True,
        confidence_level: float = 0.95,
        n_permutations: int = 1000,
        verbose: bool = False) -> Dict:
    """
    执行典型相关分析（Canonical Correlation Analysis, CCA）。
    
    典型相关分析用于研究两组变量之间的线性关系。它寻找两组变量的
    线性组合（典型变量），使得这些组合之间的相关性最大化。
    
    Parameters
    ----------
    X : array-like or pd.DataFrame
        第一组变量，形状为 (n_samples, p)
    Y : array-like or pd.DataFrame
        第二组变量，形状为 (n_samples, q)
    compute_significance : bool, default=True
        是否计算显著性检验（p值）
    confidence_level : float, default=0.95
        置信区间水平
    n_permutations : int, default=1000
        置换检验次数（用于额外的显著性验证）
    verbose : bool, default=False
        是否输出详细信息
        
    Returns
    -------
    dict
        包含以下键的字典：
        - canonical_correlations: 典型相关系数数组
        - x_weights: X的典型变量系数（权重矩阵）
        - y_weights: Y的典型变量系数（权重矩阵）
        - x_scores: X的典型变量得分
        - y_scores: Y的典型变量得分
        - significance_tests: 显著性检验结果列表
        - confidence_intervals: 置信区间列表
        - redundancy: 冗余分析结果
        - n_components: 典型变量对数
        - n_samples: 样本量
        - x_names: X的变量名（如果输入是DataFrame）
        - y_names: Y的变量名（如果输入是DataFrame）
        
    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 3)
    >>> Y = np.random.randn(100, 2)
    >>> result = cca(X, Y)
    >>> print(f"第一对典型相关系数: {result['canonical_correlations'][0]:.4f}")
    
    >>> # 使用DataFrame
    >>> import pandas as pd
    >>> df = pd.DataFrame(np.random.randn(100, 5), 
    ...                   columns=['x1', 'x2', 'x3', 'y1', 'y2'])
    >>> result = cca(df[['x1', 'x2', 'x3']], df[['y1', 'y2']])
    
    References
    ----------
    Hotelling, H. (1936). "Relations between two sets of variates"
    Biometrika, 28(3/4), 321-377
    """
    x_names = None
    y_names = None
    
    if isinstance(X, pd.DataFrame):
        x_names = X.columns.tolist()
        X = X.values
    if isinstance(Y, pd.DataFrame):
        y_names = Y.columns.tolist()
        Y = Y.values
    
    X, Y, n = _validate_cca_inputs(X, Y)
    
    p = X.shape[1]
    q = Y.shape[1]
    k = min(p, q)
    
    if verbose:
        print(f"CCA分析: X有{p}个变量, Y有{q}个变量, 样本量={n}")
        print(f"将计算{k}对典型变量")
    
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    
    canonical_corrs, A, B, eigenvalues = _compute_cca_eigen(X_centered, Y_centered)
    
    X_scores = X_centered @ A
    Y_scores = Y_centered @ B
    
    significance_tests = []
    if compute_significance:
        significance_tests = _wilks_lambda_test(canonical_corrs, n, p, q)
    
    confidence_intervals = _compute_confidence_intervals(canonical_corrs, n, confidence_level)
    
    redundancy_X_given_Y, redundancy_Y_given_X, X_var_explained, Y_var_explained = \
        _compute_redundancy(X_centered, Y_centered, A, B, canonical_corrs)
    
    result = {
        'canonical_correlations': canonical_corrs,
        'x_weights': A,
        'y_weights': B,
        'x_scores': X_scores,
        'y_scores': Y_scores,
        'eigenvalues': eigenvalues,
        'significance_tests': significance_tests,
        'confidence_intervals': confidence_intervals,
        'redundancy': {
            'x_given_y': redundancy_X_given_Y,
            'y_given_x': redundancy_Y_given_X,
            'x_variance_explained': X_var_explained,
            'y_variance_explained': Y_var_explained,
            'total_x_given_y': redundancy_X_given_Y.sum(),
            'total_y_given_x': redundancy_Y_given_X.sum()
        },
        'n_components': k,
        'n_samples': n,
        'n_x_vars': p,
        'n_y_vars': q,
        'x_names': x_names if x_names else [f'X{i+1}' for i in range(p)],
        'y_names': y_names if y_names else [f'Y{i+1}' for i in range(q)],
        'x_mean': X.mean(axis=0),
        'y_mean': Y.mean(axis=0)
    }
    
    if verbose:
        print("\n典型相关系数:")
        for i, (r, ci) in enumerate(zip(canonical_corrs, confidence_intervals)):
            print(f"  第{i+1}对: {r:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
        
        if significance_tests:
            print("\n显著性检验 (Wilks' Lambda):")
            for test in significance_tests:
                print(f"  第{test['canonical_index']}对: χ²={test['chi_square']:.4f}, "
                      f"df={test['df']}, p={test['p_value']:.4f}")
        
        print(f"\n冗余分析:")
        print(f"  X被Y解释的总方差: {result['redundancy']['total_x_given_y']:.2%}")
        print(f"  Y被X解释的总方差: {result['redundancy']['total_y_given_x']:.2%}")
    
    return result


def cca_permutation_test(X: Union[np.ndarray, pd.DataFrame],
                         Y: Union[np.ndarray, pd.DataFrame],
                         n_permutations: int = 1000,
                         random_state: Optional[int] = None) -> Dict:
    """
    使用置换检验验证CCA结果的显著性。
    
    置换检验通过随机打乱样本标签来生成零假设分布，
    然后比较观测到的典型相关系数与零假设分布。
    
    Parameters
    ----------
    X : array-like or pd.DataFrame
        第一组变量
    Y : array-like or pd.DataFrame
        第二组变量
    n_permutations : int, default=1000
        置换次数
    random_state : int, optional
        随机种子
        
    Returns
    -------
    dict
        包含置换检验结果的字典
        
    Examples
    --------
    >>> X = np.random.randn(100, 3)
    >>> Y = np.random.randn(100, 2)
    >>> result = cca_permutation_test(X, Y, n_permutations=500)
    >>> print(f"第一对典型相关系数p值: {result['p_values'][0]:.4f}")
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(Y, pd.DataFrame):
        Y = Y.values
    
    X, Y, n = _validate_cca_inputs(X, Y)
    
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    
    obs_result = cca(X, Y, compute_significance=False, verbose=False)
    obs_corrs = obs_result['canonical_correlations']
    k = len(obs_corrs)
    
    perm_corrs = np.zeros((n_permutations, k))
    
    for i in range(n_permutations):
        perm_idx = np.random.permutation(n)
        Y_perm = Y_centered[perm_idx]
        
        try:
            perm_result = cca(X_centered, Y_perm, compute_significance=False, verbose=False)
            perm_corrs[i, :len(perm_result['canonical_correlations'])] = perm_result['canonical_correlations']
        except Exception:
            pass
    
    p_values = np.zeros(k)
    for j in range(k):
        p_values[j] = (np.sum(perm_corrs[:, j] >= obs_corrs[j]) + 1) / (n_permutations + 1)
    
    return {
        'observed_correlations': obs_corrs,
        'p_values': p_values,
        'n_permutations': n_permutations,
        'permutation_distribution': perm_corrs
    }


class CCAAnalyzer:
    """
    典型相关分析器类
    
    提供完整的典型相关分析流程，包括分析、可视化和结果解释。
    
    Parameters
    ----------
    X : pd.DataFrame
        第一组变量
    Y : pd.DataFrame
        第二组变量
    verbose : bool, default=True
        是否输出详细信息
        
    Examples
    --------
    >>> analyzer = CCAAnalyzer(df[['x1', 'x2']], df[['y1', 'y2']])
    >>> result = analyzer.fit()
    >>> analyzer.summary()
    >>> analyzer.plot_weights()
    """
    
    def __init__(self, 
                 X: Union[pd.DataFrame, np.ndarray],
                 Y: Union[pd.DataFrame, np.ndarray],
                 verbose: bool = True):
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(X.shape[1])])
        if isinstance(Y, np.ndarray):
            Y = pd.DataFrame(Y, columns=[f'Y{i+1}' for i in range(Y.shape[1])])
        
        self.X = X
        self.Y = Y
        self.verbose = verbose
        self.result = None
    
    def fit(self,
            compute_significance: bool = True,
            confidence_level: float = 0.95,
            n_permutations: int = 1000) -> Dict:
        """
        执行典型相关分析。
        
        Parameters
        ----------
        compute_significance : bool, default=True
            是否计算显著性检验
        confidence_level : float, default=0.95
            置信区间水平
        n_permutations : int, default=1000
            置换检验次数
            
        Returns
        -------
        dict
            分析结果
        """
        self.result = cca(
            self.X, self.Y,
            compute_significance=compute_significance,
            confidence_level=confidence_level,
            n_permutations=n_permutations,
            verbose=self.verbose
        )
        
        return self.result
    
    def summary(self) -> str:
        """
        生成分析摘要。
        
        Returns
        -------
        str
            摘要文本
        """
        if self.result is None:
            raise ValueError("请先调用fit()方法")
        
        lines = []
        lines.append("=" * 60)
        lines.append("典型相关分析结果摘要")
        lines.append("=" * 60)
        
        lines.append(f"\n样本量: {self.result['n_samples']}")
        lines.append(f"X变量数: {self.result['n_x_vars']}")
        lines.append(f"Y变量数: {self.result['n_y_vars']}")
        lines.append(f"典型变量对数: {self.result['n_components']}")
        
        lines.append("\n典型相关系数:")
        lines.append("-" * 50)
        lines.append(f"{'序号':<6}{'相关系数':<12}{'95%置信区间':<20}{'p值':<10}")
        lines.append("-" * 50)
        
        for i, (r, ci) in enumerate(zip(
            self.result['canonical_correlations'],
            self.result['confidence_intervals']
        )):
            p_val = ""
            if self.result['significance_tests']:
                p_val = f"{self.result['significance_tests'][i]['p_value']:.4f}"
            lines.append(f"{i+1:<6}{r:<12.4f}[{ci[0]:.4f}, {ci[1]:.4f}]{'':>4}{p_val:<10}")
        
        lines.append("\n冗余分析:")
        lines.append("-" * 50)
        lines.append(f"X被Y解释的总方差: {self.result['redundancy']['total_x_given_y']:.2%}")
        lines.append(f"Y被X解释的总方差: {self.result['redundancy']['total_y_given_x']:.2%}")
        
        lines.append("\n各典型变量的方差解释比例:")
        lines.append("  X变量:")
        for i, v in enumerate(self.result['redundancy']['x_variance_explained']):
            lines.append(f"    第{i+1}对: {v:.2%}")
        lines.append("  Y变量:")
        for i, v in enumerate(self.result['redundancy']['y_variance_explained']):
            lines.append(f"    第{i+1}对: {v:.2%}")
        
        return "\n".join(lines)
    
    def get_weights(self, component: int = 1) -> pd.DataFrame:
        """
        获取指定典型变量的权重。
        
        Parameters
        ----------
        component : int
            典型变量序号（从1开始）
            
        Returns
        -------
        pd.DataFrame
            权重数据框
        """
        if self.result is None:
            raise ValueError("请先调用fit()方法")
        
        idx = component - 1
        if idx < 0 or idx >= self.result['n_components']:
            raise ValueError(f"无效的典型变量序号: {component}")
        
        weights_df = pd.DataFrame({
            'X变量': self.result['x_names'],
            f'X权重(第{component}对)': self.result['x_weights'][:, idx]
        })
        
        y_weights_df = pd.DataFrame({
            'Y变量': self.result['y_names'],
            f'Y权重(第{component}对)': self.result['y_weights'][:, idx]
        })
        
        return weights_df, y_weights_df
    
    def get_scores(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        获取典型变量得分。
        
        Returns
        -------
        tuple
            (X的典型变量得分, Y的典型变量得分)
        """
        if self.result is None:
            raise ValueError("请先调用fit()方法")
        
        x_scores = pd.DataFrame(
            self.result['x_scores'],
            columns=[f'Canonical_{i+1}' for i in range(self.result['n_components'])]
        )
        
        y_scores = pd.DataFrame(
            self.result['y_scores'],
            columns=[f'Canonical_{i+1}' for i in range(self.result['n_components'])]
        )
        
        return x_scores, y_scores
    
    def plot_weights(self, component: int = 1, figsize: Tuple[int, int] = (12, 5),
                     savefig: Optional[str] = None):
        """
        绘制典型变量权重图。
        
        Parameters
        ----------
        component : int
            典型变量序号
        figsize : tuple
            图表大小
        savefig : str, optional
            保存路径
        """
        if self.result is None:
            raise ValueError("请先调用fit()方法")
        
        import matplotlib.pyplot as plt
        
        idx = component - 1
        if idx < 0 or idx >= self.result['n_components']:
            raise ValueError(f"无效的典型变量序号: {component}")
        
        x_weights = self.result['x_weights'][:, idx]
        y_weights = self.result['y_weights'][:, idx]
        x_names = self.result['x_names']
        y_names = self.result['y_names']
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        ax1 = axes[0]
        colors1 = ['#3498db' if w >= 0 else '#e74c3c' for w in x_weights]
        ax1.barh(x_names, x_weights, color=colors1)
        ax1.set_xlabel('权重')
        ax1.set_title(f'X变量权重 (第{component}对典型变量)')
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        ax2 = axes[1]
        colors2 = ['#2ecc71' if w >= 0 else '#e67e22' for w in y_weights]
        ax2.barh(y_names, y_weights, color=colors2)
        ax2.set_xlabel('权重')
        ax2.set_title(f'Y变量权重 (第{component}对典型变量)')
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def plot_scores(self, component: int = 1, figsize: Tuple[int, int] = (8, 8),
                    savefig: Optional[str] = None):
        """
        绘制典型变量得分散点图。
        
        Parameters
        ----------
        component : int
            典型变量序号
        figsize : tuple
            图表大小
        savefig : str, optional
            保存路径
        """
        if self.result is None:
            raise ValueError("请先调用fit()方法")
        
        import matplotlib.pyplot as plt
        
        idx = component - 1
        if idx < 0 or idx >= self.result['n_components']:
            raise ValueError(f"无效的典型变量序号: {component}")
        
        x_scores = self.result['x_scores'][:, idx]
        y_scores = self.result['y_scores'][:, idx]
        r = self.result['canonical_correlations'][idx]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.scatter(x_scores, y_scores, alpha=0.6, edgecolors='white', linewidth=0.5)
        
        z = np.polyfit(x_scores, y_scores, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_scores.min(), x_scores.max(), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.8, label=f'相关系数 = {r:.4f}')
        
        ax.set_xlabel(f'X典型变量得分 (第{component}对)')
        ax.set_ylabel(f'Y典型变量得分 (第{component}对)')
        ax.set_title(f'典型变量得分散点图 (第{component}对)')
        ax.legend()
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def plot_correlations(self, figsize: Tuple[int, int] = (10, 8),
                          savefig: Optional[str] = None):
        """
        绘制典型相关系数图。
        
        Parameters
        ----------
        figsize : tuple
            图表大小
        savefig : str, optional
            保存路径
        """
        if self.result is None:
            raise ValueError("请先调用fit()方法")
        
        import matplotlib.pyplot as plt
        
        corrs = self.result['canonical_correlations']
        cis = self.result['confidence_intervals']
        n_components = len(corrs)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x_pos = range(1, n_components + 1)
        lower_errors = [corrs[i] - cis[i][0] for i in range(n_components)]
        upper_errors = [cis[i][1] - corrs[i] for i in range(n_components)]
        
        ax.bar(x_pos, corrs, color='#3498db', alpha=0.8, 
               yerr=[lower_errors, upper_errors], capsize=5)
        
        ax.set_xlabel('典型变量对')
        ax.set_ylabel('典型相关系数')
        ax.set_title('典型相关系数及95%置信区间')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'第{i}对' for i in x_pos])
        ax.set_ylim(0, 1.1)
        
        for i, (r, ci) in enumerate(zip(corrs, cis)):
            ax.annotate(f'{r:.3f}', (i + 1, r + 0.05), ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if savefig:
            plt.savefig(savefig, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def export_results(self, path: str, format: str = 'excel'):
        """
        导出分析结果。
        
        Parameters
        ----------
        path : str
            保存路径
        format : str, default='excel'
            格式：'excel' 或 'csv'
        """
        if self.result is None:
            raise ValueError("请先调用fit()方法")
        
        if format == 'excel':
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                corrs_df = pd.DataFrame({
                    '典型变量对': [f'第{i+1}对' for i in range(len(self.result['canonical_correlations']))],
                    '典型相关系数': self.result['canonical_correlations'],
                    'CI下限': [ci[0] for ci in self.result['confidence_intervals']],
                    'CI上限': [ci[1] for ci in self.result['confidence_intervals']]
                })
                if self.result['significance_tests']:
                    corrs_df['Wilks_Lambda'] = [t['wilks_lambda'] for t in self.result['significance_tests']]
                    corrs_df['Chi_Square'] = [t['chi_square'] for t in self.result['significance_tests']]
                    corrs_df['df'] = [t['df'] for t in self.result['significance_tests']]
                    corrs_df['p_value'] = [t['p_value'] for t in self.result['significance_tests']]
                corrs_df.to_excel(writer, sheet_name='典型相关系数', index=False)
                
                x_weights_df = pd.DataFrame(
                    self.result['x_weights'],
                    index=self.result['x_names'],
                    columns=[f'第{i+1}对' for i in range(self.result['n_components'])]
                )
                x_weights_df.to_excel(writer, sheet_name='X权重')
                
                y_weights_df = pd.DataFrame(
                    self.result['y_weights'],
                    index=self.result['y_names'],
                    columns=[f'第{i+1}对' for i in range(self.result['n_components'])]
                )
                y_weights_df.to_excel(writer, sheet_name='Y权重')
                
                redundancy_df = pd.DataFrame({
                    '典型变量对': [f'第{i+1}对' for i in range(self.result['n_components'])],
                    'X方差解释比例': self.result['redundancy']['x_variance_explained'],
                    'Y方差解释比例': self.result['redundancy']['y_variance_explained'],
                    'X被Y解释的冗余': self.result['redundancy']['x_given_y'],
                    'Y被X解释的冗余': self.result['redundancy']['y_given_x']
                })
                redundancy_df.to_excel(writer, sheet_name='冗余分析', index=False)
        
        elif format == 'csv':
            import os
            base, ext = os.path.splitext(path)
            
            corrs_df = pd.DataFrame({
                'canonical_pair': range(1, len(self.result['canonical_correlations']) + 1),
                'correlation': self.result['canonical_correlations'],
                'ci_lower': [ci[0] for ci in self.result['confidence_intervals']],
                'ci_upper': [ci[1] for ci in self.result['confidence_intervals']]
            })
            corrs_df.to_csv(f"{base}_correlations.csv", index=False)
            
            x_weights_df = pd.DataFrame(
                self.result['x_weights'],
                index=self.result['x_names'],
                columns=[f'pair_{i+1}' for i in range(self.result['n_components'])]
            )
            x_weights_df.to_csv(f"{base}_x_weights.csv")
            
            y_weights_df = pd.DataFrame(
                self.result['y_weights'],
                index=self.result['y_names'],
                columns=[f'pair_{i+1}' for i in range(self.result['n_components'])]
            )
            y_weights_df.to_csv(f"{base}_y_weights.csv")
        
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def __repr__(self):
        return f"CCAAnalyzer(X_vars={self.result['n_x_vars'] if self.result else self.X.shape[1]}, " \
               f"Y_vars={self.result['n_y_vars'] if self.result else self.Y.shape[1]})"
