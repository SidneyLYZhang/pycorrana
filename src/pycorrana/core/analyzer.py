"""
相关性分析核心模块
==================
提供自动化的相关性分析功能，包括自动方法选择、批量计算、可视化等。
"""


from typing import Union, Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
from scipy import stats

from ..utils.data_utils import (
    load_data, infer_types, handle_missing, 
    detect_outliers, get_column_pairs,
    is_large_data, estimate_memory_usage
)
from ..utils.stats_utils import (
    check_normality, correct_pvalues, cramers_v,
    eta_coefficient, point_biserial, interpret_correlation
)
from ..utils.large_data import (
    LargeDataConfig, smart_sample, chunked_correlation, optimize_dataframe
)
from .visualizer import CorrVisualizer
from .reporter import CorrReporter

if TYPE_CHECKING:
    import polars as pl


class CorrAnalyzer:
    """
    相关性分析器类
    
    提供完整的相关性分析流程，包括数据预处理、自动方法选择、
    批量计算、可视化和报告生成。
    
    Parameters
    ----------
    data : pd.DataFrame
        输入数据
    method : str, default='auto'
        相关性计算方法：'auto'（自动选择）、'pearson'、'spearman'、'kendall'
    missing_strategy : str, default='warn'
        缺失值处理策略：'warn'、'drop'、'fill'
    fill_method : str, optional
        填充方法：'mean'、'median'、'mode'、'knn'
    pvalue_correction : str, default='fdr_bh'
        p值校正方法
    verbose : bool, default=True
        是否输出详细信息
    large_data_config : LargeDataConfig, optional
        大数据处理配置，用于优化大数据集的计算效率
        
    Examples
    --------
    >>> analyzer = CorrAnalyzer(df)
    >>> result = analyzer.fit()
    >>> analyzer.plot_heatmap()
    
    >>> # 大数据集优化
    >>> from pycorrana.utils import LargeDataConfig
    >>> config = LargeDataConfig(sample_size=100000, auto_sample=True)
    >>> analyzer = CorrAnalyzer(large_df, large_data_config=config)
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 method: str = 'auto',
                 missing_strategy: str = 'warn',
                 fill_method: Optional[str] = None,
                 pvalue_correction: str = 'fdr_bh',
                 verbose: bool = True,
                 large_data_config: Optional[LargeDataConfig] = None):
        
        self.original_data = data.copy()
        self.data = data.copy()
        self.method = method
        self.missing_strategy = missing_strategy
        self.fill_method = fill_method
        self.pvalue_correction = pvalue_correction
        self.verbose = verbose
        
        self.large_data_config = large_data_config
        self._is_large_data = False
        self._sampled = False
        
        self.type_mapping = {}
        self.corr_matrix = None
        self.pvalue_matrix = None
        self.confidence_interval_matrix = None
        self.significant_pairs = []
        self.methods_used = {}
        
        self.visualizer = CorrVisualizer()
        self.reporter = CorrReporter()
        
        if self.verbose:
            print("=" * 60)
            print("PyCorrAna - 相关性分析器")
            print("=" * 60)
            
            mem_mb = estimate_memory_usage(self.data)
            print(f"数据规模: {len(self.data):,} 行, {len(self.data.columns)} 列, {mem_mb:.1f} MB")
            
            if is_large_data(self.data):
                self._is_large_data = True
                print("⚠️ 检测到大数据集，将使用优化策略")
                if self.large_data_config is None:
                    self.large_data_config = LargeDataConfig(verbose=self.verbose)
    
    def preprocess(self) -> 'CorrAnalyzer':
        """
        数据预处理：类型推断、缺失值处理、大数据优化。
        
        Returns
        -------
        self
        """
        if self.verbose:
            print("\n[1/4] 数据预处理...")
        
        if self._is_large_data and self.large_data_config:
            if self.large_data_config.auto_optimize:
                self.data = optimize_dataframe(self.data, verbose=self.verbose)
            
            if self.large_data_config.auto_sample:
                self.data, self._sampled = self.large_data_config.prepare_data(self.data)
                if self._sampled:
                    self.original_data = self.data.copy()
        
        self.type_mapping = infer_types(self.data)
        
        if self.verbose:
            print(f"  检测到 {len(self.data)} 行, {len(self.data.columns)} 列")
            if self._sampled:
                print(f"  (已采样，原始数据: {len(self.original_data):,} 行)")
            type_counts = {}
            for t in self.type_mapping.values():
                type_counts[t] = type_counts.get(t, 0) + 1
            for t, count in type_counts.items():
                print(f"  - {t}: {count} 列")
        
        if self.missing_strategy != 'warn':
            self.data = handle_missing(
                self.data, 
                strategy=self.missing_strategy,
                fill_method=self.fill_method,
                verbose=self.verbose
            )
        else:
            missing_ratio = self.data.isnull().sum() / len(self.data)
            if missing_ratio.any() and self.verbose:
                print("\n  ⚠️ 缺失值预警:")
                for col, ratio in missing_ratio[missing_ratio > 0].items():
                    print(f"     {col}: {ratio:.2%}")
        
        return self
    
    def compute_correlation(self, 
                           target: Optional[str] = None,
                           columns: Optional[List[str]] = None) -> 'CorrAnalyzer':
        """
        计算相关性矩阵。
        
        Parameters
        ----------
        target : str, optional
            目标变量，如果指定则只计算与目标变量的相关性
        columns : list, optional
            指定要分析的列
            
        Returns
        -------
        self
        """
        if self.verbose:
            print("\n[2/4] 计算相关性...")
        
        # 选择列
        if columns:
            self.data = self.data[columns]
            self.type_mapping = {k: v for k, v in self.type_mapping.items() 
                               if k in columns}
        
        # 获取所有列对
        pairs = get_column_pairs(self.data, self.type_mapping, target)
        
        if self.verbose:
            print(f"  需要计算 {len(pairs)} 对变量的相关性")
        
        # 初始化结果矩阵
        all_cols = [target] if target else list(self.data.columns)
        if not target:
            all_cols = list(self.data.columns)
        
        n_cols = len(all_cols)
        corr_array = np.full((n_cols, n_cols), np.nan, dtype=np.float64)
        pvalue_array = np.full((n_cols, n_cols), np.nan, dtype=np.float64)
        ci_array = np.full((n_cols, n_cols), np.nan, dtype=object)
        
        self.corr_matrix = pd.DataFrame(corr_array, index=all_cols, columns=all_cols)
        self.pvalue_matrix = pd.DataFrame(pvalue_array, index=all_cols, columns=all_cols)
        self.confidence_interval_matrix = pd.DataFrame(ci_array, index=all_cols, columns=all_cols)
        
        # 对角线为1
        for i in range(n_cols):
            self.corr_matrix.iloc[i, i] = 1.0
            self.pvalue_matrix.iloc[i, i] = 0.0
        
        # 批量计算
        pvalues_list = []
        pairs_list = []
        
        for col1, col2, pair_type in pairs:
            corr_val, p_val, method_used, ci = self._compute_pair(col1, col2, pair_type)
            
            self.corr_matrix.loc[col1, col2] = corr_val
            self.corr_matrix.loc[col2, col1] = corr_val
            self.pvalue_matrix.loc[col1, col2] = p_val
            self.pvalue_matrix.loc[col2, col1] = p_val
            self.confidence_interval_matrix.loc[col1, col2] = ci
            self.confidence_interval_matrix.loc[col2, col1] = ci
            
            self.methods_used[f"{col1}-{col2}"] = method_used
            
            pvalues_list.append(p_val)
            pairs_list.append((col1, col2, corr_val, p_val, ci))
        
        # p值校正
        if pvalues_list and self.pvalue_correction:
            corrected_pvalues = correct_pvalues(pvalues_list, self.pvalue_correction)
            
            # 更新显著性对列表
            for i, (col1, col2, corr_val, orig_p, ci) in enumerate(pairs_list):
                corr_p = corrected_pvalues[i] if i < len(corrected_pvalues) else orig_p
                
                if corr_p < 0.05:  # 显著性水平
                    self.significant_pairs.append({
                        'var1': col1,
                        'var2': col2,
                        'correlation': corr_val,
                        'p_value': corr_p,
                        'confidence_interval': ci,
                        'method': self.methods_used.get(f"{col1}-{col2}", "unknown"),
                        'interpretation': interpret_correlation(corr_val)
                    })
        
        # 按相关性强度排序
        self.significant_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        
        if self.verbose:
            print(f"  完成计算，发现 {len(self.significant_pairs)} 对显著相关变量")
        
        return self
    
    def _compute_pair(self, col1: str, col2: str, 
                     pair_type: str) -> Tuple[float, float, str, Optional[Tuple[float, float]]]:
        """
        计算单个变量对的相关性。
        
        Returns
        -------
        tuple
            (相关系数, p值, 使用的方法, 置信区间)
        """
        x = self.data[col1].dropna()
        y = self.data[col2].dropna()
        
        # 对齐数据
        common_idx = x.index.intersection(y.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(x) < 3:
            return np.nan, np.nan, "insufficient_data", None
        
        # 计算置信区间的辅助函数
        def compute_confidence_interval(r, n, method='pearson'):
            """计算相关系数的95%置信区间"""
            if abs(r) == 1:
                return (r, r)
            
            # Fisher变换
            z = np.arctanh(r)
            se = 1 / np.sqrt(n - 3)
            z_crit = stats.norm.ppf(0.975)
            
            # 计算置信区间
            ci_lower = np.tanh(z - z_crit * se)
            ci_upper = np.tanh(z + z_crit * se)
            
            # 限制在[-1, 1]范围内
            ci_lower = max(-1, ci_lower)
            ci_upper = min(1, ci_upper)
            
            return (ci_lower, ci_upper)
        
        # 根据配对类型选择方法
        if self.method == 'auto':
            if pair_type == 'numeric_numeric':
                # 数值+数值：检查正态性
                x_normal = check_normality(x)
                y_normal = check_normality(y)
                
                if x_normal and y_normal:
                    # Pearson
                    r, p = stats.pearsonr(x, y)
                    ci = compute_confidence_interval(r, len(x), 'pearson')
                    return r, p, "pearson", ci
                else:
                    # Spearman更稳健
                    r, p = stats.spearmanr(x, y)
                    ci = compute_confidence_interval(r, len(x), 'spearman')
                    return r, p, "spearman", ci
            
            elif pair_type == 'numeric_binary':
                # 数值+二分类：点双列相关
                r, p = point_biserial(x, y)
                ci = compute_confidence_interval(r, len(x))
                return r, p, "point_biserial", ci
            
            elif pair_type == 'numeric_categorical':
                # 数值+多分类：Eta系数
                eta, p = eta_coefficient(x, y)
                # Eta系数的置信区间计算（近似）
                ci = compute_confidence_interval(eta, len(x))
                return eta, p, "eta", ci
            
            elif pair_type == 'categorical_categorical':
                # 分类+分类：Cramér's V
                v, p = cramers_v(x, y)
                # Cramér's V的置信区间计算（近似）
                ci = compute_confidence_interval(v, len(x))
                return v, p, "cramers_v", ci
            
            elif pair_type == 'ordinal_ordinal':
                # 有序+有序：Kendall's Tau
                tau, p = stats.kendalltau(x, y)
                # Kendall's Tau的置信区间
                n = len(x)
                if n > 10:
                    # 使用正态近似
                    se = 1 / np.sqrt(n * (n - 1) / 2 - 1)
                    z_crit = stats.norm.ppf(0.975)
                    ci_lower = max(-1, tau - z_crit * se)
                    ci_upper = min(1, tau + z_crit * se)
                    ci = (ci_lower, ci_upper)
                else:
                    ci = (np.nan, np.nan)
                return tau, p, "kendall", ci
            
            else:
                # 默认使用Spearman
                r, p = stats.spearmanr(x, y)
                ci = compute_confidence_interval(r, len(x), 'spearman')
                return r, p, "spearman", ci
        
        else:
            # 使用用户指定的方法
            # 先转换为数值类型
            x = pd.to_numeric(x, errors='coerce')
            y = pd.to_numeric(y, errors='coerce')
            
            # 移除NaN
            valid_mask = ~(x.isna() | y.isna())
            x = x[valid_mask]
            y = y[valid_mask]
            
            if len(x) < 3:
                return np.nan, np.nan, "insufficient_data", None
            
            if self.method == 'pearson':
                r, p = stats.pearsonr(x, y)
                ci = compute_confidence_interval(r, len(x), 'pearson')
            elif self.method == 'spearman':
                r, p = stats.spearmanr(x, y)
                ci = compute_confidence_interval(r, len(x), 'spearman')
            elif self.method == 'kendall':
                tau, p = stats.kendalltau(x, y)
                # Kendall's Tau的置信区间
                n = len(x)
                if n > 10:
                    se = 1 / np.sqrt(n * (n - 1) / 2 - 1)
                    z_crit = stats.norm.ppf(0.975)
                    ci_lower = max(-1, tau - z_crit * se)
                    ci_upper = min(1, tau + z_crit * se)
                    ci = (ci_lower, ci_upper)
                else:
                    ci = (np.nan, np.nan)
                r = tau
            else:
                raise ValueError(f"未知的方法: {self.method}")
            
            return r, p, self.method, ci
    
    def fit(self, target: Optional[str] = None,
            columns: Optional[List[str]] = None) -> Dict:
        """
        执行完整分析流程。
        
        Parameters
        ----------
        target : str, optional
            目标变量
        columns : list, optional
            指定列
            
        Returns
        -------
        dict
            分析结果字典
        """
        self.preprocess()
        self.compute_correlation(target=target, columns=columns)
        
        if self.verbose:
            print("\n[3/4] 分析完成！")
        
        return {
            'correlation_matrix': self.corr_matrix,
            'pvalue_matrix': self.pvalue_matrix,
            'confidence_interval_matrix': self.confidence_interval_matrix,
            'significant_pairs': self.significant_pairs,
            'methods_used': self.methods_used,
            'type_mapping': self.type_mapping
        }
    
    def plot_heatmap(self, 
                    figsize: Tuple[int, int] = (10, 8),
                    annot: bool = True,
                    cmap: str = 'RdBu_r',
                    cluster: bool = False,
                    savefig: Optional[str] = None,
                    **kwargs):
        """
        绘制相关性热力图。
        
        Parameters
        ----------
        figsize : tuple, default=(10, 8)
            图表大小
        annot : bool, default=True
            是否显示数值标注
        cmap : str, default='RdBu_r'
            颜色映射
        cluster : bool, default=False
            是否进行层次聚类
        savefig : str, optional
            保存路径
        **kwargs
            其他参数传递给seaborn.heatmap
        """
        if self.corr_matrix is None:
            raise ValueError("请先调用fit()或compute_correlation()")
        
        return self.visualizer.plot_heatmap(
            self.corr_matrix,
            figsize=figsize,
            annot=annot,
            cmap=cmap,
            cluster=cluster,
            savefig=savefig,
            **kwargs
        )
    
    def plot_pairplot(self,
                     columns: Optional[List[str]] = None,
                     hue: Optional[str] = None,
                     savefig: Optional[str] = None,
                     **kwargs):
        """
        绘制散点图矩阵。
        
        Parameters
        ----------
        columns : list, optional
            要绘制的列，默认为所有数值列
        hue : str, optional
            用于颜色区分的列
        savefig : str, optional
            保存路径
        **kwargs
            其他参数
        """
        if columns is None:
            columns = [col for col, t in self.type_mapping.items() 
                      if t == 'numeric']
        
        return self.visualizer.plot_pairplot(
            self.data,
            columns=columns,
            hue=hue,
            savefig=savefig,
            **kwargs
        )
    
    def plot_boxplot(self,
                    numeric_col: str,
                    categorical_col: str,
                    kind: str = 'box',
                    savefig: Optional[str] = None,
                    **kwargs):
        """
        绘制数值变量按分类变量分组的箱线图/小提琴图。
        
        Parameters
        ----------
        numeric_col : str
            数值列名
        categorical_col : str
            分类列名
        kind : str, default='box'
            图表类型：'box'、'violin'、'boxen'
        savefig : str, optional
            保存路径
        **kwargs
            其他参数
        """
        return self.visualizer.plot_boxplot(
            self.data,
            numeric_col=numeric_col,
            categorical_col=categorical_col,
            kind=kind,
            savefig=savefig,
            **kwargs
        )
    
    def export_results(self, 
                      path: str,
                      format: str = 'excel'):
        """
        导出结果。
        
        Parameters
        ----------
        path : str
            保存路径
        format : str, default='excel'
            格式：'excel'、'csv'
        """
        return self.reporter.export_results(
            self.corr_matrix,
            self.pvalue_matrix,
            self.significant_pairs,
            path=path,
            format=format
        )
    
    def summary(self) -> str:
        """
        生成文本摘要。
        
        Returns
        -------
        str
            摘要文本
        """
        return self.reporter.generate_summary(
            self.significant_pairs,
            self.methods_used
        )
    
    def cca(self,
            x_vars: List[str],
            y_vars: List[str],
            compute_significance: bool = True,
            confidence_level: float = 0.95,
            n_permutations: int = 1000) -> Dict:
        """
        执行典型相关分析（Canonical Correlation Analysis, CCA）。
        
        典型相关分析用于研究两组变量之间的线性关系。它寻找两组变量的
        线性组合（典型变量），使得这些组合之间的相关性最大化。
        
        Parameters
        ----------
        x_vars : list
            第一组变量名列表
        y_vars : list
            第二组变量名列表
        compute_significance : bool, default=True
            是否计算显著性检验（p值）
        confidence_level : float, default=0.95
            置信区间水平
        n_permutations : int, default=1000
            置换检验次数
            
        Returns
        -------
        dict
            CCA分析结果，包含：
            - canonical_correlations: 典型相关系数数组
            - x_weights: X的典型变量系数
            - y_weights: Y的典型变量系数
            - x_scores: X的典型变量得分
            - y_scores: Y的典型变量得分
            - significance_tests: 显著性检验结果
            - confidence_intervals: 置信区间
            - redundancy: 冗余分析结果
            
        Examples
        --------
        >>> analyzer = CorrAnalyzer(df)
        >>> analyzer.fit()
        >>> cca_result = analyzer.cca(x_vars=['var1', 'var2'], y_vars=['var3', 'var4'])
        >>> print(cca_result['canonical_correlations'])
        
        References
        ----------
        Hotelling, H. (1936). "Relations between two sets of variates"
        Biometrika, 28(3/4), 321-377
        """
        from .cca import cca as _cca
        
        missing_x = [v for v in x_vars if v not in self.data.columns]
        missing_y = [v for v in y_vars if v not in self.data.columns]
        
        if missing_x:
            raise ValueError(f"X变量不存在于数据中: {missing_x}")
        if missing_y:
            raise ValueError(f"Y变量不存在于数据中: {missing_y}")
        
        X = self.data[x_vars]
        Y = self.data[y_vars]
        
        if self.verbose:
            print(f"\n执行典型相关分析...")
            print(f"  X变量: {x_vars}")
            print(f"  Y变量: {y_vars}")
        
        result = _cca(
            X, Y,
            compute_significance=compute_significance,
            confidence_level=confidence_level,
            n_permutations=n_permutations,
            verbose=self.verbose
        )
        
        return result
    
    def cca_summary(self,
                    x_vars: List[str],
                    y_vars: List[str],
                    compute_significance: bool = True,
                    confidence_level: float = 0.95) -> str:
        """
        执行典型相关分析并生成摘要报告。
        
        Parameters
        ----------
        x_vars : list
            第一组变量名列表
        y_vars : list
            第二组变量名列表
        compute_significance : bool, default=True
            是否计算显著性检验
        confidence_level : float, default=0.95
            置信区间水平
            
        Returns
        -------
        str
            分析摘要文本
        """
        from .cca import CCAAnalyzer
        
        missing_x = [v for v in x_vars if v not in self.data.columns]
        missing_y = [v for v in y_vars if v not in self.data.columns]
        
        if missing_x:
            raise ValueError(f"X变量不存在于数据中: {missing_x}")
        if missing_y:
            raise ValueError(f"Y变量不存在于数据中: {missing_y}")
        
        X = self.data[x_vars]
        Y = self.data[y_vars]
        
        analyzer = CCAAnalyzer(X, Y, verbose=self.verbose)
        analyzer.fit(
            compute_significance=compute_significance,
            confidence_level=confidence_level
        )
        
        return analyzer.summary()
    
    def __repr__(self):
        return f"CorrAnalyzer(data_shape={self.data.shape}, method='{self.method}')"


def quick_corr(data: Union[str, pd.DataFrame, "pl.DataFrame"],
               target: Optional[str] = None,
               columns: Optional[List[str]] = None,
               method: str = 'auto',
               missing_strategy: str = 'warn',
               fill_method: Optional[str] = None,
               pvalue_correction: str = 'fdr_bh',
               plot: bool = True,
               export: Union[bool, str] = False,
               verbose: bool = True,
               large_data_config: Optional[LargeDataConfig] = None,
               **kwargs) -> Dict:
    """
    快速相关性分析入口函数。
    
    一键完成数据加载、预处理、相关性计算、可视化和报告生成。
    
    Parameters
    ----------
    data : str, pd.DataFrame, pl.DataFrame
        数据输入（文件路径或DataFrame）
    target : str, optional
        目标变量
    columns : list, optional
        指定分析的列
    method : str, default='auto'
        相关性方法
    missing_strategy : str, default='warn'
        缺失值处理策略
    fill_method : str, optional
        填充方法
    pvalue_correction : str, default='fdr_bh'
        p值校正方法
    plot : bool, default=True
        是否自动绘图
    export : bool or str, default=False
        是否导出结果，可以是文件路径
    verbose : bool, default=True
        是否输出详细信息
    large_data_config : LargeDataConfig, optional
        大数据处理配置，用于优化大数据集的计算效率
    **kwargs
        其他参数
        
    Returns
    -------
    dict
        完整分析结果
        
    Examples
    --------
    >>> # 基本用法
    >>> result = quick_corr('data.csv')
    
    >>> # 指定目标变量
    >>> result = quick_corr(df, target='sales')
    
    >>> # 自定义参数
    >>> result = quick_corr(df, method='spearman', plot=False, export='results.xlsx')
    
    >>> # 大数据集优化
    >>> from pycorrana.utils import LargeDataConfig
    >>> config = LargeDataConfig(sample_size=50000, auto_sample=True)
    >>> result = quick_corr(large_df, large_data_config=config)
    """
    df = load_data(data)
    
    analyzer = CorrAnalyzer(
        data=df,
        method=method,
        missing_strategy=missing_strategy,
        fill_method=fill_method,
        pvalue_correction=pvalue_correction,
        verbose=verbose,
        large_data_config=large_data_config
    )
    
    result = analyzer.fit(target=target, columns=columns)
    
    if plot:
        if verbose:
            print("\n[3/4] 生成可视化...")
        
        if len(analyzer.corr_matrix) > 1:
            analyzer.plot_heatmap()
        
        numeric_cols = [col for col, t in analyzer.type_mapping.items() 
                       if t == 'numeric']
        if len(numeric_cols) >= 2 and len(numeric_cols) <= 6:
            try:
                analyzer.plot_pairplot(columns=numeric_cols)
            except Exception as e:
                if verbose:
                    print(f"  散点图矩阵生成失败: {e}")
    
    if export:
        if verbose:
            print("\n[4/4] 导出结果...")
        
        if isinstance(export, str):
            export_path = export
        else:
            export_path = 'correlation_results.xlsx'
        
        analyzer.export_results(export_path)
    
    # 输出摘要
    if verbose:
        print("\n" + analyzer.summary())
    
    return result
