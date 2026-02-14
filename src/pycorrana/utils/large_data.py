"""
大数据优化模块
==============
提供针对大数据集的采样、分块计算等优化策略。
"""

import warnings
from typing import Optional, List, Tuple, Union, Callable
import numpy as np
import pandas as pd
from pathlib import Path


LARGE_DATA_THRESHOLD_ROWS = 100_000
LARGE_DATA_THRESHOLD_MEMORY_MB = 500
SAMPLE_SIZE_DEFAULT = 50_000
CHUNK_SIZE_DEFAULT = 10_000


def estimate_memory_usage(df: pd.DataFrame) -> float:
    """
    估算DataFrame的内存使用量（MB）。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
        
    Returns
    -------
    float
        内存使用量（MB）
    """
    return df.memory_usage(deep=True).sum() / (1024 * 1024)


def is_large_data(df: pd.DataFrame, 
                  threshold_rows: int = LARGE_DATA_THRESHOLD_ROWS,
                  threshold_memory_mb: float = LARGE_DATA_THRESHOLD_MEMORY_MB) -> bool:
    """
    判断数据是否为大数据集。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    threshold_rows : int
        行数阈值
    threshold_memory_mb : float
        内存阈值（MB）
        
    Returns
    -------
    bool
        是否为大数据集
    """
    if len(df) > threshold_rows:
        return True
    
    if estimate_memory_usage(df) > threshold_memory_mb:
        return True
    
    return False


def smart_sample(df: pd.DataFrame,
                sample_size: int = SAMPLE_SIZE_DEFAULT,
                stratify_col: Optional[str] = None,
                random_state: int = 42,
                verbose: bool = True) -> pd.DataFrame:
    """
    智能采样：支持随机采样和分层采样。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    sample_size : int
        目标样本量
    stratify_col : str, optional
        用于分层采样的列名
    random_state : int
        随机种子
    verbose : bool
        是否输出信息
        
    Returns
    -------
    pd.DataFrame
        采样后的数据
    """
    n = len(df)
    
    if n <= sample_size:
        if verbose:
            print(f"数据量 ({n}) 小于采样阈值 ({sample_size})，无需采样")
        return df
    
    if stratify_col and stratify_col in df.columns:
        try:
            sample_df = df.groupby(stratify_col, group_keys=False).apply(
                lambda x: x.sample(
                    n=int(sample_size * len(x) / n),
                    random_state=random_state
                ),
                include_groups=False
            )
            if verbose:
                print(f"分层采样: {n} -> {len(sample_df)} 行 (分层列: {stratify_col})")
            return sample_df
        except Exception:
            pass
    
    sample_df = df.sample(n=sample_size, random_state=random_state)
    
    if verbose:
        print(f"随机采样: {n} -> {sample_size} 行")
    
    return sample_df


def chunked_correlation(df: pd.DataFrame,
                       columns: Optional[List[str]] = None,
                       method: str = 'pearson',
                       chunk_size: int = CHUNK_SIZE_DEFAULT,
                       verbose: bool = True) -> pd.DataFrame:
    """
    分块计算相关性矩阵（适用于大数据集）。
    
    使用增量更新算法，避免一次性加载所有数据到内存。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    columns : list, optional
        要计算的列
    method : str
        相关方法：'pearson', 'spearman', 'kendall'
    chunk_size : int
        每块大小
    verbose : bool
        是否输出进度
        
    Returns
    -------
    pd.DataFrame
        相关性矩阵
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = len(columns)
    n_rows = len(df)
    
    if method == 'pearson':
        sum_x = np.zeros(n_cols)
        sum_x2 = np.zeros(n_cols)
        sum_xy = np.zeros((n_cols, n_cols))
        count = np.zeros(n_cols)
        
        for i in range(0, n_rows, chunk_size):
            chunk = df.iloc[i:i+chunk_size][columns]
            chunk_values = chunk.values
            
            valid_mask = ~np.isnan(chunk_values)
            
            for j in range(n_cols):
                col_valid = valid_mask[:, j]
                sum_x[j] += np.nansum(chunk_values[:, j])
                sum_x2[j] += np.nansum(chunk_values[:, j] ** 2)
                count[j] += col_valid.sum()
            
            for j in range(n_cols):
                for k in range(j, n_cols):
                    valid_both = valid_mask[:, j] & valid_mask[:, k]
                    if valid_both.sum() > 0:
                        sum_xy[j, k] += np.nansum(
                            chunk_values[:, j] * chunk_values[:, k]
                        )
            
            if verbose and (i // chunk_size) % 10 == 0:
                print(f"  处理进度: {min(i+chunk_size, n_rows)}/{n_rows} 行")
        
        corr_matrix = np.eye(n_cols)
        for j in range(n_cols):
            for k in range(j+1, n_cols):
                n_jk = min(count[j], count[k])
                if n_jk > 1:
                    mean_j = sum_x[j] / count[j]
                    mean_k = sum_x[k] / count[k]
                    
                    cov_jk = (sum_xy[j, k] - n_jk * mean_j * mean_k) / (n_jk - 1)
                    var_j = (sum_x2[j] - count[j] * mean_j**2) / (count[j] - 1)
                    var_k = (sum_x2[k] - count[k] * mean_k**2) / (count[k] - 1)
                    
                    if var_j > 0 and var_k > 0:
                        corr_matrix[j, k] = cov_jk / np.sqrt(var_j * var_k)
                        corr_matrix[k, j] = corr_matrix[j, k]
        
        return pd.DataFrame(corr_matrix, index=columns, columns=columns)
    
    else:
        sample_size = min(SAMPLE_SIZE_DEFAULT, n_rows)
        if verbose:
            print(f"非Pearson方法使用采样计算 (样本量: {sample_size})")
        sample_df = df.sample(n=sample_size, random_state=42)
        
        if method == 'spearman':
            return sample_df[columns].corr(method='spearman')
        elif method == 'kendall':
            return sample_df[columns].corr(method='kendall')
        else:
            return sample_df[columns].corr()


def chunked_apply(df: pd.DataFrame,
                  func: Callable,
                  chunk_size: int = CHUNK_SIZE_DEFAULT,
                  combine_func: Optional[Callable] = None,
                  verbose: bool = True,
                  **kwargs) -> dict:
    """
    分块应用函数并合并结果。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    func : callable
        每块应用的函数，返回字典
    chunk_size : int
        每块大小
    combine_func : callable, optional
        合并结果的函数
    verbose : bool
        是否输出进度
    **kwargs
        传递给func的参数
        
    Returns
    -------
    dict
        合并后的结果
    """
    n_rows = len(df)
    results = []
    
    for i in range(0, n_rows, chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        result = func(chunk, **kwargs)
        results.append(result)
        
        if verbose:
            print(f"  处理进度: {min(i+chunk_size, n_rows)}/{n_rows} 行")
    
    if combine_func:
        return combine_func(results)
    
    return results


def optimize_dataframe(df: pd.DataFrame,
                      verbose: bool = True) -> pd.DataFrame:
    """
    优化DataFrame内存使用。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    verbose : bool
        是否输出信息
        
    Returns
    -------
    pd.DataFrame
        优化后的数据
    """
    start_memory = estimate_memory_usage(df)
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        if col_type == 'int64':
            col_min = df_optimized[col].min()
            col_max = df_optimized[col].max()
            
            if col_min >= 0:
                if col_max < 255:
                    df_optimized[col] = df_optimized[col].astype('uint8')
                elif col_max < 65535:
                    df_optimized[col] = df_optimized[col].astype('uint16')
                elif col_max < 4294967295:
                    df_optimized[col] = df_optimized[col].astype('uint32')
            else:
                if col_min > -128 and col_max < 127:
                    df_optimized[col] = df_optimized[col].astype('int8')
                elif col_min > -32768 and col_max < 32767:
                    df_optimized[col] = df_optimized[col].astype('int16')
                elif col_min > -2147483648 and col_max < 2147483647:
                    df_optimized[col] = df_optimized[col].astype('int32')
        
        elif col_type == 'float64':
            df_optimized[col] = df_optimized[col].astype('float32')
        
        elif col_type == 'object':
            unique_ratio = df_optimized[col].nunique() / len(df_optimized)
            if unique_ratio < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
    
    end_memory = estimate_memory_usage(df_optimized)
    
    if verbose:
        reduction = (1 - end_memory / start_memory) * 100
        print(f"内存优化: {start_memory:.1f}MB -> {end_memory:.1f}MB (减少 {reduction:.1f}%)")
    
    return df_optimized


class LargeDataConfig:
    """
    大数据处理配置类。
    
    Examples
    --------
    >>> config = LargeDataConfig(
    ...     sample_size=100000,
    ...     chunk_size=50000,
    ...     auto_sample=True
    ... )
    >>> analyzer = CorrAnalyzer(df, large_data_config=config)
    """
    
    def __init__(self,
                 sample_size: int = SAMPLE_SIZE_DEFAULT,
                 chunk_size: int = CHUNK_SIZE_DEFAULT,
                 auto_sample: bool = True,
                 auto_optimize: bool = True,
                 threshold_rows: int = LARGE_DATA_THRESHOLD_ROWS,
                 threshold_memory_mb: float = LARGE_DATA_THRESHOLD_MEMORY_MB,
                 stratify_column: Optional[str] = None,
                 random_state: int = 42,
                 verbose: bool = True):
        
        self.sample_size = sample_size
        self.chunk_size = chunk_size
        self.auto_sample = auto_sample
        self.auto_optimize = auto_optimize
        self.threshold_rows = threshold_rows
        self.threshold_memory_mb = threshold_memory_mb
        self.stratify_column = stratify_column
        self.random_state = random_state
        self.verbose = verbose
    
    def should_optimize(self, df: pd.DataFrame) -> bool:
        """判断是否需要优化处理"""
        return is_large_data(df, self.threshold_rows, self.threshold_memory_mb)
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """
        准备数据：根据配置进行优化和采样。
        
        Returns
        -------
        tuple
            (处理后的数据, 是否进行了采样)
        """
        sampled = False
        result_df = df.copy()
        
        if self.auto_optimize:
            result_df = optimize_dataframe(result_df, verbose=self.verbose)
        
        if self.auto_sample and self.should_optimize(result_df):
            result_df = smart_sample(
                result_df,
                sample_size=self.sample_size,
                stratify_col=self.stratify_column,
                random_state=self.random_state,
                verbose=self.verbose
            )
            sampled = True
        
        return result_df, sampled
    
    def __repr__(self):
        return (f"LargeDataConfig(sample_size={self.sample_size}, "
                f"chunk_size={self.chunk_size}, auto_sample={self.auto_sample})")
