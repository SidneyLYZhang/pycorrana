"""
数据接入与基础清洗工具
======================
提供数据加载、类型推断、缺失值处理、异常值检测等功能。
"""

import warnings
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# 尝试导入 polars，如果失败则使用占位符
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None


def load_data(data: Union[str, Path, pd.DataFrame, "pl.DataFrame"]) -> pd.DataFrame:
    """
    智能数据加载器，支持多种输入格式。
    
    Parameters
    ----------
    data : str, Path, pd.DataFrame, pl.DataFrame
        数据输入，可以是文件路径或DataFrame对象
        
    Returns
    -------
    pd.DataFrame
        标准化的pandas DataFrame
        
    Examples
    --------
    >>> df = load_data('data.csv')
    >>> df = load_data('data.xlsx')
    >>> df = load_data(polars_df)
    """
    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        
        suffix = path.suffix.lower()
        
        if suffix == '.csv':
            return pd.read_csv(path)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(path)
        elif suffix == '.parquet':
            return pd.read_parquet(path)
        elif suffix == '.json':
            return pd.read_json(path)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
    
    elif POLARS_AVAILABLE and pl is not None and isinstance(data, pl.DataFrame):
        return data.to_pandas()
    
    elif isinstance(data, pd.DataFrame):
        return data.copy()
    
    else:
        raise TypeError(f"不支持的数据类型: {type(data)}")


def infer_types(df: pd.DataFrame, 
                manual_override: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    自动推断DataFrame列的数据类型。
    
    将列区分为：
    - 'numeric': 数值型（int/float）
    - 'binary': 二分类
    - 'categorical': 多分类无序
    - 'ordinal': 有序分类
    - 'datetime': 日期时间
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    manual_override : dict, optional
        手动覆盖类型，如 {'col1': 'numeric', 'col2': 'categorical'}
        
    Returns
    -------
    dict
        列名到类型的映射字典
    """
    type_mapping = {}
    manual_override = manual_override or {}
    
    for col in df.columns:
        # 优先使用手动覆盖
        if col in manual_override:
            type_mapping[col] = manual_override[col]
            continue
        
        series = df[col]
        dtype = series.dtype
        
        # 检查是否为datetime
        if pd.api.types.is_datetime64_any_dtype(dtype):
            type_mapping[col] = 'datetime'
            continue
        
        # 检查是否为数值型
        if pd.api.types.is_numeric_dtype(dtype):
            # 检查是否为二分类（0/1或两个唯一值）
            unique_vals = series.dropna().unique()
            if len(unique_vals) == 2:
                type_mapping[col] = 'binary'
            else:
                type_mapping[col] = 'numeric'
            continue
        
        # 检查是否为布尔型
        if pd.api.types.is_bool_dtype(dtype):
            type_mapping[col] = 'binary'
            continue
        
        # 处理object/category类型
        n_unique = series.nunique(dropna=True)
        n_total = len(series)
        
        # 如果唯一值很少，可能是分类变量
        if n_unique == 2:
            type_mapping[col] = 'binary'
        elif n_unique <= min(20, n_total * 0.05):  # 少于5%或20个唯一值
            type_mapping[col] = 'categorical'
        else:
            # 尝试转换为数值
            try:
                pd.to_numeric(series, errors='raise')
                type_mapping[col] = 'numeric'
            except (ValueError, TypeError):
                type_mapping[col] = 'categorical'
    
    return type_mapping


def handle_missing(df: pd.DataFrame, 
                   strategy: str = 'warn',
                   fill_method: Optional[str] = None,
                   verbose: bool = True) -> pd.DataFrame:
    """
    缺失值处理工具。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    strategy : str, default='warn'
        处理策略：
        - 'warn': 仅输出警告信息
        - 'drop': 删除含缺失值的行
        - 'fill': 使用fill_method填充
    fill_method : str, optional
        填充方法，当strategy='fill'时使用：
        - 'mean': 均值填充（数值型）
        - 'median': 中位数填充（数值型）
        - 'mode': 众数填充
        - 'knn': KNN预测填充
    verbose : bool, default=True
        是否输出详细信息
        
    Returns
    -------
    pd.DataFrame
        处理后的数据
    """
    df_clean = df.copy()
    missing_ratio = df_clean.isnull().sum() / len(df_clean)
    
    # 输出缺失值预警
    if verbose and missing_ratio.any():
        print("=" * 50)
        print("缺失值检测报告")
        print("=" * 50)
        for col, ratio in missing_ratio[missing_ratio > 0].items():
            level = "严重" if ratio > 0.3 else "中等" if ratio > 0.1 else "轻微"
            print(f"  {col}: {ratio:.2%} ({level})")
        print("=" * 50)
    
    if strategy == 'drop':
        n_before = len(df_clean)
        df_clean = df_clean.dropna()
        n_after = len(df_clean)
        if verbose:
            print(f"已删除 {n_before - n_after} 行 ({(n_before - n_after) / n_before:.2%})")
    
    elif strategy == 'fill' and fill_method:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        if fill_method == 'mean':
            for col in numeric_cols:
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        elif fill_method == 'median':
            for col in numeric_cols:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        elif fill_method == 'mode':
            for col in df_clean.columns:
                mode_val = df_clean[col].mode()
                if len(mode_val) > 0:
                    df_clean[col].fillna(mode_val[0], inplace=True)
        elif fill_method == 'knn':
            try:
                from sklearn.impute import KNNImputer
                imputer = KNNImputer(n_neighbors=5)
                df_numeric = df_clean[numeric_cols]
                df_clean[numeric_cols] = imputer.fit_transform(df_numeric)
            except ImportError:
                warnings.warn("scikit-learn未安装，无法使用KNN填充，改用中位数填充")
                for col in numeric_cols:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        if verbose:
            print(f"已使用 {fill_method} 方法填充缺失值")
    
    return df_clean


def detect_outliers(df: pd.DataFrame, 
                   columns: Optional[List[str]] = None,
                   method: str = 'iqr',
                   visualize: bool = False,
                   figsize: Tuple[int, int] = (12, 4)) -> Dict[str, pd.Series]:
    """
    异常值检测工具。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    columns : list, optional
        要检测的列，默认为所有数值列
    method : str, default='iqr'
        检测方法：'iqr'（四分位距法）或 'zscore'
    visualize : bool, default=False
        是否显示箱线图
    figsize : tuple, default=(12, 4)
        图表大小
        
    Returns
    -------
    dict
        每列的异常值布尔掩码
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_masks = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        series = df[col].dropna()
        
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_masks[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            outlier_masks[col] = z_scores > 3
    
    # 可视化
    if visualize and outlier_masks:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        n_cols = len(columns)
        fig, axes = plt.subplots(1, n_cols, figsize=figsize)
        if n_cols == 1:
            axes = [axes]
        
        for idx, col in enumerate(columns):
            sns.boxplot(y=df[col], ax=axes[idx])
            axes[idx].set_title(f'{col} 箱线图')
        
        plt.tight_layout()
        plt.show()
    
    return outlier_masks


def get_column_pairs(df: pd.DataFrame, 
                     type_mapping: Dict[str, str],
                     target: Optional[str] = None) -> List[Tuple[str, str, str]]:
    """
    获取所有需要计算相关性的列对。
    
    Parameters
    ----------
    df : pd.DataFrame
        输入数据
    type_mapping : dict
        列类型映射
    target : str, optional
        目标变量，如果指定则只计算与目标变量的相关性
        
    Returns
    -------
    list
        列对列表，每个元素为 (col1, col2, pair_type)
    """
    columns = list(df.columns)
    pairs = []
    
    if target:
        # 只计算目标变量与其他变量的相关性
        for col in columns:
            if col != target:
                pair_type = _get_pair_type(type_mapping[target], type_mapping[col])
                pairs.append((target, col, pair_type))
    else:
        # 计算所有变量对的相关性
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                pair_type = _get_pair_type(type_mapping[col1], type_mapping[col2])
                pairs.append((col1, col2, pair_type))
    
    return pairs


def _get_pair_type(type1: str, type2: str) -> str:
    """根据两个变量的类型确定配对类型"""
    type_set = {type1, type2}
    
    if type_set == {'numeric'}:
        return 'numeric_numeric'
    elif 'numeric' in type_set and 'binary' in type_set:
        return 'numeric_binary'
    elif 'numeric' in type_set and 'categorical' in type_set:
        return 'numeric_categorical'
    elif type_set == {'binary'} or type_set == {'categorical'} or \
         ('binary' in type_set and 'categorical' in type_set):
        return 'categorical_categorical'
    elif 'ordinal' in type_set:
        return 'ordinal_ordinal'
    else:
        return 'other'
