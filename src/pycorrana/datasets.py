"""
示例数据集模块
==============
提供内置的经典数据集，方便测试和学习。
"""

import pandas as pd
import numpy as np


def load_iris() -> pd.DataFrame:
    """
    加载鸢尾花数据集。
    
    经典的分类数据集，包含3种鸢尾花的4个特征测量值。
    
    Returns
    -------
    pd.DataFrame
        鸢尾花数据集
        
    Examples
    --------
    >>> from pycorrana.datasets import load_iris
    >>> df = load_iris()
    >>> print(df.head())
    """
    try:
        from sklearn.datasets import load_iris as sklearn_load_iris
        
        iris = sklearn_load_iris()
        df = pd.DataFrame(
            iris.data,
            columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        )
        df['species'] = pd.Categorical.from_codes(
            iris.target, iris.target_names
        )
        return df
    
    except ImportError:
        # 手动创建数据
        np.random.seed(42)
        n = 150
        
        df = pd.DataFrame({
            'sepal_length': np.concatenate([
                np.random.normal(5.0, 0.3, 50),
                np.random.normal(5.9, 0.4, 50),
                np.random.normal(6.5, 0.6, 50)
            ]),
            'sepal_width': np.concatenate([
                np.random.normal(3.4, 0.3, 50),
                np.random.normal(2.8, 0.3, 50),
                np.random.normal(3.0, 0.3, 50)
            ]),
            'petal_length': np.concatenate([
                np.random.normal(1.5, 0.2, 50),
                np.random.normal(4.3, 0.4, 50),
                np.random.normal(5.5, 0.5, 50)
            ]),
            'petal_width': np.concatenate([
                np.random.normal(0.2, 0.05, 50),
                np.random.normal(1.3, 0.2, 50),
                np.random.normal(2.0, 0.3, 50)
            ]),
            'species': ['setosa'] * 50 + ['versicolor'] * 50 + ['virginica'] * 50
        })
        
        return df


def load_titanic() -> pd.DataFrame:
    """
    加载泰坦尼克号数据集。
    
    包含泰坦尼克号乘客的生存情况和相关信息。
    
    Returns
    -------
    pd.DataFrame
        泰坦尼克号数据集
        
    Examples
    --------
    >>> from pycorrana.datasets import load_titanic
    >>> df = load_titanic()
    >>> print(df.head())
    """
    try:
        import seaborn as sns
        return sns.load_dataset('titanic')
    except:
        # 创建简化版数据
        np.random.seed(42)
        n = 891
        
        df = pd.DataFrame({
            'survived': np.random.choice([0, 1], n, p=[0.6, 0.4]),
            'pclass': np.random.choice([1, 2, 3], n, p=[0.2, 0.2, 0.6]),
            'sex': np.random.choice(['male', 'female'], n, p=[0.65, 0.35]),
            'age': np.random.normal(30, 14, n).clip(0, 80),
            'sibsp': np.random.choice([0, 1, 2, 3, 4, 5, 8], n, 
                                     p=[0.68, 0.23, 0.03, 0.02, 0.01, 0.01, 0.02]),
            'parch': np.random.choice([0, 1, 2, 3, 4, 5, 6], n,
                                     p=[0.76, 0.13, 0.09, 0.01, 0.005, 0.005, 0.01]),
            'fare': np.random.exponential(32, n).clip(0, 512),
            'embarked': np.random.choice(['S', 'C', 'Q'], n, p=[0.7, 0.2, 0.1]),
            'class': np.random.choice(['First', 'Second', 'Third'], n, p=[0.2, 0.2, 0.6]),
            'who': np.random.choice(['man', 'woman', 'child'], n, p=[0.6, 0.3, 0.1]),
            'adult_male': np.random.choice([True, False], n, p=[0.6, 0.4]),
            'deck': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', None], n),
            'embark_town': np.random.choice(
                ['Southampton', 'Cherbourg', 'Queenstown'], n, p=[0.7, 0.2, 0.1]
            ),
            'alive': np.random.choice(['no', 'yes'], n, p=[0.6, 0.4]),
            'alone': np.random.choice([True, False], n, p=[0.6, 0.4]),
        })
        
        # 添加一些相关性
        df.loc[df['sex'] == 'female', 'survived'] = np.random.choice(
            [0, 1], (df['sex'] == 'female').sum(), p=[0.25, 0.75]
        )
        df.loc[df['pclass'] == 1, 'fare'] *= 3
        df.loc[df['pclass'] == 2, 'fare'] *= 1.5
        
        return df


def load_wine() -> pd.DataFrame:
    """
    加载葡萄酒数据集。
    
    包含葡萄酒的化学分析数据和类别标签。
    
    Returns
    -------
    pd.DataFrame
        葡萄酒数据集
    """
    try:
        from sklearn.datasets import load_wine as sklearn_load_wine
        
        wine = sklearn_load_wine()
        df = pd.DataFrame(wine.data, columns=wine.feature_names)
        df['target'] = wine.target
        return df
    
    except ImportError:
        np.random.seed(42)
        n = 178
        
        df = pd.DataFrame({
            'alcohol': np.concatenate([
                np.random.normal(13.7, 0.5, 59),
                np.random.normal(12.3, 0.4, 71),
                np.random.normal(13.1, 0.5, 48)
            ]),
            'malic_acid': np.random.exponential(2.5, n).clip(0.7, 5.8),
            'ash': np.random.normal(2.4, 0.3, n).clip(1.3, 3.2),
            'alcalinity_of_ash': np.random.normal(19.5, 3, n).clip(10, 30),
            'magnesium': np.random.normal(100, 15, n).clip(70, 162),
            'total_phenols': np.random.normal(2.3, 0.6, n).clip(0.9, 3.9),
            'flavanoids': np.random.normal(2.0, 1.0, n).clip(0.3, 5.1),
            'nonflavanoid_phenols': np.random.normal(0.36, 0.12, n).clip(0.1, 0.7),
            'proanthocyanins': np.random.normal(1.6, 0.6, n).clip(0.4, 3.6),
            'color_intensity': np.random.exponential(5, n).clip(1, 13),
            'hue': np.random.normal(0.96, 0.2, n).clip(0.4, 1.7),
            'od280/od315_of_diluted_wines': np.random.normal(2.6, 0.7, n).clip(1.2, 4.0),
            'proline': np.random.exponential(750, n).clip(270, 1680),
            'target': [0] * 59 + [1] * 71 + [2] * 48
        })
        
        return df


def make_correlated_data(n_samples: int = 500,
                        n_features: int = 5,
                        correlation: float = 0.7,
                        noise: float = 0.1,
                        random_state: int = 42) -> pd.DataFrame:
    """
    生成具有指定相关性的测试数据。
    
    Parameters
    ----------
    n_samples : int, default=500
        样本数量
    n_features : int, default=5
        特征数量
    correlation : float, default=0.7
        基础相关性强度
    noise : float, default=0.1
        噪声水平
    random_state : int, default=42
        随机种子
        
    Returns
    -------
    pd.DataFrame
        生成的数据集
        
    Examples
    --------
    >>> from pycorrana.datasets import make_correlated_data
    >>> df = make_correlated_data(n_samples=1000, correlation=0.8)
    >>> print(df.corr())
    """
    np.random.seed(random_state)
    
    # 生成基础变量
    x0 = np.random.randn(n_samples)
    
    data = {'X0': x0}
    
    # 生成相关变量
    for i in range(1, n_features):
        # 新变量 = 相关部分 + 独立部分
        xi = correlation * x0 + np.random.randn(n_samples) * np.sqrt(1 - correlation**2)
        xi += np.random.randn(n_samples) * noise  # 添加噪声
        data[f'X{i}'] = xi
    
    # 添加一些非线性关系
    data['Y_linear'] = 2 * x0 + np.random.randn(n_samples) * 0.5
    data['Y_quadratic'] = x0**2 + np.random.randn(n_samples) * 0.5
    data['Y_sinusoidal'] = np.sin(x0 * 2) + np.random.randn(n_samples) * 0.3
    
    # 添加分类变量
    data['Category'] = np.random.choice(['A', 'B', 'C'], n_samples)
    data['Binary'] = np.random.choice([0, 1], n_samples)
    
    return pd.DataFrame(data)


# 数据集信息
DATASET_INFO = {
    'iris': {
        'name': '鸢尾花数据集',
        'description': '包含3种鸢尾花的4个特征测量值',
        'n_samples': 150,
        'n_features': 4,
        'task': '分类',
        'source': 'UCI Machine Learning Repository'
    },
    'titanic': {
        'name': '泰坦尼克号数据集',
        'description': '泰坦尼克号乘客的生存情况和相关信息',
        'n_samples': 891,
        'n_features': 15,
        'task': '分类',
        'source': 'Kaggle'
    },
    'wine': {
        'name': '葡萄酒数据集',
        'description': '葡萄酒的化学分析数据和类别标签',
        'n_samples': 178,
        'n_features': 13,
        'task': '分类',
        'source': 'UCI Machine Learning Repository'
    }
}


def list_datasets():
    """列出所有可用数据集"""
    print("可用数据集:")
    print("-" * 60)
    for key, info in DATASET_INFO.items():
        print(f"\n{key}:")
        print(f"  名称: {info['name']}")
        print(f"  描述: {info['description']}")
        print(f"  样本数: {info['n_samples']}")
        print(f"  特征数: {info['n_features']}")
        print(f"  任务类型: {info['task']}")
        print(f"  来源: {info['source']}")
