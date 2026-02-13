"""
测试主分析器模块
"""

import unittest
import numpy as np
import pandas as pd

from pycorrana.core.analyzer import CorrAnalyzer, quick_corr
from pycorrana.datasets import make_correlated_data


class TestCorrAnalyzer(unittest.TestCase):
    """测试 CorrAnalyzer 类"""
    
    def setUp(self):
        """设置测试数据"""
        self.df = make_correlated_data(n_samples=100, n_features=4, correlation=0.7)
    
    def test_initialization(self):
        """测试初始化"""
        analyzer = CorrAnalyzer(self.df, verbose=False)
        self.assertIsNotNone(analyzer.data)
        self.assertEqual(analyzer.method, 'auto')
    
    def test_preprocess(self):
        """测试预处理"""
        analyzer = CorrAnalyzer(self.df, verbose=False)
        analyzer.preprocess()
        
        self.assertIsNotNone(analyzer.type_mapping)
        self.assertGreater(len(analyzer.type_mapping), 0)
    
    def test_compute_correlation(self):
        """测试相关性计算"""
        analyzer = CorrAnalyzer(self.df, verbose=False)
        analyzer.preprocess()
        analyzer.compute_correlation()
        
        self.assertIsNotNone(analyzer.corr_matrix)
        self.assertIsNotNone(analyzer.pvalue_matrix)
        self.assertEqual(analyzer.corr_matrix.shape[0], len(self.df.columns))
    
    def test_fit(self):
        """测试完整流程"""
        analyzer = CorrAnalyzer(self.df, verbose=False)
        result = analyzer.fit()
        
        self.assertIn('correlation_matrix', result)
        self.assertIn('pvalue_matrix', result)
        self.assertIn('significant_pairs', result)
    
    def test_target_analysis(self):
        """测试目标变量分析"""
        analyzer = CorrAnalyzer(self.df, verbose=False)
        result = analyzer.fit(target='X0')
        
        # 检查是否只包含目标变量相关的行
        self.assertEqual(analyzer.corr_matrix.shape[0], len(self.df.columns))


class TestQuickCorr(unittest.TestCase):
    """测试 quick_corr 函数"""
    
    def setUp(self):
        """设置测试数据"""
        self.df = make_correlated_data(n_samples=100, n_features=4, correlation=0.7)
    
    def test_quick_corr_basic(self):
        """测试基础功能"""
        result = quick_corr(self.df, plot=False, verbose=False)
        
        self.assertIn('correlation_matrix', result)
        self.assertIn('pvalue_matrix', result)
        self.assertIn('significant_pairs', result)
    
    def test_quick_corr_with_target(self):
        """测试指定目标变量"""
        result = quick_corr(self.df, target='X0', plot=False, verbose=False)
        
        self.assertIsNotNone(result['correlation_matrix'])
    
    def test_quick_corr_methods(self):
        """测试不同方法"""
        for method in ['pearson', 'spearman', 'kendall']:
            result = quick_corr(self.df, method=method, plot=False, verbose=False)
            self.assertIsNotNone(result['correlation_matrix'])


class TestDataTypes(unittest.TestCase):
    """测试数据类型处理"""
    
    def test_numeric_numeric(self):
        """测试数值-数值相关"""
        df = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100)
        })
        df['B'] = df['A'] * 0.8 + np.random.randn(100) * 0.2
        
        result = quick_corr(df, plot=False, verbose=False)
        corr_ab = result['correlation_matrix'].loc['A', 'B']
        
        self.assertGreater(abs(corr_ab), 0.5)  # 应该强相关
    
    def test_binary_numeric(self):
        """测试二分类-数值相关"""
        df = pd.DataFrame({
            'binary': np.random.choice([0, 1], 100),
            'numeric': np.random.randn(100)
        })
        
        result = quick_corr(df, plot=False, verbose=False)
        self.assertIsNotNone(result['correlation_matrix'])
    
    def test_categorical_categorical(self):
        """测试分类-分类相关"""
        df = pd.DataFrame({
            'A': np.random.choice(['X', 'Y', 'Z'], 100),
            'B': np.random.choice(['P', 'Q', 'R'], 100)
        })
        
        result = quick_corr(df, plot=False, verbose=False)
        self.assertIsNotNone(result['correlation_matrix'])


if __name__ == '__main__':
    unittest.main()
