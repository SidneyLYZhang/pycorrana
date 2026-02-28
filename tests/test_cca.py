"""
测试典型相关分析模块
"""

import unittest
import numpy as np
import pandas as pd

from pycorrana.core.cca import cca, cca_permutation_test, CCAAnalyzer
from pycorrana.core.analyzer import CorrAnalyzer


class TestCCA(unittest.TestCase):
    """测试CCA函数"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        n = 100
        
        self.X = np.random.randn(n, 3)
        self.Y = np.random.randn(n, 2)
        
        self.Y[:, 0] = 0.8 * self.X[:, 0] + 0.2 * np.random.randn(n)
        
        self.df_X = pd.DataFrame(self.X, columns=['X1', 'X2', 'X3'])
        self.df_Y = pd.DataFrame(self.Y, columns=['Y1', 'Y2'])
    
    def test_cca_basic(self):
        """测试基础CCA功能"""
        result = cca(self.X, self.Y, verbose=False)
        
        self.assertIn('canonical_correlations', result)
        self.assertIn('x_weights', result)
        self.assertIn('y_weights', result)
        self.assertIn('x_scores', result)
        self.assertIn('y_scores', result)
        
        k = min(self.X.shape[1], self.Y.shape[1])
        self.assertEqual(len(result['canonical_correlations']), k)
    
    def test_cca_with_dataframe(self):
        """测试使用DataFrame输入"""
        result = cca(self.df_X, self.df_Y, verbose=False)
        
        self.assertEqual(result['x_names'], ['X1', 'X2', 'X3'])
        self.assertEqual(result['y_names'], ['Y1', 'Y2'])
    
    def test_cca_significance(self):
        """测试显著性检验"""
        result = cca(self.X, self.Y, compute_significance=True, verbose=False)
        
        self.assertIn('significance_tests', result)
        self.assertGreater(len(result['significance_tests']), 0)
        
        for test in result['significance_tests']:
            self.assertIn('canonical_correlation', test)
            self.assertIn('wilks_lambda', test)
            self.assertIn('chi_square', test)
            self.assertIn('p_value', test)
    
    def test_cca_confidence_intervals(self):
        """测试置信区间"""
        result = cca(self.X, self.Y, confidence_level=0.95, verbose=False)
        
        self.assertIn('confidence_intervals', result)
        
        for ci in result['confidence_intervals']:
            self.assertEqual(len(ci), 2)
            self.assertLessEqual(ci[0], ci[1])
            self.assertGreaterEqual(ci[0], 0)
            self.assertLessEqual(ci[1], 1)
    
    def test_cca_redundancy(self):
        """测试冗余分析"""
        result = cca(self.X, self.Y, verbose=False)
        
        self.assertIn('redundancy', result)
        self.assertIn('x_given_y', result['redundancy'])
        self.assertIn('y_given_x', result['redundancy'])
        self.assertIn('total_x_given_y', result['redundancy'])
        self.assertIn('total_y_given_x', result['redundancy'])
    
    def test_cca_correlation_range(self):
        """测试典型相关系数范围"""
        result = cca(self.X, self.Y, verbose=False)
        
        for r in result['canonical_correlations']:
            self.assertGreaterEqual(r, 0)
            self.assertLessEqual(r, 1)
    
    def test_cca_sample_size(self):
        """测试样本量不足时的处理"""
        X_small = np.random.randn(2, 3)
        Y_small = np.random.randn(2, 2)
        
        with self.assertRaises(ValueError):
            cca(X_small, Y_small, verbose=False)


class TestCCAPermutationTest(unittest.TestCase):
    """测试CCA置换检验"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        n = 50
        
        self.X = np.random.randn(n, 2)
        self.Y = np.random.randn(n, 2)
    
    def test_permutation_test_basic(self):
        """测试基础置换检验"""
        result = cca_permutation_test(
            self.X, self.Y,
            n_permutations=100,
            random_state=42
        )
        
        self.assertIn('observed_correlations', result)
        self.assertIn('p_values', result)
        self.assertIn('n_permutations', result)
        
        self.assertEqual(result['n_permutations'], 100)
        self.assertEqual(len(result['p_values']), len(result['observed_correlations']))
    
    def test_permutation_test_p_value_range(self):
        """测试p值范围"""
        result = cca_permutation_test(
            self.X, self.Y,
            n_permutations=100,
            random_state=42
        )
        
        for p in result['p_values']:
            self.assertGreaterEqual(p, 0)
            self.assertLessEqual(p, 1)


class TestCCAAnalyzer(unittest.TestCase):
    """测试CCAAnalyzer类"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        n = 100
        
        X = np.random.randn(n, 3)
        Y = np.random.randn(n, 2)
        Y[:, 0] = 0.8 * X[:, 0] + 0.2 * np.random.randn(n)
        
        self.df_X = pd.DataFrame(X, columns=['X1', 'X2', 'X3'])
        self.df_Y = pd.DataFrame(Y, columns=['Y1', 'Y2'])
    
    def test_analyzer_initialization(self):
        """测试初始化"""
        analyzer = CCAAnalyzer(self.df_X, self.df_Y, verbose=False)
        
        self.assertIsNotNone(analyzer.X)
        self.assertIsNotNone(analyzer.Y)
    
    def test_analyzer_fit(self):
        """测试fit方法"""
        analyzer = CCAAnalyzer(self.df_X, self.df_Y, verbose=False)
        result = analyzer.fit()
        
        self.assertIsNotNone(analyzer.result)
        self.assertIn('canonical_correlations', result)
    
    def test_analyzer_summary(self):
        """测试summary方法"""
        analyzer = CCAAnalyzer(self.df_X, self.df_Y, verbose=False)
        analyzer.fit()
        
        summary = analyzer.summary()
        
        self.assertIsInstance(summary, str)
        self.assertIn('典型相关系数', summary)
        self.assertIn('冗余分析', summary)
    
    def test_analyzer_get_weights(self):
        """测试get_weights方法"""
        analyzer = CCAAnalyzer(self.df_X, self.df_Y, verbose=False)
        analyzer.fit()
        
        x_weights, y_weights = analyzer.get_weights(component=1)
        
        self.assertIsInstance(x_weights, pd.DataFrame)
        self.assertIsInstance(y_weights, pd.DataFrame)
    
    def test_analyzer_get_scores(self):
        """测试get_scores方法"""
        analyzer = CCAAnalyzer(self.df_X, self.df_Y, verbose=False)
        analyzer.fit()
        
        x_scores, y_scores = analyzer.get_scores()
        
        self.assertIsInstance(x_scores, pd.DataFrame)
        self.assertIsInstance(y_scores, pd.DataFrame)
        self.assertEqual(len(x_scores), len(self.df_X))
    
    def test_analyzer_invalid_component(self):
        """测试无效的典型变量序号"""
        analyzer = CCAAnalyzer(self.df_X, self.df_Y, verbose=False)
        analyzer.fit()
        
        with self.assertRaises(ValueError):
            analyzer.get_weights(component=10)


class TestCorrAnalyzerCCA(unittest.TestCase):
    """测试CorrAnalyzer中的CCA集成"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        n = 100
        
        data = np.random.randn(n, 5)
        data[:, 3] = 0.8 * data[:, 0] + 0.2 * np.random.randn(n)
        
        self.df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])
    
    def test_corr_analyzer_cca(self):
        """测试CorrAnalyzer的CCA方法"""
        analyzer = CorrAnalyzer(self.df, verbose=False)
        
        result = analyzer.cca(
            x_vars=['A', 'B'],
            y_vars=['D', 'E'],
            compute_significance=True
        )
        
        self.assertIn('canonical_correlations', result)
        self.assertIn('significance_tests', result)
        self.assertIn('confidence_intervals', result)
    
    def test_corr_analyzer_cca_summary(self):
        """测试CorrAnalyzer的CCA摘要方法"""
        analyzer = CorrAnalyzer(self.df, verbose=False)
        
        summary = analyzer.cca_summary(
            x_vars=['A', 'B'],
            y_vars=['D', 'E']
        )
        
        self.assertIsInstance(summary, str)
        self.assertIn('典型相关系数', summary)
    
    def test_corr_analyzer_cca_invalid_vars(self):
        """测试无效变量名"""
        analyzer = CorrAnalyzer(self.df, verbose=False)
        
        with self.assertRaises(ValueError):
            analyzer.cca(x_vars=['X1'], y_vars=['D'])
        
        with self.assertRaises(ValueError):
            analyzer.cca(x_vars=['A'], y_vars=['Y1'])


class TestCCAResults(unittest.TestCase):
    """测试CCA结果的正确性"""
    
    def test_known_correlation(self):
        """测试已知相关性的数据"""
        np.random.seed(42)
        n = 200
        
        X = np.random.randn(n, 2)
        Y = np.zeros((n, 2))
        
        Y[:, 0] = X[:, 0] + 0.1 * np.random.randn(n)
        Y[:, 1] = X[:, 1] + 0.1 * np.random.randn(n)
        
        result = cca(X, Y, verbose=False)
        
        self.assertGreater(result['canonical_correlations'][0], 0.9)
    
    def test_independent_variables(self):
        """测试独立变量"""
        np.random.seed(42)
        n = 200
        
        X = np.random.randn(n, 2)
        Y = np.random.randn(n, 2)
        
        result = cca(X, Y, verbose=False)
        
        self.assertLess(result['canonical_correlations'][0], 0.3)
        self.assertLess(result['redundancy']['total_x_given_y'], 0.1)
    
    def test_weights_normalization(self):
        """测试权重归一化"""
        np.random.seed(42)
        n = 100
        
        X = np.random.randn(n, 3)
        Y = np.random.randn(n, 2)
        
        result = cca(X, Y, verbose=False)
        
        X_centered = X - X.mean(axis=0)
        Y_centered = Y - Y.mean(axis=0)
        
        for i in range(result['n_components']):
            scores_x = X_centered @ result['x_weights'][:, i]
            self.assertAlmostEqual(np.std(scores_x), 1.0, places=1)


if __name__ == '__main__':
    unittest.main()
