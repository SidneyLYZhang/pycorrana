"""
基础使用示例
==============
演示 PyCorrAna 的基本功能
"""

import pandas as pd
import numpy as np

# 导入 PyCorrAna
from pycorrana import quick_corr, CorrAnalyzer
from pycorrana.datasets import load_iris, load_titanic, make_correlated_data


def example_1_quick_start():
    """示例1: 快速开始 - 一行代码完成分析"""
    print("=" * 60)
    print("示例1: 快速开始")
    print("=" * 60)
    
    # 生成测试数据
    df = make_correlated_data(n_samples=200, correlation=0.7)
    
    # 一行代码完成分析
    result = quick_corr(df, plot=False, verbose=True)
    
    print("\n相关性矩阵:")
    print(result['correlation_matrix'].round(3))
    
    print("\n显著相关对 (Top 5):")
    for pair in result['significant_pairs'][:5]:
        print(f"  {pair['var1']} vs {pair['var2']}: {pair['correlation']:.3f}")


def example_2_with_target():
    """示例2: 指定目标变量"""
    print("\n" + "=" * 60)
    print("示例2: 指定目标变量")
    print("=" * 60)
    
    # 使用鸢尾花数据集
    df = load_iris()
    
    # 只分析与 petal_length 的相关性
    result = quick_corr(
        df,
        target='petal_length',
        method='auto',
        plot=False,
        verbose=True
    )
    
    print("\n与 petal_length 显著相关的变量:")
    for pair in result['significant_pairs']:
        other = pair['var2'] if pair['var1'] == 'petal_length' else pair['var1']
        print(f"  {other}: {pair['correlation']:.3f} ({pair['interpretation']})")


def example_3_analyzer_class():
    """示例3: 使用分析器类（更灵活的控制）"""
    print("\n" + "=" * 60)
    print("示例3: 使用分析器类")
    print("=" * 60)
    
    df = make_correlated_data(n_samples=150, n_features=5)
    
    # 创建分析器
    analyzer = CorrAnalyzer(
        df,
        method='spearman',
        missing_strategy='drop',
        pvalue_correction='fdr_bh',
        verbose=True
    )
    
    # 执行分析
    analyzer.fit()
    
    # 获取结果
    print("\n相关系数矩阵:")
    print(analyzer.corr_matrix.round(3))
    
    print("\n使用的方法:")
    for pair, method in list(analyzer.methods_used.items())[:5]:
        print(f"  {pair}: {method}")
    
    # 文本摘要
    print("\n" + analyzer.summary())


def example_4_missing_values():
    """示例4: 缺失值处理"""
    print("\n" + "=" * 60)
    print("示例4: 缺失值处理")
    print("=" * 60)
    
    # 创建含缺失值的数据
    np.random.seed(42)
    df = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.randn(100)
    })
    
    # 随机插入缺失值
    df.loc[np.random.choice(100, 10, replace=False), 'A'] = np.nan
    df.loc[np.random.choice(100, 15, replace=False), 'B'] = np.nan
    
    print(f"原始数据缺失值数量: {df.isnull().sum().sum()}")
    
    # 使用 drop 策略
    result = quick_corr(
        df,
        missing_strategy='drop',
        plot=False,
        verbose=True
    )
    
    print(f"\n处理后数据形状: {result['correlation_matrix'].shape}")


def example_5_partial_corr():
    """示例5: 偏相关分析"""
    print("\n" + "=" * 60)
    print("示例5: 偏相关分析")
    print("=" * 60)
    
    from pycorrana import partial_corr, partial_corr_matrix
    
    # 创建数据
    np.random.seed(42)
    n = 200
    Z = np.random.randn(n)  # 混淆变量
    X = Z * 0.5 + np.random.randn(n)  # X 与 Z 相关
    Y = Z * 0.5 + np.random.randn(n)  # Y 与 Z 相关
    
    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    
    # 简单相关
    simple_corr = df[['X', 'Y']].corr().iloc[0, 1]
    print(f"简单相关系数: {simple_corr:.3f}")
    
    # 偏相关（控制 Z）
    result = partial_corr(df, x='X', y='Y', covars='Z')
    print(f"偏相关系数 (控制 Z): {result['partial_correlation']:.3f}")
    print(f"p值: {result['p_value']:.4f}")
    print(f"95% CI: [{result['ci_95'][0]:.3f}, {result['ci_95'][1]:.3f}]")
    
    # 偏相关矩阵
    print("\n偏相关矩阵 (控制 Z):")
    pcorr_matrix = partial_corr_matrix(df, covars='Z')
    print(pcorr_matrix.round(3))


def example_6_nonlinear():
    """示例6: 非线性依赖检测"""
    print("\n" + "=" * 60)
    print("示例6: 非线性依赖检测")
    print("=" * 60)
    
    from pycorrana import distance_correlation, mutual_info_score
    
    # 创建线性关系数据
    np.random.seed(42)
    n = 200
    x = np.random.randn(n)
    y_linear = x * 2 + np.random.randn(n) * 0.3
    y_quadratic = x**2 + np.random.randn(n) * 0.3
    
    df = pd.DataFrame({
        'x': x,
        'y_linear': y_linear,
        'y_quadratic': y_quadratic
    })
    
    # 线性关系的 Pearson 相关
    pearson_linear = df[['x', 'y_linear']].corr().iloc[0, 1]
    pearson_quad = df[['x', 'y_quadratic']].corr().iloc[0, 1]
    
    print(f"线性关系 - Pearson: {pearson_linear:.3f}")
    print(f"二次关系 - Pearson: {pearson_quad:.3f}")
    
    # 距离相关
    dcor_linear = distance_correlation(df['x'], df['y_linear'])
    dcor_quad = distance_correlation(df['x'], df['y_quadratic'])
    
    print(f"\n线性关系 - dCor: {dcor_linear['dcor']:.3f}")
    print(f"二次关系 - dCor: {dcor_quad['dcor']:.3f}")
    
    # 互信息
    mi_linear = mutual_info_score(df['x'], df['y_linear'])
    mi_quad = mutual_info_score(df['x'], df['y_quadratic'])
    
    print(f"\n线性关系 - MI: {mi_linear['mi_normalized']:.3f}")
    print(f"二次关系 - MI: {mi_quad['mi_normalized']:.3f}")


def example_7_different_methods():
    """示例7: 比较不同相关系数方法"""
    print("\n" + "=" * 60)
    print("示例7: 比较不同相关系数方法")
    print("=" * 60)
    
    # 创建数据
    np.random.seed(42)
    df = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100)
    })
    df['B'] = df['A'] * 0.8 + np.random.randn(100) * 0.2
    
    methods = ['pearson', 'spearman', 'kendall']
    
    print("\n不同方法的相关系数:")
    for method in methods:
        result = quick_corr(df, method=method, plot=False, verbose=False)
        corr = result['correlation_matrix'].loc['A', 'B']
        print(f"  {method:10s}: {corr:.4f}")


def example_8_export_results():
    """示例8: 导出结果"""
    print("\n" + "=" * 60)
    print("示例8: 导出结果")
    print("=" * 60)
    
    df = make_correlated_data(n_samples=100, n_features=4)
    
    # 使用分析器
    analyzer = CorrAnalyzer(df, verbose=False)
    analyzer.fit()
    
    # 导出为 Excel
    analyzer.export_results('correlation_results.xlsx', format='excel')
    print("已导出: correlation_results.xlsx")
    
    # 生成 HTML 报告
    html_content = analyzer.reporter.to_html(
        analyzer.corr_matrix,
        analyzer.pvalue_matrix,
        analyzer.significant_pairs
    )
    with open('correlation_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("已导出: correlation_report.html")
    
    # 生成 Markdown 报告
    md_content = analyzer.reporter.to_markdown(
        analyzer.corr_matrix,
        analyzer.significant_pairs
    )
    with open('correlation_report.md', 'w', encoding='utf-8') as f:
        f.write(md_content)
    print("已导出: correlation_report.md")


if __name__ == '__main__':
    # 运行所有示例
    example_1_quick_start()
    example_2_with_target()
    example_3_analyzer_class()
    example_4_missing_values()
    example_5_partial_corr()
    example_6_nonlinear()
    example_7_different_methods()
    example_8_export_results()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
