"""
基础使用示例
============
演示 PyCorrAna 的基本功能
"""

import os
import tempfile

import numpy as np
import pandas as pd

from pycorrana import (
    quick_corr,
    CorrAnalyzer,
    partial_corr,
    partial_corr_matrix,
    semipartial_corr,
    distance_correlation,
    mutual_info_score,
    maximal_information_coefficient,
    NonlinearAnalyzer,
)
from pycorrana.datasets import load_iris, make_correlated_data


def _print_header(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def _print_subheader(title: str):
    print("\n" + "-" * 40)
    print(f" {title}")
    print("-" * 40)


def example_1_quick_start():
    _print_header("示例1: 快速开始 - 一行代码完成分析")
    
    df = make_correlated_data(n_samples=200, correlation=0.7)
    
    result = quick_corr(df, plot=False, verbose=True)
    
    _print_subheader("相关性矩阵")
    print(result['correlation_matrix'].round(3))
    
    _print_subheader("显著相关对 (Top 5)")
    for pair in result['significant_pairs'][:5]:
        print(f"  {pair['var1']} vs {pair['var2']}: {pair['correlation']:.3f}")


def example_2_with_target():
    _print_header("示例2: 指定目标变量")
    
    df = load_iris()
    
    result = quick_corr(
        df,
        target='petal_length',
        method='auto',
        plot=False,
        verbose=True
    )
    
    _print_subheader("与 petal_length 显著相关的变量")
    for pair in result['significant_pairs']:
        other = pair['var2'] if pair['var1'] == 'petal_length' else pair['var1']
        print(f"  {other}: {pair['correlation']:.3f} ({pair['interpretation']})")


def example_3_analyzer_class():
    _print_header("示例3: 使用分析器类（更灵活的控制）")
    
    df = make_correlated_data(n_samples=150, n_features=5)
    
    analyzer = CorrAnalyzer(
        df,
        method='spearman',
        missing_strategy='drop',
        pvalue_correction='fdr_bh',
        verbose=True
    )
    
    analyzer.fit()
    
    _print_subheader("相关系数矩阵")
    print(analyzer.corr_matrix.round(3))
    
    _print_subheader("使用的方法")
    for pair, method in list(analyzer.methods_used.items())[:5]:
        print(f"  {pair}: {method}")
    
    _print_subheader("文本摘要")
    print(analyzer.summary())


def example_4_missing_values():
    _print_header("示例4: 缺失值处理")
    
    np.random.seed(42)
    df = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.randn(100)
    })
    
    df.loc[np.random.choice(100, 10, replace=False), 'A'] = np.nan
    df.loc[np.random.choice(100, 15, replace=False), 'B'] = np.nan
    
    print(f"原始数据缺失值数量: {df.isnull().sum().sum()}")
    
    result = quick_corr(
        df,
        missing_strategy='drop',
        plot=False,
        verbose=True
    )
    
    print(f"\n处理后数据形状: {result['correlation_matrix'].shape}")


def example_5_partial_corr():
    _print_header("示例5: 偏相关分析")
    
    np.random.seed(42)
    n = 200
    Z = np.random.randn(n)
    X = Z * 0.5 + np.random.randn(n)
    Y = Z * 0.5 + np.random.randn(n)
    
    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    
    simple_corr = df[['X', 'Y']].corr().iloc[0, 1]
    print(f"简单相关系数: {simple_corr:.3f}")
    
    result = partial_corr(df, x='X', y='Y', covars='Z')
    print(f"偏相关系数 (控制 Z): {result['partial_correlation']:.3f}")
    print(f"p值: {result['p_value']:.4f}")
    print(f"95% CI: [{result['ci_95'][0]:.3f}, {result['ci_95'][1]:.3f}]")
    
    _print_subheader("偏相关矩阵 (控制 Z)")
    pcorr_matrix = partial_corr_matrix(df, covars='Z')
    print(pcorr_matrix.round(3))


def example_6_semipartial_corr():
    _print_header("示例6: 半偏相关分析")
    
    np.random.seed(42)
    n = 150
    Z = np.random.randn(n)
    X = Z * 0.6 + np.random.randn(n)
    Y = X * 0.8 + Z * 0.3 + np.random.randn(n)
    
    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    
    simple_corr = df[['X', 'Y']].corr().iloc[0, 1]
    print(f"简单相关系数 (X vs Y): {simple_corr:.3f}")
    
    result = semipartial_corr(df, x='X', y='Y', covars='Z')
    print(f"半偏相关系数 (X vs Y, 从X中移除Z的影响): {result['semipartial_correlation']:.3f}")
    print(f"p值: {result['p_value']:.4f}")


def example_7_nonlinear():
    _print_header("示例7: 非线性依赖检测")
    
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
    
    pearson_linear = df[['x', 'y_linear']].corr().iloc[0, 1]
    pearson_quad = df[['x', 'y_quadratic']].corr().iloc[0, 1]
    
    print(f"线性关系 - Pearson: {pearson_linear:.3f}")
    print(f"二次关系 - Pearson: {pearson_quad:.3f}")
    
    dcor_linear = distance_correlation(df['x'], df['y_linear'])
    dcor_quad = distance_correlation(df['x'], df['y_quadratic'])
    
    print(f"\n线性关系 - dCor: {dcor_linear['dcor']:.3f}")
    print(f"二次关系 - dCor: {dcor_quad['dcor']:.3f}")
    
    mi_linear = mutual_info_score(df['x'], df['y_linear'])
    mi_quad = mutual_info_score(df['x'], df['y_quadratic'])
    
    print(f"\n线性关系 - MI: {mi_linear['mi_normalized']:.3f}")
    print(f"二次关系 - MI: {mi_quad['mi_normalized']:.3f}")


def example_8_mic():
    _print_header("示例8: 最大信息系数 (MIC)")
    
    np.random.seed(42)
    n = 200
    x = np.linspace(-3, 3, n)
    y_sine = np.sin(x) + np.random.randn(n) * 0.2
    y_linear = x + np.random.randn(n) * 0.5
    
    df = pd.DataFrame({'x': x, 'y_sine': y_sine, 'y_linear': y_linear})
    
    pearson_sine = df[['x', 'y_sine']].corr().iloc[0, 1]
    pearson_linear = df[['x', 'y_linear']].corr().iloc[0, 1]
    
    print(f"正弦关系 - Pearson: {pearson_sine:.3f}")
    print(f"线性关系 - Pearson: {pearson_linear:.3f}")
    
    mic_sine = maximal_information_coefficient(df['x'], df['y_sine'])
    mic_linear = maximal_information_coefficient(df['x'], df['y_linear'])
    
    print(f"\n正弦关系 - MIC: {mic_sine['mic']:.3f}")
    print(f"线性关系 - MIC: {mic_linear['mic']:.3f}")
    print("\n说明: MIC 能更好地检测非单调非线性关系")


def example_9_nonlinear_analyzer():
    _print_header("示例9: NonlinearAnalyzer 类")
    
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        'x': np.random.randn(n),
        'y1': np.random.randn(n),
        'y2': np.random.randn(n)
    })
    df['y1'] = df['x'] * 0.8 + np.random.randn(n) * 0.3
    df['y2'] = df['x']**2 + np.random.randn(n) * 0.3
    
    analyzer = NonlinearAnalyzer(df, verbose=True)
    analyzer.fit()
    
    _print_subheader("非线性依赖报告")
    report = analyzer.get_report()
    for pair, info in list(report.items())[:3]:
        print(f"\n{pair}:")
        print(f"  dCor: {info.get('dcor', 'N/A'):.3f}" if isinstance(info.get('dcor'), float) else f"  dCor: {info.get('dcor', 'N/A')}")
        print(f"  MI: {info.get('mi_normalized', 'N/A'):.3f}" if isinstance(info.get('mi_normalized'), float) else f"  MI: {info.get('mi_normalized', 'N/A')}")


def example_10_different_methods():
    _print_header("示例10: 比较不同相关系数方法")
    
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


def example_11_export_results():
    _print_header("示例11: 导出结果")
    
    df = make_correlated_data(n_samples=100, n_features=4)
    
    analyzer = CorrAnalyzer(df, verbose=False)
    analyzer.fit()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        excel_path = os.path.join(tmpdir, 'correlation_results.xlsx')
        analyzer.export_results(excel_path, format='excel')
        print(f"已导出: {excel_path}")
        
        html_path = os.path.join(tmpdir, 'correlation_report.html')
        html_content = analyzer.reporter.to_html(
            analyzer.corr_matrix,
            analyzer.pvalue_matrix,
            analyzer.significant_pairs
        )
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"已导出: {html_path}")
        
        md_path = os.path.join(tmpdir, 'correlation_report.md')
        md_content = analyzer.reporter.to_markdown(
            analyzer.corr_matrix,
            analyzer.significant_pairs
        )
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"已导出: {md_path}")
    
    print("\n说明: 使用临时目录演示导出功能，实际使用时可指定具体路径")


def main():
    examples = [
        ("快速开始", example_1_quick_start),
        ("指定目标变量", example_2_with_target),
        ("分析器类", example_3_analyzer_class),
        ("缺失值处理", example_4_missing_values),
        ("偏相关分析", example_5_partial_corr),
        ("半偏相关分析", example_6_semipartial_corr),
        ("非线性依赖检测", example_7_nonlinear),
        ("最大信息系数", example_8_mic),
        ("NonlinearAnalyzer类", example_9_nonlinear_analyzer),
        ("比较不同方法", example_10_different_methods),
        ("导出结果", example_11_export_results),
    ]
    
    print("\n" + "=" * 60)
    print(" PyCorrAna 基础使用示例")
    print("=" * 60)
    print("\n可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i:2d}. {name}")
    
    for _, func in examples:
        func()
    
    print("\n" + "=" * 60)
    print(" 所有示例运行完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
