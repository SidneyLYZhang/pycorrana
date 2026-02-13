#!/usr/bin/env python
"""
PyCorrAna 演示脚本
==================
展示 PyCorrAna 的核心功能
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# 导入 PyCorrAna 模块
from pycorrana import quick_corr, CorrAnalyzer
from pycorrana.core.partial_corr import partial_corr, partial_corr_matrix
from pycorrana.core.nonlinear import distance_correlation, mutual_info_score
from pycorrana.datasets import make_correlated_data, load_iris
from pycorrana.utils.data_utils import infer_types, handle_missing


def demo_basic_analysis():
    """演示基础分析功能"""
    print("\n" + "=" * 70)
    print(" " * 20 + "PyCorrAna 功能演示")
    print("=" * 70)
    
    print("\n📊 演示1: 基础相关性分析")
    print("-" * 70)
    
    # 生成测试数据
    df = make_correlated_data(n_samples=200, n_features=5, correlation=0.6)
    print(f"\n生成测试数据: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"列名: {', '.join(df.columns)}")
    
    # 一键分析
    print("\n执行 quick_corr() 分析...")
    result = quick_corr(df, plot=False, verbose=True)
    
    # 显示结果
    print("\n📈 相关系数矩阵:")
    print(result['correlation_matrix'].round(3).to_string())
    
    print("\n🔍 Top 5 显著相关对:")
    for i, pair in enumerate(result['significant_pairs'][:5], 1):
        print(f"  {i}. {pair['var1']} vs {pair['var2']}")
        print(f"     相关系数: {pair['correlation']:.4f}")
        print(f"     p值: {pair['p_value']:.2e}")
        print(f"     方法: {pair['method']}")
        print(f"     解释: {pair['interpretation']}")


def demo_auto_method_selection():
    """演示自动方法选择"""
    print("\n" + "=" * 70)
    print("📊 演示2: 自动方法选择")
    print("-" * 70)
    
    # 创建不同类型的变量
    np.random.seed(42)
    n = 150
    
    df = pd.DataFrame({
        'numeric1': np.random.randn(n),
        'numeric2': np.random.randn(n),
        'binary': np.random.choice([0, 1], n),
        'category': np.random.choice(['A', 'B', 'C'], n),
    })
    
    # 添加相关性
    df['numeric2'] = df['numeric1'] * 0.7 + np.random.randn(n) * 0.5
    
    print("\n数据类型:")
    type_mapping = infer_types(df)
    for col, t in type_mapping.items():
        print(f"  {col}: {t}")
    
    print("\n执行自动分析...")
    analyzer = CorrAnalyzer(df, method='auto', verbose=True)
    analyzer.preprocess()
    analyzer.compute_correlation()
    
    print("\n每对变量使用的方法:")
    for pair, method in analyzer.methods_used.items():
        print(f"  {pair}: {method}")


def demo_partial_correlation():
    """演示偏相关分析"""
    print("\n" + "=" * 70)
    print("📊 演示3: 偏相关分析")
    print("-" * 70)
    
    # 创建有混淆变量的数据
    np.random.seed(42)
    n = 200
    
    # Z 是混淆变量
    Z = np.random.randn(n)
    X = Z * 0.6 + np.random.randn(n)  # X 与 Z 相关
    Y = Z * 0.6 + np.random.randn(n)  # Y 与 Z 相关
    
    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    
    print("\n数据生成: X 和 Y 都受 Z 影响")
    
    # 简单相关
    simple_corr = df[['X', 'Y']].corr().iloc[0, 1]
    print(f"\n简单相关系数 (X vs Y): {simple_corr:.4f}")
    
    # 偏相关
    result = partial_corr(df, x='X', y='Y', covars='Z')
    print(f"偏相关系数 (X vs Y, 控制 Z): {result['partial_correlation']:.4f}")
    print(f"p值: {result['p_value']:.4e}")
    print(f"95% 置信区间: [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
    
    print("\n说明: 控制 Z 后，X 和 Y 的相关性显著降低，说明之前的相关主要由 Z 引起")


def demo_nonlinear_detection():
    """演示非线性依赖检测"""
    print("\n" + "=" * 70)
    print("📊 演示4: 非线性依赖检测")
    print("-" * 70)
    
    np.random.seed(42)
    n = 200
    x = np.random.randn(n)
    
    # 线性关系
    y_linear = x * 2 + np.random.randn(n) * 0.3
    
    # 二次关系（非线性）
    y_quad = x**2 + np.random.randn(n) * 0.3
    
    df = pd.DataFrame({
        'x': x,
        'y_linear': y_linear,
        'y_quadratic': y_quad
    })
    
    print("\n比较线性关系和二次关系:")
    
    # Pearson 相关
    pearson_linear = df[['x', 'y_linear']].corr().iloc[0, 1]
    pearson_quad = df[['x', 'y_quadratic']].corr().iloc[0, 1]
    
    print(f"\nPearson 相关系数:")
    print(f"  线性关系: {pearson_linear:.4f}")
    print(f"  二次关系: {pearson_quad:.4f}")
    
    # 距离相关
    dcor_linear = distance_correlation(df['x'], df['y_linear'])
    dcor_quad = distance_correlation(df['x'], df['y_quadratic'])
    
    print(f"\n距离相关系数 (dCor):")
    print(f"  线性关系: {dcor_linear['dcor']:.4f}")
    print(f"  二次关系: {dcor_quad['dcor']:.4f}")
    
    # 互信息
    mi_linear = mutual_info_score(df['x'], df['y_linear'])
    mi_quad = mutual_info_score(df['x'], df['y_quadratic'])
    
    print(f"\n归一化互信息 (MI):")
    print(f"  线性关系: {mi_linear['mi_normalized']:.4f}")
    print(f"  二次关系: {mi_quad['mi_normalized']:.4f}")
    
    print("\n说明: dCor 和 MI 能更好地检测非线性关系")


def demo_missing_value_handling():
    """演示缺失值处理"""
    print("\n" + "=" * 70)
    print("📊 演示5: 缺失值处理")
    print("-" * 70)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.randn(100)
    })
    
    # 随机插入缺失值
    missing_idx_A = np.random.choice(100, 10, replace=False)
    missing_idx_B = np.random.choice(100, 15, replace=False)
    df.loc[missing_idx_A, 'A'] = np.nan
    df.loc[missing_idx_B, 'B'] = np.nan
    
    print(f"\n原始数据缺失值:")
    print(f"  A列: {df['A'].isnull().sum()} 个")
    print(f"  B列: {df['B'].isnull().sum()} 个")
    print(f"  C列: {df['C'].isnull().sum()} 个")
    
    # 使用中位数填充
    df_filled = handle_missing(df, strategy='fill', fill_method='median', verbose=True)
    
    print(f"\n填充后缺失值:")
    print(f"  A列: {df_filled['A'].isnull().sum()} 个")
    print(f"  B列: {df_filled['B'].isnull().sum()} 个")
    print(f"  C列: {df_filled['C'].isnull().sum()} 个")


def demo_real_dataset():
    """演示真实数据集分析"""
    print("\n" + "=" * 70)
    print("📊 演示6: 真实数据集分析 (Iris)")
    print("-" * 70)
    
    df = load_iris()
    print(f"\nIris 数据集: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"特征: {', '.join(df.columns[:-1])}")
    print(f"类别: {df['species'].unique()}")
    
    # 分析
    result = quick_corr(df, plot=False, verbose=True)
    
    print("\n特征间相关性 (Top 5):")
    for i, pair in enumerate(result['significant_pairs'][:5], 1):
        if 'species' not in [pair['var1'], pair['var2']]:
            print(f"  {i}. {pair['var1']} vs {pair['var2']}: {pair['correlation']:.4f}")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("  PyCorrAna - Python Correlation Analysis Toolkit")
    print("  自动化相关性分析工具")
    print("=" * 70)
    
    # 运行所有演示
    demo_basic_analysis()
    demo_auto_method_selection()
    demo_partial_correlation()
    demo_nonlinear_detection()
    demo_missing_value_handling()
    demo_real_dataset()
    
    print("\n" + "=" * 70)
    print(" " * 20 + "演示完成！")
    print("=" * 70)
    print("\n更多功能请查看:")
    print("  - 示例代码: examples/basic_usage.py")
    print("  - 交互式工具: pycorrana-interactive")
    print("  - 命令行工具: pycorrana --help")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
