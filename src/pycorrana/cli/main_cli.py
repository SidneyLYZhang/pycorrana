"""
命令行工具主模块
================
提供分模块的CLI工具，支持数据流水线各阶段的操作。
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from ..core.analyzer import quick_corr, CorrAnalyzer
from ..core.partial_corr import partial_corr, partial_corr_matrix
from ..core.nonlinear import distance_correlation, mutual_info_score, nonlinear_dependency_report
from ..utils.data_utils import load_data, infer_types, handle_missing, detect_outliers


def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog='pycorrana',
        description='PyCorrAna - Python Correlation Analysis Toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整分析
  pycorrana analyze data.csv --target sales --export results.xlsx
  
  # 数据清洗
  pycorrana clean data.csv --dropna --output cleaned.csv
  
  # 偏相关分析
  pycorrana partial data.csv -x income -y happiness -c age,education
  
  # 非线性检测
  pycorrana nonlinear data.csv --top 20
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # ========== analyze 命令 ==========
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='执行完整的相关性分析'
    )
    analyze_parser.add_argument('input', help='输入文件路径 (CSV/Excel)')
    analyze_parser.add_argument('--target', '-t', help='目标变量')
    analyze_parser.add_argument('--columns', '-c', help='要分析的列，逗号分隔')
    analyze_parser.add_argument('--method', '-m', default='auto',
                               choices=['auto', 'pearson', 'spearman', 'kendall'],
                               help='相关性方法 (默认: auto)')
    analyze_parser.add_argument('--missing', default='warn',
                               choices=['warn', 'drop', 'fill'],
                               help='缺失值处理策略 (默认: warn)')
    analyze_parser.add_argument('--fill-method', default='median',
                               choices=['mean', 'median', 'mode', 'knn'],
                               help='填充方法 (默认: median)')
    analyze_parser.add_argument('--pvalue-correction', default='fdr_bh',
                               choices=['bonferroni', 'fdr_bh', 'fdr_by', 'holm'],
                               help='p值校正方法 (默认: fdr_bh)')
    analyze_parser.add_argument('--no-plot', action='store_true',
                               help='不生成图表')
    analyze_parser.add_argument('--export', '-e', help='导出结果路径')
    analyze_parser.add_argument('--verbose', '-v', action='store_true',
                               help='输出详细信息')
    
    # ========== clean 命令 ==========
    clean_parser = subparsers.add_parser(
        'clean',
        help='数据清洗和预处理'
    )
    clean_parser.add_argument('input', help='输入文件路径')
    clean_parser.add_argument('--output', '-o', required=True, help='输出文件路径')
    clean_parser.add_argument('--dropna', action='store_true', help='删除缺失值')
    clean_parser.add_argument('--fill', choices=['mean', 'median', 'mode', 'knn'],
                             help='填充缺失值')
    clean_parser.add_argument('--detect-outliers', action='store_true',
                             help='检测异常值')
    clean_parser.add_argument('--outlier-method', default='iqr',
                             choices=['iqr', 'zscore'],
                             help='异常值检测方法')
    
    # ========== partial 命令 ==========
    partial_parser = subparsers.add_parser(
        'partial',
        help='偏相关分析'
    )
    partial_parser.add_argument('input', help='输入文件路径')
    partial_parser.add_argument('-x', required=True, help='第一个变量')
    partial_parser.add_argument('-y', required=True, help='第二个变量')
    partial_parser.add_argument('-c', '--covars', required=True,
                               help='协变量，逗号分隔')
    partial_parser.add_argument('--method', default='pearson',
                               choices=['pearson', 'spearman'],
                               help='相关方法')
    partial_parser.add_argument('--matrix', action='store_true',
                               help='计算偏相关矩阵')
    
    # ========== nonlinear 命令 ==========
    nonlinear_parser = subparsers.add_parser(
        'nonlinear',
        help='非线性依赖检测'
    )
    nonlinear_parser.add_argument('input', help='输入文件路径')
    nonlinear_parser.add_argument('--columns', '-c', help='要分析的列，逗号分隔')
    nonlinear_parser.add_argument('--methods', default='dcor,mi',
                                 help='检测方法，逗号分隔 (dcor, mi, mic)')
    nonlinear_parser.add_argument('--top', type=int, default=10,
                                 help='显示前N个结果')
    nonlinear_parser.add_argument('--export', '-e', help='导出结果路径')
    
    # ========== info 命令 ==========
    info_parser = subparsers.add_parser(
        'info',
        help='查看数据信息'
    )
    info_parser.add_argument('input', help='输入文件路径')
    info_parser.add_argument('--types', action='store_true',
                            help='显示类型推断结果')
    info_parser.add_argument('--missing', action='store_true',
                            help='显示缺失值信息')
    
    return parser


def cmd_analyze(args):
    """执行analyze命令"""
    print(f"正在分析: {args.input}")
    
    columns = args.columns.split(',') if args.columns else None
    
    result = quick_corr(
        data=args.input,
        target=args.target,
        columns=columns,
        method=args.method,
        missing_strategy=args.missing,
        fill_method=args.fill_method if args.missing == 'fill' else None,
        pvalue_correction=args.pvalue_correction,
        plot=not args.no_plot,
        export=args.export if args.export else False,
        verbose=args.verbose
    )
    
    return 0


def cmd_clean(args):
    """执行clean命令"""
    print(f"正在清洗: {args.input}")
    
    # 加载数据
    df = load_data(args.input)
    print(f"原始数据: {len(df)} 行, {len(df.columns)} 列")
    
    # 缺失值处理
    if args.dropna:
        df = df.dropna()
        print(f"删除缺失值后: {len(df)} 行")
    elif args.fill:
        df = handle_missing(df, strategy='fill', fill_method=args.fill, verbose=True)
    
    # 异常值检测
    if args.detect_outliers:
        outliers = detect_outliers(df, method=args.outlier_method)
        total_outliers = sum(mask.sum() for mask in outliers.values())
        print(f"检测到 {total_outliers} 个异常值")
    
    # 保存
    if args.output.endswith('.csv'):
        df.to_csv(args.output, index=False)
    elif args.output.endswith('.xlsx'):
        df.to_excel(args.output, index=False)
    else:
        df.to_csv(args.output, index=False)
    
    print(f"已保存到: {args.output}")
    return 0


def cmd_partial(args):
    """执行partial命令"""
    print(f"正在执行偏相关分析: {args.input}")
    
    df = load_data(args.input)
    covars = args.covars.split(',')
    
    if args.matrix:
        # 计算偏相关矩阵
        matrix = partial_corr_matrix(df, covars=covars, method=args.method)
        print("\n偏相关矩阵:")
        print(matrix.round(4))
    else:
        # 计算单个偏相关
        result = partial_corr(df, args.x, args.y, covars, method=args.method)
        
        print(f"\n偏相关分析结果:")
        print(f"  变量: {result['x']} vs {result['y']}")
        print(f"  控制变量: {result['covariates']}")
        print(f"  偏相关系数: {result['partial_correlation']:.4f}")
        print(f"  p值: {result['p_value']:.4e}")
        print(f"  95% CI: [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
        print(f"  样本量: {result['n']}")
    
    return 0


def cmd_nonlinear(args):
    """执行nonlinear命令"""
    print(f"正在执行非线性依赖检测: {args.input}")
    
    df = load_data(args.input)
    
    columns = args.columns.split(',') if args.columns else None
    methods = args.methods.split(',')
    
    report = nonlinear_dependency_report(
        df, columns=columns, methods=methods, top_n=args.top
    )
    
    print(f"\n非线性依赖检测报告 (Top {args.top}):")
    print(report.to_string(index=False))
    
    if args.export:
        report.to_csv(args.export, index=False)
        print(f"\n已导出到: {args.export}")
    
    return 0


def cmd_info(args):
    """执行info命令"""
    df = load_data(args.input)
    
    print(f"\n数据概览: {args.input}")
    print(f"=" * 50)
    print(f"行数: {len(df)}")
    print(f"列数: {len(df.columns)}")
    print(f"\n列信息:")
    print("-" * 50)
    
    info_df = pd.DataFrame({
        '列名': df.columns,
        '类型': df.dtypes.values,
        '非空值': df.count().values,
        '缺失值': df.isnull().sum().values,
        '缺失比例': (df.isnull().sum() / len(df)).values,
        '唯一值': df.nunique().values
    })
    
    print(info_df.to_string(index=False))
    
    if args.types:
        print("\n类型推断结果:")
        print("-" * 50)
        type_mapping = infer_types(df)
        for col, t in type_mapping.items():
            print(f"  {col}: {t}")
    
    if args.missing:
        print("\n缺失值详情:")
        print("-" * 50)
        missing = df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) > 0:
            for col, count in missing.items():
                ratio = count / len(df)
                print(f"  {col}: {count} ({ratio:.2%})")
        else:
            print("  无缺失值")
    
    return 0


def main():
    """主入口函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # 执行对应命令
    commands = {
        'analyze': cmd_analyze,
        'clean': cmd_clean,
        'partial': cmd_partial,
        'nonlinear': cmd_nonlinear,
        'info': cmd_info,
    }
    
    if args.command in commands:
        try:
            return commands[args.command](args)
        except Exception as e:
            print(f"错误: {e}", file=sys.stderr)
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
