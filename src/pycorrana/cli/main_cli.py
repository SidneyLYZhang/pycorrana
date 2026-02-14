"""
命令行工具主模块
================
提供分模块的CLI工具，支持数据流水线各阶段的操作。
"""

from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table

from ..core.analyzer import quick_corr
from ..core.partial_corr import partial_corr, partial_corr_matrix
from ..core.nonlinear import nonlinear_dependency_report
from ..utils.data_utils import load_data, infer_types, handle_missing, detect_outliers

app = typer.Typer(
    name="pycorrana",
    help="PyCorrAna - Python Correlation Analysis Toolkit",
    add_completion=False,
)

console = Console()


def parse_columns(columns: Optional[str]) -> Optional[List[str]]:
    if not columns:
        return None
    return [c.strip() for c in columns.split(',')]


@app.command()
def analyze(
    input: Path = typer.Argument(..., help="输入文件路径 (CSV/Excel)", exists=True),
    target: Optional[str] = typer.Option(None, "--target", "-t", help="目标变量"),
    columns: Optional[str] = typer.Option(None, "--columns", "-c", help="要分析的列，逗号分隔"),
    method: str = typer.Option("auto", "--method", "-m", help="相关性方法",
                               case_sensitive=False),
    missing: str = typer.Option("warn", "--missing", help="缺失值处理策略"),
    fill_method: str = typer.Option("median", "--fill-method", help="填充方法"),
    pvalue_correction: str = typer.Option("fdr_bh", "--pvalue-correction", help="p值校正方法"),
    no_plot: bool = typer.Option(False, "--no-plot", help="不生成图表"),
    export: Optional[str] = typer.Option(None, "--export", "-e", help="导出结果路径"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="输出详细信息"),
):
    """
    执行完整的相关性分析

    示例:

        pycorrana analyze data.csv --target sales --export results.xlsx
    """
    console.print(f"[cyan]正在分析:[/cyan] {input}")
    
    cols = parse_columns(columns)
    
    result = quick_corr(
        data=str(input),
        target=target,
        columns=cols,
        method=method,
        missing_strategy=missing,
        fill_method=fill_method if missing == 'fill' else None,
        pvalue_correction=pvalue_correction,
        plot=not no_plot,
        export=export if export else False,
        verbose=verbose
    )
    
    raise typer.Exit(0)


@app.command()
def clean(
    input: Path = typer.Argument(..., help="输入文件路径", exists=True),
    output: str = typer.Option(..., "--output", "-o", help="输出文件路径"),
    dropna: bool = typer.Option(False, "--dropna", help="删除缺失值"),
    fill: Optional[str] = typer.Option(None, "--fill", help="填充缺失值 (mean/median/mode/knn)"),
    detect_outliers_flag: bool = typer.Option(False, "--detect-outliers", help="检测异常值"),
    outlier_method: str = typer.Option("iqr", "--outlier-method", help="异常值检测方法"),
):
    """
    数据清洗和预处理

    示例:

        pycorrana clean data.csv --dropna --output cleaned.csv
    """
    console.print(f"[cyan]正在清洗:[/cyan] {input}")
    
    df = load_data(str(input))
    console.print(f"原始数据: {len(df)} 行, {len(df.columns)} 列")
    
    if dropna:
        df = df.dropna()
        console.print(f"删除缺失值后: {len(df)} 行")
    elif fill:
        df = handle_missing(df, strategy='fill', fill_method=fill, verbose=True)
    
    if detect_outliers_flag:
        outliers = detect_outliers(df, method=outlier_method)
        total_outliers = sum(mask.sum() for mask in outliers.values())
        console.print(f"检测到 [yellow]{total_outliers}[/yellow] 个异常值")
    
    if output.endswith('.csv'):
        df.to_csv(output, index=False)
    elif output.endswith('.xlsx'):
        df.to_excel(output, index=False)
    else:
        df.to_csv(output, index=False)
    
    console.print(f"[green]已保存到:[/green] {output}")
    raise typer.Exit(0)


@app.command()
def partial(
    input: Path = typer.Argument(..., help="输入文件路径", exists=True),
    x: str = typer.Option(..., "-x", help="第一个变量"),
    y: str = typer.Option(..., "-y", help="第二个变量"),
    covars: str = typer.Option(..., "-c", "--covars", help="协变量，逗号分隔"),
    method: str = typer.Option("pearson", "--method", help="相关方法"),
    matrix: bool = typer.Option(False, "--matrix", help="计算偏相关矩阵"),
):
    """
    偏相关分析

    示例:

        pycorrana partial data.csv -x income -y happiness -c age,education
    """
    console.print(f"[cyan]正在执行偏相关分析:[/cyan] {input}")
    
    df = load_data(str(input))
    covar_list = [c.strip() for c in covars.split(',')]
    
    if matrix:
        result_matrix = partial_corr_matrix(df, covars=covar_list, method=method)
        console.print("\n[bold]偏相关矩阵:[/bold]")
        console.print(result_matrix.round(4))
    else:
        result = partial_corr(df, x, y, covar_list, method=method)
        
        console.print(f"\n[bold]偏相关分析结果:[/bold]")
        console.print(f"  变量: {result['x']} vs {result['y']}")
        console.print(f"  控制变量: {result['covariates']}")
        console.print(f"  偏相关系数: [cyan]{result['partial_correlation']:.4f}[/cyan]")
        console.print(f"  p值: {result['p_value']:.4e}")
        console.print(f"  95% CI: [{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
        console.print(f"  样本量: {result['n']}")
    
    raise typer.Exit(0)


@app.command()
def nonlinear(
    input: Path = typer.Argument(..., help="输入文件路径", exists=True),
    columns: Optional[str] = typer.Option(None, "--columns", "-c", help="要分析的列，逗号分隔"),
    methods: str = typer.Option("dcor,mi", "--methods", help="检测方法，逗号分隔 (dcor, mi, mic)"),
    top: int = typer.Option(10, "--top", help="显示前N个结果"),
    export: Optional[str] = typer.Option(None, "--export", "-e", help="导出结果路径"),
):
    """
    非线性依赖检测

    示例:

        pycorrana nonlinear data.csv --top 20
    """
    console.print(f"[cyan]正在执行非线性依赖检测:[/cyan] {input}")
    
    df = load_data(str(input))
    
    cols = parse_columns(columns)
    method_list = [m.strip() for m in methods.split(',')]
    
    report = nonlinear_dependency_report(
        df, columns=cols, methods=method_list, top_n=top
    )
    
    console.print(f"\n[bold]非线性依赖检测报告 (Top {top}):[/bold]")
    
    table = Table(show_header=True, header_style="bold magenta")
    for col in report.columns:
        table.add_column(col, style="cyan")
    
    for _, row in report.iterrows():
        table.add_row(*[str(v) for v in row.values])
    
    console.print(table)
    
    if export:
        report.to_csv(export, index=False)
        console.print(f"\n[green]已导出到:[/green] {export}")
    
    raise typer.Exit(0)


@app.command()
def info(
    input: Path = typer.Argument(..., help="输入文件路径", exists=True),
    types: bool = typer.Option(False, "--types", help="显示类型推断结果"),
    missing: bool = typer.Option(False, "--missing", help="显示缺失值信息"),
):
    """
    查看数据信息

    示例:

        pycorrana info data.csv --types --missing
    """
    df = load_data(str(input))
    
    console.print(f"\n[bold]数据概览:[/bold] {input}")
    console.print("=" * 50)
    console.print(f"行数: {len(df)}")
    console.print(f"列数: {len(df.columns)}")
    console.print(f"\n[bold]列信息:[/bold]")
    console.print("-" * 50)
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("列名", style="cyan")
    table.add_column("类型", style="green")
    table.add_column("非空值", style="yellow")
    table.add_column("缺失值", style="red")
    table.add_column("缺失比例", style="magenta")
    table.add_column("唯一值", style="blue")
    
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_ratio = missing_count / len(df)
        table.add_row(
            col,
            str(df[col].dtype),
            str(df[col].count()),
            str(missing_count),
            f"{missing_ratio:.2%}",
            str(df[col].nunique())
        )
    
    console.print(table)
    
    if types:
        console.print("\n[bold]类型推断结果:[/bold]")
        console.print("-" * 50)
        type_mapping = infer_types(df)
        type_table = Table(show_header=True, header_style="bold magenta")
        type_table.add_column("列名", style="cyan")
        type_table.add_column("推断类型", style="green")
        for col, t in type_mapping.items():
            type_table.add_row(col, t)
        console.print(type_table)
    
    if missing:
        console.print("\n[bold]缺失值详情:[/bold]")
        console.print("-" * 50)
        missing_series = df.isnull().sum()
        missing_series = missing_series[missing_series > 0].sort_values(ascending=False)
        if len(missing_series) > 0:
            missing_table = Table(show_header=True, header_style="bold magenta")
            missing_table.add_column("列名", style="cyan")
            missing_table.add_column("缺失数量", style="yellow")
            missing_table.add_column("缺失比例", style="red")
            for col, count in missing_series.items():
                ratio = count / len(df)
                missing_table.add_row(col, str(count), f"{ratio:.2%}")
            console.print(missing_table)
        else:
            console.print("[green]无缺失值[/green]")
    
    raise typer.Exit(0)


def main():
    """主入口函数 - 保持向后兼容"""
    app()


if __name__ == '__main__':
    main()
