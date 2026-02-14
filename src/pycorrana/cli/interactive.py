"""
交互式命令行工具
================
提供交互式的相关性分析流程，使用 typer 和 rich 提供更好的用户体验。
"""

import os
import sys
from typing import Optional

import pandas as pd
import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt

from ..core.analyzer import quick_corr, CorrAnalyzer
from ..core.partial_corr import PartialCorrAnalyzer
from ..core.nonlinear import NonlinearAnalyzer
from ..utils.data_utils import load_data, infer_types, handle_missing, detect_outliers

app = typer.Typer(
    name="pycorrana-interactive",
    help="PyCorrAna 交互式相关性分析工具",
    add_completion=False,
    invoke_without_command=True,
)

console = Console()


@app.callback()
def main(ctx: typer.Context):
    """PyCorrAna 交互式相关性分析工具"""
    if ctx.invoked_subcommand is None:
        ctx.invoke(start)


class InteractiveSession:
    """
    交互式分析会话
    
    管理整个交互式分析流程的状态和操作。
    """
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.data_path: Optional[str] = None
        self.analyzer: Optional[CorrAnalyzer] = None
        self.results: Optional[dict] = None
    
    def show_header(self, title: str):
        """显示标题"""
        console.clear()
        console.print(Panel.fit(
            f"[bold cyan]{title}[/bold cyan]",
            border_style="cyan",
        ))
    
    def show_welcome(self):
        """显示欢迎界面"""
        console.clear()
        console.print(Panel.fit(
            "[bold green]PyCorrAna - 交互式相关性分析工具[/bold green]\n"
            "[dim]自动化相关性分析，降低决策成本[/dim]",
            border_style="green",
        ))
        console.print()
    
    def show_data_info(self):
        """显示数据信息"""
        if self.data is None:
            console.print("[yellow]尚未加载数据[/yellow]")
            return
        
        table = Table(title="数据概览", show_header=True, header_style="bold magenta")
        table.add_column("属性", style="cyan")
        table.add_column("值", style="green")
        
        table.add_row("行数", str(len(self.data)))
        table.add_row("列数", str(len(self.data.columns)))
        table.add_row("列名", ", ".join(self.data.columns[:5]) + ("..." if len(self.data.columns) > 5 else ""))
        
        console.print(table)
    
    def load_from_file(self, path: str) -> bool:
        """从文件加载数据"""
        if not os.path.exists(path):
            console.print(f"[red]文件不存在: {path}[/red]")
            return False
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("正在加载数据...", total=None)
                self.data = load_data(path)
                self.data_path = path
            
            console.print(f"[green]✓[/green] 成功加载数据: {len(self.data)} 行, {len(self.data.columns)} 列")
            return True
        except Exception as e:
            console.print(f"[red]加载失败: {e}[/red]")
            return False
    
    def load_sample_data(self, dataset: str) -> bool:
        """加载示例数据集"""
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task(f"正在加载 {dataset} 数据集...", total=None)
                
                if dataset == "iris":
                    from sklearn.datasets import load_iris
                    iris = load_iris()
                    self.data = pd.DataFrame(
                        iris.data,
                        columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
                    )
                    self.data['species'] = pd.Categorical.from_codes(
                        iris.target, iris.target_names
                    )
                elif dataset == "titanic":
                    try:
                        import seaborn as sns
                        self.data = sns.load_dataset('titanic')
                    except Exception:
                        self.data = self._generate_sample_data()
                else:
                    self.data = self._generate_sample_data()
            
            console.print(f"[green]✓[/green] 已加载示例数据集: {len(self.data)} 行, {len(self.data.columns)} 列")
            return True
        except Exception as e:
            console.print(f"[red]加载失败: {e}[/red]")
            return False
    
    def _generate_sample_data(self, n: int = 200) -> pd.DataFrame:
        """生成示例数据"""
        np.random.seed(42)
        df = pd.DataFrame({
            'A': np.random.randn(n),
            'B': np.random.randn(n),
            'C': np.random.randn(n),
            'D': np.random.choice(['X', 'Y', 'Z'], n),
            'E': np.random.choice([0, 1], n),
        })
        df['B'] = df['A'] * 0.7 + np.random.randn(n) * 0.5
        df['C'] = df['A']**2 + np.random.randn(n) * 0.3
        return df


session = InteractiveSession()


@app.command()
def start():
    """启动交互式分析会话"""
    session.show_welcome()
    
    if not step_load_data():
        console.print("\n[yellow]已取消数据加载[/yellow]")
        raise typer.Exit(0)
    
    while True:
        choice = show_main_menu()
        
        if choice is None:
            console.print("\n[green]感谢使用 PyCorrAna！再见！[/green]")
            break
        
        try:
            execute_action(choice)
        except Exception as e:
            console.print(f"\n[red]错误: {e}[/red]")
            if Confirm.ask("是否继续？", default=True):
                continue
            else:
                break


def show_main_menu() -> Optional[str]:
    """显示主菜单"""
    console.print()
    table = Table(show_header=True, header_style="bold blue", box=None)
    table.add_column("选项", style="cyan", width=6)
    table.add_column("功能", style="white")
    
    options = [
        ("1", "执行完整分析", "full"),
        ("2", "数据探索", "explore"),
        ("3", "数据清洗", "clean"),
        ("4", "相关性分析", "corr"),
        ("5", "偏相关分析", "partial"),
        ("6", "非线性依赖检测", "nonlinear"),
        ("7", "可视化", "visualize"),
        ("8", "导出结果", "export"),
    ]
    
    for opt, desc, _ in options:
        table.add_row(opt, desc)
    table.add_row("0", "[dim]退出[/dim]")
    
    console.print(table)
    
    choice = Prompt.ask("\n请选择", choices=["0", "1", "2", "3", "4", "5", "6", "7", "8"], default="1")
    
    if choice == "0":
        return None
    
    return options[int(choice) - 1][2]


def step_load_data() -> bool:
    """数据加载步骤"""
    session.show_header("步骤 1: 数据加载")
    
    console.print("\n选择数据来源:")
    table = Table(show_header=False, box=None)
    table.add_column("选项", style="cyan", width=6)
    table.add_column("来源", style="white")
    table.add_row("1", "从文件加载 (CSV/Excel)")
    table.add_row("2", "使用示例数据集")
    console.print(table)
    
    source = Prompt.ask("请选择", choices=["1", "2"], default="1")
    
    if source == "1":
        path = Prompt.ask("请输入文件路径")
        return session.load_from_file(path)
    else:
        console.print("\n选择示例数据集:")
        table = Table(show_header=False, box=None)
        table.add_column("选项", style="cyan", width=6)
        table.add_column("数据集", style="white")
        table.add_row("1", "鸢尾花数据集 (Iris)")
        table.add_row("2", "泰坦尼克号数据集 (Titanic)")
        table.add_row("3", "随机生成的测试数据")
        console.print(table)
        
        dataset_choice = Prompt.ask("请选择", choices=["1", "2", "3"], default="1")
        dataset_map = {"1": "iris", "2": "titanic", "3": "random"}
        return session.load_sample_data(dataset_map[dataset_choice])


def execute_action(action: str):
    """执行选中的操作"""
    actions = {
        "full": action_full_analysis,
        "explore": action_explore,
        "clean": action_clean,
        "corr": action_correlation,
        "partial": action_partial,
        "nonlinear": action_nonlinear,
        "visualize": action_visualize,
        "export": action_export,
    }
    
    if action in actions:
        actions[action]()


def action_full_analysis():
    """执行完整分析"""
    session.show_header("执行完整分析")
    
    console.print("\n本流程将自动完成：")
    console.print("  1. 数据预处理")
    console.print("  2. 相关性计算")
    console.print("  3. 可视化生成")
    console.print("  4. 结果导出")
    
    console.print("\n选择相关方法:")
    table = Table(show_header=False, box=None)
    table.add_column("选项", style="cyan", width=6)
    table.add_column("方法", style="white")
    table.add_row("1", "自动选择 (推荐)")
    table.add_row("2", "Pearson")
    table.add_row("3", "Spearman")
    console.print(table)
    
    method_choice = Prompt.ask("请选择", choices=["1", "2", "3"], default="1")
    method_map = {"1": "auto", "2": "pearson", "3": "spearman"}
    method = method_map[method_choice]
    
    export = Confirm.ask("是否导出结果？", default=True)
    export_path = None
    if export:
        export_path = Prompt.ask("导出路径", default="correlation_results.xlsx")
    
    console.print("\n[cyan]开始分析...[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("正在执行相关性分析...", total=None)
        
        session.results = quick_corr(
            session.data,
            method=method,
            plot=True,
            export=export_path if export else False,
            verbose=False
        )
    
    console.print("[green]✓[/green] 分析完成！")
    
    if session.analyzer is None and hasattr(session, 'results'):
        pass


def action_explore():
    """数据探索"""
    session.show_header("数据探索")
    
    if session.data is None:
        console.print("[red]请先加载数据！[/red]")
        return
    
    console.print(f"\n[cyan]数据形状:[/cyan] {session.data.shape}")
    
    table = Table(title="\n数据类型", show_header=True, header_style="bold magenta")
    table.add_column("列名", style="cyan")
    table.add_column("类型", style="green")
    for col, dtype in session.data.dtypes.items():
        table.add_row(col, str(dtype))
    console.print(table)
    
    console.print("\n[bold]描述性统计:[/bold]")
    desc_table = Table(show_header=True, header_style="bold magenta")
    desc_table.add_column("统计量", style="cyan")
    for col in session.data.select_dtypes(include=[np.number]).columns[:5]:
        desc_table.add_column(col, style="green")
    
    desc = session.data.describe()
    for stat in ['count', 'mean', 'std', 'min', 'max']:
        row = [stat]
        for col in desc.columns[:5]:
            row.append(f"{desc.loc[stat, col]:.4f}" if pd.notna(desc.loc[stat, col]) else "NaN")
        desc_table.add_row(*row)
    console.print(desc_table)
    
    console.print("\n[bold]缺失值统计:[/bold]")
    missing = session.data.isnull().sum()
    missing_pct = (missing / len(session.data) * 100).round(2)
    
    missing_table = Table(show_header=True, header_style="bold magenta")
    missing_table.add_column("列名", style="cyan")
    missing_table.add_column("缺失数量", style="yellow")
    missing_table.add_column("缺失比例%", style="red")
    
    for col in missing[missing > 0].index:
        missing_table.add_row(col, str(missing[col]), f"{missing_pct[col]:.2f}")
    
    if missing_table.row_count > 0:
        console.print(missing_table)
    else:
        console.print("[green]无缺失值[/green]")
    
    console.print("\n[bold]自动类型推断:[/bold]")
    type_mapping = infer_types(session.data)
    type_table = Table(show_header=True, header_style="bold magenta")
    type_table.add_column("列名", style="cyan")
    type_table.add_column("推断类型", style="green")
    for col, t in type_mapping.items():
        type_table.add_row(col, t)
    console.print(type_table)


def action_clean():
    """数据清洗"""
    session.show_header("数据清洗")
    
    if session.data is None:
        console.print("[red]请先加载数据！[/red]")
        return
    
    console.print("\n选择清洗操作:")
    table = Table(show_header=False, box=None)
    table.add_column("选项", style="cyan", width=6)
    table.add_column("操作", style="white")
    table.add_row("1", "处理缺失值")
    table.add_row("2", "检测异常值")
    table.add_row("3", "删除列")
    console.print(table)
    
    action = Prompt.ask("请选择", choices=["1", "2", "3"], default="1")
    
    if action == "1":
        console.print("\n选择缺失值处理策略:")
        table = Table(show_header=False, box=None)
        table.add_column("选项", style="cyan", width=6)
        table.add_column("策略", style="white")
        table.add_row("1", "删除含缺失值的行")
        table.add_row("2", "用均值填充")
        table.add_row("3", "用中位数填充")
        table.add_row("4", "用众数填充")
        table.add_row("5", "用KNN预测填充")
        console.print(table)
        
        strategy = Prompt.ask("请选择", choices=["1", "2", "3", "4", "5"], default="1")
        
        if strategy == "1":
            n_before = len(session.data)
            session.data = session.data.dropna()
            n_after = len(session.data)
            console.print(f"[green]✓[/green] 已删除 {n_before - n_after} 行")
        else:
            fill_methods = {"2": "mean", "3": "median", "4": "mode", "5": "knn"}
            session.data = handle_missing(
                session.data,
                strategy='fill',
                fill_method=fill_methods[strategy],
                verbose=True
            )
            console.print("[green]✓[/green] 缺失值处理完成")
    
    elif action == "2":
        console.print("\n选择异常值检测方法:")
        table = Table(show_header=False, box=None)
        table.add_column("选项", style="cyan", width=6)
        table.add_column("方法", style="white")
        table.add_row("1", "IQR方法（四分位距）")
        table.add_row("2", "Z-Score方法")
        console.print(table)
        
        method_choice = Prompt.ask("请选择", choices=["1", "2"], default="1")
        method = "iqr" if method_choice == "1" else "zscore"
        
        numeric_cols = session.data.select_dtypes(include=[np.number]).columns.tolist()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("正在检测异常值...", total=None)
            outliers = detect_outliers(session.data, columns=numeric_cols, method=method)
        
        outlier_table = Table(title="异常值检测结果", show_header=True, header_style="bold magenta")
        outlier_table.add_column("列名", style="cyan")
        outlier_table.add_column("异常值数量", style="yellow")
        
        total_outliers = 0
        for col, mask in outliers.items():
            n_outliers = mask.sum()
            if n_outliers > 0:
                outlier_table.add_row(col, str(n_outliers))
                total_outliers += n_outliers
        
        console.print(outlier_table)
        console.print(f"\n共检测到 [yellow]{total_outliers}[/yellow] 个异常值")
    
    elif action == "3":
        console.print(f"\n当前列: {', '.join(session.data.columns)}")
        cols = Prompt.ask("请输入要删除的列名（逗号分隔）")
        cols_to_drop = [c.strip() for c in cols.split(',')]
        session.data = session.data.drop(columns=cols_to_drop, errors='ignore')
        console.print(f"[green]✓[/green] 已删除列: {', '.join(cols_to_drop)}")


def action_correlation():
    """相关性分析"""
    session.show_header("相关性分析")
    
    if session.data is None:
        console.print("[red]请先加载数据！[/red]")
        return
    
    console.print("\n选择相关方法:")
    table = Table(show_header=False, box=None)
    table.add_column("选项", style="cyan", width=6)
    table.add_column("方法", style="white")
    table.add_row("1", "自动选择 (推荐)")
    table.add_row("2", "Pearson相关")
    table.add_row("3", "Spearman相关")
    table.add_row("4", "Kendall相关")
    console.print(table)
    
    method_choice = Prompt.ask("请选择", choices=["1", "2", "3", "4"], default="1")
    method_map = {"1": "auto", "2": "pearson", "3": "spearman", "4": "kendall"}
    method = method_map[method_choice]
    
    has_target = Confirm.ask("是否指定目标变量？", default=False)
    target = None
    if has_target:
        console.print(f"\n可用列: {', '.join(session.data.columns)}")
        target = Prompt.ask("请输入目标变量名")
    
    console.print("\n[cyan]正在执行相关性分析...[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("计算相关性矩阵...", total=None)
        
        session.analyzer = CorrAnalyzer(
            session.data,
            method=method,
            verbose=False
        )
        session.results = session.analyzer.fit(target=target)
    
    console.print("[green]✓[/green] 分析完成！")
    
    if session.analyzer.significant_pairs:
        console.print(f"\n发现 [green]{len(session.analyzer.significant_pairs)}[/green] 对显著相关变量")
        
        sig_table = Table(title="显著相关变量对 (Top 10)", show_header=True, header_style="bold magenta")
        sig_table.add_column("变量1", style="cyan")
        sig_table.add_column("变量2", style="cyan")
        sig_table.add_column("相关系数", style="yellow")
        sig_table.add_column("p值", style="green")
        sig_table.add_column("解释", style="white")
        
        for pair in session.analyzer.significant_pairs[:10]:
            sig_table.add_row(
                pair['var1'],
                pair['var2'],
                f"{pair['correlation']:.4f}",
                f"{pair['p_value']:.4e}",
                pair.get('interpretation', '')
            )
        console.print(sig_table)


def action_partial():
    """偏相关分析"""
    session.show_header("偏相关分析")
    
    if session.data is None:
        console.print("[red]请先加载数据！[/red]")
        return
    
    console.print(f"\n可用列: {', '.join(session.data.columns)}")
    
    x = Prompt.ask("请输入第一个变量")
    y = Prompt.ask("请输入第二个变量")
    covars_str = Prompt.ask("请输入控制变量（逗号分隔）")
    covars = [c.strip() for c in covars_str.split(',')]
    
    console.print("\n选择方法:")
    table = Table(show_header=False, box=None)
    table.add_column("选项", style="cyan", width=6)
    table.add_column("方法", style="white")
    table.add_row("1", "Pearson")
    table.add_row("2", "Spearman")
    console.print(table)
    
    method_choice = Prompt.ask("请选择", choices=["1", "2"], default="1")
    method = "pearson" if method_choice == "1" else "spearman"
    
    console.print("\n[cyan]正在计算偏相关...[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("计算偏相关系数...", total=None)
        
        analyzer = PartialCorrAnalyzer(session.data, verbose=False)
        result = analyzer.fit(x, y, covars, method)
    
    console.print("[green]✓[/green] 计算完成！")
    
    result_table = Table(title="偏相关分析结果", show_header=True, header_style="bold magenta")
    result_table.add_column("指标", style="cyan")
    result_table.add_column("值", style="green")
    
    result_table.add_row("变量 X", result['x'])
    result_table.add_row("变量 Y", result['y'])
    result_table.add_row("控制变量", ", ".join(result['covariates']))
    result_table.add_row("偏相关系数", f"{result['partial_correlation']:.4f}")
    result_table.add_row("p值", f"{result['p_value']:.4e}")
    result_table.add_row("95% CI", f"[{result['ci_95'][0]:.4f}, {result['ci_95'][1]:.4f}]")
    result_table.add_row("样本量", str(result['n']))
    
    console.print(result_table)


def action_nonlinear():
    """非线性依赖检测"""
    session.show_header("非线性依赖检测")
    
    if session.data is None:
        console.print("[red]请先加载数据！[/red]")
        return
    
    top_n = IntPrompt.ask("显示前 N 个结果", default=10)
    
    console.print("\n[cyan]正在计算距离相关、互信息等非线性指标...[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("计算非线性依赖指标...", total=None)
        
        analyzer = NonlinearAnalyzer(session.data, verbose=False)
        report = analyzer.generate_report(top_n=top_n)
    
    console.print("[green]✓[/green] 计算完成！")
    
    if report is not None and len(report) > 0:
        nl_table = Table(title=f"非线性依赖检测报告 (Top {top_n})", show_header=True, header_style="bold magenta")
        for col in report.columns:
            nl_table.add_column(col, style="cyan")
        
        for _, row in report.head(top_n).iterrows():
            nl_table.add_row(*[str(v) for v in row.values])
        
        console.print(nl_table)
    else:
        console.print("[yellow]未检测到显著的非线性关系[/yellow]")


def action_visualize():
    """可视化"""
    session.show_header("可视化")
    
    if session.analyzer is None or session.analyzer.corr_matrix is None:
        console.print("[red]请先执行相关性分析！[/red]")
        return
    
    console.print("\n选择可视化类型:")
    table = Table(show_header=False, box=None)
    table.add_column("选项", style="cyan", width=6)
    table.add_column("类型", style="white")
    table.add_row("1", "相关性热力图")
    table.add_row("2", "散点图矩阵")
    table.add_row("3", "相关网络图")
    table.add_row("4", "显著相关对条形图")
    console.print(table)
    
    viz_choice = Prompt.ask("请选择", choices=["1", "2", "3", "4"], default="1")
    
    try:
        if viz_choice == "1":
            cluster = Confirm.ask("是否进行层次聚类？", default=False)
            session.analyzer.plot_heatmap(cluster=cluster)
            console.print("[green]✓[/green] 热力图已生成")
        
        elif viz_choice == "2":
            numeric_cols = [c for c, t in session.analyzer.type_mapping.items() if t == 'numeric']
            if len(numeric_cols) >= 2:
                session.analyzer.plot_pairplot(columns=numeric_cols[:6])
                console.print("[green]✓[/green] 散点图矩阵已生成")
            else:
                console.print("[yellow]数值列不足，无法绘制散点图矩阵[/yellow]")
        
        elif viz_choice == "3":
            threshold = FloatPrompt.ask("请输入网络图阈值", default=0.5)
            session.analyzer.visualizer.plot_correlation_network(
                session.analyzer.corr_matrix,
                threshold=threshold
            )
            console.print("[green]✓[/green] 相关网络图已生成")
        
        elif viz_choice == "4":
            session.analyzer.visualizer.plot_significant_pairs(
                session.analyzer.significant_pairs
            )
            console.print("[green]✓[/green] 显著相关对条形图已生成")
    except Exception as e:
        console.print(f"[red]可视化失败: {e}[/red]")


def action_export():
    """导出结果"""
    session.show_header("导出结果")
    
    if session.analyzer is None or session.analyzer.corr_matrix is None:
        console.print("[red]请先执行相关性分析！[/red]")
        return
    
    console.print("\n选择导出格式:")
    table = Table(show_header=False, box=None)
    table.add_column("选项", style="cyan", width=6)
    table.add_column("格式", style="white")
    table.add_row("1", "Excel格式 (.xlsx)")
    table.add_row("2", "CSV格式")
    table.add_row("3", "HTML报告")
    table.add_row("4", "Markdown报告")
    console.print(table)
    
    format_choice = Prompt.ask("请选择", choices=["1", "2", "3", "4"], default="1")
    format_map = {"1": "excel", "2": "csv", "3": "html", "4": "markdown"}
    export_format = format_map[format_choice]
    
    ext = "xlsx" if export_format == "excel" else export_format
    default_name = f"correlation_results.{ext}"
    path = Prompt.ask("请输入保存路径", default=default_name)
    
    try:
        if export_format in ['excel', 'csv']:
            session.analyzer.export_results(path, format=export_format)
            console.print(f"[green]✓[/green] 已导出到: {path}")
        
        elif export_format == 'html':
            html_content = session.analyzer.reporter.to_html(
                session.analyzer.corr_matrix,
                session.analyzer.pvalue_matrix,
                session.analyzer.significant_pairs
            )
            with open(path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            console.print(f"[green]✓[/green] 已保存HTML报告: {path}")
        
        elif export_format == 'markdown':
            md_content = session.analyzer.reporter.to_markdown(
                session.analyzer.corr_matrix,
                session.analyzer.significant_pairs
            )
            with open(path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            console.print(f"[green]✓[/green] 已保存Markdown报告: {path}")
    except Exception as e:
        console.print(f"[red]导出失败: {e}[/red]")


def interactive_mode():
    """
    启动交互式模式
    
    这是主入口函数，可以通过命令行调用。
    """
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]已取消操作[/yellow]")
        sys.exit(0)
    except EOFError:
        console.print("\n\n[yellow]输入结束[/yellow]")
        sys.exit(0)


if __name__ == '__main__':
    interactive_mode()
