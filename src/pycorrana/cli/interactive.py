"""
交互式命令行工具
================
提供交互式的相关性分析流程，使用 typer 和 rich 提供更好的用户体验。
"""

import os
import sys
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm, IntPrompt, FloatPrompt

from .. import __version__
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


def show_menu(title: str, options: List[Tuple[str, str]], add_exit: bool = True) -> Optional[str]:
    """
    显示菜单并获取用户选择
    
    Parameters
    ----------
    title : str
        菜单标题
    options : List[Tuple[str, str]]
        选项列表，每个元素为 (显示文本, 返回值)
    add_exit : bool
        是否添加退出选项
    
    Returns
    -------
    Optional[str]
        用户选择的返回值，退出时返回 None
    """
    console.print()
    table = Table(show_header=True, header_style="bold blue", box=None)
    table.add_column("选项", style="cyan", width=6)
    table.add_column(title, style="white")
    
    choices = []
    for i, (desc, _) in enumerate(options, 1):
        table.add_row(str(i), desc)
        choices.append(str(i))
    
    if add_exit:
        table.add_row("0", "[dim]退出/返回[/dim]")
        choices.append("0")
    
    console.print(table)
    
    choice = Prompt.ask("\n请选择", choices=choices, default="1")
    
    if choice == "0":
        return None
    
    return options[int(choice) - 1][1]


def show_header(title: str):
    """显示标题"""
    console.clear()
    console.print(Panel.fit(
        f"[bold cyan]{title}[/bold cyan]",
        border_style="cyan",
    ))


def show_data_table(title: str, data: Dict[str, Any], style: str = "cyan"):
    """显示数据表格"""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("属性", style=style)
    table.add_column("值", style="green")
    
    for key, value in data.items():
        table.add_row(key, str(value))
    
    console.print(table)


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
    
    def show_welcome(self):
        """显示欢迎界面"""
        console.clear()
        console.print(Panel.fit(
            f"[bold green]PyCorrAna - 交互式相关性分析工具[/bold green]\n"
            f"[dim]版本 {__version__} | 自动化相关性分析，降低决策成本[/dim]",
            border_style="green",
        ))
        console.print()
    
    def show_data_info(self):
        """显示数据信息"""
        if self.data is None:
            console.print("[yellow]尚未加载数据[/yellow]")
            return
        
        show_data_table("数据概览", {
            "行数": len(self.data),
            "列数": len(self.data.columns),
            "列名": ", ".join(self.data.columns[:5]) + ("..." if len(self.data.columns) > 5 else "")
        })
    
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
    
    def require_data(self) -> bool:
        """检查是否已加载数据"""
        if self.data is None:
            console.print("[red]请先加载数据！[/red]")
            return False
        return True
    
    def require_analyzer(self) -> bool:
        """检查是否已执行分析"""
        if self.analyzer is None or self.analyzer.corr_matrix is None:
            console.print("[red]请先执行相关性分析！[/red]")
            return False
        return True


session = InteractiveSession()


def step_load_data() -> bool:
    """数据加载步骤"""
    show_header("步骤 1: 数据加载")
    
    choice = show_menu("选择数据来源", [
        ("从文件加载 (CSV/Excel)", "file"),
        ("使用示例数据集", "sample"),
    ])
    
    if choice is None:
        return False
    
    if choice == "file":
        path = Prompt.ask("请输入文件路径")
        return session.load_from_file(path)
    else:
        dataset_choice = show_menu("选择示例数据集", [
            ("鸢尾花数据集 (Iris)", "iris"),
            ("泰坦尼克号数据集 (Titanic)", "titanic"),
            ("随机生成的测试数据", "random"),
        ])
        
        if dataset_choice is None:
            return False
        
        return session.load_sample_data(dataset_choice)


def action_full_analysis():
    """执行完整分析"""
    show_header("执行完整分析")
    
    if not session.require_data():
        return
    
    console.print("\n本流程将自动完成：")
    console.print("  1. 数据预处理")
    console.print("  2. 相关性计算")
    console.print("  3. 可视化生成")
    console.print("  4. 结果导出")
    
    method_choice = show_menu("选择相关方法", [
        ("自动选择 (推荐)", "auto"),
        ("Pearson", "pearson"),
        ("Spearman", "spearman"),
    ])
    
    if method_choice is None:
        return
    
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
            method=method_choice,
            plot=True,
            export=export_path if export else False,
            verbose=False
        )
    
    console.print("[green]✓[/green] 分析完成！")


def action_explore():
    """数据探索"""
    show_header("数据探索")
    
    if not session.require_data():
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
    numeric_cols = session.data.select_dtypes(include=[np.number]).columns[:5]
    for col in numeric_cols:
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
    show_header("数据清洗")
    
    if not session.require_data():
        return
    
    action = show_menu("选择清洗操作", [
        ("处理缺失值", "missing"),
        ("检测异常值", "outliers"),
        ("删除列", "dropcols"),
    ])
    
    if action is None:
        return
    
    if action == "missing":
        strategy = show_menu("选择缺失值处理策略", [
            ("删除含缺失值的行", "drop"),
            ("用均值填充", "mean"),
            ("用中位数填充", "median"),
            ("用众数填充", "mode"),
            ("用KNN预测填充", "knn"),
        ])
        
        if strategy is None:
            return
        
        if strategy == "drop":
            n_before = len(session.data)
            session.data = session.data.dropna()
            n_after = len(session.data)
            console.print(f"[green]✓[/green] 已删除 {n_before - n_after} 行")
        else:
            session.data = handle_missing(
                session.data,
                strategy='fill',
                fill_method=strategy,
                verbose=True
            )
            console.print("[green]✓[/green] 缺失值处理完成")
    
    elif action == "outliers":
        method = show_menu("选择异常值检测方法", [
            ("IQR方法（四分位距）", "iqr"),
            ("Z-Score方法", "zscore"),
        ])
        
        if method is None:
            return
        
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
    
    elif action == "dropcols":
        console.print(f"\n当前列: {', '.join(session.data.columns)}")
        cols = Prompt.ask("请输入要删除的列名（逗号分隔）")
        cols_to_drop = [c.strip() for c in cols.split(',')]
        session.data = session.data.drop(columns=cols_to_drop, errors='ignore')
        console.print(f"[green]✓[/green] 已删除列: {', '.join(cols_to_drop)}")


def action_correlation():
    """相关性分析"""
    show_header("相关性分析")
    
    if not session.require_data():
        return
    
    method = show_menu("选择相关方法", [
        ("自动选择 (推荐)", "auto"),
        ("Pearson相关", "pearson"),
        ("Spearman相关", "spearman"),
        ("Kendall相关", "kendall"),
    ])
    
    if method is None:
        return
    
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
    show_header("偏相关分析")
    
    if not session.require_data():
        return
    
    console.print(f"\n可用列: {', '.join(session.data.columns)}")
    
    x = Prompt.ask("请输入第一个变量")
    y = Prompt.ask("请输入第二个变量")
    covars_str = Prompt.ask("请输入控制变量（逗号分隔）")
    covars = [c.strip() for c in covars_str.split(',')]
    
    method = show_menu("选择方法", [
        ("Pearson", "pearson"),
        ("Spearman", "spearman"),
    ])
    
    if method is None:
        return
    
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
    show_header("非线性依赖检测")
    
    if not session.require_data():
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
    show_header("可视化")
    
    if not session.require_analyzer():
        return
    
    viz_choice = show_menu("选择可视化类型", [
        ("相关性热力图", "heatmap"),
        ("散点图矩阵", "pairplot"),
        ("相关网络图", "network"),
        ("显著相关对条形图", "bar"),
    ])
    
    if viz_choice is None:
        return
    
    try:
        if viz_choice == "heatmap":
            cluster = Confirm.ask("是否进行层次聚类？", default=False)
            session.analyzer.plot_heatmap(cluster=cluster)
            console.print("[green]✓[/green] 热力图已生成")
        
        elif viz_choice == "pairplot":
            numeric_cols = [c for c, t in session.analyzer.type_mapping.items() if t == 'numeric']
            if len(numeric_cols) >= 2:
                session.analyzer.plot_pairplot(columns=numeric_cols[:6])
                console.print("[green]✓[/green] 散点图矩阵已生成")
            else:
                console.print("[yellow]数值列不足，无法绘制散点图矩阵[/yellow]")
        
        elif viz_choice == "network":
            threshold = FloatPrompt.ask("请输入网络图阈值", default=0.5)
            session.analyzer.visualizer.plot_correlation_network(
                session.analyzer.corr_matrix,
                threshold=threshold
            )
            console.print("[green]✓[/green] 相关网络图已生成")
        
        elif viz_choice == "bar":
            session.analyzer.visualizer.plot_significant_pairs(
                session.analyzer.significant_pairs
            )
            console.print("[green]✓[/green] 显著相关对条形图已生成")
    except Exception as e:
        console.print(f"[red]可视化失败: {e}[/red]")


def action_export():
    """导出结果"""
    show_header("导出结果")
    
    if not session.require_analyzer():
        return
    
    export_format = show_menu("选择导出格式", [
        ("Excel格式 (.xlsx)", "excel"),
        ("CSV格式", "csv"),
        ("HTML报告", "html"),
        ("Markdown报告", "markdown"),
    ])
    
    if export_format is None:
        return
    
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


ACTIONS = {
    "full": action_full_analysis,
    "explore": action_explore,
    "clean": action_clean,
    "corr": action_correlation,
    "partial": action_partial,
    "nonlinear": action_nonlinear,
    "visualize": action_visualize,
    "export": action_export,
}

MAIN_MENU_OPTIONS = [
    ("执行完整分析", "full"),
    ("数据探索", "explore"),
    ("数据清洗", "clean"),
    ("相关性分析", "corr"),
    ("偏相关分析", "partial"),
    ("非线性依赖检测", "nonlinear"),
    ("可视化", "visualize"),
    ("导出结果", "export"),
]


@app.callback()
def main(ctx: typer.Context):
    """PyCorrAna 交互式相关性分析工具"""
    if ctx.invoked_subcommand is None:
        ctx.invoke(start)


@app.command()
def start():
    """启动交互式分析会话"""
    session.show_welcome()
    
    if not step_load_data():
        console.print("\n[yellow]已取消数据加载[/yellow]")
        raise typer.Exit(0)
    
    while True:
        choice = show_menu("功能", MAIN_MENU_OPTIONS)
        
        if choice is None:
            console.print("\n[green]感谢使用 PyCorrAna！再见！[/green]")
            break
        
        try:
            ACTIONS[choice]()
        except Exception as e:
            console.print(f"\n[red]错误: {e}[/red]")
            if not Confirm.ask("是否继续？", default=True):
                break


@app.command()
def version():
    """显示版本信息"""
    console.print(f"[cyan]PyCorrAna[/cyan] 交互式工具版本 [green]{__version__}[/green]")


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
