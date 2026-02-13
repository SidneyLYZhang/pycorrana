"""
可视化核心模块
==============
提供相关性热力图、散点图矩阵、箱线图等可视化功能。
"""

import warnings
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class CorrVisualizer:
    """
    相关性可视化器
    
    提供多种相关性可视化图表。
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
        """
        Parameters
        ----------
        style : str, default='seaborn-v0_8-whitegrid'
            matplotlib样式
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        
        self.default_figsize = (10, 8)
        self.default_dpi = 100
    
    def plot_heatmap(self,
                    corr_matrix: pd.DataFrame,
                    figsize: Tuple[int, int] = (10, 8),
                    annot: bool = True,
                    fmt: str = '.2f',
                    cmap: str = 'RdBu_r',
                    center: float = 0,
                    vmin: float = -1,
                    vmax: float = 1,
                    cluster: bool = False,
                    mask_upper: bool = False,
                    title: Optional[str] = None,
                    savefig: Optional[str] = None,
                    dpi: int = 100,
                    **kwargs) -> plt.Figure:
        """
        绘制相关性热力图。
        
        Parameters
        ----------
        corr_matrix : pd.DataFrame
            相关性矩阵
        figsize : tuple, default=(10, 8)
            图表大小
        annot : bool, default=True
            是否显示数值标注
        fmt : str, default='.2f'
            数值格式
        cmap : str, default='RdBu_r'
            颜色映射
        center : float, default=0
            颜色中心值
        vmin, vmax : float, default=-1, 1
            颜色范围
        cluster : bool, default=False
            是否进行层次聚类
        mask_upper : bool, default=False
            是否只显示下三角
        title : str, optional
            图表标题
        savefig : str, optional
            保存路径
        dpi : int, default=100
            分辨率
        **kwargs
            其他参数传递给seaborn.heatmap
            
        Returns
        -------
        plt.Figure
            matplotlib图表对象
        """
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # 准备数据
        plot_data = corr_matrix.copy()
        
        # 层次聚类
        if cluster:
            try:
                from scipy.cluster.hierarchy import linkage, dendrogram
                from scipy.spatial.distance import squareform
                
                # 计算距离矩阵
                dist_matrix = 1 - np.abs(plot_data.fillna(0))
                linkage_matrix = linkage(squareform(dist_matrix), method='average')
                
                # 获取聚类顺序
                dendro = dendrogram(linkage_matrix, no_plot=True)
                order = dendro['leaves']
                
                plot_data = plot_data.iloc[order, order]
            except Exception as e:
                warnings.warn(f"聚类失败: {e}，使用原始顺序")
        
        # 创建掩码（只显示下三角）
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(plot_data, dtype=bool), k=1)
        
        # 绘制热力图
        sns.heatmap(
            plot_data,
            mask=mask,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            center=center,
            vmin=vmin,
            vmax=vmax,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
            **kwargs
        )
        
        # 设置标题
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax.set_title('相关性热力图', fontsize=14, fontweight='bold')
        
        # 旋转标签
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # 保存
        if savefig:
            plt.savefig(savefig, dpi=dpi, bbox_inches='tight',
                       format=savefig.split('.')[-1] if '.' in savefig else 'png')
            print(f"  图表已保存: {savefig}")
        
        plt.show()
        return fig
    
    def plot_pairplot(self,
                     data: pd.DataFrame,
                     columns: Optional[List[str]] = None,
                     hue: Optional[str] = None,
                     diag_kind: str = 'kde',
                     kind: str = 'scatter',
                     corner: bool = False,
                     height: float = 2.5,
                     aspect: float = 1,
                     savefig: Optional[str] = None,
                     **kwargs) -> sns.PairGrid:
        """
        绘制散点图矩阵。
        
        Parameters
        ----------
        data : pd.DataFrame
            输入数据
        columns : list, optional
            要绘制的列
        hue : str, optional
            用于颜色区分的列
        diag_kind : str, default='kde'
            对角线图表类型：'kde'、'hist'
        kind : str, default='scatter'
            非对角线图表类型：'scatter'、'reg'
        corner : bool, default=False
            是否只绘制下三角
        height : float, default=2.5
            每个子图的高度
        aspect : float, default=1
            宽高比
        savefig : str, optional
            保存路径
        **kwargs
            其他参数传递给seaborn.pairplot
            
        Returns
        -------
        sns.PairGrid
            seaborn PairGrid对象
        """
        if columns:
            plot_data = data[columns + ([hue] if hue else [])]
        else:
            plot_data = data
        
        g = sns.pairplot(
            plot_data,
            hue=hue,
            diag_kind=diag_kind,
            kind=kind,
            corner=corner,
            height=height,
            aspect=aspect,
            plot_kws={'alpha': 0.6, 's': 30},
            diag_kws={'fill': True},
            **kwargs
        )
        
        g.fig.suptitle('散点图矩阵', y=1.02, fontsize=14, fontweight='bold')
        
        # 保存
        if savefig:
            plt.savefig(savefig, dpi=100, bbox_inches='tight',
                       format=savefig.split('.')[-1] if '.' in savefig else 'png')
            print(f"  图表已保存: {savefig}")
        
        plt.show()
        return g
    
    def plot_boxplot(self,
                    data: pd.DataFrame,
                    numeric_col: str,
                    categorical_col: str,
                    kind: str = 'box',
                    figsize: Tuple[int, int] = (10, 6),
                    palette: str = 'Set2',
                    show_points: bool = False,
                    savefig: Optional[str] = None,
                    **kwargs) -> plt.Figure:
        """
        绘制数值变量按分类变量分组的箱线图/小提琴图。
        
        Parameters
        ----------
        data : pd.DataFrame
            输入数据
        numeric_col : str
            数值列名
        categorical_col : str
            分类列名
        kind : str, default='box'
            图表类型：'box'、'violin'、'boxen'、'strip'、'swarm'
        figsize : tuple, default=(10, 6)
            图表大小
        palette : str, default='Set2'
            颜色调色板
        show_points : bool, default=False
            是否显示原始数据点
        savefig : str, optional
            保存路径
        **kwargs
            其他参数
            
        Returns
        -------
        plt.Figure
            matplotlib图表对象
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # 根据类型选择绘图函数
        if kind == 'box':
            sns.boxplot(
                data=data,
                x=categorical_col,
                y=numeric_col,
                palette=palette,
                ax=ax,
                **kwargs
            )
        elif kind == 'violin':
            sns.violinplot(
                data=data,
                x=categorical_col,
                y=numeric_col,
                palette=palette,
                ax=ax,
                **kwargs
            )
        elif kind == 'boxen':
            sns.boxenplot(
                data=data,
                x=categorical_col,
                y=numeric_col,
                palette=palette,
                ax=ax,
                **kwargs
            )
        elif kind == 'strip':
            sns.stripplot(
                data=data,
                x=categorical_col,
                y=numeric_col,
                palette=palette,
                ax=ax,
                **kwargs
            )
        elif kind == 'swarm':
            sns.swarmplot(
                data=data,
                x=categorical_col,
                y=numeric_col,
                palette=palette,
                ax=ax,
                **kwargs
            )
        else:
            raise ValueError(f"未知的图表类型: {kind}")
        
        # 叠加数据点
        if show_points and kind in ['box', 'violin', 'boxen']:
            sns.stripplot(
                data=data,
                x=categorical_col,
                y=numeric_col,
                color='black',
                alpha=0.3,
                size=3,
                ax=ax
            )
        
        ax.set_title(f'{numeric_col} by {categorical_col}', fontsize=14, fontweight='bold')
        ax.set_xlabel(categorical_col, fontsize=12)
        ax.set_ylabel(numeric_col, fontsize=12)
        
        # 旋转x轴标签
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 保存
        if savefig:
            plt.savefig(savefig, dpi=100, bbox_inches='tight',
                       format=savefig.split('.')[-1] if '.' in savefig else 'png')
            print(f"  图表已保存: {savefig}")
        
        plt.show()
        return fig
    
    def plot_correlation_network(self,
                                corr_matrix: pd.DataFrame,
                                threshold: float = 0.5,
                                figsize: Tuple[int, int] = (12, 12),
                                savefig: Optional[str] = None,
                                **kwargs) -> plt.Figure:
        """
        绘制相关性网络图。
        
        Parameters
        ----------
        corr_matrix : pd.DataFrame
            相关性矩阵
        threshold : float, default=0.5
            显示连接的阈值
        figsize : tuple, default=(12, 12)
            图表大小
        savefig : str, optional
            保存路径
        **kwargs
            其他参数
            
        Returns
        -------
        plt.Figure
            matplotlib图表对象
        """
        try:
            import networkx as nx
        except ImportError:
            warnings.warn("networkx未安装，无法绘制网络图")
            return None
        
        # 创建图
        G = nx.Graph()
        
        # 添加节点
        for col in corr_matrix.columns:
            G.add_node(col)
        
        # 添加边（只添加强相关）
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # 避免重复
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) >= threshold:
                        G.add_edge(col1, col2, weight=abs(corr_val))
        
        # 绘制
        fig, ax = plt.subplots(figsize=figsize)
        
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # 节点大小基于连接数
        node_sizes = [G.degree(node) * 500 + 300 for node in G.nodes()]
        
        # 边宽度基于相关性强度
        edges = G.edges()
        weights = [G[u][v]['weight'] * 3 for u, v in edges]
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
        
        ax.set_title(f'相关性网络图 (|r| >= {threshold})', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        # 保存
        if savefig:
            plt.savefig(savefig, dpi=100, bbox_inches='tight',
                       format=savefig.split('.')[-1] if '.' in savefig else 'png')
            print(f"  图表已保存: {savefig}")
        
        plt.show()
        return fig
    
    def plot_significant_pairs(self,
                              significant_pairs: List[dict],
                              top_n: int = 20,
                              figsize: Tuple[int, int] = (10, 8),
                              savefig: Optional[str] = None,
                              **kwargs) -> plt.Figure:
        """
        绘制显著相关对条形图。
        
        Parameters
        ----------
        significant_pairs : list
            显著相关对列表
        top_n : int, default=20
            显示前N个
        figsize : tuple, default=(10, 8)
            图表大小
        savefig : str, optional
            保存路径
        **kwargs
            其他参数
            
        Returns
        -------
        plt.Figure
            matplotlib图表对象
        """
        if not significant_pairs:
            warnings.warn("没有显著相关对可绘制")
            return None
        
        # 取前N个
        pairs = significant_pairs[:top_n]
        
        # 准备数据
        labels = [f"{p['var1']}\nvs\n{p['var2']}" for p in pairs]
        values = [abs(p['correlation']) for p in pairs]
        colors = ['green' if p['correlation'] > 0 else 'red' for p in pairs]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.barh(range(len(labels)), values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('|Correlation|', fontsize=12)
        ax.set_title(f'Top {top_n} Significant Correlations', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # 保存
        if savefig:
            plt.savefig(savefig, dpi=100, bbox_inches='tight',
                       format=savefig.split('.')[-1] if '.' in savefig else 'png')
            print(f"  图表已保存: {savefig}")
        
        plt.show()
        return fig
