"""
结果导出与报告模块
==================
提供结果导出、文本摘要生成等功能。
"""

import warnings
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
import pandas as pd


class CorrReporter:
    """
    相关性报告生成器
    
    提供结果导出和文本摘要功能。
    """
    
    def __init__(self):
        self.interpretation_guide = {
            (0.0, 0.1): ("极弱相关", "几乎无线性关联"),
            (0.1, 0.3): ("弱相关", "存在轻微线性关联"),
            (0.3, 0.5): ("中等相关", "存在中等程度线性关联"),
            (0.5, 0.7): ("强相关", "存在较强线性关联"),
            (0.7, 0.9): ("很强相关", "存在很强线性关联"),
            (0.9, 1.0): ("极强相关", "几乎完全线性相关"),
        }
    
    def export_results(self,
                      corr_matrix: pd.DataFrame,
                      pvalue_matrix: pd.DataFrame,
                      significant_pairs: List[dict],
                      path: str,
                      format: str = 'excel') -> str:
        """
        导出分析结果。
        
        Parameters
        ----------
        corr_matrix : pd.DataFrame
            相关系数矩阵
        pvalue_matrix : pd.DataFrame
            p值矩阵
        significant_pairs : list
            显著相关对列表
        path : str
            保存路径
        format : str, default='excel'
            格式：'excel'、'csv'
            
        Returns
        -------
        str
            保存的文件路径
        """
        if format.lower() in ['excel', 'xlsx', 'xls']:
            return self._export_excel(
                corr_matrix, pvalue_matrix, significant_pairs, path
            )
        elif format.lower() == 'csv':
            return self._export_csv(
                corr_matrix, pvalue_matrix, significant_pairs, path
            )
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def _export_excel(self,
                     corr_matrix: pd.DataFrame,
                     pvalue_matrix: pd.DataFrame,
                     significant_pairs: List[dict],
                     path: str) -> str:
        """导出为Excel格式"""
        # 确保路径有正确后缀
        if not path.endswith('.xlsx'):
            path += '.xlsx'
        
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            # 相关系数矩阵
            corr_matrix.to_excel(writer, sheet_name='Correlation_Matrix')
            
            # p值矩阵
            pvalue_matrix.to_excel(writer, sheet_name='PValue_Matrix')
            
            # 显著相关对
            if significant_pairs:
                sig_df = pd.DataFrame(significant_pairs)
                sig_df = sig_df.sort_values('correlation', key=abs, ascending=False)
                sig_df.to_excel(writer, sheet_name='Significant_Pairs', index=False)
            
            # 摘要信息
            summary_data = {
                'Metric': [
                    '分析时间',
                    '变量数量',
                    '显著相关对数量',
                    '强相关对数量 (|r| >= 0.7)',
                    '中等相关对数量 (0.3 <= |r| < 0.7)',
                    '弱相关对数量 (|r| < 0.3)',
                ],
                'Value': [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    len(corr_matrix.columns),
                    len(significant_pairs),
                    len([p for p in significant_pairs if abs(p['correlation']) >= 0.7]),
                    len([p for p in significant_pairs if 0.3 <= abs(p['correlation']) < 0.7]),
                    len([p for p in significant_pairs if abs(p['correlation']) < 0.3]),
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"  结果已导出: {path}")
        return path
    
    def _export_csv(self,
                   corr_matrix: pd.DataFrame,
                   pvalue_matrix: pd.DataFrame,
                   significant_pairs: List[dict],
                   path: str) -> str:
        """导出为CSV格式（多个文件）"""
        import os
        
        base_path = path.replace('.csv', '')
        
        # 相关系数矩阵
        corr_path = f"{base_path}_correlation.csv"
        corr_matrix.to_csv(corr_path)
        
        # p值矩阵
        pval_path = f"{base_path}_pvalues.csv"
        pvalue_matrix.to_csv(pval_path)
        
        # 显著相关对
        if significant_pairs:
            sig_path = f"{base_path}_significant.csv"
            sig_df = pd.DataFrame(significant_pairs)
            sig_df = sig_df.sort_values('correlation', key=abs, ascending=False)
            sig_df.to_csv(sig_path, index=False)
        
        print(f"  结果已导出到目录: {os.path.dirname(os.path.abspath(base_path))}")
        return base_path
    
    def generate_summary(self,
                        significant_pairs: List[dict],
                        methods_used: Optional[Dict[str, str]] = None,
                        max_pairs_display: int = 10) -> str:
        """
        生成文本摘要。
        
        Parameters
        ----------
        significant_pairs : list
            显著相关对列表
        methods_used : dict, optional
            使用的方法记录
        max_pairs_display : int, default=10
            最多显示的变量对数量
            
        Returns
        -------
        str
            摘要文本
        """
        lines = []
        lines.append("=" * 60)
        lines.append("相关性分析摘要")
        lines.append("=" * 60)
        
        # 总体统计
        lines.append(f"\n📊 总体统计:")
        lines.append(f"  - 显著相关对总数: {len(significant_pairs)}")
        
        if significant_pairs:
            # 按强度分类
            strong = [p for p in significant_pairs if abs(p['correlation']) >= 0.7]
            moderate = [p for p in significant_pairs if 0.3 <= abs(p['correlation']) < 0.7]
            weak = [p for p in significant_pairs if abs(p['correlation']) < 0.3]
            
            lines.append(f"  - 强相关 (|r| >= 0.7): {len(strong)} 对")
            lines.append(f"  - 中等相关 (0.3 <= |r| < 0.7): {len(moderate)} 对")
            lines.append(f"  - 弱相关 (|r| < 0.3): {len(weak)} 对")
            
            # Top相关对
            lines.append(f"\n🔝 Top {min(max_pairs_display, len(significant_pairs))} 显著相关对:")
            lines.append("-" * 60)
            
            for i, pair in enumerate(significant_pairs[:max_pairs_display], 1):
                var1, var2 = pair['var1'], pair['var2']
                corr = pair['correlation']
                pval = pair['p_value']
                interp = pair['interpretation']
                method = pair.get('method', 'unknown')
                
                direction = "正相关" if corr > 0 else "负相关"
                sig_marker = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                
                lines.append(f"{i}. {var1} vs {var2}")
                lines.append(f"   相关系数: {corr:.4f} {sig_marker}")
                lines.append(f"   p值: {pval:.4e}")
                lines.append(f"   方法: {method}")
                lines.append(f"   解释: {direction}, {interp}")
                lines.append("")
        
        # 方法统计
        if methods_used:
            lines.append("\n📋 使用的方法统计:")
            method_counts = {}
            for method in methods_used.values():
                method_counts[method] = method_counts.get(method, 0) + 1
            
            for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  - {method}: {count} 次")
        
        # 解释指南
        lines.append("\n📖 相关系数解释指南:")
        lines.append("-" * 60)
        lines.append("  |r| < 0.1: 极弱相关（几乎无线性关联）")
        lines.append("  0.1 <= |r| < 0.3: 弱相关（轻微线性关联）")
        lines.append("  0.3 <= |r| < 0.5: 中等相关（中等程度线性关联）")
        lines.append("  0.5 <= |r| < 0.7: 强相关（较强线性关联）")
        lines.append("  0.7 <= |r| < 0.9: 很强相关（很强线性关联）")
        lines.append("  |r| >= 0.9: 极强相关（几乎完全线性相关）")
        lines.append("")
        lines.append("  显著性标记: *** p<0.001, ** p<0.01, * p<0.05")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def to_markdown(self,
                   corr_matrix: pd.DataFrame,
                   significant_pairs: List[dict],
                   title: str = "相关性分析报告") -> str:
        """
        生成Markdown格式报告。
        
        Parameters
        ----------
        corr_matrix : pd.DataFrame
            相关系数矩阵
        significant_pairs : list
            显著相关对列表
        title : str, default='相关性分析报告'
            报告标题
            
        Returns
        -------
        str
            Markdown格式文本
        """
        lines = []
        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 概述
        lines.append("## 概述")
        lines.append("")
        lines.append(f"- **变量数量**: {len(corr_matrix.columns)}")
        lines.append(f"- **显著相关对**: {len(significant_pairs)}")
        lines.append("")
        
        # 显著相关对表格
        if significant_pairs:
            lines.append("## 显著相关对")
            lines.append("")
            lines.append("| 排名 | 变量1 | 变量2 | 相关系数 | p值 | 方法 | 解释 |")
            lines.append("|------|-------|-------|----------|-----|------|------|")
            
            for i, pair in enumerate(significant_pairs[:20], 1):
                var1 = pair['var1']
                var2 = pair['var2']
                corr = f"{pair['correlation']:.4f}"
                pval = f"{pair['p_value']:.2e}"
                method = pair.get('method', '-')
                interp = pair['interpretation']
                
                lines.append(f"| {i} | {var1} | {var2} | {corr} | {pval} | {method} | {interp} |")
            
            lines.append("")
        
        # 相关系数矩阵
        lines.append("## 相关系数矩阵")
        lines.append("")
        lines.append(corr_matrix.round(3).to_markdown())
        lines.append("")
        
        # 解释指南
        lines.append("## 解释指南")
        lines.append("")
        lines.append("| 相关系数范围 | 强度 | 说明 |")
        lines.append("|--------------|------|------|")
        lines.append("| |r| < 0.1 | 极弱 | 几乎无线性关联 |")
        lines.append("| 0.1 ≤ |r| < 0.3 | 弱 | 轻微线性关联 |")
        lines.append("| 0.3 ≤ |r| < 0.5 | 中等 | 中等程度线性关联 |")
        lines.append("| 0.5 ≤ |r| < 0.7 | 强 | 较强线性关联 |")
        lines.append("| 0.7 ≤ |r| < 0.9 | 很强 | 很强线性关联 |")
        lines.append("| |r| ≥ 0.9 | 极强 | 几乎完全线性相关 |")
        lines.append("")
        
        return "\n".join(lines)
    
    def to_html(self,
               corr_matrix: pd.DataFrame,
               pvalue_matrix: pd.DataFrame,
               significant_pairs: List[dict],
               title: str = "相关性分析报告") -> str:
        """
        生成HTML格式报告。
        
        Parameters
        ----------
        corr_matrix : pd.DataFrame
            相关系数矩阵
        pvalue_matrix : pd.DataFrame
            p值矩阵
        significant_pairs : list
            显著相关对列表
        title : str, default='相关性分析报告'
            报告标题
            
        Returns
        -------
        str
            HTML格式文本
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .positive {{
            color: green;
        }}
        .negative {{
            color: red;
        }}
        .summary {{
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .timestamp {{
            color: #888;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p class="timestamp">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2>概述</h2>
        <p><strong>变量数量:</strong> {len(corr_matrix.columns)}</p>
        <p><strong>显著相关对:</strong> {len(significant_pairs)}</p>
    </div>
"""
        
        # 显著相关对表格
        if significant_pairs:
            html += """
    <h2>显著相关对</h2>
    <table>
        <tr>
            <th>排名</th>
            <th>变量1</th>
            <th>变量2</th>
            <th>相关系数</th>
            <th>p值</th>
            <th>方法</th>
            <th>解释</th>
        </tr>
"""
            for i, pair in enumerate(significant_pairs[:20], 1):
                corr_class = 'positive' if pair['correlation'] > 0 else 'negative'
                html += f"""
        <tr>
            <td>{i}</td>
            <td>{pair['var1']}</td>
            <td>{pair['var2']}</td>
            <td class="{corr_class}">{pair['correlation']:.4f}</td>
            <td>{pair['p_value']:.2e}</td>
            <td>{pair.get('method', '-')}</td>
            <td>{pair['interpretation']}</td>
        </tr>
"""
            html += "    </table>"
        
        # 相关系数矩阵
        html += """
    <h2>相关系数矩阵</h2>
    <table>
        <tr>
            <th>变量</th>
"""
        for col in corr_matrix.columns:
            html += f"            <th>{col}</th>\n"
        html += "        </tr>\n"
        
        for idx, row in corr_matrix.iterrows():
            html += f"        <tr>\n            <td><strong>{idx}</strong></td>\n"
            for val in row:
                corr_class = 'positive' if val > 0 else 'negative' if val < 0 else ''
                html += f'            <td class="{corr_class}">{val:.3f}</td>\n'
            html += "        </tr>\n"
        
        html += """    </table>
</body>
</html>
"""
        
        return html
