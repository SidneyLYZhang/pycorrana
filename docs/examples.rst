.. _examples:

================
示例代码
================

本节提供 PyCorrAna 的各种使用示例。

基础示例
========

快速分析 CSV 文件
-----------------

.. code-block:: python

   from pycorrana import quick_corr

   result = quick_corr('sales_data.csv')
   
   print(result['significant_pairs'][:5])

分析 Excel 数据
---------------

.. code-block:: python

   from pycorrana import quick_corr

   result = quick_corr(
       'data.xlsx',
       target='revenue',
       export='correlation_results.xlsx'
   )

指定分析方法
------------

.. code-block:: python

   from pycorrana import CorrAnalyzer

   analyzer = CorrAnalyzer(
       df,
       method='spearman',
       missing_strategy='fill',
       fill_method='median'
   )
   
   result = analyzer.fit()
   analyzer.plot_heatmap()

数据分析流程
============

完整分析流程
------------

.. code-block:: python

   import pandas as pd
   from pycorrana import CorrAnalyzer

   df = pd.read_csv('data.csv')
   
   analyzer = CorrAnalyzer(df, verbose=True)
   
   analyzer.preprocess()
   
   analyzer.compute_correlation(target='target_column')
   
   result = {
       'correlation_matrix': analyzer.corr_matrix,
       'pvalue_matrix': analyzer.pvalue_matrix,
       'significant_pairs': analyzer.significant_pairs
   }
   
   analyzer.plot_heatmap(figsize=(14, 12), cluster=True)
   
   analyzer.export_results('results.xlsx')

分步分析
--------

.. code-block:: python

   from pycorrana import CorrAnalyzer
   from pycorrana.utils.data_utils import infer_types, handle_missing

   type_mapping = infer_types(df)
   print("数据类型:", type_mapping)
   
   df_clean = handle_missing(df, strategy='fill', fill_method='mean')
   
   analyzer = CorrAnalyzer(df_clean, method='auto')
   
   result = analyzer.fit(columns=['var1', 'var2', 'var3', 'target'])
   
   print(analyzer.summary())

可视化示例
==========

自定义热力图
------------

.. code-block:: python

   from pycorrana import CorrAnalyzer

   analyzer = CorrAnalyzer(df)
   analyzer.fit()
   
   analyzer.plot_heatmap(
       figsize=(16, 14),
       annot=True,
       fmt='.2f',
       cmap='coolwarm',
       center=0,
       vmin=-1,
       vmax=1,
       linewidths=0.5,
       cluster=True,
       cluster_method='average',
       savefig='heatmap.png',
       dpi=300
   )

散点图矩阵
----------

.. code-block:: python

   analyzer.plot_pairplot(
       columns=['age', 'income', 'education', 'score'],
       hue='gender',
       diag_kind='kde',
       corner=True,
       savefig='pairplot.png'
   )

分组箱线图
----------

.. code-block:: python

   analyzer.plot_boxplot(
       numeric_col='salary',
       categorical_col='department',
       kind='violin',
       savefig='salary_by_dept.png'
   )

相关网络图
----------

.. code-block:: python

   analyzer.visualizer.plot_correlation_network(
       analyzer.corr_matrix,
       threshold=0.4,
       node_size=1000,
       layout='circular',
       savefig='network.png'
   )

显著相关对条形图
----------------

.. code-block:: python

   analyzer.visualizer.plot_significant_pairs(
       analyzer.significant_pairs,
       top_n=15,
       savefig='top_correlations.png'
   )

偏相关分析示例
==============

控制单个协变量
--------------

.. code-block:: python

   from pycorrana import partial_corr

   r, p = partial_corr(
       df,
       x='income',
       y='happiness',
       covars='age'
   )
   
   print(f"偏相关系数: {r:.4f}, p值: {p:.4f}")

控制多个协变量
--------------

.. code-block:: python

   from pycorrana import partial_corr

   covars = ['age', 'education', 'gender', 'location']
   
   r, p = partial_corr(
       df,
       x='income',
       y='happiness',
       covars=covars
   )

偏相关矩阵
----------

.. code-block:: python

   from pycorrana import partial_corr_matrix

   matrix = partial_corr_matrix(
       df,
       covars=['age'],
       columns=['income', 'health', 'happiness', 'social']
   )
   
   print(matrix)

使用 PartialCorrAnalyzer
------------------------

.. code-block:: python

   from pycorrana import PartialCorrAnalyzer

   analyzer = PartialCorrAnalyzer(df, covars=['age', 'education'])
   
   result = analyzer.fit(x='income', y='happiness')
   
   matrix = analyzer.compute_matrix(columns=['income', 'health', 'happiness'])

非线性分析示例
==============

距离相关
--------

.. code-block:: python

   from pycorrana import distance_correlation
   import numpy as np

   x = np.random.randn(100)
   y = x ** 2 + np.random.randn(100) * 0.1
   
   dcor = distance_correlation(x, y)
   print(f"距离相关系数: {dcor:.4f}")

互信息分析
----------

.. code-block:: python

   from pycorrana import mutual_info_score

   mi = mutual_info_score(df['feature1'], df['feature2'])
   print(f"互信息: {mi:.4f}")

最大信息系数
最大信息系数
------------

.. code-block:: python

   from pycorrana import maximal_information_coefficient

   mic = maximal_information_coefficient(df['x'], df['y'])
   print(f"MIC: {mic['mic']:.4f}")

.. note::

   **性能说明**：当前 MIC 实现为纯 Python 版本，计算速度较慢。对于大数据集，建议先采样：

   .. code-block:: python

      from pycorrana.utils import smart_sample

      # 采样后再计算 MIC
      sampled_df = smart_sample(df, sample_size=500)
      mic = maximal_information_coefficient(sampled_df['x'], sampled_df['y'])

非线性依赖报告
--------------

.. code-block:: python

   from pycorrana import nonlinear_dependency_report

   report = nonlinear_dependency_report(
       df,
       top_n=20,
       methods=['dcor', 'mic']
   )
   
   print(report)

使用 NonlinearAnalyzer
----------------------

.. code-block:: python

   from pycorrana import NonlinearAnalyzer

   analyzer = NonlinearAnalyzer(df)
   
   result = analyzer.analyze_all(top_n=10)
   
   analyzer.plot_nonlinear_pairs(savefig='nonlinear.png')

示例数据集使用
==============

鸢尾花数据集
------------

.. code-block:: python

   from pycorrana import load_iris, quick_corr

   iris = load_iris()
   
   result = quick_corr(iris, target='species')

泰坦尼克数据集
--------------

.. code-block:: python

   from pycorrana import load_titanic, CorrAnalyzer

   titanic = load_titanic()
   
   analyzer = CorrAnalyzer(
       titanic,
       missing_strategy='fill',
       fill_method='median'
   )
   
   result = analyzer.fit(target='survived')

葡萄酒数据集
------------

.. code-block:: python

   from pycorrana import load_wine, quick_corr

   wine = load_wine()
   
   result = quick_corr(wine, plot=True)

生成模拟数据
------------

.. code-block:: python

   from pycorrana import make_correlated_data, CorrAnalyzer

   df = make_correlated_data(
       n_samples=500,
       n_features=8,
       correlation_strength=0.6,
       noise_level=0.2
   )
   
   analyzer = CorrAnalyzer(df)
   result = analyzer.fit()
   analyzer.plot_heatmap(cluster=True)

高级用法
========

自定义分析流程
--------------

.. code-block:: python

   from pycorrana import CorrAnalyzer
   from pycorrana.utils.data_utils import load_data, infer_types
   from pycorrana.utils.stats_utils import correct_pvalues

   df = load_data('data.csv')
   
   type_mapping = infer_types(df)
   numeric_cols = [k for k, v in type_mapping.items() if v == 'numeric']
   
   analyzer = CorrAnalyzer(df[numeric_cols], method='pearson')
   result = analyzer.fit()
   
   pvalues = result['pvalue_matrix'].values.flatten()
   pvalues = pvalues[~np.isnan(pvalues)]
   corrected = correct_pvalues(pvalues.tolist(), method='bonferroni')

批量处理多个文件
----------------

.. code-block:: python

   import os
   from pycorrana import quick_corr

   data_dir = 'data/'
   output_dir = 'results/'
   
   for filename in os.listdir(data_dir):
       if filename.endswith('.csv'):
           filepath = os.path.join(data_dir, filename)
           output_path = os.path.join(output_dir, f'{filename}_results.xlsx')
           
           result = quick_corr(
               filepath,
               export=output_path,
               plot=False,
               verbose=False
           )
           
           print(f"Processed: {filename}")

结合 pandas 分析
----------------

.. code-block:: python

   import pandas as pd
   from pycorrana import CorrAnalyzer

   df = pd.read_csv('data.csv')
   
   grouped = df.groupby('category')
   
   for name, group in grouped:
       print(f"\n=== Group: {name} ===")
       analyzer = CorrAnalyzer(group)
       result = analyzer.fit()
       print(analyzer.summary())
