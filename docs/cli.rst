.. _cli:

==================
命令行工具
==================

PyCorrAna 提供了功能完整的命令行工具，无需编写代码即可完成相关性分析。

安装后的命令
============

安装 PyCorrAna 后，会提供两个命令：

- ``pycorrana`` - 主命令行工具
- ``pycorrana-interactive`` - 交互式分析工具

主命令行工具
============

基本语法
--------

.. code-block:: bash

   pycorrana <command> [options]

可用命令
--------

analyze - 完整分析
~~~~~~~~~~~~~~~~~~

执行完整的相关性分析：

.. code-block:: bash

   pycorrana analyze data.csv [options]

选项：

- ``--target, -t TARGET`` - 目标变量名
- ``--method, -m METHOD`` - 相关系数方法（auto/pearson/spearman/kendall）
- ``--columns, -c COL1,COL2`` - 指定分析的列，逗号分隔
- ``--export, -e PATH`` - 导出结果路径
- ``--no-plot`` - 不生成图表
- ``--missing STRATEGY`` - 缺失值处理策略（warn/drop/fill）
- ``--fill-method METHOD`` - 填充方法（mean/median/mode/knn）
- ``--pvalue-correction METHOD`` - p值校正方法
- ``--verbose, -v`` - 输出详细信息

示例：

.. code-block:: bash

   pycorrana analyze data.csv --target sales --export results.xlsx

   pycorrana analyze data.csv --method spearman --verbose

   pycorrana analyze data.csv --columns age,income,education --export output/

clean - 数据清洗
~~~~~~~~~~~~~~~~

数据清洗和预处理：

.. code-block:: bash

   pycorrana clean data.csv [options]

选项：

- ``--output, -o PATH`` - 输出文件路径（必需）
- ``--dropna`` - 删除缺失值
- ``--fill METHOD`` - 填充缺失值方法（mean/median/mode/knn）
- ``--detect-outliers`` - 检测异常值
- ``--outlier-method METHOD`` - 异常值检测方法（iqr/zscore）

示例：

.. code-block:: bash

   pycorrana clean data.csv --dropna --output cleaned.csv

   pycorrana clean data.csv --fill knn --output filled.csv

   pycorrana clean data.csv --detect-outliers --outlier-method iqr --output clean.csv

partial - 偏相关分析
~~~~~~~~~~~~~~~~~~~~

执行偏相关分析：

.. code-block:: bash

   pycorrana partial data.csv [options]

选项：

- ``-x VAR`` - 第一个变量（必需）
- ``-y VAR`` - 第二个变量（必需）
- ``-c, --covars VAR1,VAR2`` - 协变量列表，逗号分隔（必需）
- ``--method METHOD`` - 相关方法（pearson/spearman）
- ``--matrix`` - 计算偏相关矩阵

示例：

.. code-block:: bash

   pycorrana partial data.csv -x income -y happiness -c age,education

   pycorrana partial data.csv -x income -y happiness -c age --method spearman

   pycorrana partial data.csv -c age,education --matrix

nonlinear - 非线性检测
~~~~~~~~~~~~~~~~~~~~~~

检测非线性依赖关系：

.. code-block:: bash

   pycorrana nonlinear data.csv [options]

选项：

- ``--columns, -c COL1,COL2`` - 指定分析的列，逗号分隔
- ``--methods, -m METHODS`` - 检测方法，逗号分隔（dcor/mi/mic）
- ``--top N`` - 显示前 N 个最强的非线性关系（默认 10）
- ``--export, -e PATH`` - 导出结果路径

示例：

.. code-block:: bash

   pycorrana nonlinear data.csv --top 20

   pycorrana nonlinear data.csv -c col1,col2,col3 -m dcor,mic

   pycorrana nonlinear data.xlsx --top 15 --export nonlinear_results.csv

info - 数据信息
~~~~~~~~~~~~~~~

查看数据基本信息：

.. code-block:: bash

   pycorrana info data.csv [options]

选项：

- ``--types`` - 显示类型推断结果
- ``--missing`` - 显示缺失值详细信息

示例：

.. code-block:: bash

   pycorrana info data.csv

   pycorrana info data.xlsx --types --missing

输出包括：

- 数据维度（行数、列数）
- 列名和数据类型
- 非空值数量
- 缺失值数量和比例
- 唯一值数量
- 类型推断结果（可选）
- 缺失值详情（可选）

交互式工具
==========

启动交互式分析
--------------

.. code-block:: bash

   pycorrana-interactive

交互式流程
----------

启动后会进入交互式菜单：

.. code-block:: text

   ╭──────────────────────────────────────────╮
   │  PyCorrAna - 交互式相关性分析工具        │
   │  自动化相关性分析，降低决策成本          │
   ╰──────────────────────────────────────────╯
   
   请选择操作：
   1. 加载数据
   2. 执行完整分析
   3. 数据探索
   4. 数据清洗
   5. 相关性分析
   6. 偏相关分析
   7. 非线性依赖检测
   8. 可视化
   9. 导出结果
   0. 退出
   
   请输入选项：

数据加载
--------

选择"加载数据"后，可以：

1. 输入文件路径
2. 选择示例数据集（iris、titanic、wine）
3. 生成模拟数据

完整分析
--------

选择"执行完整分析"会自动完成：

1. 数据预处理
2. 相关性计算
3. 显著性检验
4. 生成可视化
5. 输出摘要

数据探索
--------

提供数据探索功能：

- 查看数据前几行
- 数据类型统计
- 缺失值分析
- 描述性统计
- 相关性预览

数据清洗
--------

交互式数据清洗：

- 处理缺失值
- 删除异常值
- 类型转换
- 重命名列

可视化选项
----------

交互式可视化：

- 热力图
- 散点图矩阵
- 箱线图
- 相关网络图
- 自定义图表

导出选项
--------

多种导出格式：

- Excel 文件
- CSV 文件
- HTML 报告
- 图片文件

帮助信息
========

查看帮助
--------

.. code-block:: bash

   pycorrana --help
   pycorrana analyze --help
   pycorrana partial --help
   pycorrana nonlinear --help

查看版本
--------

.. code-block:: bash

   pycorrana --version
