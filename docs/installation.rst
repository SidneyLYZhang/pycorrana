.. _installation:

============
安装指南
============

系统要求
========

PyCorrAna 需要以下环境：

- Python >= 3.10
- 支持 Python 3.10, 3.11, 3.12, 3.13

安装方法
========

使用 pip 安装
-------------

最简单的安装方式是使用 pip：

.. code-block:: bash

   pip install pycorrana

使用 uv 安装
------------

如果你使用 uv 作为包管理器：

.. code-block:: bash

   uv pip install pycorrana

从源码安装
----------

如果你需要最新的开发版本或想参与开发：

.. code-block:: bash

   git clone https://github.com/sidneylyzhang/pycorrana.git
   cd pycorrana
   pip install -e .

安装可选依赖
============

PyCorrAna 提供了几个可选依赖组：

开发依赖
--------

用于开发和测试：

.. code-block:: bash

   pip install pycorrana[dev]

包含：

- black - 代码格式化
- flake8 - 代码检查
- mypy - 类型检查
- pytest - 测试框架
- pytest-cov - 测试覆盖率

文档依赖
--------

用于构建文档：

.. code-block:: bash

   pip install pycorrana[docs]

包含：

- sphinx - 文档生成工具
- sphinx-rtd-theme - Read the Docs 主题

交互式环境依赖
--------------

用于 Jupyter 交互式分析：

.. code-block:: bash

   pip install pycorrana[interactive]

包含：

- ipython - 增强的 Python 交互环境
- jupyter - Jupyter Notebook

安装所有依赖
------------

一次性安装所有可选依赖：

.. code-block:: bash

   pip install pycorrana[all]

核心依赖
========

PyCorrAna 依赖以下核心库：

.. list-table::
   :header-rows: 1

   * - 库名
     - 版本要求
     - 用途
   * - numpy
     - >= 1.21.0
     - 数组计算
   * - pandas
     - >= 1.3.0
     - 数据处理
   * - polars
     - >= 0.15.0
     - 高性能数据处理
   * - scipy
     - >= 1.7.0
     - 统计检验
   * - matplotlib
     - >= 3.5.0
     - 基础绘图
   * - seaborn
     - >= 0.11.0
     - 高级可视化
   * - statsmodels
     - >= 0.13.0
     - 统计模型
   * - scikit-learn
     - >= 1.0.0
     - 机器学习工具
   * - openpyxl
     - >= 3.0.0
     - Excel 文件支持

验证安装
========

安装完成后，可以验证是否安装成功：

.. code-block:: python

   import pycorrana
   print(pycorrana.__version__)

如果输出版本号（如 ``0.1.0``），则表示安装成功。

你还可以测试基本功能：

.. code-block:: python

   from pycorrana import quick_corr, load_iris
   
   df = load_iris()
   result = quick_corr(df, plot=False)

常见问题
========

安装速度慢
----------

建议使用国内镜像源：

.. code-block:: bash

   pip install pycorrana -i https://pypi.tuna.tsinghua.edu.cn/simple

依赖冲突
--------

如果遇到依赖冲突，建议创建新的虚拟环境：

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # 或 venv\Scripts\activate  # Windows
   pip install pycorrana

Windows 安装问题
----------------

如果在 Windows 上安装遇到编译问题，可以尝试：

1. 确保安装了 Visual C++ Build Tools
2. 使用 conda 安装依赖库：

.. code-block:: bash

   conda install numpy pandas scipy matplotlib seaborn
   pip install pycorrana
