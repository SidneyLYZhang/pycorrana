.. _contributing:

============
贡献指南
============

感谢你对 PyCorrAna 项目感兴趣！本文档将帮助你了解如何为项目做出贡献。

行为准则
========

请阅读并遵守我们的行为准则。我们致力于提供友好、安全和欢迎的环境。

如何贡献
========

报告问题
--------

如果你发现了 bug 或有功能建议，请在 GitHub Issues 中提交：

https://github.com/sidneylyzhang/pycorrana/issues

提交问题时请包含：

1. 问题的简要描述
2. 复现步骤
3. 预期行为
4. 实际行为
5. 环境信息（Python 版本、操作系统等）
6. 如有可能，提供最小可复现示例

提交代码
--------

1. Fork 项目仓库

2. 克隆你的 Fork：

.. code-block:: bash

   git clone https://github.com/yourusername/pycorrana.git
   cd pycorrana

3. 创建开发环境：

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # 或 venv\Scripts\activate  # Windows
   pip install -e ".[dev]"

4. 创建功能分支：

.. code-block:: bash

   git checkout -b feature/your-feature-name

5. 进行修改并添加测试

6. 运行测试：

.. code-block:: bash

   pytest

7. 运行代码检查：

.. code-block:: bash

   black src tests
   flake8 src tests
   mypy src

8. 提交更改：

.. code-block:: bash

   git add .
   git commit -m "描述你的更改"

9. 推送到 Fork：

.. code-block:: bash

   git push origin feature/your-feature-name

10. 创建 Pull Request

开发指南
========

代码风格
--------

我们使用以下工具保持代码质量：

- **black** - 代码格式化
- **flake8** - 代码检查
- **mypy** - 类型检查

在提交代码前，请确保：

.. code-block:: bash

   black src tests
   flake8 src tests
   mypy src

文档字符串
----------

使用 NumPy 风格的文档字符串：

.. code-block:: python

   def function_name(param1, param2):
       """
       函数的简要描述。

       Parameters
       ----------
       param1 : type
           参数1的描述
       param2 : type, optional
           参数2的描述

       Returns
       -------
       return_type
           返回值的描述

       Examples
       --------
       >>> function_name(1, 2)
       3
       """
       pass

测试
----

- 所有新功能都需要添加测试
- 使用 pytest 框架
- 测试文件放在 ``tests/`` 目录
- 测试覆盖率应保持在高水平

运行测试：

.. code-block:: bash

   pytest

   pytest --cov=pycorrana

文档
----

文档使用 Sphinx 构建，放在 ``docs/`` 目录。

构建文档：

.. code-block:: bash

   cd docs
   make html

项目结构
========

.. code-block:: text

   pycorrana/
   ├── src/pycorrana/     # 源代码
   │   ├── core/          # 核心模块
   │   ├── utils/         # 工具函数
   │   └── cli/           # 命令行工具
   ├── tests/             # 测试
   ├── docs/              # 文档
   └── examples/          # 示例代码

发布流程
========

1. 更新版本号
2. 更新 CHANGELOG
3. 创建 Git 标签
4. 构建发布包
5. 发布到 PyPI

联系方式
========

- 项目主页：https://github.com/sidneylyzhang/pycorrana
- 问题反馈：https://github.com/sidneylyzhang/pycorrana/issues
- 作者：Sidney Zhang <zly@lyzhang.me>

感谢你的贡献！
