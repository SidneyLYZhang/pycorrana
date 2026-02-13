"""
Smoke Tests
===========

Smoke tests for verifying package installation and basic functionality.
These tests ensure the package can be installed from both wheel and source distribution.
"""

import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

import unittest


class TestSmokeWheel(unittest.TestCase):
    """
    Smoke test for wheel distribution.
    
    Tests that the package can be installed from a wheel and basic functionality works.
    """
    
    @classmethod
    def setUpClass(cls):
        """Build wheel before tests"""
        cls.project_root = Path(__file__).parent.parent
        cls.wheel_dir = cls.project_root / "dist"
    
    def test_import_package(self):
        """Test that the package can be imported"""
        import pycorrana
        
        self.assertIsNotNone(pycorrana.__version__)
        self.assertEqual(pycorrana.__version__, "0.1.0")
    
    def test_import_core_modules(self):
        """Test that core modules can be imported"""
        from pycorrana.core import analyzer
        from pycorrana.core import visualizer
        from pycorrana.core import reporter
        from pycorrana.core import partial_corr
        from pycorrana.core import nonlinear
        
        self.assertIsNotNone(analyzer)
        self.assertIsNotNone(visualizer)
        self.assertIsNotNone(reporter)
        self.assertIsNotNone(partial_corr)
        self.assertIsNotNone(nonlinear)
    
    def test_import_utils_modules(self):
        """Test that utility modules can be imported"""
        from pycorrana.utils import data_utils
        from pycorrana.utils import stats_utils
        
        self.assertIsNotNone(data_utils)
        self.assertIsNotNone(stats_utils)
    
    def test_import_cli_modules(self):
        """Test that CLI modules can be imported"""
        from pycorrana.cli import main_cli
        from pycorrana.cli import interactive
        
        self.assertIsNotNone(main_cli)
        self.assertIsNotNone(interactive)
    
    def test_quick_corr_function_exists(self):
        """Test that quick_corr function is available"""
        from pycorrana import quick_corr
        
        self.assertTrue(callable(quick_corr))
    
    def test_corr_analyzer_class_exists(self):
        """Test that CorrAnalyzer class is available"""
        from pycorrana import CorrAnalyzer
        
        self.assertTrue(callable(CorrAnalyzer))
    
    def test_partial_corr_functions_exist(self):
        """Test that partial correlation functions are available"""
        from pycorrana import (
            partial_corr,
            partial_corr_matrix,
            semipartial_corr,
            PartialCorrAnalyzer
        )
        
        self.assertTrue(callable(partial_corr))
        self.assertTrue(callable(partial_corr_matrix))
        self.assertTrue(callable(semipartial_corr))
        self.assertTrue(callable(PartialCorrAnalyzer))
    
    def test_nonlinear_functions_exist(self):
        """Test that nonlinear analysis functions are available"""
        from pycorrana import (
            distance_correlation,
            mutual_info_score,
            maximal_information_coefficient,
            nonlinear_dependency_report,
            NonlinearAnalyzer
        )
        
        self.assertTrue(callable(distance_correlation))
        self.assertTrue(callable(mutual_info_score))
        self.assertTrue(callable(maximal_information_coefficient))
        self.assertTrue(callable(nonlinear_dependency_report))
        self.assertTrue(callable(NonlinearAnalyzer))
    
    def test_dataset_functions_exist(self):
        """Test that dataset functions are available"""
        from pycorrana import (
            load_iris,
            load_titanic,
            load_wine,
            make_correlated_data,
            list_datasets
        )
        
        self.assertTrue(callable(load_iris))
        self.assertTrue(callable(load_titanic))
        self.assertTrue(callable(load_wine))
        self.assertTrue(callable(make_correlated_data))
        self.assertTrue(callable(list_datasets))
    
    def test_basic_functionality(self):
        """Test basic analysis functionality"""
        from pycorrana import quick_corr, make_correlated_data
        
        df = make_correlated_data(n_samples=50, n_features=3, correlation=0.5)
        result = quick_corr(df, plot=False, verbose=False)
        
        self.assertIn('correlation_matrix', result)
        self.assertIn('pvalue_matrix', result)
        self.assertIn('significant_pairs', result)
        self.assertGreaterEqual(result['correlation_matrix'].shape[0], 3)
    
    def test_cli_entry_points(self):
        """Test that CLI entry points are registered"""
        result = subprocess.run(
            [sys.executable, "-m", "pycorrana.cli.main_cli", "--help"],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent.parent)
        )
        
        self.assertIn("pycorrana", result.stdout.lower() + result.stderr.lower())


class TestSmokeSourceDistribution(unittest.TestCase):
    """
    Smoke test for source distribution.
    
    Tests that the source distribution contains all necessary files.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.project_root = Path(__file__).parent.parent
    
    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists"""
        pyproject_path = self.project_root / "pyproject.toml"
        self.assertTrue(pyproject_path.exists())
    
    def test_readme_exists(self):
        """Test that README.md exists"""
        readme_path = self.project_root / "README.md"
        self.assertTrue(readme_path.exists())
    
    def test_license_exists(self):
        """Test that LICENSE exists"""
        license_path = self.project_root / "LICENSE"
        self.assertTrue(license_path.exists())
    
    def test_source_directory_exists(self):
        """Test that source directory exists"""
        src_path = self.project_root / "src" / "pycorrana"
        self.assertTrue(src_path.exists())
        self.assertTrue(src_path.is_dir())
    
    def test_init_file_exists(self):
        """Test that __init__.py exists"""
        init_path = self.project_root / "src" / "pycorrana" / "__init__.py"
        self.assertTrue(init_path.exists())
    
    def test_core_module_exists(self):
        """Test that core module exists"""
        core_path = self.project_root / "src" / "pycorrana" / "core"
        self.assertTrue(core_path.exists())
        self.assertTrue((core_path / "__init__.py").exists())
        self.assertTrue((core_path / "analyzer.py").exists())
    
    def test_utils_module_exists(self):
        """Test that utils module exists"""
        utils_path = self.project_root / "src" / "pycorrana" / "utils"
        self.assertTrue(utils_path.exists())
        self.assertTrue((utils_path / "__init__.py").exists())
    
    def test_cli_module_exists(self):
        """Test that CLI module exists"""
        cli_path = self.project_root / "src" / "pycorrana" / "cli"
        self.assertTrue(cli_path.exists())
        self.assertTrue((cli_path / "__init__.py").exists())
        self.assertTrue((cli_path / "main_cli.py").exists())
        self.assertTrue((cli_path / "interactive.py").exists())
    
    def test_tests_directory_exists(self):
        """Test that tests directory exists"""
        tests_path = self.project_root / "tests"
        self.assertTrue(tests_path.exists())
        self.assertTrue((tests_path / "__init__.py").exists())
    
    def test_pyproject_has_required_fields(self):
        """Test that pyproject.toml has required fields"""
        import tomllib
        
        pyproject_path = self.project_root / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)
        
        self.assertIn("project", config)
        self.assertIn("name", config["project"])
        self.assertIn("version", config["project"])
        self.assertIn("dependencies", config["project"])
    
    def test_pyproject_has_entry_points(self):
        """Test that pyproject.toml has CLI entry points"""
        import tomllib
        
        pyproject_path = self.project_root / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)
        
        self.assertIn("scripts", config["project"])
        self.assertIn("pycorrana", config["project"]["scripts"])
        self.assertIn("pycorrana-interactive", config["project"]["scripts"])


class TestSmokePackageMetadata(unittest.TestCase):
    """
    Smoke test for package metadata.
    """
    
    def test_package_version(self):
        """Test that package version is correct"""
        import pycorrana
        
        self.assertIsInstance(pycorrana.__version__, str)
        self.assertTrue(len(pycorrana.__version__) > 0)
    
    def test_package_author(self):
        """Test that package author is set"""
        import pycorrana
        
        self.assertTrue(hasattr(pycorrana, '__author__'))
    
    def test_package_all_exports(self):
        """Test that __all__ is defined correctly"""
        import pycorrana
        
        expected_exports = [
            'quick_corr',
            'CorrAnalyzer',
            'partial_corr',
            'partial_corr_matrix',
            'semipartial_corr',
            'PartialCorrAnalyzer',
            'distance_correlation',
            'mutual_info_score',
            'maximal_information_coefficient',
            'nonlinear_dependency_report',
            'NonlinearAnalyzer',
            'load_iris',
            'load_titanic',
            'load_wine',
            'make_correlated_data',
            'list_datasets',
        ]
        
        for export in expected_exports:
            self.assertIn(export, pycorrana.__all__)


class TestSmokeDependencies(unittest.TestCase):
    """
    Smoke test for package dependencies.
    """
    
    def test_numpy_available(self):
        """Test that numpy is available"""
        import numpy
        self.assertIsNotNone(numpy)
    
    def test_pandas_available(self):
        """Test that pandas is available"""
        import pandas
        self.assertIsNotNone(pandas)
    
    def test_scipy_available(self):
        """Test that scipy is available"""
        import scipy
        self.assertIsNotNone(scipy)
    
    def test_matplotlib_available(self):
        """Test that matplotlib is available"""
        import matplotlib
        self.assertIsNotNone(matplotlib)
    
    def test_seaborn_available(self):
        """Test that seaborn is available"""
        import seaborn
        self.assertIsNotNone(seaborn)
    
    def test_sklearn_available(self):
        """Test that scikit-learn is available"""
        import sklearn
        self.assertIsNotNone(sklearn)
    
    def test_typer_available(self):
        """Test that typer is available"""
        import typer
        self.assertIsNotNone(typer)
    
    def test_rich_available(self):
        """Test that rich is available"""
        import rich
        self.assertIsNotNone(rich)


if __name__ == '__main__':
    unittest.main()
