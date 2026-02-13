#!/usr/bin/env python
"""
Smoke Test Script
=================

Standalone smoke test script for verifying package installation.
Used in GitHub Actions to test wheel and source distributions.

This script tests that:
1. The package can be imported
2. All core modules are accessible
3. Basic functionality works
4. No crucial files are missing

Usage:
    python smoke_test.py

Exit codes:
    0 - All tests passed
    1 - One or more tests failed
"""

import sys
import traceback
from typing import Callable, List, Tuple


def test_import_package() -> Tuple[bool, str]:
    """Test that the package can be imported."""
    try:
        import pycorrana
        return True, f"Package imported successfully (version: {pycorrana.__version__})"
    except ImportError as e:
        return False, f"Failed to import package: {e}"


def test_import_core_analyzer() -> Tuple[bool, str]:
    """Test that core analyzer module can be imported."""
    try:
        from pycorrana.core import analyzer
        return True, "Core analyzer module imported successfully"
    except ImportError as e:
        return False, f"Failed to import core.analyzer: {e}"


def test_import_core_visualizer() -> Tuple[bool, str]:
    """Test that visualizer module can be imported."""
    try:
        from pycorrana.core import visualizer
        return True, "Visualizer module imported successfully"
    except ImportError as e:
        return False, f"Failed to import core.visualizer: {e}"


def test_import_core_reporter() -> Tuple[bool, str]:
    """Test that reporter module can be imported."""
    try:
        from pycorrana.core import reporter
        return True, "Reporter module imported successfully"
    except ImportError as e:
        return False, f"Failed to import core.reporter: {e}"


def test_import_core_partial_corr() -> Tuple[bool, str]:
    """Test that partial correlation module can be imported."""
    try:
        from pycorrana.core import partial_corr
        return True, "Partial correlation module imported successfully"
    except ImportError as e:
        return False, f"Failed to import core.partial_corr: {e}"


def test_import_core_nonlinear() -> Tuple[bool, str]:
    """Test that nonlinear module can be imported."""
    try:
        from pycorrana.core import nonlinear
        return True, "Nonlinear module imported successfully"
    except ImportError as e:
        return False, f"Failed to import core.nonlinear: {e}"


def test_import_utils_data_utils() -> Tuple[bool, str]:
    """Test that data_utils module can be imported."""
    try:
        from pycorrana.utils import data_utils
        return True, "Data utils module imported successfully"
    except ImportError as e:
        return False, f"Failed to import utils.data_utils: {e}"


def test_import_utils_stats_utils() -> Tuple[bool, str]:
    """Test that stats_utils module can be imported."""
    try:
        from pycorrana.utils import stats_utils
        return True, "Stats utils module imported successfully"
    except ImportError as e:
        return False, f"Failed to import utils.stats_utils: {e}"


def test_import_cli_main() -> Tuple[bool, str]:
    """Test that CLI main module can be imported."""
    try:
        from pycorrana.cli import main_cli
        return True, "CLI main module imported successfully"
    except ImportError as e:
        return False, f"Failed to import cli.main_cli: {e}"


def test_import_cli_interactive() -> Tuple[bool, str]:
    """Test that CLI interactive module can be imported."""
    try:
        from pycorrana.cli import interactive
        return True, "CLI interactive module imported successfully"
    except ImportError as e:
        return False, f"Failed to import cli.interactive: {e}"


def test_quick_corr_exists() -> Tuple[bool, str]:
    """Test that quick_corr function is available."""
    try:
        from pycorrana import quick_corr
        assert callable(quick_corr), "quick_corr is not callable"
        return True, "quick_corr function is available and callable"
    except (ImportError, AssertionError) as e:
        return False, f"quick_corr test failed: {e}"


def test_corr_analyzer_exists() -> Tuple[bool, str]:
    """Test that CorrAnalyzer class is available."""
    try:
        from pycorrana import CorrAnalyzer
        assert callable(CorrAnalyzer), "CorrAnalyzer is not callable"
        return True, "CorrAnalyzer class is available"
    except (ImportError, AssertionError) as e:
        return False, f"CorrAnalyzer test failed: {e}"


def test_partial_corr_functions() -> Tuple[bool, str]:
    """Test that partial correlation functions are available."""
    try:
        from pycorrana import (
            partial_corr,
            partial_corr_matrix,
            semipartial_corr,
            PartialCorrAnalyzer,
        )
        assert callable(partial_corr), "partial_corr is not callable"
        assert callable(partial_corr_matrix), "partial_corr_matrix is not callable"
        assert callable(semipartial_corr), "semipartial_corr is not callable"
        assert callable(PartialCorrAnalyzer), "PartialCorrAnalyzer is not callable"
        return True, "All partial correlation functions are available"
    except (ImportError, AssertionError) as e:
        return False, f"Partial correlation functions test failed: {e}"


def test_nonlinear_functions() -> Tuple[bool, str]:
    """Test that nonlinear analysis functions are available."""
    try:
        from pycorrana import (
            distance_correlation,
            mutual_info_score,
            maximal_information_coefficient,
            nonlinear_dependency_report,
            NonlinearAnalyzer,
        )
        assert callable(distance_correlation), "distance_correlation is not callable"
        assert callable(mutual_info_score), "mutual_info_score is not callable"
        assert callable(maximal_information_coefficient), "maximal_information_coefficient is not callable"
        assert callable(nonlinear_dependency_report), "nonlinear_dependency_report is not callable"
        assert callable(NonlinearAnalyzer), "NonlinearAnalyzer is not callable"
        return True, "All nonlinear analysis functions are available"
    except (ImportError, AssertionError) as e:
        return False, f"Nonlinear functions test failed: {e}"


def test_dataset_functions() -> Tuple[bool, str]:
    """Test that dataset functions are available."""
    try:
        from pycorrana import (
            load_iris,
            load_titanic,
            load_wine,
            make_correlated_data,
            list_datasets,
        )
        assert callable(load_iris), "load_iris is not callable"
        assert callable(load_titanic), "load_titanic is not callable"
        assert callable(load_wine), "load_wine is not callable"
        assert callable(make_correlated_data), "make_correlated_data is not callable"
        assert callable(list_datasets), "list_datasets is not callable"
        return True, "All dataset functions are available"
    except (ImportError, AssertionError) as e:
        return False, f"Dataset functions test failed: {e}"


def test_basic_functionality() -> Tuple[bool, str]:
    """Test basic analysis functionality."""
    try:
        from pycorrana import quick_corr, make_correlated_data
        
        df = make_correlated_data(n_samples=50, n_features=3, correlation=0.5)
        result = quick_corr(df, plot=False, verbose=False)
        
        assert 'correlation_matrix' in result, "Missing correlation_matrix in result"
        assert 'pvalue_matrix' in result, "Missing pvalue_matrix in result"
        assert 'significant_pairs' in result, "Missing significant_pairs in result"
        assert result['correlation_matrix'].shape[0] >= 3, "Correlation matrix too small"
        
        return True, f"Basic functionality works (matrix shape: {result['correlation_matrix'].shape})"
    except Exception as e:
        return False, f"Basic functionality test failed: {e}"


def test_version_attribute() -> Tuple[bool, str]:
    """Test that version attribute is set correctly."""
    try:
        import pycorrana
        assert hasattr(pycorrana, '__version__'), "Missing __version__ attribute"
        assert isinstance(pycorrana.__version__, str), "__version__ is not a string"
        assert len(pycorrana.__version__) > 0, "__version__ is empty"
        return True, f"Version attribute is set: {pycorrana.__version__}"
    except (AssertionError, AttributeError) as e:
        return False, f"Version attribute test failed: {e}"


def test_all_exports() -> Tuple[bool, str]:
    """Test that __all__ exports are correct."""
    try:
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
        
        missing = []
        for export in expected_exports:
            if export not in pycorrana.__all__:
                missing.append(export)
        
        if missing:
            return False, f"Missing exports: {missing}"
        
        return True, f"All {len(expected_exports)} expected exports are present"
    except Exception as e:
        return False, f"Exports test failed: {e}"


def test_dependencies() -> Tuple[bool, str]:
    """Test that all required dependencies are available."""
    dependencies = [
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'seaborn',
        'sklearn',
        'typer',
        'rich',
    ]
    
    missing = []
    for dep in dependencies:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    if missing:
        return False, f"Missing dependencies: {missing}"
    
    return True, f"All {len(dependencies)} dependencies are available"


def run_tests() -> int:
    """Run all smoke tests and return exit code."""
    tests: List[Tuple[str, Callable[[], Tuple[bool, str]]]] = [
        ("Import package", test_import_package),
        ("Import core.analyzer", test_import_core_analyzer),
        ("Import core.visualizer", test_import_core_visualizer),
        ("Import core.reporter", test_import_core_reporter),
        ("Import core.partial_corr", test_import_core_partial_corr),
        ("Import core.nonlinear", test_import_core_nonlinear),
        ("Import utils.data_utils", test_import_utils_data_utils),
        ("Import utils.stats_utils", test_import_utils_stats_utils),
        ("Import cli.main_cli", test_import_cli_main),
        ("Import cli.interactive", test_import_cli_interactive),
        ("quick_corr function", test_quick_corr_exists),
        ("CorrAnalyzer class", test_corr_analyzer_exists),
        ("Partial correlation functions", test_partial_corr_functions),
        ("Nonlinear functions", test_nonlinear_functions),
        ("Dataset functions", test_dataset_functions),
        ("Basic functionality", test_basic_functionality),
        ("Version attribute", test_version_attribute),
        ("__all__ exports", test_all_exports),
        ("Dependencies", test_dependencies),
    ]
    
    print("=" * 60)
    print("PyCorrAna Smoke Test")
    print("=" * 60)
    print()
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            success, message = test_func()
            status = "✓ PASS" if success else "✗ FAIL"
            color_start = "\033[92m" if success else "\033[91m"
            color_end = "\033[0m"
            
            print(f"{color_start}{status}{color_end} {name}")
            print(f"       {message}")
            
            if success:
                passed += 1
            else:
                failed += 1
                
        except Exception as e:
            failed += 1
            print(f"\033[91m✗ FAIL\033[0m {name}")
            print(f"       Unexpected error: {e}")
            traceback.print_exc()
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_tests())
