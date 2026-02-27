"""Test script to verify confidence interval functionality"""

from pycorrana import quick_corr, make_correlated_data, CorrAnalyzer
import pandas as pd

# Test 1: quick_corr function
print("=== Testing quick_corr function ===")
df = make_correlated_data(n_samples=100, n_features=4, correlation=0.7)
result = quick_corr(df, plot=False, verbose=False)

print("✓ quick_corr returned result")
print(f"  - Correlation matrix shape: {result['correlation_matrix'].shape}")
print(f"  - P-value matrix shape: {result['pvalue_matrix'].shape}")
print(f"  - Confidence interval matrix shape: {result['confidence_interval_matrix'].shape}")
print(f"  - Number of significant pairs: {len(result['significant_pairs'])}")

if result['significant_pairs']:
    first_pair = result['significant_pairs'][0]
    print(f"  - First pair: {first_pair['var1']} vs {first_pair['var2']}")
    print(f"    Correlation: {first_pair['correlation']:.4f}")
    print(f"    P-value: {first_pair['p_value']:.4f}")
    print(f"    Confidence interval: {first_pair['confidence_interval']}")
    print(f"    Method: {first_pair['method']}")

# Test 2: CorrAnalyzer class
print("\n=== Testing CorrAnalyzer class ===")
analyzer = CorrAnalyzer(df, verbose=False)
analyzer_result = analyzer.fit()

print("✓ CorrAnalyzer.fit() returned result")
print(f"  - Correlation matrix shape: {analyzer_result['correlation_matrix'].shape}")
print(f"  - P-value matrix shape: {analyzer_result['pvalue_matrix'].shape}")
print(f"  - Confidence interval matrix shape: {analyzer_result['confidence_interval_matrix'].shape}")
print(f"  - Number of significant pairs: {len(analyzer_result['significant_pairs'])}")

# Test 3: Test with different methods
print("\n=== Testing different correlation methods ===")
methods = ['pearson', 'spearman', 'kendall']
for method in methods:
    analyzer = CorrAnalyzer(df, method=method, verbose=False)
    result = analyzer.fit()
    print(f"  ✓ {method}: {len(result['significant_pairs'])} significant pairs")
    if result['significant_pairs']:
        ci = result['significant_pairs'][0]['confidence_interval']
        print(f"    First pair CI: {ci}")

print("\n✅ All tests passed!")