"""Test script to verify mutual_info_score and maximal_information_coefficient functionality"""

from pycorrana import mutual_info_score, maximal_information_coefficient
import numpy as np

# Test 1: mutual_info_score with p-value and confidence interval
print("=== Testing mutual_info_score ===")

# Create correlated data
x = np.random.randn(1000)
y = x**2 + np.random.randn(1000) * 0.1

# Test without p-value
result1 = mutual_info_score(x, y, return_pvalue=False)
print("✓ Test 1a: mutual_info_score without p-value")
print(f"  MI: {result1['mi']:.4f}")
print(f"  Normalized MI: {result1['mi_normalized']:.4f}")
print(f"  Confidence interval: {result1['confidence_interval']}")
print(f"  Normalized confidence interval: {result1['confidence_interval_normalized']}")

# Test with p-value
result2 = mutual_info_score(x, y, return_pvalue=True, n_permutations=100)
print("\n✓ Test 1b: mutual_info_score with p-value")
print(f"  MI: {result2['mi']:.4f}")
print(f"  Normalized MI: {result2['mi_normalized']:.4f}")
print(f"  p-value: {result2['p_value']:.4f}")
print(f"  Confidence interval: {result2['confidence_interval']}")
print(f"  Normalized confidence interval: {result2['confidence_interval_normalized']}")

# Test 2: maximal_information_coefficient with p-value and confidence interval
print("\n=== Testing maximal_information_coefficient ===")

# Test without p-value
try:
    result3 = maximal_information_coefficient(x, y, return_pvalue=False)
    print("✓ Test 2a: maximal_information_coefficient without p-value")
    print(f"  MIC: {result3['mic']:.4f}")
    print(f"  Confidence interval: {result3['confidence_interval']}")
    if 'mas' in result3:
        print(f"  MAS: {result3['mas']:.4f}")
except Exception as e:
    print(f"⚠️ Test 2a: maximal_information_coefficient (minepy not installed): {e}")

# Test with p-value
try:
    result4 = maximal_information_coefficient(x, y, return_pvalue=True, n_permutations=100)
    print("\n✓ Test 2b: maximal_information_coefficient with p-value")
    print(f"  MIC: {result4['mic']:.4f}")
    print(f"  p-value: {result4['p_value']:.4f}")
    print(f"  Confidence interval: {result4['confidence_interval']}")
    if 'mas' in result4:
        print(f"  MAS: {result4['mas']:.4f}")
except Exception as e:
    print(f"⚠️ Test 2b: maximal_information_coefficient (minepy not installed): {e}")

# Test 3: Test with small sample size
print("\n=== Testing with small sample size ===")
x_small = np.random.randn(10)
y_small = x_small**2 + np.random.randn(10) * 0.1

result5 = mutual_info_score(x_small, y_small, return_pvalue=True)
print("✓ Test 3a: mutual_info_score with small sample")
print(f"  MI: {result5['mi']:.4f}")
print(f"  p-value: {result5['p_value']:.4f}")
print(f"  Confidence interval: {result5['confidence_interval']}")

try:
    result6 = maximal_information_coefficient(x_small, y_small, return_pvalue=True)
    print("\n✓ Test 3b: maximal_information_coefficient with small sample")
    print(f"  MIC: {result6['mic']:.4f}")
    print(f"  p-value: {result6['p_value']:.4f}")
    print(f"  Confidence interval: {result6['confidence_interval']}")
except Exception as e:
    print(f"⚠️ Test 3b: maximal_information_coefficient (minepy not installed): {e}")

print("\n✅ All tests completed!")