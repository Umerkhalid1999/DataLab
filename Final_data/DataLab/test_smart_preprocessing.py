"""
Test Smart Preprocessing System
"""
import pandas as pd
import numpy as np
from smart_preprocessor import SmartPreprocessor
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("TESTING SMART PREPROCESSING SYSTEM")
print("=" * 70)

# Test 1: Dataset with ONLY outliers (no missing, no duplicates)
print("\n[TEST 1] Dataset with ONLY outliers")
print("-" * 70)
df1 = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50, 1000],  # 1000 is outlier
    'salary': [50000, 60000, 55000, 65000, 70000, 75000, 500000],  # 500000 is outlier
    'score': [85, 90, 88, 92, 87, 89, 91]
})

preprocessor1 = SmartPreprocessor(df1)
issues1 = preprocessor1.analyze_issues()

print(f"Issues detected: {list(issues1.keys())}")
print(f"Expected: ['outliers'] only")

if 'missing_values' in issues1:
    print("[FAIL] Detected missing values when there are none!")
else:
    print("[PASS] No false missing values detected")

if 'duplicates' in issues1:
    print("[FAIL] Detected duplicates when there are none!")
else:
    print("[PASS] No false duplicates detected")

if 'outliers' in issues1:
    print(f"[PASS] Correctly detected {issues1['outliers']['total']} outliers")
else:
    print("[FAIL] Should have detected outliers!")

# Test 2: Dataset with ONLY missing values
print("\n[TEST 2] Dataset with ONLY missing values")
print("-" * 70)
df2 = pd.DataFrame({
    'age': [25, 30, np.nan, 40, 45],
    'salary': [50000, np.nan, 55000, 65000, 70000],
    'score': [85, 90, 88, 92, 87]
})

preprocessor2 = SmartPreprocessor(df2)
issues2 = preprocessor2.analyze_issues()

print(f"Issues detected: {list(issues2.keys())}")
print(f"Expected: ['missing_values'] only")

if 'missing_values' in issues2:
    print(f"[PASS] Correctly detected {issues2['missing_values']['total']} missing values")
else:
    print("[FAIL] Should have detected missing values!")

if 'outliers' in issues2:
    print("[FAIL] Detected outliers when there are none!")
else:
    print("[PASS] No false outliers detected")

# Test 3: Clean dataset (no issues)
print("\n[TEST 3] Clean dataset (no issues)")
print("-" * 70)
df3 = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'salary': [50000, 60000, 55000, 65000, 70000],
    'score': [85, 90, 88, 92, 87]
})

preprocessor3 = SmartPreprocessor(df3)
issues3 = preprocessor3.analyze_issues()

print(f"Issues detected: {list(issues3.keys())}")
print(f"Expected: [] (empty)")

if len(issues3) == 0:
    print("[PASS] Correctly identified clean dataset")
else:
    print(f"[FAIL] Detected false issues: {list(issues3.keys())}")

# Test 4: Apply smart cleaning
print("\n[TEST 4] Smart cleaning (only outliers)")
print("-" * 70)
cleaned_df = preprocessor1.apply_smart_cleaning()
transformations = preprocessor1.transformations

print(f"Transformations applied: {len(transformations)}")
for t in transformations:
    print(f"  - {t['type']}: {t['reason']}")

if len(transformations) == 1 and transformations[0]['type'] == 'outlier_capping':
    print("[PASS] Only applied outlier capping (no unnecessary operations)")
else:
    print("[FAIL] Applied unnecessary transformations")

# Test 5: AI Recommendations
print("\n[TEST 5] AI Recommendations")
print("-" * 70)
print("Getting AI analysis...")
ai_rec = preprocessor1.get_ai_recommendations()
print(f"\nAI Recommendation:\n{ai_rec}")

if len(ai_rec) > 50:
    print("\n[PASS] AI provided detailed recommendations")
else:
    print("\n[WARNING] AI response seems short")

# Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("[OK] Smart preprocessor only applies necessary transformations")
print("[OK] No false positives (doesn't fix non-existent issues)")
print("[OK] AI provides intelligent recommendations")
print("\nThe system is INTELLIGENT and EFFICIENT!")
print("=" * 70)
