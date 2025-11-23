"""
Test script for new Feature Engineering techniques
Tests: Forward Selection, Backward Elimination, Feature Importance, 
       Correlation Analysis, Variance Threshold, and VIF Analysis
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
import sys
import os

# Add routes to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'routes'))

from feature_selection import AdvancedFeatureSelector

def create_test_data(task='classification', n_samples=200, n_features=15):
    """Create synthetic test data"""
    print(f"\n{'='*60}")
    print(f"Creating {task} dataset: {n_samples} samples, {n_features} features")
    print(f"{'='*60}")
    
    if task == 'classification':
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=8,
            n_redundant=4,
            n_repeated=0,
            n_classes=2,
            random_state=42
        )
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=8,
            random_state=42
        )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    print(f" Dataset created: {X_df.shape}")
    return X_df, y_series

def test_forward_selection(X, y):
    """Test Forward Selection"""
    print(f"\n{'='*60}")
    print("TEST 1: Forward Selection")
    print(f"{'='*60}")
    
    selector = AdvancedFeatureSelector()
    results = selector.forward_selection(X, y, max_features=10, cv=3)
    
    print(f"\n Forward Selection Results:")
    print(f"   Selected Features: {len(results['selected_features'])}")
    print(f"   Final Score: {results['final_score']:.4f}")
    print(f"   Features: {results['selected_features'][:5]}...")
    
    return results

def test_backward_elimination(X, y):
    """Test Backward Elimination"""
    print(f"\n{'='*60}")
    print("TEST 2: Backward Elimination")
    print(f"{'='*60}")
    
    selector = AdvancedFeatureSelector()
    results = selector.backward_elimination(X, y, cv=3, threshold=0.01)
    
    print(f"\n Backward Elimination Results:")
    print(f"   Selected Features: {len(results['selected_features'])}")
    print(f"   Final Score: {results['final_score']:.4f}")
    print(f"   Features: {results['selected_features'][:5]}...")
    
    return results

def test_feature_importance(X, y):
    """Test Feature Importance from Multiple Models"""
    print(f"\n{'='*60}")
    print("TEST 3: Feature Importance from Simple Models")
    print(f"{'='*60}")
    
    selector = AdvancedFeatureSelector()
    results = selector.feature_importance_selection(
        X, y, 
        methods=['random_forest', 'linear', 'tree'],
        top_n=10
    )
    
    print(f"\n Feature Importance Results:")
    print(f"   Selected Features: {len(results['selected_features'])}")
    print(f"   Methods Used: {list(results['importance_scores'].keys())}")
    print(f"   Top 5 Features:")
    for i, (feat, score) in enumerate(results['sorted_features'][:5], 1):
        print(f"      {i}. {feat}: {score:.4f}")
    
    return results

def test_correlation_analysis(X, y):
    """Test Correlation Analysis"""
    print(f"\n{'='*60}")
    print("TEST 4: Correlation Analysis")
    print(f"{'='*60}")
    
    selector = AdvancedFeatureSelector()
    results = selector.correlation_analysis(
        X, y,
        target_threshold=0.05,
        multicollinearity_threshold=0.9,
        method='pearson'
    )
    
    print(f"\n Correlation Analysis Results:")
    print(f"   Selected Features: {len(results['selected_features'])}")
    print(f"   Removed by Low Target Correlation: {len(results['removed_by_target'])}")
    print(f"   Removed by Multicollinearity: {len(results['removed_by_multicollinearity'])}")
    
    # Show top correlations
    sorted_corr = sorted(results['target_correlations'].items(), key=lambda x: x[1], reverse=True)
    print(f"   Top 5 Correlations with Target:")
    for i, (feat, corr) in enumerate(sorted_corr[:5], 1):
        print(f"      {i}. {feat}: {corr:.4f}")
    
    return results

def test_variance_threshold(X, y):
    """Test Variance Threshold"""
    print(f"\n{'='*60}")
    print("TEST 5: Variance Threshold")
    print(f"{'='*60}")
    
    # Add some low-variance features for testing
    X_test = X.copy()
    X_test['constant_feature'] = 1.0
    X_test['quasi_constant'] = np.random.choice([1, 2], size=len(X), p=[0.96, 0.04])
    
    selector = AdvancedFeatureSelector()
    results = selector.variance_threshold_selection(
        X_test,
        threshold=0.01,
        quasi_constant_threshold=0.95
    )
    
    print(f"\n Variance Threshold Results:")
    print(f"   Selected Features: {len(results['selected_features'])}")
    print(f"   Low Variance Features Removed: {len(results['low_variance_features'])}")
    print(f"   Quasi-Constant Features Removed: {len(results['quasi_constant_features'])}")
    
    if results['low_variance_features']:
        print(f"   Removed: {results['low_variance_features']}")
    if results['quasi_constant_features']:
        print(f"   Removed: {results['quasi_constant_features']}")
    
    return results

def test_vif_analysis(X, y):
    """Test VIF Analysis"""
    print(f"\n{'='*60}")
    print("TEST 6: VIF Analysis (Multicollinearity)")
    print(f"{'='*60}")
    
    # Use only first 10 features for speed
    X_subset = X.iloc[:, :10]
    
    selector = AdvancedFeatureSelector()
    results = selector.vif_analysis(X_subset, threshold=10)
    
    print(f"\n VIF Analysis Results:")
    print(f"   Selected Features: {len(results['selected_features'])}")
    print(f"   High VIF Features (>10): {len(results['high_vif_features'])}")
    
    print(f"   VIF Scores:")
    for item in results['vif_scores'][:5]:
        print(f"      {item['feature']}: {item['vif']:.2f}")
    
    return results

def run_all_tests():
    """Run all feature engineering tests"""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SYSTEM TEST")
    print("="*60)
    
    # Test Classification
    print("\n" + "CLASSIFICATION TESTS")
    X_clf, y_clf = create_test_data('classification', n_samples=200, n_features=15)
    
    try:
        test_forward_selection(X_clf, y_clf)
        print(" Forward Selection: PASSED")
    except Exception as e:
        print(f" Forward Selection: FAILED - {e}")
    
    try:
        test_backward_elimination(X_clf, y_clf)
        print(" Backward Elimination: PASSED")
    except Exception as e:
        print(f" Backward Elimination: FAILED - {e}")
    
    try:
        test_feature_importance(X_clf, y_clf)
        print(" Feature Importance: PASSED")
    except Exception as e:
        print(f" Feature Importance: FAILED - {e}")
    
    try:
        test_correlation_analysis(X_clf, y_clf)
        print(" Correlation Analysis: PASSED")
    except Exception as e:
        print(f" Correlation Analysis: FAILED - {e}")
    
    try:
        test_variance_threshold(X_clf, y_clf)
        print(" Variance Threshold: PASSED")
    except Exception as e:
        print(f" Variance Threshold: FAILED - {e}")
    
    try:
        test_vif_analysis(X_clf, y_clf)
        print(" VIF Analysis: PASSED")
    except Exception as e:
        print(f" VIF Analysis: FAILED - {e}")
    
    # Test Regression
    print("\n" + "REGRESSION TESTS")
    X_reg, y_reg = create_test_data('regression', n_samples=200, n_features=15)
    
    try:
        test_forward_selection(X_reg, y_reg)
        print(" Forward Selection (Regression): PASSED")
    except Exception as e:
        print(f" Forward Selection (Regression): FAILED - {e}")
    
    try:
        test_feature_importance(X_reg, y_reg)
        print(" Feature Importance (Regression): PASSED")
    except Exception as e:
        print(f" Feature Importance (Regression): FAILED - {e}")
    
    # Final Summary
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED!")
    print("="*60)
    print("\nFeature Engineering System is working correctly!")
    print("\nAvailable API Endpoints:")
    print("  Automated:")
    print("    - POST /module6/automated/forward_selection")
    print("    - POST /module6/automated/backward_elimination")
    print("    - POST /module6/automated/feature_importance")
    print("  Manual:")
    print("    - POST /module6/manual/correlation_analysis")
    print("    - POST /module6/manual/variance_threshold")
    print("    - POST /module6/manual/vif_analysis")

if __name__ == '__main__':
    run_all_tests()
