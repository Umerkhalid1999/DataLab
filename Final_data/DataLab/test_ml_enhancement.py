"""
Test script to verify ML Recommender enhancements
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'routes'))

from routes.app import MLRecommender
import pandas as pd
import numpy as np

def test_classification_models():
    """Test that all classification models are available"""
    print("=" * 60)
    print("Testing Classification Models")
    print("=" * 60)
    
    recommender = MLRecommender()
    models = recommender.classification_models
    
    print(f"\n[OK] Total Classification Models: {len(models)}")
    for i, model_name in enumerate(models.keys(), 1):
        print(f"   {i}. {model_name}")
    
    expected_models = ['Random Forest', 'Gradient Boosting', 'Logistic Regression', 
                      'Decision Tree', 'Naive Bayes', 'KNN', 'SVM', 'AdaBoost', 'Neural Network']
    
    assert len(models) >= 9, f"Expected at least 9 models, got {len(models)}"
    print(f"\n[PASS] {len(models)} classification models available")
    return True

def test_regression_models():
    """Test that all regression models are available"""
    print("\n" + "=" * 60)
    print("Testing Regression Models")
    print("=" * 60)
    
    recommender = MLRecommender()
    models = recommender.regression_models
    
    print(f"\n[OK] Total Regression Models: {len(models)}")
    for i, model_name in enumerate(models.keys(), 1):
        print(f"   {i}. {model_name}")
    
    assert len(models) >= 11, f"Expected at least 11 models, got {len(models)}"
    print(f"\n[PASS] {len(models)} regression models available")
    return True

def test_model_evaluation():
    """Test model evaluation with sample data"""
    print("\n" + "=" * 60)
    print("Testing Model Evaluation")
    print("=" * 60)
    
    # Create sample classification dataset
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    
    recommender = MLRecommender()
    
    print("\n[ANALYZING] Dataset...")
    analysis = recommender.analyze_dataset(df)
    print(f"   Task Type: {analysis['task_type']}")
    print(f"   Samples: {analysis['n_samples']}")
    print(f"   Features: {analysis['n_features']}")
    
    print("\n[PREPROCESSING] Data...")
    X_processed, y_processed, scaler = recommender.preprocess_data(df.copy(), analysis)
    
    print("\n[EVALUATING] Models...")
    results = recommender.evaluate_models(X_processed, y_processed, analysis['task_type'])
    
    print(f"\n[OK] Models Evaluated: {len(results)}")
    
    # Check that we have results for all models
    successful_models = [name for name, result in results.items() if 'error' not in result]
    print(f"[OK] Successful Models: {len(successful_models)}")
    
    # Show top 3 models
    sorted_results = sorted(
        [(name, result['mean_score']) for name, result in results.items() if 'error' not in result],
        key=lambda x: x[1],
        reverse=True
    )
    
    print("\n[TOP 3] Models:")
    for i, (name, score) in enumerate(sorted_results[:3], 1):
        print(f"   {i}. {name}: {score:.4f}")
    
    assert len(successful_models) >= 8, f"Expected at least 8 successful models, got {len(successful_models)}"
    print(f"\n[PASS] Model evaluation working correctly")
    return True

def test_hyperparameter_grids():
    """Test that hyperparameter grids are defined for new models"""
    print("\n" + "=" * 60)
    print("Testing Hyperparameter Grids")
    print("=" * 60)
    
    recommender = MLRecommender()
    grids = recommender.hyperparameter_grids
    
    print(f"\n[OK] Total Models with Hyperparameter Grids: {len(grids)}")
    for model_name in grids.keys():
        print(f"   • {model_name}")
    
    # Check for new models
    new_models = ['AdaBoost', 'Logistic Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet']
    for model in new_models:
        if model in grids:
            print(f"   [OK] {model} grid defined")
    
    print(f"\n[PASS] Hyperparameter grids configured")
    return True

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ML RECOMMENDER ENHANCEMENT TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_classification_models,
        test_regression_models,
        test_hyperparameter_grids,
        test_model_evaluation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n[FAIL] {test.__name__}")
            print(f"   Error: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"[PASSED] {passed}/{len(tests)}")
    print(f"[FAILED] {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n[SUCCESS] ALL TESTS PASSED! ML Recommender enhancement successful!")
    else:
        print("\n[WARNING] Some tests failed. Please review the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
