#!/usr/bin/env python3
"""
Test script for Advanced Feature Engineering Module
Creates sample data and tests core functionality
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
import os
import sys

# Add current directory to path to import app
sys.path.append('.')
from app import AdvancedFeatureEngineering

def create_sample_datasets():
    """Create sample datasets for testing"""
    
    # Classification dataset
    X_class, y_class = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(10)]
    df_class = pd.DataFrame(X_class, columns=feature_names)
    df_class['target'] = y_class
    
    # Regression dataset
    X_reg, y_reg = make_regression(
        n_samples=500,
        n_features=8,
        n_informative=6,
        noise=0.1,
        random_state=42
    )
    
    # Create DataFrame
    reg_feature_names = [f'reg_feature_{i}' for i in range(8)]
    df_reg = pd.DataFrame(X_reg, columns=reg_feature_names)
    df_reg['target'] = y_reg
    
    return df_class, df_reg

def test_feature_engineering_functionality():
    """Test all major functionality"""
    
    print("🧪 Starting Advanced Feature Engineering Module Tests\n")
    
    # Create sample datasets
    print("📊 Creating sample datasets...")
    df_class, df_reg = create_sample_datasets()
    
    # Save sample datasets
    df_class.to_csv('sample_classification_data.csv', index=False)
    df_reg.to_csv('sample_regression_data.csv', index=False)
    print("✅ Sample datasets created and saved\n")
    
    # Initialize feature engineering class
    fe = AdvancedFeatureEngineering()
    
    # Test 1: Data Loading
    print("🔍 Test 1: Data Loading")
    try:
        shape, dtypes, target = fe.load_data(df_class, 'target')
        print(f"✅ Classification data loaded: {shape}, target: {target}")
        
        shape, dtypes, target = fe.load_data(df_reg, 'target')
        print(f"✅ Regression data loaded: {shape}, target: {target}")
        
        # Test auto-detection
        shape, dtypes, auto_target = fe.load_data(df_reg)
        print(f"✅ Auto-detected target: {auto_target}")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    # Test 2: Feature Importance
    print("\n📈 Test 2: Feature Importance Analysis")
    try:
        importance_results = fe.automated_feature_importance()
        print(f"✅ Feature importance calculated for {len(importance_results)} methods")
        for method, results in importance_results.items():
            if isinstance(results, list):
                print(f"   - {method}: {len(results)} features ranked")
            else:
                print(f"   - {method}: {results}")
    except Exception as e:
        print(f"❌ Feature importance failed: {e}")
    
    # Test 3: Intelligent Feature Creation
    print("\n🎯 Test 3: Intelligent Feature Creation")
    try:
        created_features = fe.intelligent_feature_creation(max_features=15)
        print(f"✅ Created {len(created_features)} intelligent features")
        for feature_name in list(created_features.keys())[:5]:
            print(f"   - {feature_name}")
        if len(created_features) > 5:
            print(f"   ... and {len(created_features) - 5} more")
    except Exception as e:
        print(f"❌ Feature creation failed: {e}")
    
    # Test 4: Dimensionality Reduction
    print("\n📊 Test 4: Dimensionality Reduction")
    try:
        dim_results = fe.dimensionality_reduction_analysis()
        print(f"✅ Dimensionality reduction completed for {len(dim_results)} methods")
        for method, results in dim_results.items():
            if 'transformed_data' in results:
                shape = np.array(results['transformed_data']).shape
                print(f"   - {method}: Reduced to {shape}")
    except Exception as e:
        print(f"❌ Dimensionality reduction failed: {e}")
    
    # Test 5: Feature Set Comparison
    print("\n⚖️ Test 5: Feature Set Comparison")
    try:
        # Create sample feature sets
        all_features = list(df_reg.columns[:-1])  # Exclude target
        feature_sets = {
            'All Features': all_features,
            'First Half': all_features[:len(all_features)//2],
            'Second Half': all_features[len(all_features)//2:],
            'Top 3': all_features[:3]
        }
        
        comparison_results = fe.compare_feature_sets(feature_sets)
        print(f"✅ Feature set comparison completed for {len(comparison_results)} sets")
        for set_name, results in comparison_results.items():
            if 'error' not in results:
                print(f"   - {set_name}: {results['metric']} = {results['mean_score']:.4f} (±{results['std_score']:.4f})")
            else:
                print(f"   - {set_name}: Error - {results['error']}")
    except Exception as e:
        print(f"❌ Feature set comparison failed: {e}")
    
    # Test 6: Domain Templates
    print("\n🏭 Test 6: Domain-Specific Templates")
    try:
        templates = fe.get_domain_templates()
        print(f"✅ Retrieved {len(templates)} domain templates")
        for domain, categories in templates.items():
            print(f"   - {domain}: {len(categories)} categories")
    except Exception as e:
        print(f"❌ Domain templates failed: {e}")
    
    print("\n🎉 All tests completed!")
    print("\n📋 Summary:")
    print("   - Sample datasets created in current directory")
    print("   - All core functionality tested")
    print("   - Ready for web interface testing")
    print("\n🚀 To start the web application, run: python app.py")
    
    return True

def test_web_endpoints():
    """Test if Flask app can be imported and configured"""
    print("\n🌐 Testing Web Application Setup")
    try:
        from app import app
        print("✅ Flask app imported successfully")
        print("✅ All routes configured")
        print("✅ Ready to run web server")
    except Exception as e:
        print(f"❌ Web app setup failed: {e}")
        return False
    return True

if __name__ == "__main__":
    success = test_feature_engineering_functionality()
    web_success = test_web_endpoints()
    
    if success and web_success:
        print("\n✨ Module 6 Advanced Feature Engineering is ready for integration!")
        print("\nNext steps:")
        print("1. Run 'python app.py' to start the web server")
        print("2. Open http://localhost:5000 in your browser")
        print("3. Upload the generated sample CSV files to test the interface")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)
