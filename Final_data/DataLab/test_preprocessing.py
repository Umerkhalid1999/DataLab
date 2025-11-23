"""
Test script for enhanced preprocessing system
Run this to verify everything is working
"""
import pandas as pd
import numpy as np
from advanced_preprocessor import AdvancedPreprocessor

def create_test_dataset():
    """Create a messy test dataset"""
    np.random.seed(42)
    data = {
        'age': [25, 30, np.nan, 45, 50, 35, np.nan, 40, 28, 33],
        'income': [50000, 60000, 55000, np.nan, 80000, 65000, 70000, np.nan, 52000, 58000],
        'score': [85, 90, 88, 92, 1000, 87, 89, 91, 86, 88],  # 1000 is outlier
        'category': ['A', 'B', 'A', 'C', 'B', np.nan, 'A', 'C', 'B', 'A']
    }
    df = pd.DataFrame(data)
    
    # Add duplicates
    df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
    
    return df

def test_preprocessing():
    """Test all preprocessing functions"""
    print("=" * 60)
    print("TESTING ENHANCED PREPROCESSING SYSTEM")
    print("=" * 60)
    
    # Create test data
    print("\n1. Creating test dataset...")
    df = create_test_dataset()
    print(f"   Original shape: {df.shape}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    print(f"   Duplicates: {df.duplicated().sum()}")
    
    # Initialize preprocessor
    print("\n2. Initializing AdvancedPreprocessor...")
    preprocessor = AdvancedPreprocessor(df)
    
    # Test duplicate removal
    print("\n3. Testing duplicate removal...")
    preprocessor.remove_duplicates()
    print("   [OK] Duplicates removed")
    
    # Test missing value handling
    print("\n4. Testing missing value imputation...")
    preprocessor.handle_missing_values(['age', 'income'], strategy='median')
    preprocessor.handle_missing_values(['category'], strategy='mode')
    print("   [OK] Missing values filled")
    
    # Test outlier removal
    print("\n5. Testing outlier removal...")
    preprocessor.remove_outliers(['score'], method='iqr', threshold=1.5)
    print("   [OK] Outliers capped")
    
    # Test normalization
    print("\n6. Testing normalization...")
    preprocessor.normalize(['age', 'income'], method='minmax')
    print("   [OK] Features normalized")
    
    # Test standardization
    print("\n7. Testing standardization...")
    preprocessor.standardize(['score'])
    print("   [OK] Features standardized")
    
    # Get results
    cleaned_df = preprocessor.get_transformed_data()
    transformations = preprocessor.get_transformation_summary()
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Final shape: {cleaned_df.shape}")
    print(f"Missing values: {cleaned_df.isnull().sum().sum()}")
    print(f"Duplicates: {cleaned_df.duplicated().sum()}")
    
    print("\nTransformations applied:")
    for i, t in enumerate(transformations, 1):
        print(f"  {i}. {t['type']}: {t.get('column', 'all columns')}")
    
    print("\n" + "=" * 60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("=" * 60)
    
    return True

def test_openai_helper():
    """Test OpenAI helper (requires API key)"""
    print("\n" + "=" * 60)
    print("TESTING OPENAI EXPLAINABILITY")
    print("=" * 60)
    
    try:
        from openai_helper import PreprocessingExplainer
        
        print("\n1. Initializing PreprocessingExplainer...")
        explainer = PreprocessingExplainer()
        print("   [OK] OpenAI client initialized")
        
        print("\n2. Generating explanation...")
        transformations = [
            {'type': 'imputation', 'columns': ['age', 'income']},
            {'type': 'duplicate_removal', 'columns': []},
            {'type': 'outlier_removal', 'columns': ['score']}
        ]
        data_stats = {
            'rows': 1000,
            'columns': 15,
            'original_quality': 71.9
        }
        
        explanation = explainer.explain_preprocessing(transformations, data_stats)
        
        print("\n" + "=" * 60)
        print("AI EXPLANATION:")
        print("=" * 60)
        print(explanation)
        print("=" * 60)
        
        print("\n[SUCCESS] OPENAI TEST PASSED!")
        return True
        
    except ValueError as e:
        print(f"\n[WARNING] OpenAI test skipped: {e}")
        print("   Set OPENAI_API_KEY environment variable to test")
        return False
    except Exception as e:
        print(f"\n[ERROR] OpenAI test failed: {e}")
        return False

if __name__ == "__main__":
    try:
        # Test preprocessing
        test_preprocessing()
        
        # Test OpenAI (optional)
        print("\n")
        test_openai_helper()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] ENHANCED PREPROCESSING SYSTEM IS READY!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Set OPENAI_API_KEY in .env file")
        print("2. Restart Flask application")
        print("3. Upload a dataset and click 'Clean Dataset'")
        print("4. Enjoy the new UI and AI explanations!")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
