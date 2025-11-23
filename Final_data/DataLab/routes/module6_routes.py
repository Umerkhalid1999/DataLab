# routes/module6_routes.py - Module 6 Feature Engineering Routes
from flask import Blueprint, render_template, request, jsonify, send_file, session
import pandas as pd
import numpy as np
import json
import io
import base64
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Import new feature selection module
try:
    from routes.feature_selection import AdvancedFeatureSelector
except ImportError:
    from feature_selection import AdvancedFeatureSelector

# Import Explainable AI modules
try:
    from routes.module6_enhancements import ExplainableAI, ProgressTracker
    from routes.decision_logger import (
        decision_logger,
        explain_pca_vs_tsne_choice,
        explain_feature_selection_strategy,
        explain_why_sampling_decision,
        explain_cv_folds_choice
    )
except ImportError:
    try:
        from module6_enhancements import ExplainableAI, ProgressTracker
        from decision_logger import (
            decision_logger,
            explain_pca_vs_tsne_choice,
            explain_feature_selection_strategy,
            explain_why_sampling_decision,
            explain_cv_folds_choice
        )
    except ImportError:
        print("Warning: Explainable AI modules not found. Running without explanations.")
        ExplainableAI = None
        ProgressTracker = None
        decision_logger = None

# OpenAI integration
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Blueprint
module6_bp = Blueprint('module6', __name__, url_prefix='/module6')

# Global datasets reference (will be set by main app)
datasets = {}

def set_datasets_reference(global_datasets):
    """Set reference to global datasets dictionary"""
    global datasets
    datasets = global_datasets

def get_user_id():
    """Get user ID from session (similar to main app logic)"""
    return session.get('user_id', 'default_user')

# Initialize OpenAI client
class LLMAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY', 'your_openai_api_key_here')
        self.model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        self.temperature = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))
        
        if self.api_key and self.api_key != 'your_openai_api_key_here':
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
            self.enabled = True
        else:
            self.enabled = False
            print("Warning: OpenAI API key not found. LLM features will be disabled.")
    
    def generate_insights(self, prompt, max_tokens=500):
        """Generate insights using OpenAI GPT"""
        if not self.enabled:
            return "LLM analysis unavailable - please set OPENAI_API_KEY in .env file"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert data scientist and feature engineering specialist. Provide clear, actionable insights and recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM analysis error: {str(e)}"
    
    def analyze_dataset_characteristics(self, dataset_info):
        """Generate LLM insights about dataset characteristics"""
        prompt = f"""
        As an expert data scientist, analyze this dataset and provide structured insights:
        
        **Dataset Overview:**
        - Shape: {dataset_info['shape']} (rows × columns)
        - Target: {dataset_info['target_column']}
        - Features: {list(dataset_info['dtypes'].keys())}
        - Missing Values: {dataset_info['missing_values']}
        - Sample Data: {dataset_info['sample_data'][:2]}
        
        **Required Analysis:**
        Provide a comprehensive analysis in markdown format with these sections:
        
        ## 📊 Data Quality Assessment
        - Overall data quality score (1-10)
        - Specific quality issues identified
        - Missing value patterns and impact
        
        ## 🎯 Domain Recognition
        - What domain/industry does this represent?
        - Key domain characteristics identified
        - Expected target variable behavior
        
        ## ⚙️ Intelligent Decisions Required
        **For each technique, decide YES/NO with reasoning:**
        - **Feature Scaling**: Required? (YES/NO + reason)
        - **Dimensionality Reduction**: Required? (YES/NO + reason)  
        - **Feature Creation**: Required? (YES/NO + reason)
        - **Outlier Handling**: Required? (YES/NO + reason)
        - **Categorical Encoding**: Required? (YES/NO + reason)
        
        ## 🚨 Risk Assessment
        - Data leakage risks
        - Bias concerns
        - Overfitting warnings
        
        Use clear markdown formatting with headers, bullets, and **bold** text.
        """
        
        return self.generate_insights(prompt, max_tokens=800)
    
    def analyze_feature_importance(self, importance_results, target_column):
        """Generate LLM insights about feature importance patterns"""
        if (not importance_results or 
            not isinstance(importance_results, dict) or 
            len(importance_results) == 0 or 
            any('error' in str(v) for v in importance_results.values())):
            return "Cannot analyze feature importance due to errors in the results."
        
        # Extract top features from each method
        top_features_summary = {}
        for method, results in importance_results.items():
            if isinstance(results, list) and len(results) > 0:
                top_features_summary[method] = [feat for feat, score in results[:5]]
        
        prompt = f"""
        Analyze these feature importance rankings for target '{target_column}':
        
        {json.dumps(top_features_summary, indent=2)}
        
        Please provide:
        1. Which features consistently rank high across methods?
        2. Are there any surprising or concerning patterns?
        3. What might these importance patterns suggest about the underlying relationships?
        4. Recommendations for feature selection strategy
        5. Any domain-specific insights based on feature names
        
        Keep response focused and practical.
        """
        
        return self.generate_insights(prompt)
    
    def suggest_domain_features(self, dataset_info, sample_data):
        """Suggest domain-specific features based on column names and data"""
        column_names = dataset_info['columns']
        target = dataset_info['target_column']
        
        prompt = f"""
        Based on these column names and target variable, suggest domain-specific feature engineering:
        
        Columns: {column_names}
        Target: {target}
        Sample data types: {dataset_info['dtypes']}
        
        Please identify:
        1. What domain/industry this dataset likely represents
        2. Domain-specific feature engineering opportunities
        3. Important interactions or ratios to create
        4. Time-based features if applicable
        5. Categorical encoding strategies
        6. Feature scaling recommendations
        
        Provide specific, implementable suggestions.
        """
        
        return self.generate_insights(prompt, max_tokens=600)
    
    def explain_model_performance(self, comparison_results):
        """Explain model performance differences using LLM"""
        if (not comparison_results or 
            not isinstance(comparison_results, dict) or 
            len(comparison_results) == 0 or 
            any('error' in str(v) for v in comparison_results.values())):
            return "Cannot analyze performance due to errors in the results."
        
        # Sort results by performance
        valid_results = {k: v for k, v in comparison_results.items() if 'error' not in v}
        if not valid_results:
            return "No valid performance results to analyze."
        
        prompt = f"""
        Analyze these feature set performance results:
        
        {json.dumps(valid_results, indent=2)}
        
        Please explain:
        1. Why certain feature sets perform better than others?
        2. What does this suggest about feature redundancy or importance?
        3. Are there signs of overfitting or underfitting?
        4. Recommendations for the optimal feature set
        5. Next steps for model improvement
        
        Provide clear, actionable explanations.
        """
        
        return self.generate_insights(prompt)
    
    def generate_comprehensive_recommendations(self, all_results):
        """Generate comprehensive strategic recommendations"""
        # Extract key insights from all analysis
        dataset_info = all_results.get('dataset_info', {})
        feature_importance = all_results.get('feature_importance', {})
        feature_comparison = all_results.get('feature_comparison', {})
        
        # Get best performing feature set
        best_feature_set = "Unknown"
        best_performance = "Unknown"
        if (feature_comparison and 
            isinstance(feature_comparison, dict) and 
            len(feature_comparison) > 0 and 
            not any('error' in str(v) for v in feature_comparison.values())):
            valid_results = {k: v for k, v in feature_comparison.items() if 'error' not in v}
            if valid_results:
                sorted_results = sorted(valid_results.items(), key=lambda x: x[1].get('mean_score', 0), reverse=True)
                best_feature_set = sorted_results[0][0]
                best_performance = f"{sorted_results[0][1]['mean_score']:.4f} {sorted_results[0][1]['metric']}"
        
        prompt = f"""
        As a senior ML consultant, provide comprehensive strategic recommendations for this project:
        
        **Project Context:**
        - Dataset: {dataset_info.get('shape')} samples, {len(dataset_info.get('columns', []))} features
        - Target: {dataset_info.get('target_column')}
        - Best Feature Set: {best_feature_set}
        - Best Performance: {best_performance}
        - Features Created: {len(all_results.get('created_features', {}))}
        
        **Provide Strategic Recommendations in markdown format:**
        
        # 🎯 Executive Summary
        - Project viability assessment
        - Expected performance range
        - Key success factors
        
        # 📋 Implementation Roadmap
        ## Phase 1: Immediate Actions (Week 1-2)
        - Specific tasks to start immediately
        - Resource requirements
        
        ## Phase 2: Model Development (Week 3-4)
        - Model selection strategy
        - Feature engineering pipeline
        
        ## Phase 3: Validation & Testing (Week 5-6)
        - Validation approach
        - Performance benchmarks
        
        # 🏗️ Technical Architecture
        ## Recommended Feature Pipeline
        - Final feature set to use
        - Preprocessing steps required
        - Feature engineering transformations
        
        ## Model Selection Strategy
        - Primary algorithm recommendation
        - Alternative algorithms to test
        - Hyperparameter tuning approach
        
        # 🚀 Production Deployment
        ## Infrastructure Requirements
        - Hardware/software needs
        - Scaling considerations
        
        ## Monitoring & Maintenance
        - Key metrics to track
        - Model drift detection
        - Retraining schedule
        
        # ⚠️ Risk Mitigation
        ## Technical Risks
        - Overfitting prevention
        - Data quality issues
        
        ## Business Risks
        - Performance degradation
        - Bias and fairness concerns
        
        # 💰 Success Metrics & ROI
        - Key performance indicators
        - Business impact measurement
        - Expected ROI timeline
        
        Provide specific, actionable recommendations with timeline estimates.
        """
        
        return self.generate_insights(prompt, max_tokens=1200)
    
    def make_intelligent_decisions(self, dataset_info, analysis_results):
        """LLM makes intelligent decisions about what techniques to apply"""
        prompt = f"""
        As an expert ML engineer, analyze this dataset and make intelligent decisions:
        
        **Dataset Context:**
        - Shape: {dataset_info['shape']}
        - Features: {list(dataset_info['dtypes'].keys())}
        - Target: {dataset_info['target_column']}
        - Missing Values: {dataset_info['missing_values']}
        
        **Analysis Results Available:**
        - Feature Importance: {bool(analysis_results.get('feature_importance') and len(analysis_results.get('feature_importance', {})) > 0)}
        - Created Features: {len(analysis_results.get('created_features', {}))} new features
        - Dimensionality Reduction: {bool(analysis_results.get('dimensionality_reduction') and len(analysis_results.get('dimensionality_reduction', {})) > 0)}
        - Performance Comparison: {bool(analysis_results.get('feature_comparison') and len(analysis_results.get('feature_comparison', {})) > 0)}
        
        **Make These Critical Decisions (respond in JSON format):**
        
        {{
            "apply_feature_scaling": {{
                "decision": true/false,
                "reason": "why this decision was made",
                "method": "standardization/normalization/none"
            }},
            "apply_dimensionality_reduction": {{
                "decision": true/false,
                "reason": "why this decision was made",
                "method": "pca/tsne/none",
                "n_components": number
            }},
            "apply_outlier_removal": {{
                "decision": true/false,
                "reason": "why this decision was made",
                "method": "iqr/zscore/isolation_forest/none"
            }},
            "recommended_features": {{
                "feature_set": "best performing feature set name",
                "features": ["list", "of", "recommended", "features"],
                "reasoning": "why these features were selected"
            }},
            "preprocessing_pipeline": {{
                "steps": ["step1", "step2", "step3"],
                "order": "sequence of operations"
            }},
            "confidence_score": 0.85,
            "next_actions": ["immediate action 1", "immediate action 2"]
        }}
        
        Base decisions on data characteristics, performance results, and ML best practices.
        """
        
        try:
            decisions_text = self.generate_insights(prompt, max_tokens=800)
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', decisions_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "Could not extract JSON decisions", "raw_response": decisions_text}
        except Exception as e:
            return {"error": f"Decision making failed: {str(e)}"}

# Initialize LLM analyzer
llm_analyzer = LLMAnalyzer()

# Initialize Explainable AI
if ExplainableAI is not None:
    explainer = ExplainableAI()
    print("Explainable AI initialized successfully")
else:
    explainer = None
    print("Running without Explainable AI")

def make_json_serializable(obj):
    """Convert pandas/numpy objects to JSON-serializable format and handle ALL NaN values"""
    import math

    # Handle None first
    if obj is None:
        return None

    # Check for array-like objects BEFORE using pd.isna() to avoid ambiguity error
    if isinstance(obj, np.ndarray):
        # Convert array and handle NaN recursively
        arr_list = obj.tolist()
        return make_json_serializable(arr_list)

    if isinstance(obj, pd.DataFrame):
        # Replace NaN with None before converting to dict
        return obj.replace([np.inf, -np.inf, np.nan], None).fillna(None).to_dict('records')

    if isinstance(obj, pd.Series):
        # Replace NaN with None before converting to list
        return obj.replace([np.inf, -np.inf, np.nan], None).fillna(None).tolist()

    # Now safe to check pd.isna() on scalar values
    try:
        if pd.isna(obj):
            return None
    except (ValueError, TypeError):
        # If pd.isna() fails, continue with other checks
        pass

    # Handle string "NaN" values
    if isinstance(obj, str) and obj.lower() in ['nan', 'inf', '-inf', 'infinity', '-infinity']:
        return None

    # Handle NaN and infinity for floats
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)

    # Handle integers
    if isinstance(obj, np.integer):
        return int(obj)

    # Handle collections
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)

    # Handle other types
    elif isinstance(obj, (pd.Int64Dtype, pd.Float64Dtype)):
        return str(obj)
    elif hasattr(obj, 'dtype'):
        return str(obj)
    else:
        return obj

class AdvancedFeatureEngineering:
    def __init__(self):
        self.data = None
        self.target_column = None
        self.feature_importance_results = {}
        self.created_features = {}
        self.dimensionality_results = {}
        self.all_results = {}
        
    def auto_detect_target(self, data):
        """Automatically detect the most likely target column"""
        target_candidates = []
        
        # Look for common target column names
        common_targets = ['target', 'label', 'class', 'y', 'output', 'prediction', 'result', 
                         'price', 'value', 'score', 'rating', 'outcome', 'response']
        
        for col in data.columns:
            col_lower = col.lower()
            if any(target in col_lower for target in common_targets):
                target_candidates.append((col, 10))  # High priority
        
        # If no obvious target, look for the last column (common convention)
        if not target_candidates:
            target_candidates.append((data.columns[-1], 5))
        
        # Look for columns with fewer unique values (likely categorical targets)
        for col in data.columns:
            if data[col].dtype in ['object', 'category']:
                unique_ratio = data[col].nunique() / len(data)
                if unique_ratio < 0.1:  # Less than 10% unique values
                    target_candidates.append((col, 8))
            elif data[col].dtype in ['int64', 'float64']:
                unique_ratio = data[col].nunique() / len(data)
                if unique_ratio < 0.05:  # Very few unique values for numeric
                    target_candidates.append((col, 6))
        
        # Return the column with highest priority
        if target_candidates:
            target_candidates.sort(key=lambda x: x[1], reverse=True)
            return target_candidates[0][0]
        
        # Fallback to last column
        return data.columns[-1]

    def load_data(self, file_path_or_dataframe, target_column=None):
        """Load data and automatically detect target column if not provided"""
        if isinstance(file_path_or_dataframe, str):
            if file_path_or_dataframe.endswith('.csv'):
                self.data = pd.read_csv(file_path_or_dataframe)
            elif file_path_or_dataframe.endswith('.xlsx'):
                self.data = pd.read_excel(file_path_or_dataframe)
        else:
            self.data = file_path_or_dataframe.copy()
        
        # Auto-detect target column if not provided
        if target_column is None:
            self.target_column = self.auto_detect_target(self.data)
        else:
            self.target_column = target_column
            
        return self.data.shape, self.data.dtypes.to_dict(), self.target_column
    
    def automated_feature_importance(self, methods=['random_forest', 'xgboost', 'mutual_info', 'correlation']):
        """Calculate feature importance using multiple methods"""
        if self.data is None:
            raise ValueError("No data loaded")
            
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        
        # Handle categorical variables
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        results = {}
        
        if 'random_forest' in methods:
            if y.dtype == 'object' or len(y.unique()) < 10:
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
            
            rf.fit(X_encoded, y)
            rf_importance = dict(zip(X_encoded.columns, rf.feature_importances_))
            results['random_forest'] = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)
        
        if 'xgboost' in methods:
            try:
                if y.dtype == 'object' or len(y.unique()) < 10:
                    xgb_model = xgb.XGBClassifier(random_state=42)
                else:
                    xgb_model = xgb.XGBRegressor(random_state=42)
                
                xgb_model.fit(X_encoded, y)
                xgb_importance = dict(zip(X_encoded.columns, xgb_model.feature_importances_))
                results['xgboost'] = sorted(xgb_importance.items(), key=lambda x: x[1], reverse=True)
            except Exception as e:
                results['xgboost'] = f"Error: {str(e)}"
        
        if 'mutual_info' in methods:
            try:
                if y.dtype == 'object':
                    # For classification, convert to numeric
                    y_encoded = pd.Categorical(y).codes
                    mi_scores = mutual_info_regression(X_encoded, y_encoded)
                else:
                    mi_scores = mutual_info_regression(X_encoded, y)
                
                mi_importance = dict(zip(X_encoded.columns, mi_scores))
                results['mutual_info'] = sorted(mi_importance.items(), key=lambda x: x[1], reverse=True)
            except Exception as e:
                results['mutual_info'] = f"Error: {str(e)}"
        
        if 'correlation' in methods:
            try:
                if y.dtype == 'object':
                    y_encoded = pd.Categorical(y).codes
                else:
                    y_encoded = y

                correlations = {}
                for col in X_encoded.columns:
                    try:
                        # Handle constant columns and NaN values
                        col_data = X_encoded[col].fillna(0)
                        if col_data.std() == 0:  # Constant column
                            correlations[col] = 0.0
                        else:
                            corr, _ = pearsonr(col_data, y_encoded)
                            # Replace NaN with 0
                            correlations[col] = 0.0 if (pd.isna(corr) or np.isnan(corr) or np.isinf(corr)) else abs(corr)
                    except:
                        correlations[col] = 0.0

                results['correlation'] = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            except Exception as e:
                results['correlation'] = f"Error: {str(e)}"
        
        self.feature_importance_results = results
        return results
    
    def intelligent_feature_creation(self, max_features=20):
        """Create intelligent features based on data relationships"""
        if self.data is None:
            raise ValueError("No data loaded")

        X = self.data.drop(columns=[self.target_column])
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

        # CRITICAL: Fill NaN values before any feature creation to prevent errors
        X = X.fillna(0)

        created_features = {}

        # 1. Polynomial features for numeric columns (NaN-safe)
        if len(numeric_columns) >= 2:
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            # Ensure no NaN before polynomial transformation
            X_poly_input = X[numeric_columns[:min(5, len(numeric_columns))]].fillna(0)
            poly_features = poly.fit_transform(X_poly_input)
            poly_feature_names = poly.get_feature_names_out(numeric_columns[:min(5, len(numeric_columns))])
            
            # Add only interaction terms (not squares)
            for i, name in enumerate(poly_feature_names):
                if ' ' in name and '^2' not in name:  # Interaction terms only
                    created_features[f'poly_{name}'] = poly_features[:, i]
        
        # 2. Ratio features (with NaN prevention)
        for i, col1 in enumerate(numeric_columns[:5]):
            for col2 in numeric_columns[i+1:6]:
                try:
                    # Check for zeros and handle division safely
                    if (X[col2] != 0).all() and not X[col2].isnull().any():
                        ratio = X[col1] / X[col2]
                        # Replace inf with nan, then fill with 0
                        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
                        created_features[f'ratio_{col1}_{col2}'] = ratio
                except Exception as e:
                    # Skip this ratio if any error occurs
                    continue

        # 3. Log transformations (with NaN prevention)
        for col in numeric_columns[:5]:
            try:
                # Check if all values are positive
                if (X[col] > 0).all() and not X[col].isnull().any():
                    log_vals = np.log(X[col])
                    # Replace inf/nan with 0
                    log_vals = log_vals.replace([np.inf, -np.inf], np.nan).fillna(0)
                    created_features[f'log_{col}'] = log_vals
            except Exception as e:
                # Skip this log transformation if any error occurs
                continue
        
        # 4. Binning features (with error handling)
        for col in numeric_columns[:3]:
            try:
                binned = pd.cut(X[col], bins=5, labels=False)
                # Fill NaN values that may result from binning
                created_features[f'binned_{col}'] = binned.fillna(0)
            except Exception as e:
                # Skip if binning fails
                continue

        # 5. Statistical features (with NaN handling)
        if len(numeric_columns) >= 2:
            try:
                created_features['mean_all_numeric'] = X[numeric_columns].mean(axis=1).fillna(0)
                created_features['std_all_numeric'] = X[numeric_columns].std(axis=1).fillna(0)
                created_features['max_all_numeric'] = X[numeric_columns].max(axis=1).fillna(0)
                created_features['min_all_numeric'] = X[numeric_columns].min(axis=1).fillna(0)
            except Exception as e:
                # Skip if statistical features fail
                pass
        
        # Limit to max_features
        feature_items = list(created_features.items())[:max_features]
        self.created_features = dict(feature_items)
        
        return {name: f"Created feature: {name}" for name in self.created_features.keys()}
    
    def dimensionality_reduction_analysis(self, methods=['pca', 'tsne'], n_components=2):
        """Perform dimensionality reduction with visual explanations"""
        if self.data is None:
            raise ValueError("No data loaded")

        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # Handle categorical variables
        X_encoded = pd.get_dummies(X, drop_first=True)

        # CRITICAL: Fill NaN values before scaling and PCA/t-SNE
        X_encoded = X_encoded.fillna(0)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)
        
        results = {}
        
        if 'pca' in methods:
            pca = PCA(n_components=min(n_components, X_scaled.shape[1]))
            X_pca = pca.fit_transform(X_scaled)
            
            results['pca'] = {
                'transformed_data': X_pca,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
                'components': pca.components_.tolist()
            }
        
        if 'tsne' in methods and X_scaled.shape[0] > 50:  # t-SNE needs sufficient samples
            tsne = TSNE(n_components=min(n_components, 3), random_state=42, perplexity=min(30, X_scaled.shape[0]-1))
            X_tsne = tsne.fit_transform(X_scaled)
            
            results['tsne'] = {
                'transformed_data': X_tsne
            }
        
        self.dimensionality_results = results
        return results
    
    def compare_feature_sets(self, feature_sets, model_type='auto', cv_folds=3):
        """Compare different feature sets and their performance impact (OPTIMIZED)"""
        if self.data is None:
            raise ValueError("No data loaded")

        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # Determine model type
        if model_type == 'auto':
            if y.dtype == 'object' or len(y.unique()) < 10:
                model_type = 'classification'
            else:
                model_type = 'regression'

        results = {}

        for set_name, features in feature_sets.items():
            try:
                # Get feature subset
                available_features = [f for f in features if f in X.columns]
                if not available_features:
                    results[set_name] = {'error': 'No valid features found'}
                    continue

                X_subset = X[available_features]
                X_encoded = pd.get_dummies(X_subset, drop_first=True)

                # Train model and get performance (OPTIMIZED: Reduced estimators from 100 to 50)
                if model_type == 'classification':
                    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                    scores = cross_val_score(model, X_encoded, y, cv=cv_folds, scoring='accuracy', n_jobs=-1)
                    metric = 'accuracy'
                else:
                    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                    scores = cross_val_score(model, X_encoded, y, cv=cv_folds, scoring='neg_mean_squared_error', n_jobs=-1)
                    scores = -scores  # Convert to positive MSE
                    metric = 'mse'

                results[set_name] = {
                    'mean_score': float(np.mean(scores)),
                    'std_score': float(np.std(scores)),
                    'metric': metric,
                    'n_features': len(available_features),
                    'features_used': available_features
                }

            except Exception as e:
                results[set_name] = {'error': str(e)}
        
        return results
    
    def calculate_data_quality_score(self, data, target_column=None):
        """Calculate comprehensive data quality metrics"""
        if data is None or len(data) == 0:
            return {'error': 'No data provided'}

        try:
            metrics = {}

            # 1. Completeness Score (missing values)
            total_cells = data.shape[0] * data.shape[1]
            missing_cells = data.isnull().sum().sum()
            completeness_score = ((total_cells - missing_cells) / total_cells) * 100
            metrics['completeness_percentage'] = round(completeness_score, 2)
            metrics['missing_values_count'] = int(missing_cells)
            metrics['missing_values_percentage'] = round((missing_cells / total_cells) * 100, 2)

            # 2. Consistency Score (data type consistency)
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            consistency_issues = 0
            for col in numeric_cols:
                # Check for inf values
                if np.isinf(data[col]).any():
                    consistency_issues += 1
                # Check for extreme outliers (beyond 5 std)
                if len(data[col].dropna()) > 0:
                    mean_val = data[col].mean()
                    std_val = data[col].std()
                    if std_val > 0:
                        outliers = ((data[col] - mean_val).abs() > 5 * std_val).sum()
                        if outliers > len(data) * 0.05:  # More than 5% outliers
                            consistency_issues += 1

            consistency_score = max(0, 100 - (consistency_issues / len(data.columns) * 10))
            metrics['consistency_percentage'] = round(consistency_score, 2)
            metrics['consistency_issues'] = consistency_issues

            # 3. Uniqueness Score (duplicate rows)
            total_rows = len(data)
            unique_rows = len(data.drop_duplicates())
            uniqueness_score = (unique_rows / total_rows) * 100
            metrics['uniqueness_percentage'] = round(uniqueness_score, 2)
            metrics['duplicate_rows'] = total_rows - unique_rows

            # 4. Validity Score (valid ranges for numeric data)
            validity_issues = 0
            for col in numeric_cols:
                # Check for negative values where they shouldn't be
                if data[col].min() < 0 and 'age' in col.lower():
                    validity_issues += 1
                # Check for unrealistic ranges
                if 'age' in col.lower() and (data[col].max() > 150 or data[col].min() < 0):
                    validity_issues += 1

            validity_score = max(0, 100 - (validity_issues / len(data.columns) * 15))
            metrics['validity_percentage'] = round(validity_score, 2)
            metrics['validity_issues'] = validity_issues

            # 5. Feature Quality Score (correlation with target if available)
            if target_column and target_column in data.columns:
                X = data.drop(columns=[target_column])
                y = data[target_column]
                numeric_features = X.select_dtypes(include=[np.number]).columns

                if len(numeric_features) > 0 and y.dtype in [np.number, 'int64', 'float64']:
                    correlations = []
                    for col in numeric_features:
                        try:
                            col_data = X[col].fillna(0)
                            if col_data.std() > 0:
                                corr, _ = pearsonr(col_data, y)
                                if not (pd.isna(corr) or np.isnan(corr) or np.isinf(corr)):
                                    correlations.append(abs(corr))
                        except:
                            pass

                    if correlations:
                        avg_correlation = np.mean(correlations)
                        feature_quality_score = min(100, avg_correlation * 100 * 2)  # Scale up
                        metrics['feature_quality_percentage'] = round(feature_quality_score, 2)
                        metrics['avg_feature_correlation'] = round(avg_correlation, 4)
                    else:
                        metrics['feature_quality_percentage'] = 50.0
                        metrics['avg_feature_correlation'] = 0.0
                else:
                    metrics['feature_quality_percentage'] = 50.0
                    metrics['avg_feature_correlation'] = 0.0
            else:
                metrics['feature_quality_percentage'] = 50.0
                metrics['avg_feature_correlation'] = 0.0

            # 6. Overall Data Quality Score (weighted average)
            overall_score = (
                completeness_score * 0.30 +  # 30% weight
                consistency_score * 0.20 +    # 20% weight
                uniqueness_score * 0.15 +     # 15% weight
                validity_score * 0.15 +       # 15% weight
                metrics['feature_quality_percentage'] * 0.20  # 20% weight
            )

            metrics['overall_quality_score'] = round(overall_score, 2)
            metrics['quality_grade'] = self._get_quality_grade(overall_score)

            # Additional statistics
            metrics['total_rows'] = int(total_rows)
            metrics['total_columns'] = int(data.shape[1])
            metrics['numeric_features'] = int(len(numeric_cols))
            metrics['categorical_features'] = int(len(data.select_dtypes(include=['object', 'category']).columns))

            return metrics

        except Exception as e:
            return {'error': f'Quality calculation failed: {str(e)}'}

    def _get_quality_grade(self, score):
        """Convert quality score to letter grade"""
        if score >= 90:
            return 'A+ (Excellent)'
        elif score >= 80:
            return 'A (Very Good)'
        elif score >= 70:
            return 'B (Good)'
        elif score >= 60:
            return 'C (Fair)'
        elif score >= 50:
            return 'D (Poor)'
        else:
            return 'F (Very Poor)'

    def get_domain_templates(self):
        """Get domain-specific feature templates"""
        templates = {
            'financial': {
                'ratios': ['debt_to_equity', 'current_ratio', 'quick_ratio', 'return_on_equity'],
                'moving_averages': ['sma_20', 'sma_50', 'ema_12', 'ema_26'],
                'volatility': ['rolling_std', 'bollinger_bands', 'rsi'],
                'technical_indicators': ['macd', 'momentum', 'williams_r']
            },
            'healthcare': {
                'vital_ratios': ['bmi', 'pulse_pressure', 'map', 'shock_index'],
                'age_interactions': ['age_bp_interaction', 'age_cholesterol_interaction'],
                'risk_scores': ['framingham_score', 'apache_score'],
                'categorical_encodings': ['gender_age_group', 'comorbidity_count']
            },
            'ecommerce': {
                'customer_behavior': ['avg_order_value', 'purchase_frequency', 'days_since_last_order'],
                'product_features': ['price_category', 'category_popularity', 'seasonal_demand'],
                'engagement': ['page_views_per_session', 'cart_abandonment_rate', 'conversion_rate'],
                'temporal': ['hour_of_day', 'day_of_week', 'month_seasonality']
            },
            'marketing': {
                'campaign_metrics': ['ctr', 'cpm', 'roas', 'ltv_cac_ratio'],
                'audience_features': ['demographic_score', 'interest_affinity', 'lookalike_score'],
                'channel_features': ['channel_mix', 'attribution_weight', 'cross_channel_interaction'],
                'timing': ['optimal_send_time', 'frequency_cap', 'recency_score']
            },
            'manufacturing': {
                'quality_metrics': ['defect_rate', 'yield_percentage', 'first_pass_yield'],
                'process_features': ['cycle_time', 'setup_time', 'oee'],
                'maintenance': ['mtbf', 'mttr', 'preventive_maintenance_score'],
                'operational': ['capacity_utilization', 'throughput_rate', 'bottleneck_index']
            }
        }
        return templates
    
    def create_optimized_dataset(self, intelligent_decisions):
        """Create optimized dataset based on LLM decisions"""
        if self.data is None:
            raise ValueError("No data loaded")

        try:
            # Calculate BEFORE quality metrics
            print("📊 Calculating quality score BEFORE preprocessing...")
            quality_before = self.calculate_data_quality_score(self.data, self.target_column)

            # Start with original data
            optimized_data = self.data.copy()
            processing_log = []
            
            # 1. Feature Selection based on LLM recommendation
            if 'recommended_features' in intelligent_decisions:
                recommended_features = intelligent_decisions['recommended_features'].get('features', [])
                # Include target column
                if self.target_column not in recommended_features:
                    recommended_features.append(self.target_column)
                
                # Filter to features that exist in the dataset
                available_features = [f for f in recommended_features if f in optimized_data.columns]
                optimized_data = optimized_data[available_features]
                processing_log.append(f"✅ Selected {len(available_features)} features based on LLM recommendation")
            
            # 2. Add created features if recommended
            if (hasattr(self, 'created_features') and 
                len(self.created_features) > 0 and 
                intelligent_decisions.get('apply_feature_scaling', {}).get('decision')):
                # Add top created features
                created_df = pd.DataFrame(self.created_features)
                created_df.index = optimized_data.index
                # Add top 5 created features
                top_created = list(created_df.columns)[:5]
                for feature in top_created:
                    optimized_data[feature] = created_df[feature]
                processing_log.append(f"✅ Added {len(top_created)} engineered features")
            
            # 3. Handle outliers if recommended
            outlier_decision = intelligent_decisions.get('apply_outlier_removal', {})
            if outlier_decision.get('decision'):
                method = outlier_decision.get('method', 'iqr')
                if method == 'iqr':
                    # Remove outliers using IQR method
                    numeric_columns = optimized_data.select_dtypes(include=[np.number]).columns
                    for col in numeric_columns:
                        if col != self.target_column:
                            Q1 = optimized_data[col].quantile(0.25)
                            Q3 = optimized_data[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            outlier_mask = (optimized_data[col] >= lower_bound) & (optimized_data[col] <= upper_bound)
                            optimized_data = optimized_data[outlier_mask]
                    processing_log.append(f"✅ Removed outliers using {method.upper()} method")
            
            # 4. Apply feature scaling if recommended
            scaling_decision = intelligent_decisions.get('apply_feature_scaling', {})
            if scaling_decision.get('decision'):
                method = scaling_decision.get('method', 'standardization')
                numeric_columns = optimized_data.select_dtypes(include=[np.number]).columns
                feature_columns = [col for col in numeric_columns if col != self.target_column]
                
                if method == 'standardization':
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    optimized_data[feature_columns] = scaler.fit_transform(optimized_data[feature_columns])
                elif method == 'normalization':
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    optimized_data[feature_columns] = scaler.fit_transform(optimized_data[feature_columns])
                
                processing_log.append(f"✅ Applied {method} to {len(feature_columns)} features")
            
            # 5. Apply dimensionality reduction if recommended
            dim_reduction_decision = intelligent_decisions.get('apply_dimensionality_reduction', {})
            if dim_reduction_decision.get('decision'):
                method = dim_reduction_decision.get('method', 'pca')
                n_components = dim_reduction_decision.get('n_components', 2)
                
                numeric_columns = optimized_data.select_dtypes(include=[np.number]).columns
                feature_columns = [col for col in numeric_columns if col != self.target_column]
                
                if method == 'pca' and len(feature_columns) > n_components:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=n_components)
                    pca_features = pca.fit_transform(optimized_data[feature_columns])
                    
                    # Replace original features with PCA components
                    pca_columns = [f'PCA_Component_{i+1}' for i in range(n_components)]
                    pca_df = pd.DataFrame(pca_features, columns=pca_columns, index=optimized_data.index)
                    
                    # Keep target and add PCA components
                    optimized_data = pd.concat([pca_df, optimized_data[[self.target_column]]], axis=1)
                    processing_log.append(f"✅ Applied {method.upper()} reduction to {n_components} components")
            
            # 6. Final cleanup
            optimized_data = optimized_data.dropna()
            processing_log.append(f"✅ Final dataset shape: {optimized_data.shape}")

            # Calculate AFTER quality metrics
            print("📊 Calculating quality score AFTER preprocessing...")
            quality_after = self.calculate_data_quality_score(optimized_data, self.target_column)

            # Calculate improvements
            quality_improvement = {}
            if 'error' not in quality_before and 'error' not in quality_after:
                quality_improvement = {
                    'overall_score_before': quality_before.get('overall_quality_score', 0),
                    'overall_score_after': quality_after.get('overall_quality_score', 0),
                    'score_improvement': round(
                        quality_after.get('overall_quality_score', 0) -
                        quality_before.get('overall_quality_score', 0), 2
                    ),
                    'percentage_improvement': round(
                        ((quality_after.get('overall_quality_score', 0) -
                          quality_before.get('overall_quality_score', 0)) /
                         max(quality_before.get('overall_quality_score', 1), 1)) * 100, 2
                    ),
                    'grade_before': quality_before.get('quality_grade', 'Unknown'),
                    'grade_after': quality_after.get('quality_grade', 'Unknown'),
                    'completeness_improvement': round(
                        quality_after.get('completeness_percentage', 0) -
                        quality_before.get('completeness_percentage', 0), 2
                    ),
                    'consistency_improvement': round(
                        quality_after.get('consistency_percentage', 0) -
                        quality_before.get('consistency_percentage', 0), 2
                    ),
                    'feature_quality_improvement': round(
                        quality_after.get('feature_quality_percentage', 0) -
                        quality_before.get('feature_quality_percentage', 0), 2
                    )
                }

                # Add improvement summary to processing log
                processing_log.append(f"\n📈 QUALITY IMPROVEMENTS:")
                processing_log.append(f"   Overall Score: {quality_improvement['overall_score_before']:.1f}% → {quality_improvement['overall_score_after']:.1f}% (+{quality_improvement['score_improvement']:.1f}%)")
                processing_log.append(f"   Grade: {quality_improvement['grade_before']} → {quality_improvement['grade_after']}")
                processing_log.append(f"   Completeness: +{quality_improvement['completeness_improvement']:.1f}%")
                processing_log.append(f"   Consistency: +{quality_improvement['consistency_improvement']:.1f}%")
                processing_log.append(f"   Feature Quality: +{quality_improvement['feature_quality_improvement']:.1f}%")

            # Convert DataFrame to serializable format
            optimized_data_serializable = optimized_data.to_dict('records')

            return {
                'optimized_dataset': optimized_data_serializable,
                'processing_log': processing_log,
                'original_shape': self.data.shape,
                'final_shape': optimized_data.shape,
                'improvement_summary': f"Optimized from {self.data.shape} to {optimized_data.shape}",
                'columns': list(optimized_data.columns),
                'quality_before': quality_before,
                'quality_after': quality_after,
                'quality_improvement': quality_improvement
            }
            
        except Exception as e:
            return {'error': f"Dataset optimization failed: {str(e)}"}
    
    def run_complete_analysis(self):
        """Run all feature engineering analysis automatically with optimized performance"""
        if self.data is None:
            raise ValueError("No data loaded")

        # Performance optimization: Sample large datasets
        data_to_analyze = self.data
        is_sampled = False
        if len(self.data) > 5000:
            print(f"⚡ Dataset has {len(self.data)} rows. Sampling 5000 rows for faster analysis...")
            data_to_analyze = self.data.sample(n=5000, random_state=42)
            is_sampled = True
            # Temporarily replace data for analysis
            original_data = self.data
            self.data = data_to_analyze

        # Prepare dataset info with proper serialization
        sample_data = data_to_analyze.head().copy()
        for col in sample_data.columns:
            if sample_data[col].dtype == 'object':
                sample_data[col] = sample_data[col].astype(str)
            else:
                # Convert to float where possible, keep as string otherwise
                try:
                    sample_data[col] = sample_data[col].astype(float)
                except:
                    sample_data[col] = sample_data[col].astype(str)

        results = {
            'dataset_info': {
                'shape': data_to_analyze.shape,
                'original_shape': original_data.shape if is_sampled else data_to_analyze.shape,
                'is_sampled': is_sampled,
                'target_column': self.target_column,
                'columns': list(data_to_analyze.columns),
                'dtypes': {col: str(dtype) for col, dtype in data_to_analyze.dtypes.to_dict().items()},
                'missing_values': {col: int(count) for col, count in data_to_analyze.isnull().sum().to_dict().items()},
                'sample_data': sample_data.to_dict('records')
            }
        }

        print("🚀 Starting optimized feature engineering analysis...")

        # 0. Calculate Initial Data Quality Score
        print("📊 Calculating initial data quality score...")
        try:
            initial_quality = self.calculate_data_quality_score(data_to_analyze, self.target_column)
            results['initial_quality_score'] = initial_quality
            if 'error' not in initial_quality:
                print(f"✅ Initial Quality Score: {initial_quality.get('overall_quality_score', 0):.1f}% ({initial_quality.get('quality_grade', 'Unknown')})")
        except Exception as e:
            results['initial_quality_score'] = {'error': str(e)}
            print(f"⚠️ Initial quality score calculation failed: {e}")

        # 1. Feature Importance Analysis (OPTIMIZED: Use faster methods only)
        print("📊 Analyzing feature importance...")
        try:
            # Only use fast methods: Random Forest and Correlation
            importance_results = self.automated_feature_importance(methods=['random_forest', 'correlation'])
            results['feature_importance'] = importance_results
            print("✅ Feature importance analysis completed")
        except Exception as e:
            results['feature_importance'] = {'error': str(e)}
            print(f"⚠️ Feature importance failed: {e}")

        # 2. Intelligent Feature Creation (OPTIMIZED: Reduced from 20 to 10 features)
        print("🎯 Creating intelligent features...")
        try:
            created_features = self.intelligent_feature_creation(max_features=10)
            results['created_features'] = created_features
            print(f"✅ Created {len(created_features)} intelligent features")
        except Exception as e:
            results['created_features'] = {'error': str(e)}
            print(f"⚠️ Feature creation failed: {e}")

        # 3. Dimensionality Reduction (OPTIMIZED: Skip t-SNE as it's very slow, only use PCA)
        print("📈 Performing dimensionality reduction...")
        try:
            dim_results = self.dimensionality_reduction_analysis(methods=['pca'], n_components=2)
            results['dimensionality_reduction'] = dim_results
            print("✅ Dimensionality reduction completed")
        except Exception as e:
            results['dimensionality_reduction'] = {'error': str(e)}
            print(f"⚠️ Dimensionality reduction failed: {e}")

        # Restore original data if sampled
        if is_sampled:
            self.data = original_data
        
        # 4. Automatic Feature Set Comparison (OPTIMIZED: Reduced feature sets)
        print("⚖️ Comparing different feature sets...")
        try:
            # Create automatic feature sets based on importance (REDUCED to only top 10 from best method)
            if ('feature_importance' in results and
                results['feature_importance'] and
                not any('error' in str(v) for v in results['feature_importance'].values())):
                # Get top features only from the first method (fastest)
                feature_sets = {}

                # Only compare top 10 vs all features (reduced from multiple methods)
                for method, scores in list(results['feature_importance'].items())[:1]:  # Only first method
                    if isinstance(scores, list) and len(scores) > 0:
                        top_features = [feat for feat, score in scores[:10]]
                        feature_sets['top_10_features'] = top_features
                        break

                feature_sets['all_features'] = list(self.data.drop(columns=[self.target_column]).columns)

                comparison_results = self.compare_feature_sets(feature_sets, cv_folds=3)  # Reduced from 5 to 3 folds
                results['feature_comparison'] = comparison_results
                print(f"✅ Compared {len(feature_sets)} feature sets")
            else:
                results['feature_comparison'] = {'error': 'No feature importance results available'}
        except Exception as e:
            results['feature_comparison'] = {'error': str(e)}
            print(f"⚠️ Feature comparison failed: {e}")

        # 5. Domain Template Suggestions (SKIP for faster analysis)
        print("🏭 Skipping domain templates for faster analysis...")
        results['domain_templates'] = {'message': 'Skipped for performance optimization'}

        # 6. LLM-Powered Analysis (OPTIMIZED: Reduced to only 2 most important analyses)
        print("🤖 Generating AI insights...")
        try:
            llm_insights = {}

            # Only do the most important analyses to speed up
            # Dataset characteristics analysis
            print("   📊 Analyzing dataset characteristics...")
            llm_insights['dataset_analysis'] = llm_analyzer.analyze_dataset_characteristics(results['dataset_info'])

            # Comprehensive recommendations (combines multiple insights)
            print("   💡 Generating comprehensive recommendations...")
            llm_insights['comprehensive_recommendations'] = llm_analyzer.generate_comprehensive_recommendations(results)

            # Mark other analyses as skipped for performance
            llm_insights['importance_analysis'] = "Skipped for performance - check feature importance results above"
            llm_insights['domain_suggestions'] = "Skipped for performance - use feature creation results"
            llm_insights['performance_explanation'] = "Skipped for performance - check feature comparison results"
            llm_insights['intelligent_decisions'] = {'message': 'Using automated feature selection based on importance scores'}

            results['llm_insights'] = llm_insights
            print("✅ AI insights generated successfully")

        except Exception as e:
            results['llm_insights'] = {'error': str(e)}
            print(f"⚠️ AI insights failed: {e}")
        
        # 7. Create Optimized Dataset
        print("🔧 Creating optimized dataset...")
        try:
            if 'llm_insights' in results and 'intelligent_decisions' in results['llm_insights']:
                optimized_result = self.create_optimized_dataset(results['llm_insights']['intelligent_decisions'])
                results['optimized_dataset'] = optimized_result
                print("✅ Optimized dataset created successfully")
            else:
                results['optimized_dataset'] = {'error': 'No intelligent decisions available for optimization'}
                print("⚠️ Dataset optimization skipped")
                
        except Exception as e:
            results['optimized_dataset'] = {'error': str(e)}
            print(f"⚠️ Dataset optimization failed: {e}")
        
        # Store complete results
        self.all_results = results
        print("🎉 Complete analysis finished!")
        
        return results

# Initialize the feature engineering class
feature_engine = AdvancedFeatureEngineering()

@module6_bp.route('/')
def index():
    return render_template('module6_automated.html')

@module6_bp.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Feature Engineering Module is running'})

@module6_bp.route('/old')
def old_index():
    return render_template('module6_enhanced.html')

@module6_bp.route('/manual')
def manual_index():
    return render_template('module6_new.html')

@module6_bp.route('/available_datasets')
def get_available_datasets():
    """Get list of user's available datasets for dropdown"""
    try:
        user_id = get_user_id()
        user_datasets = datasets.get(user_id, [])

        # Format for dropdown
        dataset_options = []
        for ds in user_datasets:
            dataset_options.append({
                'id': ds['id'],
                'name': ds['name'],
                'rows': ds['rows'],
                'columns': ds['columns'],
                'file_type': ds['file_type']
            })

        return jsonify({
            'datasets': dataset_options,
            'count': len(dataset_options)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/load_dataset/<int:dataset_id>', methods=['POST'])
def load_existing_dataset(dataset_id):
    """Load an existing dataset from DataLab storage"""
    try:
        user_id = get_user_id()
        user_datasets = datasets.get(user_id, [])

        # Find the dataset
        dataset = next((ds for ds in user_datasets if ds['id'] == dataset_id), None)
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404

        # Read the dataset file
        file_path = dataset['file_path']
        file_type = dataset['file_type']

        if file_type == 'csv':
            df = pd.read_csv(file_path)
        elif file_type == 'excel':
            df = pd.read_excel(file_path)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400

        # Load data into feature engine (use existing target detection)
        shape, dtypes, detected_target = feature_engine.load_data(df)

        # Convert pandas dtypes to JSON-serializable format
        dtypes_serializable = {col: str(dtype) for col, dtype in dtypes.items()}

        # Convert sample data to handle pandas dtypes (prevent NaN generation)
        sample_data = df.head().copy()
        sample_records = []
        for _, row in sample_data.iterrows():
            record = {}
            for col in sample_data.columns:
                val = row[col]
                # Safely convert values, replacing NaN/inf with None
                if pd.isna(val):
                    record[col] = None
                elif isinstance(val, (np.integer, np.int64, np.int32)):
                    record[col] = int(val)
                elif isinstance(val, (np.floating, np.float64, np.float32, float)):
                    if np.isnan(val) or np.isinf(val):
                        record[col] = None
                    else:
                        record[col] = float(val)
                else:
                    record[col] = str(val)
            sample_records.append(record)

        return jsonify({
            'message': f'Dataset "{dataset["name"]}" loaded successfully',
            'shape': shape,
            'columns': list(df.columns),
            'dtypes': dtypes_serializable,
            'detected_target': detected_target,
            'sample_data': sample_records,
            'dataset_info': {
                'id': dataset['id'],
                'name': dataset['name'],
                'rows': dataset['rows'],
                'columns': dataset['columns']
            }
        })

    except Exception as e:
        return jsonify({'error': f'Error loading dataset: {str(e)}'}), 500

@module6_bp.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read the file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        # Load data into feature engine (auto-detect target)
        shape, dtypes, detected_target = feature_engine.load_data(df)
        
        # Convert pandas dtypes to JSON-serializable format
        dtypes_serializable = {col: str(dtype) for col, dtype in dtypes.items()}
        
        # Convert sample data to handle pandas dtypes (prevent NaN generation)
        sample_data = df.head().copy()
        sample_records = []
        for _, row in sample_data.iterrows():
            record = {}
            for col in sample_data.columns:
                val = row[col]
                # Safely convert values, replacing NaN/inf with None
                if pd.isna(val):
                    record[col] = None
                elif isinstance(val, (np.integer, np.int64, np.int32)):
                    record[col] = int(val)
                elif isinstance(val, (np.floating, np.float64, np.float32, float)):
                    if np.isnan(val) or np.isinf(val):
                        record[col] = None
                    else:
                        record[col] = float(val)
                else:
                    record[col] = str(val)
            sample_records.append(record)

        return jsonify({
            'message': 'File uploaded successfully',
            'shape': shape,
            'columns': list(df.columns),
            'dtypes': dtypes_serializable,
            'detected_target': detected_target,
            'sample_data': sample_records
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def safe_json_encoder(obj):
    """Ultra-safe JSON encoder that handles ALL edge cases"""
    import math

    # Handle None
    if obj is None:
        return None

    # Handle pandas NA
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass

    # Handle numeric NaN/Inf
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)

    # Handle numpy integers
    if isinstance(obj, np.integer):
        return int(obj)

    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Handle other types
    return str(obj)

@module6_bp.route('/analyze', methods=['POST'])
def run_complete_analysis():
    try:
        if feature_engine.data is None:
            return jsonify({'error': 'No data uploaded yet'}), 400

        print("🚀 Starting Feature Engineering analysis...")

        # Run complete automated analysis
        results = feature_engine.run_complete_analysis()

        print("📊 Analysis complete, generating explanations...")

        # Generate comprehensive explanations
        explanations = {}
        if explainer is not None:
            try:
                # Generate explanations for each part of the analysis
                if 'feature_importance' in results and results['feature_importance']:
                    explanations['feature_importance'] = explainer.explain_feature_importance(results['feature_importance'])

                if 'created_features' in results and results['created_features']:
                    explanations['feature_creation'] = explainer.explain_feature_creation(results['created_features'])

                if 'dimensionality_reduction' in results and results['dimensionality_reduction']:
                    for method, data in results['dimensionality_reduction'].items():
                        explanations[f'dimensionality_{method}'] = explainer.explain_dimensionality_reduction(method, data)

                if 'feature_comparison' in results and results['feature_comparison']:
                    explanations['model_comparison'] = explainer.explain_model_comparison(results['feature_comparison'])

                # Add decision-based explanations
                if decision_logger is not None:
                    # Explain why PCA was used
                    dataset_size = len(feature_engine.data) if feature_engine.data is not None else 0
                    n_features = len(feature_engine.data.columns) if feature_engine.data is not None else 0
                    explanations['pca_vs_tsne'] = explain_pca_vs_tsne_choice(dataset_size, n_features, purpose="model_training")

                    # Explain sampling decision
                    explanations['sampling_decision'] = explain_why_sampling_decision(dataset_size, threshold=5000)

                    # Explain cross-validation choice
                    explanations['cv_folds'] = explain_cv_folds_choice(n_folds=5)

                # Add processing steps explanation
                processing_steps = [
                    "Data Quality Assessment",
                    "Feature Importance Calculation",
                    "Intelligent Feature Creation",
                    "Dimensionality Reduction",
                    "Feature Set Comparison",
                    "Optimization and Export"
                ]
                explanations['processing_steps'] = explainer.generate_processing_steps_explanation(processing_steps)

                print("✅ Explanations generated successfully")
            except Exception as exp_err:
                print(f"⚠️ Error generating explanations: {exp_err}")
                explanations = {'error': f'Could not generate explanations: {str(exp_err)}'}
        else:
            explanations = {'info': 'Explainable AI not available'}

        # Add explanations to results
        results['explanations'] = explanations

        print("📊 Serializing results...")

        # Make results JSON serializable - deep clean ALL NaN values
        serializable_results = make_json_serializable(results)

        # ULTRA-SAFE: Force convert to JSON string with custom encoder, then parse back
        print("🔒 Applying ultra-safe JSON encoding...")
        try:
            # Convert to JSON string with safe encoder
            json_str = json.dumps(serializable_results, default=safe_json_encoder, allow_nan=False)
            # Parse back to ensure it's valid
            serializable_results = json.loads(json_str)
        except Exception as json_err:
            print(f"⚠️ JSON encoding error: {json_err}")
            # Absolute fallback: convert everything to strings
            json_str = json.dumps(serializable_results, default=str, allow_nan=False)
            serializable_results = json.loads(json_str)

        print("✅ Feature Engineering analysis complete!")

        return jsonify({
            'message': 'Complete analysis finished successfully',
            'results': serializable_results
        })

    except Exception as e:
        print(f"❌ Feature Engineering Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/feature_importance', methods=['POST'])
def calculate_feature_importance():
    try:
        data = request.json
        methods = data.get('methods', ['random_forest', 'xgboost', 'mutual_info', 'correlation'])
        
        results = feature_engine.automated_feature_importance(methods)
        
        return jsonify({
            'results': results,
            'message': 'Feature importance calculated successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/create_features', methods=['POST'])
def create_intelligent_features():
    try:
        data = request.json
        max_features = data.get('max_features', 20)
        
        results = feature_engine.intelligent_feature_creation(max_features)
        
        return jsonify({
            'results': results,
            'message': f'Created {len(results)} intelligent features'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/dimensionality_reduction', methods=['POST'])
def perform_dimensionality_reduction():
    try:
        data = request.json
        methods = data.get('methods', ['pca', 'tsne'])
        n_components = data.get('n_components', 2)
        
        results = feature_engine.dimensionality_reduction_analysis(methods, n_components)
        
        # Convert numpy arrays to lists for JSON serialization
        for method, result in results.items():
            if 'transformed_data' in result:
                result['transformed_data'] = result['transformed_data'].tolist()
        
        return jsonify({
            'results': results,
            'message': 'Dimensionality reduction completed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/compare_features', methods=['POST'])
def compare_feature_sets():
    try:
        data = request.json
        feature_sets = data.get('feature_sets', {})
        model_type = data.get('model_type', 'auto')
        
        results = feature_engine.compare_feature_sets(feature_sets, model_type)
        
        return jsonify({
            'results': results,
            'message': 'Feature set comparison completed successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/domain_templates')
def get_domain_templates():
    try:
        templates = feature_engine.get_domain_templates()
        return jsonify({
            'templates': templates,
            'message': 'Domain templates retrieved successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/download/optimized_dataset', methods=['GET', 'POST'])
def download_optimized_dataset():
    """Download the optimized dataset as CSV"""
    try:
        if feature_engine.all_results is None:
            return jsonify({'error': 'No analysis results available'}), 400

        optimized_result = feature_engine.all_results.get('optimized_dataset')
        if not optimized_result or optimized_result.get('error'):
            return jsonify({'error': 'No optimized dataset available'}), 400

        # Get the optimized dataset
        optimized_data = optimized_result.get('optimized_dataset', [])
        if not optimized_data:
            return jsonify({'error': 'Optimized dataset is empty'}), 400

        # Convert to DataFrame
        df = pd.DataFrame(optimized_data)

        # Create CSV response
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='optimized_dataset.csv'
        )

    except Exception as e:
        return jsonify({'error': f'Error downloading optimized dataset: {str(e)}'}), 500

@module6_bp.route('/download/feature_importance', methods=['GET'])
def download_feature_importance():
    """Download feature importance results as CSV"""
    try:
        if feature_engine.all_results is None:
            return jsonify({'error': 'No analysis results available'}), 400

        feature_importance = feature_engine.all_results.get('feature_importance')
        if not feature_importance or isinstance(feature_importance, dict) and feature_importance.get('error'):
            return jsonify({'error': 'No feature importance results available'}), 400

        # Create CSV content
        output = io.StringIO()
        output.write('Feature,Method,Importance_Score\n')

        if isinstance(feature_importance, dict):
            for method, scores in feature_importance.items():
                if isinstance(scores, list):
                    for feature, score in scores:
                        output.write(f'"{feature}","{method}",{score}\n')
                elif not isinstance(scores, str) or 'error' not in scores.lower():
                    # Handle other formats
                    output.write(f'"{method}","{str(scores)}",0\n')

        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='feature_importance.csv'
        )

    except Exception as e:
        return jsonify({'error': f'Error downloading feature importance: {str(e)}'}), 500

@module6_bp.route('/download/analysis_results', methods=['GET'])
def download_analysis_results():
    """Download complete analysis results as JSON"""
    try:
        if feature_engine.all_results is None:
            return jsonify({'error': 'No analysis results available'}), 400

        # Make results JSON serializable
        serializable_results = make_json_serializable(feature_engine.all_results)

        # Convert to JSON string
        json_output = json.dumps(serializable_results, indent=2)

        return send_file(
            io.BytesIO(json_output.encode('utf-8')),
            mimetype='application/json',
            as_attachment=True,
            download_name='feature_engineering_analysis.json'
        )

    except Exception as e:
        return jsonify({'error': f'Error downloading analysis results: {str(e)}'}), 500

@module6_bp.route('/download/feature_list', methods=['GET'])
def download_feature_list():
    """Download list of created features as CSV"""
    try:
        if feature_engine.all_results is None:
            return jsonify({'error': 'No analysis results available'}), 400

        created_features = feature_engine.all_results.get('created_features')
        if not created_features or isinstance(created_features, dict) and created_features.get('error'):
            return jsonify({'error': 'No created features available'}), 400

        # Create CSV content
        output = io.StringIO()
        output.write('Feature_Name,Type\n')

        if isinstance(created_features, dict):
            for feature_name, description in created_features.items():
                feature_type = 'engineered'
                if 'ratio_' in feature_name:
                    feature_type = 'ratio'
                elif 'log_' in feature_name:
                    feature_type = 'log_transformation'
                elif 'poly_' in feature_name:
                    feature_type = 'polynomial'
                elif 'binned_' in feature_name:
                    feature_type = 'binning'
                elif 'mean_' in feature_name or 'std_' in feature_name or 'max_' in feature_name or 'min_' in feature_name:
                    feature_type = 'statistical'

                output.write(f'"{feature_name}","{feature_type}"\n')

        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='created_features.csv'
        )

    except Exception as e:
        return jsonify({'error': f'Error downloading feature list: {str(e)}'}), 500

# ============================================================================
# MANUAL FEATURE ENGINEERING ROUTES
# ============================================================================

@module6_bp.route('/manual/log_transform', methods=['POST'])
def manual_log_transform():
    """Apply logarithmic transformation to selected column"""
    try:
        if feature_engine.data is None:
            return jsonify({'error': 'No data loaded'}), 400

        data = request.json
        column = data.get('column')
        base = data.get('base', 'e')  # 'e', '10', or '2'

        if not column or column not in feature_engine.data.columns:
            return jsonify({'error': 'Invalid column specified'}), 400

        # Check if column is numeric
        if not pd.api.types.is_numeric_dtype(feature_engine.data[column]):
            return jsonify({'error': 'Column must be numeric for log transformation'}), 400

        # Check for non-positive values
        if (feature_engine.data[column] <= 0).any():
            return jsonify({'error': 'Column contains non-positive values. Log transformation requires positive values.'}), 400

        # Apply transformation
        new_column_name = f'log_{column}'
        if base == 'e':
            feature_engine.data[new_column_name] = np.log(feature_engine.data[column])
        elif base == '10':
            feature_engine.data[new_column_name] = np.log10(feature_engine.data[column])
        elif base == '2':
            feature_engine.data[new_column_name] = np.log2(feature_engine.data[column])
        else:
            return jsonify({'error': 'Invalid base specified'}), 400

        # Get statistics
        stats = {
            'original_mean': float(feature_engine.data[column].mean()),
            'original_std': float(feature_engine.data[column].std()),
            'transformed_mean': float(feature_engine.data[new_column_name].mean()),
            'transformed_std': float(feature_engine.data[new_column_name].std()),
            'new_column': new_column_name
        }

        return jsonify({
            'message': f'Log transformation applied successfully',
            'new_column': new_column_name,
            'new_feature_name': new_column_name,  # For frontend compatibility
            'statistics': stats,
            'sample_data': feature_engine.data[[column, new_column_name]].head().to_dict('records')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/manual/power_transform', methods=['POST'])
def manual_power_transform():
    """Apply power transformation to selected column"""
    try:
        if feature_engine.data is None:
            return jsonify({'error': 'No data loaded'}), 400

        data = request.json
        column = data.get('column')
        power = data.get('power', 2)  # Default to square

        if not column or column not in feature_engine.data.columns:
            return jsonify({'error': 'Invalid column specified'}), 400

        # Check if column is numeric
        if not pd.api.types.is_numeric_dtype(feature_engine.data[column]):
            return jsonify({'error': 'Column must be numeric for power transformation'}), 400

        # Apply transformation
        power_names = {0.5: 'sqrt', 2: 'square', 3: 'cube'}
        power_name = power_names.get(power, f'power{power}')
        new_column_name = f'{power_name}_{column}'

        feature_engine.data[new_column_name] = np.power(feature_engine.data[column], power)

        # Get statistics
        stats = {
            'original_mean': float(feature_engine.data[column].mean()),
            'original_std': float(feature_engine.data[column].std()),
            'transformed_mean': float(feature_engine.data[new_column_name].mean()),
            'transformed_std': float(feature_engine.data[new_column_name].std()),
            'new_column': new_column_name,
            'power': power
        }

        return jsonify({
            'message': f'Power transformation (^{power}) applied successfully',
            'new_column': new_column_name,
            'new_feature_name': new_column_name,  # For frontend compatibility
            'statistics': stats,
            'sample_data': feature_engine.data[[column, new_column_name]].head().to_dict('records')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/manual/ratio_feature', methods=['POST'])
def manual_ratio_feature():
    """Create ratio feature from two columns"""
    try:
        if feature_engine.data is None:
            return jsonify({'error': 'No data loaded'}), 400

        data = request.json
        numerator = data.get('numerator')
        denominator = data.get('denominator')

        if not numerator or numerator not in feature_engine.data.columns:
            return jsonify({'error': 'Invalid numerator column'}), 400

        if not denominator or denominator not in feature_engine.data.columns:
            return jsonify({'error': 'Invalid denominator column'}), 400

        # Check if columns are numeric
        if not pd.api.types.is_numeric_dtype(feature_engine.data[numerator]):
            return jsonify({'error': 'Numerator column must be numeric'}), 400

        if not pd.api.types.is_numeric_dtype(feature_engine.data[denominator]):
            return jsonify({'error': 'Denominator column must be numeric'}), 400

        # Check for zeros in denominator
        if (feature_engine.data[denominator] == 0).any():
            return jsonify({'error': 'Denominator contains zero values'}), 400

        # Create ratio
        new_column_name = f'ratio_{numerator}_{denominator}'
        feature_engine.data[new_column_name] = feature_engine.data[numerator] / feature_engine.data[denominator]

        # Replace inf values with NaN and then fill
        feature_engine.data[new_column_name] = feature_engine.data[new_column_name].replace([np.inf, -np.inf], np.nan)

        # Get statistics
        stats = {
            'mean': float(feature_engine.data[new_column_name].mean()),
            'std': float(feature_engine.data[new_column_name].std()),
            'min': float(feature_engine.data[new_column_name].min()),
            'max': float(feature_engine.data[new_column_name].max()),
            'new_column': new_column_name
        }

        return jsonify({
            'message': f'Ratio feature created successfully',
            'new_column': new_column_name,
            'new_feature_name': new_column_name,  # For frontend compatibility
            'statistics': stats,
            'sample_data': feature_engine.data[[numerator, denominator, new_column_name]].head().to_dict('records')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/manual/interaction_feature', methods=['POST'])
def manual_interaction_feature():
    """Create interaction feature (multiplication) from two columns"""
    try:
        if feature_engine.data is None:
            return jsonify({'error': 'No data loaded'}), 400

        data = request.json
        column1 = data.get('column1')
        column2 = data.get('column2')

        if not column1 or column1 not in feature_engine.data.columns:
            return jsonify({'error': 'Invalid first column'}), 400

        if not column2 or column2 not in feature_engine.data.columns:
            return jsonify({'error': 'Invalid second column'}), 400

        # Check if columns are numeric
        if not pd.api.types.is_numeric_dtype(feature_engine.data[column1]):
            return jsonify({'error': 'First column must be numeric'}), 400

        if not pd.api.types.is_numeric_dtype(feature_engine.data[column2]):
            return jsonify({'error': 'Second column must be numeric'}), 400

        # Create interaction
        new_column_name = f'interaction_{column1}_{column2}'
        feature_engine.data[new_column_name] = feature_engine.data[column1] * feature_engine.data[column2]

        # Get statistics
        stats = {
            'mean': float(feature_engine.data[new_column_name].mean()),
            'std': float(feature_engine.data[new_column_name].std()),
            'min': float(feature_engine.data[new_column_name].min()),
            'max': float(feature_engine.data[new_column_name].max()),
            'new_column': new_column_name
        }

        return jsonify({
            'message': f'Interaction feature created successfully',
            'new_column': new_column_name,
            'new_feature_name': new_column_name,  # For frontend compatibility
            'statistics': stats,
            'sample_data': feature_engine.data[[column1, column2, new_column_name]].head().to_dict('records')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/manual/binning', methods=['POST'])
def manual_binning():
    """Apply binning to selected column"""
    try:
        if feature_engine.data is None:
            return jsonify({'error': 'No data loaded'}), 400

        data = request.json
        column = data.get('column')
        n_bins = data.get('n_bins', 5)
        strategy = data.get('strategy', 'equal_width')  # 'equal_width', 'equal_frequency', 'quantile'

        if not column or column not in feature_engine.data.columns:
            return jsonify({'error': 'Invalid column specified'}), 400

        # Check if column is numeric
        if not pd.api.types.is_numeric_dtype(feature_engine.data[column]):
            return jsonify({'error': 'Column must be numeric for binning'}), 400

        # Apply binning
        new_column_name = f'binned_{column}'

        if strategy == 'equal_width':
            feature_engine.data[new_column_name] = pd.cut(feature_engine.data[column], bins=n_bins, labels=False)
        elif strategy == 'equal_frequency':
            feature_engine.data[new_column_name] = pd.qcut(feature_engine.data[column], q=n_bins, labels=False, duplicates='drop')
        elif strategy == 'quantile':
            feature_engine.data[new_column_name] = pd.qcut(feature_engine.data[column], q=n_bins, labels=False, duplicates='drop')
        else:
            return jsonify({'error': 'Invalid binning strategy'}), 400

        # Fill NaN values that may result from binning
        feature_engine.data[new_column_name] = feature_engine.data[new_column_name].fillna(0)

        # Get bin statistics
        bin_counts = feature_engine.data[new_column_name].value_counts().sort_index().to_dict()

        return jsonify({
            'message': f'Binning applied successfully using {strategy} strategy',
            'new_column': new_column_name,
            'n_bins': n_bins,
            'strategy': strategy,
            'bin_counts': {int(k): int(v) for k, v in bin_counts.items()},
            'sample_data': feature_engine.data[[column, new_column_name]].head().to_dict('records')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/manual/scaling', methods=['POST'])
def manual_scaling():
    """Apply scaling to selected columns"""
    try:
        if feature_engine.data is None:
            return jsonify({'error': 'No data loaded'}), 400

        data = request.json
        columns = data.get('columns', [])
        method = data.get('method', 'standardization')  # 'standardization', 'minmax', 'robust', 'maxabs'

        if not columns:
            return jsonify({'error': 'No columns specified'}), 400

        # Validate columns
        invalid_columns = [col for col in columns if col not in feature_engine.data.columns]
        if invalid_columns:
            return jsonify({'error': f'Invalid columns: {invalid_columns}'}), 400

        # Check if columns are numeric
        non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(feature_engine.data[col])]
        if non_numeric:
            return jsonify({'error': f'Non-numeric columns: {non_numeric}'}), 400

        # Apply scaling
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

        if method == 'standardization':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'maxabs':
            scaler = MaxAbsScaler()
        else:
            return jsonify({'error': 'Invalid scaling method'}), 400

        # Create new columns with scaled values
        scaled_data = scaler.fit_transform(feature_engine.data[columns])
        new_columns = []

        for i, col in enumerate(columns):
            new_col_name = f'scaled_{col}'
            feature_engine.data[new_col_name] = scaled_data[:, i]
            new_columns.append(new_col_name)

        # Get statistics
        stats = {}
        for orig_col, new_col in zip(columns, new_columns):
            stats[orig_col] = {
                'original_mean': float(feature_engine.data[orig_col].mean()),
                'original_std': float(feature_engine.data[orig_col].std()),
                'scaled_mean': float(feature_engine.data[new_col].mean()),
                'scaled_std': float(feature_engine.data[new_col].std()),
                'new_column': new_col
            }

        return jsonify({
            'message': f'{method} scaling applied successfully',
            'method': method,
            'new_columns': new_columns,
            'statistics': stats
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/manual/select_features', methods=['POST'])
def manual_select_features():
    """Select specific features and create a new dataset"""
    try:
        if feature_engine.data is None:
            return jsonify({'error': 'No data loaded'}), 400

        data = request.json
        selected_features = data.get('features', [])

        if not selected_features:
            return jsonify({'error': 'No features selected'}), 400

        # Validate features
        invalid_features = [f for f in selected_features if f not in feature_engine.data.columns]
        if invalid_features:
            return jsonify({'error': f'Invalid features: {invalid_features}'}), 400

        # Create subset with selected features
        original_shape = feature_engine.data.shape

        # Store the selection (don't modify original data yet)
        selected_data = feature_engine.data[selected_features].copy()

        return jsonify({
            'message': f'Selected {len(selected_features)} features',
            'selected_features': selected_features,
            'original_shape': original_shape,
            'new_shape': selected_data.shape,
            'sample_data': selected_data.head().to_dict('records')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/manual/get_current_features', methods=['GET'])
def get_current_features():
    """Get list of current features in the dataset"""
    try:
        if feature_engine.data is None:
            return jsonify({'error': 'No data loaded'}), 400

        features_info = []
        for col in feature_engine.data.columns:
            features_info.append({
                'name': col,
                'dtype': str(feature_engine.data[col].dtype),
                'is_numeric': pd.api.types.is_numeric_dtype(feature_engine.data[col]),
                'null_count': int(feature_engine.data[col].isnull().sum()),
                'unique_count': int(feature_engine.data[col].nunique())
            })

        return jsonify({
            'features': features_info,
            'total_features': len(features_info),
            'shape': feature_engine.data.shape
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/manual/download_dataset', methods=['POST'])
def download_manual_dataset():
    """Download the current dataset with all manual transformations"""
    try:
        if feature_engine.data is None:
            return jsonify({'error': 'No data loaded'}), 400

        # Create CSV response
        output = io.StringIO()
        feature_engine.data.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='manual_engineered_dataset.csv'
        )

    except Exception as e:
        return jsonify({'error': f'Error downloading dataset: {str(e)}'}), 500

@module6_bp.route('/download/manual_dataset', methods=['GET'])
def download_manual_dataset_get():
    """Download the current dataset with all manual transformations (GET version)"""
    try:
        if feature_engine.data is None:
            return jsonify({'error': 'No data loaded'}), 400

        # Create CSV response
        output = io.StringIO()
        feature_engine.data.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='manual_engineered_dataset.csv'
        )

    except Exception as e:
        return jsonify({'error': f'Error downloading dataset: {str(e)}'}), 500

@module6_bp.route('/manual/reset_dataset', methods=['POST'])
def reset_manual_dataset():
    """Reset dataset to original state (reload from file)"""
    try:
        # This would need to reload the original dataset
        # For now, return a message that user should reload the dataset
        return jsonify({
            'message': 'Please reload your dataset to reset all transformations'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# END MANUAL FEATURE ENGINEERING ROUTES
# ============================================================================

# ============================================================================
# NEW ADVANCED FEATURE SELECTION ROUTES
# ============================================================================

@module6_bp.route('/run_automated', methods=['POST'])
def run_fully_automated():
    """Fully Automated Feature Selection - Runs ALL methods and recommends best"""
    try:
        if feature_engine.data is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        X = feature_engine.data.drop(columns=[feature_engine.target_column])
        y = feature_engine.data[feature_engine.target_column]
        X_encoded = pd.get_dummies(X, drop_first=True).fillna(0)
        
        selector = AdvancedFeatureSelector()
        all_results = {}
        detailed_logs = []
        
        print("\n" + "="*60)
        print("RUNNING FULLY AUTOMATED FEATURE SELECTION (3 METHODS)")
        print("="*60)
        print(f"Dataset: {X_encoded.shape[0]} samples, {X_encoded.shape[1]} features")
        print(f"Target: {feature_engine.target_column}")
        print(f"Task Type: {'Classification' if y.dtype == 'object' or y.nunique() < 10 else 'Regression'}")
        detailed_logs.append(f"Dataset: {X_encoded.shape[0]} samples, {X_encoded.shape[1]} features")
        detailed_logs.append(f"Target: {feature_engine.target_column}")
        
        # 1. Forward Selection
        print("\n" + "="*60)
        print("[1/3] FORWARD SELECTION")
        print("="*60)
        detailed_logs.append("\n[1/3] FORWARD SELECTION")
        try:
            result = selector.forward_selection(X_encoded, y, max_features=5, cv=5)
            all_results['forward_selection'] = {
                'selected_features': result['selected_features'],
                'n_features': result['n_features'],
                'score': result['final_score'],
                'cv_details': result.get('cv_details', []),
                'cv_folds': result.get('cv_folds', 5),
                'scoring_metric': result.get('scoring_metric', 'accuracy'),
                'method': 'Forward Selection',
                'description': 'Iteratively adds features that improve model performance'
            }
            detailed_logs.append(f"Selected {result['n_features']} features with CV score: {result['final_score']:.4f}")
            print(f"\nFORWARD SELECTION COMPLETE: {result['n_features']} features selected")
        except Exception as e:
            all_results['forward_selection'] = {'error': str(e)}
            detailed_logs.append(f"ERROR: {str(e)}")
            print(f"ERROR: {str(e)}")
        
        # 2. Backward Elimination
        print("\n" + "="*60)
        print("[2/3] BACKWARD ELIMINATION")
        print("="*60)
        detailed_logs.append("\n[2/3] BACKWARD ELIMINATION")
        try:
            result = selector.backward_elimination(X_encoded, y, cv=5, threshold=0.005)
            all_results['backward_elimination'] = {
                'selected_features': result['selected_features'],
                'n_features': result['n_features'],
                'score': result['final_score'],
                'cv_details': result.get('cv_details', []),
                'cv_folds': result.get('cv_folds', 5),
                'scoring_metric': result.get('scoring_metric', 'accuracy'),
                'method': 'Backward Elimination',
                'description': 'Iteratively removes features that hurt model performance'
            }
            detailed_logs.append(f"Selected {result['n_features']} features with CV score: {result['final_score']:.4f}")
            print(f"\nBACKWARD ELIMINATION COMPLETE: {result['n_features']} features selected")
        except Exception as e:
            all_results['backward_elimination'] = {'error': str(e)}
            detailed_logs.append(f"ERROR: {str(e)}")
            print(f"ERROR: {str(e)}")
        
        # 3. Stepwise Selection
        print("\n" + "="*60)
        print("[3/3] STEPWISE SELECTION")
        print("="*60)
        detailed_logs.append("\n[3/3] STEPWISE SELECTION")
        try:
            result = selector.stepwise_selection(X_encoded, y, max_features=5, cv=5, threshold=0.005)
            all_results['stepwise_selection'] = {
                'selected_features': result['selected_features'],
                'n_features': result['n_features'],
                'score': result['final_score'],
                'cv_details': result.get('cv_details', []),
                'cv_folds': result.get('cv_folds', 5),
                'scoring_metric': result.get('scoring_metric', 'accuracy'),
                'method': 'Stepwise Selection',
                'description': 'Combines forward and backward: adds best features, removes worst features'
            }
            detailed_logs.append(f"Selected {result['n_features']} features with CV score: {result['final_score']:.4f}")
            print(f"\nSTEPWISE SELECTION COMPLETE: {result['n_features']} features selected")
        except Exception as e:
            all_results['stepwise_selection'] = {'error': str(e)}
            detailed_logs.append(f"ERROR: {str(e)}")
            print(f"ERROR: {str(e)}")
        
        # Calculate best method
        print("\n" + "="*60)
        print("CALCULATING BEST METHOD")
        print("="*60)
        detailed_logs.append("\n" + "="*60)
        detailed_logs.append("CALCULATING BEST METHOD")
        detailed_logs.append("="*60)
        
        best_method = None
        best_score = -np.inf
        scoring_details = []
        
        # Score each method
        for method_name, result in all_results.items():
            if 'error' in result:
                continue
            
            # Calculate composite score
            n_features = result['n_features']
            model_score = result.get('score', 0)
            
            # Scoring criteria:
            # 1. Model performance (if available) - 50%
            # 2. Feature reduction (fewer is better) - 30%
            # 3. Method reliability - 20%
            
            if model_score:
                performance_score = abs(model_score) * 0.5
            else:
                performance_score = 0
            
            # Feature reduction score (prefer 30-50% of original features)
            original_features = len(X_encoded.columns)
            ideal_ratio = 0.4
            actual_ratio = n_features / original_features
            reduction_score = (1 - abs(actual_ratio - ideal_ratio)) * 0.3
            
            # Method reliability scores (based on literature and empirical studies)
            reliability_scores = {
                'forward_selection': 0.9,  # High: Greedy but effective
                'backward_elimination': 0.9,  # High: Comprehensive evaluation
                'stepwise_selection': 0.95  # Highest: Combines both approaches
            }
            reliability_score = reliability_scores.get(method_name, 0.5) * 0.2
            
            composite_score = performance_score + reduction_score + reliability_score
            result['composite_score'] = float(composite_score)
            result['score_breakdown'] = {
                'performance': float(performance_score),
                'reduction': float(reduction_score),
                'reliability': float(reliability_score)
            }
            
            # Log scoring details
            score_detail = {
                'method': method_name,
                'n_features': n_features,
                'cv_score': model_score if model_score else 'N/A',
                'composite_score': float(composite_score),
                'breakdown': {
                    'performance': f"{performance_score:.4f} (50% weight)",
                    'reduction': f"{reduction_score:.4f} (30% weight)",
                    'reliability': f"{reliability_score:.4f} (20% weight)"
                }
            }
            scoring_details.append(score_detail)
            
            print(f"\n{method_name.upper().replace('_', ' ')}:")
            print(f"  Features: {n_features}/{original_features} ({n_features/original_features*100:.1f}%)")
            if model_score:
                print(f"  CV Score: {model_score:.4f}")
            print(f"  Composite Score: {composite_score:.4f}")
            print(f"    - Performance: {performance_score:.4f} (50%)")
            print(f"    - Reduction: {reduction_score:.4f} (30%)")
            print(f"    - Reliability: {reliability_score:.4f} (20%)")
            
            detailed_logs.append(f"{method_name}: {n_features} features, composite score: {composite_score:.4f}")
            
            if composite_score > best_score:
                best_score = composite_score
                best_method = method_name
        
        # Prepare recommendation
        recommendation = {
            'best_method': best_method,
            'best_method_details': all_results[best_method] if best_method else None,
            'all_results': all_results,
            'original_features': len(X_encoded.columns),
            'scoring_details': scoring_details,
            'detailed_logs': detailed_logs,
            'recommendation_reason': f"Selected based on composite score of {best_score:.3f} considering model performance (50%), feature reduction (30%), and method reliability (20%)"
        }

        print("\n" + "="*60)
        print("RECOMMENDATION")
        print("="*60)
        print(f"BEST METHOD: {best_method.upper().replace('_', ' ')}")
        print(f"COMPOSITE SCORE: {best_score:.4f}")
        print(f"SELECTED FEATURES: {all_results[best_method]['n_features']}/{len(X_encoded.columns)}")
        print(f"REASON: {recommendation['recommendation_reason']}")
        print("="*60)

        detailed_logs.append(f"\nRECOMMENDATION: {best_method} with {all_results[best_method]['n_features']} features")

        # Generate comprehensive explanations
        print("📊 Generating explanations...")
        explanations = {}
        try:
            # Get dataset info
            dataset_size = len(feature_engine.data) if feature_engine.data is not None else 0
            n_features = len(X_encoded.columns)

            # Explain why PCA was used
            if explain_pca_vs_tsne_choice is not None:
                explanations['pca_vs_tsne'] = explain_pca_vs_tsne_choice(dataset_size, n_features, purpose="model_training")

            # Explain sampling decision
            if explain_why_sampling_decision is not None:
                explanations['sampling_decision'] = explain_why_sampling_decision(dataset_size, threshold=5000)

            # Explain cross-validation choice
            if explain_cv_folds_choice is not None:
                explanations['cv_folds'] = explain_cv_folds_choice(n_folds=5)

            print("✅ Explanations generated successfully")
        except Exception as exp_err:
            print(f"⚠️ Error generating explanations: {exp_err}")
            explanations = {'error': f'Could not generate explanations: {str(exp_err)}'}

        # Add explanations to recommendation
        recommendation['explanations'] = explanations

        return jsonify({
            'message': 'Automated feature selection completed',
            'recommendation': recommendation
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nERROR: {str(e)}")
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/automated/forward_selection', methods=['POST'])
def automated_forward_selection():
    """Automated Forward Selection"""
    try:
        if feature_engine.data is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        data = request.json
        max_features = data.get('max_features', None)
        cv_folds = data.get('cv_folds', 3)
        
        X = feature_engine.data.drop(columns=[feature_engine.target_column])
        y = feature_engine.data[feature_engine.target_column]
        
        # Handle categorical variables
        X_encoded = pd.get_dummies(X, drop_first=True).fillna(0)
        
        selector = AdvancedFeatureSelector()
        results = selector.forward_selection(X_encoded, y, max_features=max_features, cv=cv_folds)
        
        return jsonify({
            'message': 'Forward selection completed',
            'results': results,
            'method': 'forward_selection'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/automated/backward_elimination', methods=['POST'])
def automated_backward_elimination():
    """Automated Backward Elimination"""
    try:
        if feature_engine.data is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        data = request.json
        cv_folds = data.get('cv_folds', 3)
        threshold = data.get('threshold', 0.01)
        
        X = feature_engine.data.drop(columns=[feature_engine.target_column])
        y = feature_engine.data[feature_engine.target_column]
        
        # Handle categorical variables
        X_encoded = pd.get_dummies(X, drop_first=True).fillna(0)
        
        selector = AdvancedFeatureSelector()
        results = selector.backward_elimination(X_encoded, y, cv=cv_folds, threshold=threshold)
        
        return jsonify({
            'message': 'Backward elimination completed',
            'results': results,
            'method': 'backward_elimination'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/automated/feature_importance', methods=['POST'])
def automated_feature_importance():
    """Feature Importance from Simple Models"""
    try:
        if feature_engine.data is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        data = request.json
        methods = data.get('methods', ['random_forest', 'linear', 'tree'])
        top_n = data.get('top_n', None)
        threshold = data.get('threshold', None)
        
        X = feature_engine.data.drop(columns=[feature_engine.target_column])
        y = feature_engine.data[feature_engine.target_column]
        
        # Handle categorical variables
        X_encoded = pd.get_dummies(X, drop_first=True).fillna(0)
        
        selector = AdvancedFeatureSelector()
        results = selector.feature_importance_selection(X_encoded, y, methods=methods, top_n=top_n, threshold=threshold)
        
        # Make results JSON serializable
        results['importance_scores'] = {k: {feat: float(score) for feat, score in v.items()} 
                                       for k, v in results['importance_scores'].items()}
        results['aggregated_scores'] = {k: float(v) for k, v in results['aggregated_scores'].items()}
        results['sorted_features'] = [(f, float(s)) for f, s in results['sorted_features']]
        
        return jsonify({
            'message': 'Feature importance analysis completed',
            'results': results,
            'method': 'feature_importance'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/manual/correlation_analysis', methods=['POST'])
def manual_correlation_analysis():
    """Correlation Analysis for Manual Feature Selection"""
    try:
        if feature_engine.data is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        data = request.json
        target_threshold = data.get('target_threshold', 0.05)
        multicollinearity_threshold = data.get('multicollinearity_threshold', 0.9)
        method = data.get('method', 'pearson')
        
        X = feature_engine.data.drop(columns=[feature_engine.target_column])
        y = feature_engine.data[feature_engine.target_column]
        
        # Only use numeric features
        X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
        
        selector = AdvancedFeatureSelector()
        results = selector.correlation_analysis(X_numeric, y, target_threshold, multicollinearity_threshold, method)
        
        # Make results JSON serializable
        results['target_correlations'] = {k: float(v) for k, v in results['target_correlations'].items()}
        
        return jsonify({
            'message': 'Correlation analysis completed',
            'results': results,
            'method': 'correlation_analysis'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/manual/variance_threshold', methods=['POST'])
def manual_variance_threshold():
    """Variance Threshold for Manual Feature Selection"""
    try:
        if feature_engine.data is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        data = request.json
        threshold = data.get('threshold', 0.01)
        quasi_constant_threshold = data.get('quasi_constant_threshold', 0.95)
        
        X = feature_engine.data.drop(columns=[feature_engine.target_column])
        
        # Only use numeric features
        X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
        
        selector = AdvancedFeatureSelector()
        results = selector.variance_threshold_selection(X_numeric, threshold, quasi_constant_threshold)
        
        # Make results JSON serializable
        results['variances'] = {k: float(v) for k, v in results['variances'].items()}
        
        return jsonify({
            'message': 'Variance threshold analysis completed',
            'results': results,
            'method': 'variance_threshold'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/manual/vif_analysis', methods=['POST'])
def manual_vif_analysis():
    """VIF Analysis for Multicollinearity Detection"""
    try:
        if feature_engine.data is None:
            return jsonify({'error': 'No data loaded'}), 400
        
        data = request.json
        threshold = data.get('threshold', 10)
        
        X = feature_engine.data.drop(columns=[feature_engine.target_column])
        
        # Only use numeric features
        X_numeric = X.select_dtypes(include=[np.number]).fillna(0)
        
        # Limit to first 20 features for performance
        if X_numeric.shape[1] > 20:
            X_numeric = X_numeric.iloc[:, :20]
        
        selector = AdvancedFeatureSelector()
        results = selector.vif_analysis(X_numeric, threshold)
        
        # Make results JSON serializable
        for item in results['vif_scores']:
            if np.isinf(item['vif']):
                item['vif'] = 999.99
            else:
                item['vif'] = float(item['vif'])
        
        return jsonify({
            'message': 'VIF analysis completed',
            'results': results,
            'method': 'vif_analysis'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/download_selected_features', methods=['POST'])
def download_selected_features():
    """Download selected features as CSV file"""
    try:
        data = request.json
        selected_features = data.get('selected_features', [])
        method_name = data.get('method_name', 'feature_selection')
        
        if not selected_features:
            return jsonify({'error': 'No features selected'}), 400
        
        if feature_engine.data is None:
            return jsonify({'error': 'No dataset loaded'}), 400
        
        # Include target column if not already included
        if feature_engine.target_column not in selected_features:
            selected_features.append(feature_engine.target_column)
        
        # Create new dataset with selected features
        new_dataset = feature_engine.data[selected_features].copy()
        
        # Create CSV in memory
        output = io.StringIO()
        new_dataset.to_csv(output, index=False)
        output.seek(0)
        
        # Return as downloadable file
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{method_name}_selected_features.csv'
        )
    
    except Exception as e:
        import traceback
        print(f"\n=== ERROR IN EXPORT ===")
        traceback.print_exc()
        print(f"=== END ERROR ===")
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/generate_notebook', methods=['POST'])
def generate_notebook():
    """Generate Jupyter notebook with feature selection code"""
    try:
        data = request.json
        method_name = data.get('method_name', 'feature_selection')
        selected_features = data.get('selected_features', [])
        
        # Create notebook content
        notebook = {
            'cells': [
                {
                    'cell_type': 'markdown',
                    'metadata': {},
                    'source': [f'# Feature Selection: {method_name}\n',
                              f'\nSelected {len(selected_features)} features using {method_name}\n']
                },
                {
                    'cell_type': 'code',
                    'execution_count': None,
                    'metadata': {},
                    'outputs': [],
                    'source': [
                        'import pandas as pd\n',
                        'import numpy as np\n',
                        'from sklearn.model_selection import train_test_split\n',
                        'from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n'
                    ]
                },
                {
                    'cell_type': 'markdown',
                    'metadata': {},
                    'source': ['## Load Data\n']
                },
                {
                    'cell_type': 'code',
                    'execution_count': None,
                    'metadata': {},
                    'outputs': [],
                    'source': [
                        '# Load your dataset\n',
                        'df = pd.read_csv("your_dataset.csv")\n',
                        f'\n# Selected features from {method_name}\n',
                        f'selected_features = {selected_features}\n',
                        '\n# Create feature subset\n',
                        'X = df[selected_features]\n',
                        f'y = df["{feature_engine.target_column}"]\n'
                    ]
                },
                {
                    'cell_type': 'markdown',
                    'metadata': {},
                    'source': ['## Train Model\n']
                },
                {
                    'cell_type': 'code',
                    'execution_count': None,
                    'metadata': {},
                    'outputs': [],
                    'source': [
                        '# Split data\n',
                        'X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n',
                        '\n# Train model\n',
                        'model = RandomForestClassifier(n_estimators=100, random_state=42)\n',
                        'model.fit(X_train, y_train)\n',
                        '\n# Evaluate\n',
                        'score = model.score(X_test, y_test)\n',
                        'print(f"Model Accuracy: {score:.4f}")\n'
                    ]
                }
            ],
            'metadata': {
                'kernelspec': {
                    'display_name': 'Python 3',
                    'language': 'python',
                    'name': 'python3'
                },
                'language_info': {
                    'name': 'python',
                    'version': '3.8.0'
                }
            },
            'nbformat': 4,
            'nbformat_minor': 4
        }
        
        return jsonify({
            'notebook': notebook,
            'filename': f'{method_name}_feature_selection.ipynb'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# END NEW ADVANCED FEATURE SELECTION ROUTES
# ============================================================================

@module6_bp.route('/visualize/<viz_type>')
def create_visualization(viz_type):
    try:
        plt.figure(figsize=(12, 8))
        
        if viz_type == 'feature_importance':
            if not feature_engine.feature_importance_results:
                return jsonify({'error': 'No feature importance results available'}), 400
            
            # Create subplot for each method
            methods = list(feature_engine.feature_importance_results.keys())
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, method in enumerate(methods[:4]):
                if isinstance(feature_engine.feature_importance_results[method], list):
                    features, scores = zip(*feature_engine.feature_importance_results[method][:10])
                    axes[i].barh(range(len(features)), scores)
                    axes[i].set_yticks(range(len(features)))
                    axes[i].set_yticklabels(features)
                    axes[i].set_title(f'{method.replace("_", " ").title()} Importance')
                    axes[i].invert_yaxis()
            
            plt.tight_layout()
            
        elif viz_type == 'pca_explained_variance':
            if 'pca' not in feature_engine.dimensionality_results:
                return jsonify({'error': 'No PCA results available'}), 400
            
            variance_ratio = feature_engine.dimensionality_results['pca']['explained_variance_ratio']
            cumulative_variance = feature_engine.dimensionality_results['pca']['cumulative_variance']
            
            plt.subplot(1, 2, 1)
            plt.bar(range(1, len(variance_ratio) + 1), variance_ratio)
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
            plt.title('Explained Variance by Component')
            
            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('Cumulative Explained Variance')
            plt.grid(True)
            
        elif viz_type == 'dimensionality_scatter':
            if not feature_engine.dimensionality_results:
                return jsonify({'error': 'No dimensionality reduction results available'}), 400
            
            methods = list(feature_engine.dimensionality_results.keys())
            fig, axes = plt.subplots(1, len(methods), figsize=(6*len(methods), 6))
            if len(methods) == 1:
                axes = [axes]
            
            for i, method in enumerate(methods):
                data = np.array(feature_engine.dimensionality_results[method]['transformed_data'])
                scatter = axes[i].scatter(data[:, 0], data[:, 1], alpha=0.6, c=range(len(data)), cmap='viridis')
                axes[i].set_title(f'{method.upper()} Visualization')
                axes[i].set_xlabel(f'{method.upper()} Component 1')
                axes[i].set_ylabel(f'{method.upper()} Component 2')
                plt.colorbar(scatter, ax=axes[i])
        
        # Save plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_string = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return jsonify({
            'image': f'data:image/png;base64,{img_string}',
            'message': 'Visualization created successfully'
        })
        
    except Exception as e:
        plt.close()
        return jsonify({'error': str(e)}), 500
