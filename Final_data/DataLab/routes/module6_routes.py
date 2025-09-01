# routes/module6_routes.py - Module 6 Feature Engineering Routes
from flask import Blueprint, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import json
import io
import base64
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

# OpenAI integration
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create Blueprint
module6_bp = Blueprint('module6', __name__, url_prefix='/module6')

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
            print("⚠️ OpenAI API key not found. LLM features will be disabled.")
    
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

def make_json_serializable(obj):
    """Convert pandas/numpy objects to JSON-serializable format"""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, (pd.Int64Dtype, pd.Float64Dtype)):
        return str(obj)
    elif hasattr(obj, 'dtype') and not isinstance(obj, (pd.DataFrame, pd.Series)):
        return str(obj)
    elif not isinstance(obj, (pd.DataFrame, pd.Series)) and pd.isna(obj):
        return None
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
                    corr, _ = pearsonr(X_encoded[col], y_encoded)
                    correlations[col] = abs(corr)
                
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
        
        created_features = {}
        
        # 1. Polynomial features for numeric columns
        if len(numeric_columns) >= 2:
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            poly_features = poly.fit_transform(X[numeric_columns[:min(5, len(numeric_columns))]])
            poly_feature_names = poly.get_feature_names_out(numeric_columns[:min(5, len(numeric_columns))])
            
            # Add only interaction terms (not squares)
            for i, name in enumerate(poly_feature_names):
                if ' ' in name and '^2' not in name:  # Interaction terms only
                    created_features[f'poly_{name}'] = poly_features[:, i]
        
        # 2. Ratio features
        for i, col1 in enumerate(numeric_columns[:5]):
            for col2 in numeric_columns[i+1:6]:
                if X[col2].ne(0).all():
                    created_features[f'ratio_{col1}_{col2}'] = X[col1] / X[col2]
        
        # 3. Log transformations
        for col in numeric_columns[:5]:
            if X[col].gt(0).all():
                created_features[f'log_{col}'] = np.log(X[col])
        
        # 4. Binning features
        for col in numeric_columns[:3]:
            created_features[f'binned_{col}'] = pd.cut(X[col], bins=5, labels=False)
        
        # 5. Statistical features
        if len(numeric_columns) >= 2:
            created_features['mean_all_numeric'] = X[numeric_columns].mean(axis=1)
            created_features['std_all_numeric'] = X[numeric_columns].std(axis=1)
            created_features['max_all_numeric'] = X[numeric_columns].max(axis=1)
            created_features['min_all_numeric'] = X[numeric_columns].min(axis=1)
        
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
    
    def compare_feature_sets(self, feature_sets, model_type='auto'):
        """Compare different feature sets and their performance impact"""
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
                
                # Train model and get performance
                if model_type == 'classification':
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    scores = cross_val_score(model, X_encoded, y, cv=5, scoring='accuracy')
                    metric = 'accuracy'
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    scores = cross_val_score(model, X_encoded, y, cv=5, scoring='neg_mean_squared_error')
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
            
            # Convert DataFrame to serializable format
            optimized_data_serializable = optimized_data.to_dict('records')
            
            return {
                'optimized_dataset': optimized_data_serializable,
                'processing_log': processing_log,
                'original_shape': self.data.shape,
                'final_shape': optimized_data.shape,
                'improvement_summary': f"Optimized from {self.data.shape} to {optimized_data.shape}",
                'columns': list(optimized_data.columns)
            }
            
        except Exception as e:
            return {'error': f"Dataset optimization failed: {str(e)}"}
    
    def run_complete_analysis(self):
        """Run all feature engineering analysis automatically"""
        if self.data is None:
            raise ValueError("No data loaded")
        
        # Prepare dataset info with proper serialization
        sample_data = self.data.head().copy()
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
                'shape': self.data.shape,
                'target_column': self.target_column,
                'columns': list(self.data.columns),
                'dtypes': {col: str(dtype) for col, dtype in self.data.dtypes.to_dict().items()},
                'missing_values': {col: int(count) for col, count in self.data.isnull().sum().to_dict().items()},
                'sample_data': sample_data.to_dict('records')
            }
        }
        
        print("🚀 Starting comprehensive feature engineering analysis...")
        
        # 1. Feature Importance Analysis
        print("📊 Analyzing feature importance...")
        try:
            importance_results = self.automated_feature_importance()
            results['feature_importance'] = importance_results
            print("✅ Feature importance analysis completed")
        except Exception as e:
            results['feature_importance'] = {'error': str(e)}
            print(f"⚠️ Feature importance failed: {e}")
        
        # 2. Intelligent Feature Creation
        print("🎯 Creating intelligent features...")
        try:
            created_features = self.intelligent_feature_creation(max_features=20)
            results['created_features'] = created_features
            print(f"✅ Created {len(created_features)} intelligent features")
        except Exception as e:
            results['created_features'] = {'error': str(e)}
            print(f"⚠️ Feature creation failed: {e}")
        
        # 3. Dimensionality Reduction
        print("📈 Performing dimensionality reduction...")
        try:
            dim_results = self.dimensionality_reduction_analysis()
            results['dimensionality_reduction'] = dim_results
            print("✅ Dimensionality reduction completed")
        except Exception as e:
            results['dimensionality_reduction'] = {'error': str(e)}
            print(f"⚠️ Dimensionality reduction failed: {e}")
        
        # 4. Automatic Feature Set Comparison
        print("⚖️ Comparing different feature sets...")
        try:
            # Create automatic feature sets based on importance
            if ('feature_importance' in results and 
                results['feature_importance'] and 
                not any('error' in str(v) for v in results['feature_importance'].values())):
                # Get top features from different methods
                feature_sets = {'all_features': list(self.data.drop(columns=[self.target_column]).columns)}
                
                # Add top features from each method
                for method, scores in results['feature_importance'].items():
                    if isinstance(scores, list) and len(scores) > 0:
                        top_features = [feat for feat, score in scores[:10]]
                        feature_sets[f'top_10_{method}'] = top_features
                        feature_sets[f'top_5_{method}'] = top_features[:5]
                
                comparison_results = self.compare_feature_sets(feature_sets)
                results['feature_comparison'] = comparison_results
                print(f"✅ Compared {len(feature_sets)} feature sets")
            else:
                results['feature_comparison'] = {'error': 'No feature importance results available'}
        except Exception as e:
            results['feature_comparison'] = {'error': str(e)}
            print(f"⚠️ Feature comparison failed: {e}")
        
        # 5. Domain Template Suggestions
        print("🏭 Getting domain-specific suggestions...")
        try:
            templates = self.get_domain_templates()
            results['domain_templates'] = templates
            print("✅ Domain templates retrieved")
        except Exception as e:
            results['domain_templates'] = {'error': str(e)}
            print(f"⚠️ Domain templates failed: {e}")
        
        # 6. LLM-Powered Analysis
        print("🤖 Generating AI insights...")
        try:
            llm_insights = {}
            
            # Dataset characteristics analysis
            print("   📊 Analyzing dataset characteristics...")
            llm_insights['dataset_analysis'] = llm_analyzer.analyze_dataset_characteristics(results['dataset_info'])
            
            # Feature importance insights
            if ('feature_importance' in results and 
                results['feature_importance'] and 
                not any('error' in str(v) for v in results['feature_importance'].values())):
                print("   ⭐ Analyzing feature importance patterns...")
                llm_insights['importance_analysis'] = llm_analyzer.analyze_feature_importance(
                    results['feature_importance'], 
                    results['dataset_info']['target_column']
                )
            
            # Domain-specific suggestions
            print("   🎯 Generating domain-specific suggestions...")
            llm_insights['domain_suggestions'] = llm_analyzer.suggest_domain_features(
                results['dataset_info'], 
                results['dataset_info']['sample_data']
            )
            
            # Performance explanation
            if ('feature_comparison' in results and 
                results['feature_comparison'] and 
                not any('error' in str(v) for v in results['feature_comparison'].values())):
                print("   📈 Explaining model performance...")
                llm_insights['performance_explanation'] = llm_analyzer.explain_model_performance(
                    results['feature_comparison']
                )
            
            # Intelligent decision making
            print("   🧠 Making intelligent decisions...")
            llm_insights['intelligent_decisions'] = llm_analyzer.make_intelligent_decisions(
                results['dataset_info'], 
                results
            )
            
            # Comprehensive recommendations
            print("   💡 Generating comprehensive recommendations...")
            llm_insights['comprehensive_recommendations'] = llm_analyzer.generate_comprehensive_recommendations(results)
            
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
    return render_template('module6_simple_index.html')

@module6_bp.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Feature Engineering Module is running'})

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
        
        # Convert sample data to handle pandas dtypes
        sample_data = df.head().copy()
        for col in sample_data.columns:
            if sample_data[col].dtype == 'object':
                sample_data[col] = sample_data[col].astype(str)
            else:
                sample_data[col] = sample_data[col].astype(float, errors='ignore')
        
        return jsonify({
            'message': 'File uploaded successfully',
            'shape': shape,
            'columns': list(df.columns),
            'dtypes': dtypes_serializable,
            'detected_target': detected_target,
            'sample_data': sample_data.to_dict('records')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@module6_bp.route('/analyze', methods=['POST'])
def run_complete_analysis():
    try:
        if feature_engine.data is None:
            return jsonify({'error': 'No data uploaded yet'}), 400
        
        # Run complete automated analysis
        results = feature_engine.run_complete_analysis()
        
        # Make results JSON serializable
        serializable_results = make_json_serializable(results)
        
        return jsonify({
            'message': 'Complete analysis finished successfully',
            'results': serializable_results
        })
        
    except Exception as e:
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
