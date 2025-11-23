from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import openai
import json
import time
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class MLRecommender:
    def __init__(self, openai_api_key=None):
        if openai_api_key:
            openai.api_key = openai_api_key
            
        # Enhanced: Train 10+ models for comprehensive analysis
        self.classification_models = {
            'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Naive Bayes': GaussianNB(),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'SVM': SVC(random_state=42, kernel='rbf'),
            'AdaBoost': AdaBoostClassifier(random_state=42, n_estimators=100),
            'Neural Network': MLPClassifier(random_state=42, max_iter=500, hidden_layer_sizes=(100,))
        }

        self.regression_models = {
            'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=100),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'Lasso Regression': Lasso(random_state=42),
            'ElasticNet': ElasticNet(random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'SVR': SVR(kernel='rbf'),
            'AdaBoost': AdaBoostRegressor(random_state=42, n_estimators=100),
            'Neural Network': MLPRegressor(random_state=42, max_iter=500, hidden_layer_sizes=(100,))
        }
        
        # Three-tier tuning grids: normal, semi-deep, deep
        self.hyperparameter_grids = {
            'normal': self._get_normal_grids(),
            'semi_deep': self._get_semi_deep_grids(),
            'deep': self._get_deep_grids()
        }
    
    def _get_normal_grids(self):
        """Fast tuning - 5-15 seconds"""
        return {
            'Random Forest': {
                'classification': {'n_estimators': [100, 200], 'max_depth': [10, None]},
                'regression': {'n_estimators': [100, 200], 'max_depth': [10, None]}
            },
            'Gradient Boosting': {
                'classification': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]},
                'regression': {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]}
            },
            'SVM': {
                'classification': {'C': [1, 10], 'kernel': ['rbf']},
                'regression': {'C': [1, 10], 'kernel': ['rbf']}
            },
            'SVR': {'regression': {'C': [1, 10], 'kernel': ['rbf']}},
            'KNN': {
                'classification': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
                'regression': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
            },
            'AdaBoost': {
                'classification': {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0]},
                'regression': {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0]}
            },
            'Neural Network': {
                'classification': {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]},
                'regression': {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]}
            },
            'Logistic Regression': {
                'classification': {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}
            },
            'Decision Tree': {
                'classification': {'max_depth': [5, 10, None], 'min_samples_split': [2, 5]},
                'regression': {'max_depth': [5, 10, None], 'min_samples_split': [2, 5]}
            },
            'Linear Regression': {
                'regression': {'fit_intercept': [True, False]}
            },
            'Ridge Regression': {'regression': {'alpha': [0.1, 1.0, 10.0]}},
            'Lasso Regression': {'regression': {'alpha': [0.01, 0.1, 1.0]}},
            'ElasticNet': {'regression': {'alpha': [0.1, 1.0], 'l1_ratio': [0.3, 0.5, 0.7]}},
            'Naive Bayes': {
                'classification': {'var_smoothing': [1e-9, 1e-8, 1e-7]}
            }
        }
    
    def _get_semi_deep_grids(self):
        """Medium tuning - 30-90 seconds"""
        return {
            'Random Forest': {
                'classification': {'n_estimators': [50, 100, 200, 300], 'max_depth': [5, 10, 15, 20, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
                'regression': {'n_estimators': [50, 100, 200, 300], 'max_depth': [5, 10, 15, 20, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
            },
            'Gradient Boosting': {
                'classification': {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [3, 5, 7, 9], 'subsample': [0.8, 0.9, 1.0]},
                'regression': {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [3, 5, 7, 9], 'subsample': [0.8, 0.9, 1.0]}
            },
            'SVM': {
                'classification': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto']},
                'regression': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto']}
            },
            'SVR': {'regression': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto']}},
            'KNN': {
                'classification': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']},
                'regression': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}
            },
            'AdaBoost': {
                'classification': {'n_estimators': [50, 100, 200], 'learning_rate': [0.5, 1.0, 1.5]},
                'regression': {'n_estimators': [50, 100, 200], 'learning_rate': [0.5, 1.0, 1.5]}
            },
            'Neural Network': {
                'classification': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001, 0.01], 'activation': ['relu', 'tanh']},
                'regression': {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001, 0.01], 'activation': ['relu', 'tanh']}
            },
            'Logistic Regression': {'classification': {'C': [0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear', 'saga']}},
            'Decision Tree': {
                'classification': {'max_depth': [5, 10, 15, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
                'regression': {'max_depth': [5, 10, 15, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
            },
            'Linear Regression': {
                'regression': {'fit_intercept': [True, False], 'copy_X': [True, False]}
            },
            'Ridge Regression': {'regression': {'alpha': [0.1, 1.0, 10.0, 100.0], 'solver': ['auto', 'svd']}},
            'Lasso Regression': {'regression': {'alpha': [0.01, 0.1, 1.0, 10.0], 'max_iter': [1000, 5000]}},
            'ElasticNet': {'regression': {'alpha': [0.1, 1.0, 10.0], 'l1_ratio': [0.3, 0.5, 0.7], 'max_iter': [1000, 5000]}},
            'Naive Bayes': {
                'classification': {'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]}
            }
        }
    
    def _get_deep_grids(self):
        """Comprehensive tuning - 90-300 seconds"""
        return {
            'Random Forest': {
                'classification': {'n_estimators': [50, 100, 200, 300, 500], 'max_depth': [5, 10, 15, 20, 30, None], 'min_samples_split': [2, 5, 10, 15], 'min_samples_leaf': [1, 2, 4, 6], 'max_features': ['sqrt', 'log2', None]},
                'regression': {'n_estimators': [50, 100, 200, 300, 500], 'max_depth': [5, 10, 15, 20, 30, None], 'min_samples_split': [2, 5, 10, 15], 'min_samples_leaf': [1, 2, 4, 6], 'max_features': ['sqrt', 'log2', None]}
            },
            'Gradient Boosting': {
                'classification': {'n_estimators': [50, 100, 200, 300, 500], 'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2], 'max_depth': [3, 5, 7, 9, 11], 'subsample': [0.7, 0.8, 0.9, 1.0], 'min_samples_split': [2, 5, 10]},
                'regression': {'n_estimators': [50, 100, 200, 300, 500], 'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2], 'max_depth': [3, 5, 7, 9, 11], 'subsample': [0.7, 0.8, 0.9, 1.0], 'min_samples_split': [2, 5, 10]}
            },
            'SVM': {
                'classification': {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['rbf', 'linear', 'poly'], 'gamma': ['scale', 'auto', 0.001, 0.01]},
                'regression': {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['rbf', 'linear', 'poly'], 'gamma': ['scale', 'auto', 0.001, 0.01]}
            },
            'SVR': {'regression': {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['rbf', 'linear', 'poly'], 'gamma': ['scale', 'auto', 0.001, 0.01], 'epsilon': [0.01, 0.1, 0.2]}},
            'KNN': {
                'classification': {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 20], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan', 'minkowski'], 'p': [1, 2, 3]},
                'regression': {'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 20], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan', 'minkowski'], 'p': [1, 2, 3]}
            },
            'AdaBoost': {
                'classification': {'n_estimators': [50, 100, 200, 300, 500], 'learning_rate': [0.01, 0.1, 0.5, 1.0, 1.5, 2.0]},
                'regression': {'n_estimators': [50, 100, 200, 300, 500], 'learning_rate': [0.01, 0.1, 0.5, 1.0, 1.5, 2.0]}
            },
            'Neural Network': {
                'classification': {'hidden_layer_sizes': [(50,), (100,), (150,), (200,), (50, 50), (100, 50), (100, 100)], 'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1], 'activation': ['relu', 'tanh', 'logistic'], 'learning_rate': ['constant', 'adaptive'], 'solver': ['adam', 'sgd']},
                'regression': {'hidden_layer_sizes': [(50,), (100,), (150,), (200,), (50, 50), (100, 50), (100, 100)], 'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1], 'activation': ['relu', 'tanh'], 'learning_rate': ['constant', 'adaptive'], 'solver': ['adam', 'sgd']}
            },
            'Logistic Regression': {'classification': {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'solver': ['lbfgs', 'liblinear', 'saga'], 'max_iter': [500, 1000, 2000], 'penalty': ['l2']}},
            'Decision Tree': {
                'classification': {'max_depth': [3, 5, 10, 15, 20, 30, None], 'min_samples_split': [2, 5, 10, 15, 20], 'min_samples_leaf': [1, 2, 4, 6, 8], 'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random']},
                'regression': {'max_depth': [3, 5, 10, 15, 20, 30, None], 'min_samples_split': [2, 5, 10, 15, 20], 'min_samples_leaf': [1, 2, 4, 6, 8], 'criterion': ['squared_error', 'absolute_error'], 'splitter': ['best', 'random']}
            },
            'Linear Regression': {
                'regression': {'fit_intercept': [True, False], 'copy_X': [True, False], 'positive': [True, False]}
            },
            'Ridge Regression': {'regression': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], 'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']}},
            'Lasso Regression': {'regression': {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0], 'max_iter': [1000, 5000, 10000], 'selection': ['cyclic', 'random']}},
            'ElasticNet': {'regression': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0], 'l1_ratio': [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9], 'max_iter': [1000, 5000, 10000]}},
            'Naive Bayes': {
                'classification': {'var_smoothing': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]}
            },
            'Linear Regression': {
                'regression': {'fit_intercept': [True, False], 'copy_X': [True, False], 'positive': [True, False]}
            }
        }
    
    def analyze_dataset(self, df):
        """Comprehensive dataset analysis"""
        target_col = df.columns[-1]
        feature_cols = df.columns[:-1]
        
        # Determine task type - Fix for binary classification
        unique_targets = df[target_col].nunique()
        # Binary classification: exactly 2 unique values
        # Multi-class: 3-20 unique values for small datasets
        is_classification = (unique_targets == 2) or (unique_targets <= 20 and unique_targets > 2 and df[target_col].dtype in ['object', 'int64', 'bool'])
        
        # Feature analysis
        numerical_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Data quality metrics
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        
        # Feature correlations for numerical features
        correlations = {}
        if len(numerical_features) > 1:
            corr_matrix = df[numerical_features].corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            correlations['high_correlation_pairs'] = high_corr_pairs
        
        analysis = {
            'n_samples': int(len(df)),
            'n_features': int(len(feature_cols)),
            'missing_values': int(df.isnull().sum().sum()),
            'missing_percentage': float(missing_percentage),
            'categorical_features': len(categorical_features),
            'numerical_features': len(numerical_features),
            'task_type': 'classification' if is_classification else 'regression',
            'target_classes': int(unique_targets) if is_classification else None,
            'class_balance': {str(k): int(v) for k, v in df[target_col].value_counts().to_dict().items()} if is_classification else None,
            'feature_names': feature_cols.tolist(),
            'target_name': target_col,
            'data_size_category': 'small' if len(df) < 1000 else 'medium' if len(df) < 10000 else 'large',
            'dimensionality': 'low' if len(feature_cols) < 10 else 'medium' if len(feature_cols) < 50 else 'high',
            'correlations': correlations
        }
        return analysis
    
    def preprocess_data(self, df, analysis):
        """Advanced preprocessing with feature engineering"""
        df_processed = df.copy()
        
        # Handle categorical variables
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != df_processed.columns[-1]:  # Don't encode target if it's categorical
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        
        # Handle target encoding for classification
        if analysis['task_type'] == 'classification' and df_processed.iloc[:, -1].dtype == 'object':
            le_target = LabelEncoder()
            df_processed.iloc[:, -1] = le_target.fit_transform(df_processed.iloc[:, -1])
        
        # Handle missing values intelligently
        for col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                if df_processed[col].dtype in ['object']:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
                else:
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
        
        X = df_processed.iloc[:, :-1]
        y = df_processed.iloc[:, -1]
        
        # Feature selection for high-dimensional data
        if analysis['dimensionality'] == 'high':
            k_features = min(20, X.shape[1])
            if analysis['task_type'] == 'classification':
                selector = SelectKBest(f_classif, k=k_features)
            else:
                selector = SelectKBest(f_regression, k=k_features)
            X = selector.fit_transform(X, y)
        
        # Robust scaling for better performance
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y, scaler
    
    def evaluate_models(self, X, y, task_type):
        """Enhanced: Comprehensive model evaluation with 5-fold CV"""
        models = self.classification_models if task_type == 'classification' else self.regression_models
        results = {}

        print(f"[INFO] Evaluating {len(models)} models with 5-fold cross-validation...")

        # Split data for detailed evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if task_type == 'classification' else None)

        for name, model in models.items():
            try:
                print(f"  [TRAINING] {name}...")
                start_time = time.time()

                # Use cv=5 for robust evaluation
                if task_type == 'classification':
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)

                    # Fit model for additional metrics
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Calculate additional metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                    additional_metrics = {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1)
                    }

                else:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
                    
                    # Fit model for additional metrics
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate additional metrics
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    
                    additional_metrics = {
                        'r2_score': float(r2),
                        'mse': float(mse),
                        'mae': float(mae),
                        'rmse': float(rmse)
                    }
                
                training_time = time.time() - start_time
                
                results[name] = {
                    'mean_score': float(cv_scores.mean()),
                    'std_score': float(cv_scores.std()),
                    'cv_scores': [float(s) for s in cv_scores],
                    'training_time': float(training_time),
                    'additional_metrics': additional_metrics
                }
                
            except Exception as e:
                results[name] = {
                    'mean_score': 0.0,
                    'std_score': 0.0,
                    'cv_scores': [0.0] * 5,
                    'training_time': 0.0,
                    'error': str(e),
                    'additional_metrics': {}
                }
        
        return results
    
    def get_gpt_analysis(self, model_name, dataset_analysis, performance_results):
        """Get intelligent analysis from GPT-3.5-turbo"""
        if not hasattr(openai, 'api_key') or not openai.api_key or openai.api_key == "sk-your-openai-api-key-here":
            return None
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai.api_key)
            
            prompt = f"""As an ML expert, analyze {model_name} for this dataset in 2-3 sentences:

Dataset: {dataset_analysis['n_samples']} samples, {dataset_analysis['n_features']} features, {dataset_analysis['task_type']}
Performance: {performance_results['mean_score']:.3f} CV score, {performance_results['training_time']:.2f}s training

Explain why this model works well/poorly for this data."""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"GPT Analysis Error: {e}")
            return None
    
    def tune_hyperparameters(self, model_name, X, y, task_type, tuning_level='normal'):
        """Perform hyperparameter tuning using GridSearchCV"""
        models = self.classification_models if task_type == 'classification' else self.regression_models
        
        if model_name not in models:
            return {'error': 'Model not found'}
        
        model = models[model_name]
        param_grid = self.hyperparameter_grids.get(tuning_level, {}).get(model_name, {}).get(task_type, {})
        
        if not param_grid:
            return {'error': 'No hyperparameter grid defined for this model'}
        
        try:
            start_time = time.time()
            
            # Use appropriate scoring metric
            scoring = 'accuracy' if task_type == 'classification' else 'r2'
            
            # Perform grid search with reduced CV for speed
            grid_search = GridSearchCV(
                model, param_grid, cv=3, scoring=scoring, n_jobs=-1, verbose=0
            )
            
            grid_search.fit(X, y)
            
            tuning_time = time.time() - start_time
            
            # Get best parameters and score
            best_params = grid_search.best_params_
            best_score = float(grid_search.best_score_)
            
            # Convert numpy types to Python types
            for key, value in best_params.items():
                if isinstance(value, (np.integer, np.int64)):
                    best_params[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    best_params[key] = float(value)
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'tuning_time': float(tuning_time),
                'total_combinations': len(grid_search.cv_results_['params'])
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_suitability_score(self, model_name, dataset_analysis, performance):
        """Advanced suitability scoring with comprehensive factors"""
        base_score = performance['mean_score'] * 100
        
        # Initialize adjustment factors
        adjustments = {
            'performance_stability': 0,
            'data_size_fit': 0,
            'dimensionality_fit': 0,
            'training_efficiency': 0,
            'interpretability': 0
        }
        
        # Performance stability (lower std is better)
        if performance['std_score'] < 0.05:
            adjustments['performance_stability'] = 5
        elif performance['std_score'] > 0.15:
            adjustments['performance_stability'] = -5
        
        # Data size appropriateness
        data_size = dataset_analysis['data_size_category']
        if model_name in ['Random Forest', 'Gradient Boosting', 'Neural Network', 'AdaBoost']:
            if data_size == 'large':
                adjustments['data_size_fit'] = 5
            elif data_size == 'small':
                adjustments['data_size_fit'] = -3
        elif model_name in ['Naive Bayes', 'KNN']:
            if data_size == 'small':
                adjustments['data_size_fit'] = 3
            elif data_size == 'large':
                adjustments['data_size_fit'] = -5
        elif model_name in ['SVM', 'SVR']:
            if data_size == 'small' or data_size == 'medium':
                adjustments['data_size_fit'] = 2
            elif data_size == 'large':
                adjustments['data_size_fit'] = -7
        
        # Dimensionality appropriateness
        dimensionality = dataset_analysis['dimensionality']
        if model_name in ['SVM', 'SVR']:
            if dimensionality == 'high':
                adjustments['dimensionality_fit'] = -8
            else:
                adjustments['dimensionality_fit'] = 3
        elif model_name in ['Random Forest', 'Gradient Boosting', 'AdaBoost']:
            if dimensionality == 'high':
                adjustments['dimensionality_fit'] = 5
        elif model_name in ['Lasso Regression', 'ElasticNet']:
            if dimensionality == 'high':
                adjustments['dimensionality_fit'] = 4
        
        # Training efficiency (faster is better for deployment)
        training_time = performance.get('training_time', 1)
        if training_time < 1:
            adjustments['training_efficiency'] = 3
        elif training_time > 10:
            adjustments['training_efficiency'] = -3
        
        # Interpretability bonus
        interpretable_models = ['Decision Tree', 'Linear Regression', 'Logistic Regression', 'Naive Bayes', 'Ridge Regression', 'Lasso Regression']
        if model_name in interpretable_models:
            adjustments['interpretability'] = 2
        
        # Calculate final score
        total_adjustment = sum(adjustments.values())
        final_score = max(0, min(100, base_score + total_adjustment))
        
        # Create detailed justification
        justification_parts = [f"Base CV Score: {base_score:.1f}%"]
        for factor, value in adjustments.items():
            if value != 0:
                justification_parts.append(f"{factor.replace('_', ' ').title()}: {value:+d}")
        justification_parts.append(f"Final Suitability: {final_score:.1f}%")
        
        return final_score, " | ".join(justification_parts)
    
    def evaluate_tuned_model(self, model_name, X, y, task_type, best_params):
        """Re-evaluate model with optimized parameters"""
        models = self.classification_models if task_type == 'classification' else self.regression_models
        
        if model_name not in models:
            return {'error': 'Model not found'}
        
        try:
            # Create model with optimized parameters
            model = models[model_name]
            model.set_params(**best_params)
            
            # Split data for evaluation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if task_type == 'classification' else None)
            
            start_time = time.time()
            
            # Cross-validation scores
            if task_type == 'classification':
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                # Fit model for additional metrics
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate additional metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                additional_metrics = {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1)
                }
                
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                # Fit model for additional metrics
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate additional metrics
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                
                additional_metrics = {
                    'r2_score': float(r2),
                    'mse': float(mse),
                    'mae': float(mae),
                    'rmse': float(rmse)
                }
            
            training_time = time.time() - start_time
            
            return {
                'mean_score': float(cv_scores.mean()),
                'std_score': float(cv_scores.std()),
                'cv_scores': [float(s) for s in cv_scores],
                'training_time': float(training_time),
                'additional_metrics': additional_metrics
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def generate_notebook(self, model_name, model_params, dataset_info):
        """Generate Jupyter notebook with model implementation"""
        
        # Map model names to sklearn imports
        model_imports = {
            'Random Forest': 'from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor',
            'Gradient Boosting': 'from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor',
            'AdaBoost': 'from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor',
            'Logistic Regression': 'from sklearn.linear_model import LogisticRegression',
            'Linear Regression': 'from sklearn.linear_model import LinearRegression',
            'SVM': 'from sklearn.svm import SVC, SVR',
            'SVR': 'from sklearn.svm import SVR',
            'Decision Tree': 'from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor',
            'KNN': 'from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor',
            'Naive Bayes': 'from sklearn.naive_bayes import GaussianNB',
            'Neural Network': 'from sklearn.neural_network import MLPClassifier, MLPRegressor',
            'Ridge Regression': 'from sklearn.linear_model import Ridge',
            'Lasso Regression': 'from sklearn.linear_model import Lasso',
            'ElasticNet': 'from sklearn.linear_model import ElasticNet'
        }
        
        task_type = dataset_info.get('task_type', 'classification')
        
        notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# {model_name} Model Implementation\n",
                        f"\n",
                        f"This notebook contains the optimized {model_name} model for your {task_type} task.\n",
                        f"\n",
                        f"## Dataset Information\n",
                        f"- Samples: {dataset_info.get('n_samples', 'N/A')}\n",
                        f"- Features: {dataset_info.get('n_features', 'N/A')}\n",
                        f"- Task Type: {task_type.title()}\n"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Import required libraries\n",
                        "import pandas as pd\n",
                        "import numpy as np\n",
                        f"{model_imports.get(model_name, 'from sklearn.ensemble import RandomForestClassifier')}\n",
                        "from sklearn.model_selection import train_test_split, cross_val_score\n",
                        "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
                        "from sklearn.metrics import classification_report, confusion_matrix\n" if task_type == 'classification' else "from sklearn.metrics import mean_squared_error, r2_score\n",
                        "import matplotlib.pyplot as plt\n",
                        "import seaborn as sns\n"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Load your dataset\n",
                        "# Replace 'your_dataset.csv' with your actual file path\n",
                        "df = pd.read_csv('your_dataset.csv')\n",
                        "print(f'Dataset shape: {df.shape}')\n",
                        "df.head()"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Data preprocessing\n",
                        "# Handle categorical variables\n",
                        "for col in df.select_dtypes(include=['object']).columns:\n",
                        "    if col != df.columns[-1]:  # Don't encode target if it's categorical\n",
                        "        le = LabelEncoder()\n",
                        "        df[col] = le.fit_transform(df[col].astype(str))\n",
                        "\n",
                        "# Handle missing values\n",
                        "df = df.fillna(df.median())\n",
                        "\n",
                        "# Separate features and target\n",
                        "X = df.iloc[:, :-1]\n",
                        "y = df.iloc[:, -1]\n",
                        "\n",
                        "# Scale features\n",
                        "scaler = StandardScaler()\n",
                        "X_scaled = scaler.fit_transform(X)\n",
                        "\n",
                        "print(f'Features shape: {X_scaled.shape}')\n",
                        "print(f'Target shape: {y.shape}')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Split the data\n",
                        "X_train, X_test, y_train, y_test = train_test_split(\n",
                        "    X_scaled, y, test_size=0.2, random_state=42" + (", stratify=y" if task_type == 'classification' else "") + "\n",
                        ")\n",
                        "\n",
                        "print(f'Training set: {X_train.shape[0]} samples')\n",
                        "print(f'Test set: {X_test.shape[0]} samples')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        f"# Initialize optimized {model_name} model\n",
                        f"model = {self._get_model_constructor(model_name, task_type)}(\n",
                        *[f"    {k}={repr(v)},\n" for k, v in model_params.items()],
                        "    random_state=42\n" if 'random_state' not in model_params else "",
                        ")\n",
                        "\n",
                        "# Train the model\n",
                        "model.fit(X_train, y_train)\n",
                        "\n",
                        "# Make predictions\n",
                        "y_pred = model.predict(X_test)\n",
                        "\n",
                        "print('Model trained successfully!')"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Evaluate the model\n",
                        "cv_scores = cross_val_score(model, X_train, y_train, cv=5)\n",
                        "\n",
                        "print(f'Cross-validation scores: {cv_scores}')\n",
                        "print(f'Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')\n",
                        "\n"
                    ] + (
                        [
                            "# Classification metrics\n",
                            "from sklearn.metrics import accuracy_score, classification_report\n",
                            "\n",
                            "accuracy = accuracy_score(y_test, y_pred)\n",
                            "print(f'Test Accuracy: {accuracy:.4f}')\n",
                            "\n",
                            "print('\\nClassification Report:')\n",
                            "print(classification_report(y_test, y_pred))"
                        ] if task_type == 'classification' else [
                            "# Regression metrics\n",
                            "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n",
                            "\n",
                            "mse = mean_squared_error(y_test, y_pred)\n",
                            "mae = mean_absolute_error(y_test, y_pred)\n",
                            "r2 = r2_score(y_test, y_pred)\n",
                            "\n",
                            "print(f'Mean Squared Error: {mse:.4f}')\n",
                            "print(f'Mean Absolute Error: {mae:.4f}')\n",
                            "print(f'R² Score: {r2:.4f}')"
                        ]
                    )
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Visualize results\n",
                        "plt.figure(figsize=(12, 4))\n",
                        "\n"
                    ] + (
                        [
                            "# Confusion Matrix\n",
                            "plt.subplot(1, 2, 1)\n",
                            "cm = confusion_matrix(y_test, y_pred)\n",
                            "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\n",
                            "plt.title('Confusion Matrix')\n",
                            "plt.ylabel('True Label')\n",
                            "plt.xlabel('Predicted Label')\n",
                            "\n",
                            "# Feature Importance (if available)\n",
                            "plt.subplot(1, 2, 2)\n",
                            "if hasattr(model, 'feature_importances_'):\n",
                            "    importance = model.feature_importances_\n",
                            "    indices = np.argsort(importance)[::-1][:10]\n",
                            "    plt.bar(range(len(indices)), importance[indices])\n",
                            "    plt.title('Top 10 Feature Importances')\n",
                            "    plt.xlabel('Feature Index')\n",
                            "    plt.ylabel('Importance')\n",
                            "else:\n",
                            "    plt.text(0.5, 0.5, 'Feature importance\\nnot available', ha='center', va='center')\n",
                            "    plt.title('Feature Importance')\n",
                            "\n",
                            "plt.tight_layout()\n",
                            "plt.show()"
                        ] if task_type == 'classification' else [
                            "# Actual vs Predicted\n",
                            "plt.subplot(1, 2, 1)\n",
                            "plt.scatter(y_test, y_pred, alpha=0.6)\n",
                            "plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\n",
                            "plt.xlabel('Actual Values')\n",
                            "plt.ylabel('Predicted Values')\n",
                            "plt.title('Actual vs Predicted')\n",
                            "\n",
                            "# Residuals\n",
                            "plt.subplot(1, 2, 2)\n",
                            "residuals = y_test - y_pred\n",
                            "plt.scatter(y_pred, residuals, alpha=0.6)\n",
                            "plt.axhline(y=0, color='r', linestyle='--')\n",
                            "plt.xlabel('Predicted Values')\n",
                            "plt.ylabel('Residuals')\n",
                            "plt.title('Residual Plot')\n",
                            "\n",
                            "plt.tight_layout()\n",
                            "plt.show()"
                        ]
                    )
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Next Steps\n",
                        "\n",
                        "You can now:\n",
                        "1. Experiment with different parameters\n",
                        "2. Try feature engineering\n",
                        "3. Implement cross-validation strategies\n",
                        "4. Deploy the model for production use\n",
                        "\n",
                        "Feel free to modify and experiment with this code!"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        return notebook
    
    def _get_model_constructor(self, model_name, task_type):
        """Get the appropriate model constructor string"""
        constructors = {
            'Random Forest': 'RandomForestClassifier' if task_type == 'classification' else 'RandomForestRegressor',
            'Gradient Boosting': 'GradientBoostingClassifier' if task_type == 'classification' else 'GradientBoostingRegressor',
            'AdaBoost': 'AdaBoostClassifier' if task_type == 'classification' else 'AdaBoostRegressor',
            'Logistic Regression': 'LogisticRegression',
            'Linear Regression': 'LinearRegression',
            'SVM': 'SVC' if task_type == 'classification' else 'SVR',
            'SVR': 'SVR',
            'Decision Tree': 'DecisionTreeClassifier' if task_type == 'classification' else 'DecisionTreeRegressor',
            'KNN': 'KNeighborsClassifier' if task_type == 'classification' else 'KNeighborsRegressor',
            'Naive Bayes': 'GaussianNB',
            'Neural Network': 'MLPClassifier' if task_type == 'classification' else 'MLPRegressor',
            'Ridge Regression': 'Ridge',
            'Lasso Regression': 'Lasso',
            'ElasticNet': 'ElasticNet'
        }
        return constructors.get(model_name, 'RandomForestClassifier')

# Initialize with OpenAI API key - you'll need to set this
OPENAI_API_KEY = "sk-your-openai-api-key-here"  # Replace with your actual API key
recommender = MLRecommender(openai_api_key=OPENAI_API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['dataset']
        df = pd.read_csv(file)
        
        # Analyze dataset
        analysis = recommender.analyze_dataset(df)
        
        # Preprocess data
        X, y, scaler = recommender.preprocess_data(df.copy(), analysis)
        
        # Evaluate models
        results = recommender.evaluate_models(X, y, analysis['task_type'])
        
        # Calculate suitability scores and get GPT analysis
        recommendations = []
        for model_name, performance in results.items():
            if 'error' not in performance:
                suitability_score, justification = recommender.calculate_suitability_score(
                    model_name, analysis, performance
                )
                
                # Get GPT analysis
                gpt_analysis = recommender.get_gpt_analysis(model_name, analysis, performance)
                
                recommendations.append({
                    'model': model_name,
                    'performance': performance,
                    'suitability_score': float(suitability_score),
                    'justification': justification,
                    'gpt_analysis': gpt_analysis,
                    'can_tune': model_name in recommender.hyperparameter_grids
                })
        
        # Sort by suitability score
        recommendations.sort(key=lambda x: x['suitability_score'], reverse=True)
        
        # Cache data for tuning
        tune_hyperparameters.cached_data = (X, y, analysis['task_type'])
        
        # Add best model details
        best_model = recommendations[0] if recommendations else None
        
        return jsonify({
            'dataset_analysis': analysis,
            'recommendations': recommendations,
            'best_model': {
                'name': best_model['model'],
                'score': best_model['suitability_score'],
                'performance': best_model['performance'],
                'why_best': f"Achieved highest suitability score of {best_model['suitability_score']:.1f}% with {best_model['performance']['mean_score']:.3f} CV score and {best_model['performance']['training_time']:.2f}s training time."
            } if best_model else None
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/tune', methods=['POST'])
def tune_hyperparameters():
    try:
        data = request.get_json()
        model_name = data['model_name']
        
        # Get cached data from session (in production, use proper caching)
        if not hasattr(tune_hyperparameters, 'cached_data'):
            return jsonify({'error': 'No dataset cached. Please re-analyze your dataset first.'}), 400
        
        X, y, task_type = tune_hyperparameters.cached_data
        
        # Perform real hyperparameter tuning
        tuning_results = recommender.tune_hyperparameters(model_name, X, y, task_type)
        
        if 'error' in tuning_results:
            return jsonify({'error': tuning_results['error']}), 400
        
        # Re-evaluate model with optimized parameters
        updated_performance = recommender.evaluate_tuned_model(model_name, X, y, task_type, tuning_results['best_params'])
        
        return jsonify({
            'message': 'Hyperparameter tuning completed',
            'model_name': model_name,
            'tuning_results': tuning_results,
            'updated_performance': updated_performance
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/export-notebook', methods=['POST'])
def export_notebook():
    try:
        data = request.get_json()
        model_name = data['model_name']
        model_params = data.get('model_params', {})
        dataset_info = data.get('dataset_info', {})
        
        # Generate notebook content
        notebook_content = recommender.generate_notebook(model_name, model_params, dataset_info)
        
        return jsonify({
            'notebook_content': notebook_content,
            'filename': f'{model_name.lower().replace(" ", "_")}_model.ipynb'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/jupyterlite')
def jupyterlite():
    return render_template('jupyterlite.html')

if __name__ == '__main__':
    app.run(debug=True)