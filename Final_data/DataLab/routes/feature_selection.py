# routes/feature_selection.py - Advanced Feature Selection Techniques
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr, spearmanr
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureSelector:
    """Advanced feature selection using modern techniques"""
    
    def __init__(self, task_type='auto'):
        self.task_type = task_type
        self.selected_features = []
        self.feature_scores = {}
        
    def _determine_task_type(self, y):
        """Auto-detect classification vs regression"""
        if self.task_type != 'auto':
            return self.task_type
        
        if y.dtype == 'object' or y.nunique() < 10:
            return 'classification'
        return 'regression'
    
    def forward_selection(self, X, y, max_features=None, cv=10, scoring=None, progress_callback=None):
        """
        Forward Selection: Start with no features, add one at a time
        """
        task_type = self._determine_task_type(y)
        
        if max_features is None:
            max_features = X.shape[1]
        
        # Choose model and scoring
        if task_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            scoring = scoring or 'accuracy'
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            scoring = scoring or 'neg_mean_squared_error'
        
        selected = []
        remaining = list(X.columns)
        best_score = -np.inf
        scores_history = []
        cv_details = []
        
        print(f" Forward Selection: Starting with 0 features...")
        print(f"   Using {cv}-fold cross-validation with {scoring} metric")
        
        for i in range(min(max_features, len(remaining))):
            best_feature = None
            best_new_score = -np.inf
            best_cv_scores = None
            
            for feature in remaining:
                current_features = selected + [feature]
                X_subset = X[current_features]
                
                # Cross-validation score with detailed fold results
                fold_scores = cross_val_score(model, X_subset, y, cv=cv, scoring=scoring, n_jobs=-1)
                mean_score = np.mean(fold_scores)
                
                if mean_score > best_new_score:
                    best_new_score = mean_score
                    best_feature = feature
                    best_cv_scores = fold_scores
            
            # Check if adding feature improves performance
            if best_new_score > best_score:
                selected.append(best_feature)
                remaining.remove(best_feature)
                best_score = best_new_score
                
                # Store detailed CV results
                cv_detail = {
                    'iteration': i + 1,
                    'feature_added': best_feature,
                    'n_features': len(selected),
                    'cv_scores': [float(s) for s in best_cv_scores],
                    'mean_score': float(best_score),
                    'std_score': float(np.std(best_cv_scores))
                }
                cv_details.append(cv_detail)
                scores_history.append({'features': len(selected), 'score': best_score})
                
                # Print detailed progress
                print(f"   Added '{best_feature}' | Features: {len(selected)} | CV Mean: {best_score:.4f} (+/- {np.std(best_cv_scores):.4f})")
                print(f"      Fold scores: {[f'{s:.4f}' for s in best_cv_scores]}")
            else:
                print(f"   No improvement. Stopping at {len(selected)} features.")
                break
        
        return {
            'selected_features': selected,
            'final_score': best_score,
            'scores_history': scores_history,
            'cv_details': cv_details,
            'n_features': len(selected),
            'cv_folds': cv,
            'scoring_metric': scoring
        }
    
    def stepwise_selection(self, X, y, max_features=None, cv=10, scoring=None, threshold=0.01, progress_callback=None):
        """
        Stepwise Selection: Combination of forward and backward selection
        Alternates between adding best features and removing worst features
        """
        task_type = self._determine_task_type(y)
        
        if max_features is None:
            max_features = X.shape[1]
        
        # Choose model and scoring
        if task_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            scoring = scoring or 'accuracy'
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            scoring = scoring or 'neg_mean_squared_error'
        
        selected = []
        remaining = list(X.columns)
        best_score = -np.inf
        scores_history = []
        cv_details = []
        iteration = 0
        
        print(f" Stepwise Selection: Starting with 0 features...")
        print(f"   Using {cv}-fold cross-validation with {scoring} metric")
        
        max_iterations = 8
        
        while len(selected) < max_features and len(remaining) > 0 and iteration < max_iterations:
            iteration += 1
            
            # FORWARD STEP: Add best feature
            best_feature = None
            best_new_score = -np.inf
            best_cv_scores = None
            
            for feature in remaining:
                current_features = selected + [feature]
                X_subset = X[current_features]
                fold_scores = cross_val_score(model, X_subset, y, cv=cv, scoring=scoring, n_jobs=-1)
                mean_score = np.mean(fold_scores)
                
                if mean_score > best_new_score:
                    best_new_score = mean_score
                    best_feature = feature
                    best_cv_scores = fold_scores
            
            # Check if adding improves
            if best_new_score > best_score:
                selected.append(best_feature)
                remaining.remove(best_feature)
                best_score = best_new_score
                
                cv_detail = {
                    'iteration': iteration,
                    'feature_added': best_feature,
                    'n_features': len(selected),
                    'cv_scores': [float(s) for s in best_cv_scores],
                    'mean_score': float(best_score),
                    'std_score': float(np.std(best_cv_scores))
                }
                cv_details.append(cv_detail)
                scores_history.append({'features': len(selected), 'score': best_score})
                print(f"   [FORWARD] Added '{best_feature}' | Features: {len(selected)} | CV: {best_score:.4f} (±{np.std(best_cv_scores):.4f})")
            else:
                print(f"   [FORWARD] No improvement, stopping.")
                break
            
            # BACKWARD STEP: Only every 2 iterations for speed
            if len(selected) > 2 and iteration % 2 == 0:
                worst_feature = None
                best_after_removal = -np.inf
                best_cv_scores_removal = None
                
                for feature in selected:
                    current_features = [f for f in selected if f != feature]
                    X_subset = X[current_features]
                    fold_scores = cross_val_score(model, X_subset, y, cv=cv, scoring=scoring, n_jobs=-1)
                    mean_score = np.mean(fold_scores)
                    
                    if mean_score > best_after_removal:
                        best_after_removal = mean_score
                        worst_feature = feature
                        best_cv_scores_removal = fold_scores
                
                # Remove if it maintains/improves performance
                if best_after_removal >= best_score - threshold:
                    selected.remove(worst_feature)
                    remaining.append(worst_feature)
                    best_score = best_after_removal
                    
                    cv_detail = {
                        'iteration': iteration,
                        'feature_removed': worst_feature,
                        'n_features': len(selected),
                        'cv_scores': [float(s) for s in best_cv_scores_removal],
                        'mean_score': float(best_score),
                        'std_score': float(np.std(best_cv_scores_removal))
                    }
                    cv_details.append(cv_detail)
                    scores_history.append({'features': len(selected), 'score': best_score})
                    print(f"   [BACKWARD] Removed '{worst_feature}' | Features: {len(selected)} | CV: {best_score:.4f} (±{np.std(best_cv_scores_removal):.4f})")
        
        return {
            'selected_features': selected,
            'final_score': best_score,
            'scores_history': scores_history,
            'cv_details': cv_details,
            'n_features': len(selected),
            'cv_folds': cv,
            'scoring_metric': scoring
        }
    
    def backward_elimination(self, X, y, cv=10, scoring=None, threshold=0.01, progress_callback=None):
        """
        Backward Elimination: Start with all features, remove least important
        """
        task_type = self._determine_task_type(y)
        
        # Choose model and scoring
        if task_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            scoring = scoring or 'accuracy'
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            scoring = scoring or 'neg_mean_squared_error'
        
        selected = list(X.columns)
        cv_details = []
        
        # Get baseline score with all features
        fold_scores = cross_val_score(model, X[selected], y, cv=cv, scoring=scoring, n_jobs=-1)
        best_score = np.mean(fold_scores)
        scores_history = [{'features': len(selected), 'score': best_score}]
        
        print(f" Backward Elimination: Starting with {len(selected)} features")
        print(f"   Using {cv}-fold cross-validation with {scoring} metric")
        print(f"   Baseline CV Mean: {best_score:.4f} (+/- {np.std(fold_scores):.4f})")
        print(f"   Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
        
        iteration = 0
        while len(selected) > 1:
            worst_feature = None
            best_new_score = -np.inf
            best_cv_scores = None
            
            for feature in selected:
                current_features = [f for f in selected if f != feature]
                X_subset = X[current_features]
                
                # Cross-validation score
                fold_scores = cross_val_score(model, X_subset, y, cv=cv, scoring=scoring, n_jobs=-1)
                mean_score = np.mean(fold_scores)
                
                if mean_score > best_new_score:
                    best_new_score = mean_score
                    worst_feature = feature
                    best_cv_scores = fold_scores
            
            # Check if removing feature maintains or improves performance
            if best_new_score >= best_score - threshold:
                selected.remove(worst_feature)
                iteration += 1
                
                # Store detailed CV results
                cv_detail = {
                    'iteration': iteration,
                    'feature_removed': worst_feature,
                    'n_features': len(selected),
                    'cv_scores': [float(s) for s in best_cv_scores],
                    'mean_score': float(best_new_score),
                    'std_score': float(np.std(best_cv_scores))
                }
                cv_details.append(cv_detail)
                
                best_score = best_new_score
                scores_history.append({'features': len(selected), 'score': best_score})
                
                print(f"   Removed '{worst_feature}' | Features: {len(selected)} | CV Mean: {best_score:.4f} (+/- {np.std(best_cv_scores):.4f})")
                print(f"      Fold scores: {[f'{s:.4f}' for s in best_cv_scores]}")
            else:
                print(f"   Performance degraded. Stopping at {len(selected)} features.")
                break
        
        return {
            'selected_features': selected,
            'final_score': best_score,
            'scores_history': scores_history,
            'cv_details': cv_details,
            'n_features': len(selected),
            'cv_folds': cv,
            'scoring_metric': scoring
        }
    
    def feature_importance_selection(self, X, y, methods=['random_forest', 'linear', 'tree'], top_n=None, threshold=None, cv=20, scoring=None):
        """
        Feature Importance from Simple Models with CV validation
        """
        task_type = self._determine_task_type(y)
        importance_scores = {}
        cv_details = []
        
        print(f" Calculating feature importance using {len(methods)} methods...")
        
        # Choose scoring metric
        if task_type == 'classification':
            scoring = scoring or 'accuracy'
        else:
            scoring = scoring or 'neg_mean_squared_error'
        
        # Random Forest Importance
        if 'random_forest' in methods:
            if task_type == 'classification':
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            rf.fit(X, y)
            importance_scores['random_forest'] = dict(zip(X.columns, rf.feature_importances_))
            print(f"   Random Forest importance calculated")
        
        # Linear Model Coefficients
        if 'linear' in methods:
            if task_type == 'classification':
                model = LogisticRegression(max_iter=1000, random_state=42)
            else:
                model = LinearRegression()
            
            model.fit(X, y)
            if task_type == 'classification':
                coef = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
            else:
                coef = np.abs(model.coef_)
            importance_scores['linear'] = dict(zip(X.columns, coef))
            print(f"   Linear model coefficients calculated")
        
        # Decision Tree Importance
        if 'tree' in methods:
            if task_type == 'classification':
                tree = DecisionTreeClassifier(random_state=42)
            else:
                tree = DecisionTreeRegressor(random_state=42)
            
            tree.fit(X, y)
            importance_scores['tree'] = dict(zip(X.columns, tree.feature_importances_))
            print(f"   Decision tree importance calculated")
        
        # Ridge/Lasso for regularization-based importance
        if 'ridge' in methods or 'lasso' in methods:
            if task_type == 'regression':
                if 'ridge' in methods:
                    ridge = Ridge(alpha=1.0, random_state=42)
                    ridge.fit(X, y)
                    importance_scores['ridge'] = dict(zip(X.columns, np.abs(ridge.coef_)))
                    print(f"   Ridge coefficients calculated")
                
                if 'lasso' in methods:
                    lasso = Lasso(alpha=0.1, random_state=42, max_iter=1000)
                    lasso.fit(X, y)
                    importance_scores['lasso'] = dict(zip(X.columns, np.abs(lasso.coef_)))
                    print(f"   Lasso coefficients calculated")
        
        # Aggregate scores across methods
        aggregated_scores = {}
        for feature in X.columns:
            scores = [importance_scores[method][feature] for method in importance_scores.keys()]
            aggregated_scores[feature] = np.mean(scores)
        
        # Sort by importance
        sorted_features = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select features based on top_n or threshold
        if top_n:
            selected = [f for f, s in sorted_features[:top_n]]
        elif threshold:
            selected = [f for f, s in sorted_features if s >= threshold]
        else:
            # Default: keep features with above-median importance
            median_score = np.median(list(aggregated_scores.values()))
            selected = [f for f, s in sorted_features if s >= median_score]
        
        # Validate selected features with CV
        if task_type == 'classification':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        fold_scores = cross_val_score(model, X[selected], y, cv=cv, scoring=scoring, n_jobs=-1)
        cv_detail = {
            'iteration': 1,
            'feature_added': f'Top {len(selected)} by importance',
            'n_features': len(selected),
            'cv_scores': [float(s) for s in fold_scores],
            'mean_score': float(np.mean(fold_scores)),
            'std_score': float(np.std(fold_scores))
        }
        cv_details.append(cv_detail)
        print(f"   Selected {len(selected)} features | CV Mean: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
        
        return {
            'selected_features': selected,
            'importance_scores': importance_scores,
            'aggregated_scores': aggregated_scores,
            'sorted_features': sorted_features,
            'n_features': len(selected),
            'cv_details': cv_details,
            'cv_folds': cv,
            'scoring_metric': scoring
        }
    
    def correlation_analysis(self, X, y, target_threshold=0.05, multicollinearity_threshold=0.9, method='pearson', cv=20, scoring=None):
        """
        Correlation Analysis for feature selection with CV validation
        - Remove features with low correlation to target
        - Remove highly correlated features (multicollinearity)
        """
        task_type = self._determine_task_type(y)
        cv_details = []
        
        # Choose scoring metric
        if task_type == 'classification':
            scoring = scoring or 'accuracy'
        else:
            scoring = scoring or 'neg_mean_squared_error'
        
        print(f" Performing correlation analysis...")
        
        # 1. Target correlation
        target_correlations = {}
        for col in X.columns:
            try:
                if method == 'pearson':
                    corr, _ = pearsonr(X[col].fillna(0), y)
                else:  # spearman
                    corr, _ = spearmanr(X[col].fillna(0), y)
                target_correlations[col] = abs(corr) if not np.isnan(corr) else 0
            except:
                target_correlations[col] = 0
        
        # Filter by target correlation
        features_by_target = [f for f, corr in target_correlations.items() if corr >= target_threshold]
        print(f"   {len(features_by_target)}/{len(X.columns)} features pass target correlation threshold (|r| >= {target_threshold})")
        
        # 2. Multicollinearity check
        if len(features_by_target) > 1:
            X_subset = X[features_by_target]
            corr_matrix = X_subset.corr().abs()
            
            # Find highly correlated pairs
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = set()
            
            for column in upper_triangle.columns:
                correlated_features = upper_triangle[column][upper_triangle[column] > multicollinearity_threshold].index.tolist()
                if correlated_features:
                    # Keep the feature with higher target correlation
                    for corr_feature in correlated_features:
                        if target_correlations[column] >= target_correlations[corr_feature]:
                            to_drop.add(corr_feature)
                        else:
                            to_drop.add(column)
            
            final_features = [f for f in features_by_target if f not in to_drop]
            print(f"   Removed {len(to_drop)} features due to multicollinearity (r > {multicollinearity_threshold})")
        else:
            final_features = features_by_target
            to_drop = set()
        
        # Validate selected features with CV
        if len(final_features) > 0:
            if task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            fold_scores = cross_val_score(model, X[final_features], y, cv=cv, scoring=scoring, n_jobs=-1)
            cv_detail = {
                'iteration': 1,
                'feature_added': f'{len(final_features)} correlated features',
                'n_features': len(final_features),
                'cv_scores': [float(s) for s in fold_scores],
                'mean_score': float(np.mean(fold_scores)),
                'std_score': float(np.std(fold_scores))
            }
            cv_details.append(cv_detail)
            print(f"   Selected {len(final_features)} features | CV Mean: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
        
        return {
            'selected_features': final_features,
            'target_correlations': target_correlations,
            'removed_by_target': [f for f in X.columns if f not in features_by_target],
            'removed_by_multicollinearity': list(to_drop),
            'n_features': len(final_features),
            'cv_details': cv_details,
            'cv_folds': cv,
            'scoring_metric': scoring
        }
    
    def variance_threshold_selection(self, X, y, threshold=0.01, quasi_constant_threshold=0.95, cv=20, scoring=None):
        """
        Variance Threshold: Remove low-variance and quasi-constant features with CV validation
        """
        task_type = self._determine_task_type(y)
        cv_details = []
        
        # Choose scoring metric
        if task_type == 'classification':
            scoring = scoring or 'accuracy'
        else:
            scoring = scoring or 'neg_mean_squared_error'
        
        print(f" Applying variance threshold analysis...")
        
        # Calculate variance for each feature
        variances = X.var()
        
        # Remove near-zero variance features
        low_variance_features = variances[variances < threshold].index.tolist()
        
        # Remove quasi-constant features (95% same value)
        quasi_constant_features = []
        for col in X.columns:
            if col not in low_variance_features:
                value_counts = X[col].value_counts(normalize=True)
                if len(value_counts) > 0 and value_counts.iloc[0] >= quasi_constant_threshold:
                    quasi_constant_features.append(col)
        
        # Combine removals
        features_to_remove = set(low_variance_features + quasi_constant_features)
        selected_features = [f for f in X.columns if f not in features_to_remove]
        
        print(f"   Removed {len(low_variance_features)} low-variance features (var < {threshold})")
        print(f"   Removed {len(quasi_constant_features)} quasi-constant features ({quasi_constant_threshold*100}% same value)")
        
        # Validate selected features with CV
        if len(selected_features) > 0:
            if task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            fold_scores = cross_val_score(model, X[selected_features], y, cv=cv, scoring=scoring, n_jobs=-1)
            cv_detail = {
                'iteration': 1,
                'feature_added': f'{len(selected_features)} high-variance features',
                'n_features': len(selected_features),
                'cv_scores': [float(s) for s in fold_scores],
                'mean_score': float(np.mean(fold_scores)),
                'std_score': float(np.std(fold_scores))
            }
            cv_details.append(cv_detail)
            print(f"   Selected {len(selected_features)} features | CV Mean: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
        
        return {
            'selected_features': selected_features,
            'low_variance_features': low_variance_features,
            'quasi_constant_features': quasi_constant_features,
            'variances': variances.to_dict(),
            'n_features': len(selected_features),
            'cv_details': cv_details,
            'cv_folds': cv,
            'scoring_metric': scoring
        }
    
    def univariate_statistical_tests(self, X, y, significance_level=0.05, cv=20, scoring=None):
        """
        Univariate Statistical Tests for feature selection with CV validation
        """
        task_type = self._determine_task_type(y)
        cv_details = []
        
        # Choose scoring metric
        if task_type == 'classification':
            scoring = scoring or 'accuracy'
        else:
            scoring = scoring or 'neg_mean_squared_error'
        
        print(f"Performing univariate statistical tests...")
        
        from sklearn.feature_selection import chi2, f_classif, f_regression, mutual_info_classif, mutual_info_regression
        from scipy.stats import chi2_contingency
        
        selected_features = []
        p_values = {}
        scores = {}
        
        if task_type == 'classification':
            # For classification: use chi2 for categorical, f_classif for continuous
            try:
                # F-test (ANOVA)
                f_scores, p_vals = f_classif(X, y)
                for i, col in enumerate(X.columns):
                    p_values[col] = p_vals[i]
                    scores[col] = f_scores[i]
                    if p_vals[i] < significance_level:
                        selected_features.append(col)
                print(f"  F-test: {len(selected_features)} features with p < {significance_level}")
            except Exception as e:
                print(f"  F-test failed: {e}")
        else:
            # For regression: use f_regression
            try:
                f_scores, p_vals = f_regression(X, y)
                for i, col in enumerate(X.columns):
                    p_values[col] = p_vals[i]
                    scores[col] = f_scores[i]
                    if p_vals[i] < significance_level:
                        selected_features.append(col)
                print(f"  F-test: {len(selected_features)} features with p < {significance_level}")
            except Exception as e:
                print(f"  F-test failed: {e}")
        
        # Validate selected features with CV
        if len(selected_features) > 0:
            if task_type == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            fold_scores = cross_val_score(model, X[selected_features], y, cv=cv, scoring=scoring, n_jobs=-1)
            cv_detail = {
                'iteration': 1,
                'feature_added': f'{len(selected_features)} significant features (p<{significance_level})',
                'n_features': len(selected_features),
                'cv_scores': [float(s) for s in fold_scores],
                'mean_score': float(np.mean(fold_scores)),
                'std_score': float(np.std(fold_scores))
            }
            cv_details.append(cv_detail)
            print(f"  Selected {len(selected_features)} features | CV Mean: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
        
        return {
            'selected_features': selected_features,
            'p_values': p_values,
            'scores': scores,
            'n_features': len(selected_features),
            'significance_level': significance_level,
            'cv_details': cv_details,
            'cv_folds': cv,
            'scoring_metric': scoring
        }
    
    def vif_analysis(self, X, threshold=10):
        """
        Variance Inflation Factor (VIF) for multicollinearity detection
        """
        if not STATSMODELS_AVAILABLE:
            return {
                'error': 'statsmodels not installed. Install with: pip install statsmodels',
                'selected_features': list(X.columns),
                'n_features': len(X.columns)
            }
        
        print(f" Calculating VIF for multicollinearity...")
        
        vif_data = []
        features = list(X.columns)
        
        for i, feature in enumerate(features):
            try:
                vif = variance_inflation_factor(X.values, i)
                vif_data.append({'feature': feature, 'vif': vif})
            except:
                vif_data.append({'feature': feature, 'vif': np.inf})
        
        vif_df = pd.DataFrame(vif_data)
        
        # Remove features with high VIF
        high_vif_features = vif_df[vif_df['vif'] > threshold]['feature'].tolist()
        selected_features = [f for f in features if f not in high_vif_features]
        
        print(f"   {len(high_vif_features)} features have VIF > {threshold} (multicollinearity)")
        
        return {
            'selected_features': selected_features,
            'vif_scores': vif_df.to_dict('records'),
            'high_vif_features': high_vif_features,
            'n_features': len(selected_features)
        }
