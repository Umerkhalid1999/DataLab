"""
INTELLIGENT AI-POWERED PREPROCESSING SYSTEM
Crystal-clear transparency with detailed explanations for every decision
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

class IntelligentPreprocessor:
    """AI-powered preprocessing with complete transparency"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.original_df = df.copy()
        self.analysis_report = {}
        self.transformation_log = []
        self.ai_insights = {}
        
    def analyze_with_ai(self):
        """Deep analysis using AI to understand data quality"""
        # 1. Detect all issues
        issues = self._detect_all_issues()
        
        # 2. Get AI recommendations for each issue
        for issue_type, details in issues.items():
            ai_explanation = self._get_ai_explanation(issue_type, details)
            self.ai_insights[issue_type] = ai_explanation
        
        self.analysis_report = {
            'issues_detected': issues,
            'ai_insights': self.ai_insights,
            'dataset_shape': self.df.shape,
            'columns': list(self.df.columns)
        }
        
        return self.analysis_report
    
    def _detect_all_issues(self):
        """Comprehensive issue detection"""
        issues = {}
        
        # 1. Missing Values Analysis
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            missing_details = {}
            for col in missing[missing > 0].index:
                missing_details[col] = {
                    'count': int(missing[col]),
                    'percentage': round((missing[col] / len(self.df)) * 100, 2),
                    'dtype': str(self.df[col].dtype)
                }
            issues['missing_values'] = {
                'total_missing': int(missing.sum()),
                'affected_columns': missing_details,
                'severity': 'HIGH' if missing.sum() > len(self.df) * 0.1 else 'MEDIUM'
            }
        
        # 2. Duplicate Rows Analysis
        duplicates = self.df.duplicated()
        if duplicates.sum() > 0:
            issues['duplicates'] = {
                'count': int(duplicates.sum()),
                'percentage': round((duplicates.sum() / len(self.df)) * 100, 2),
                'severity': 'HIGH' if duplicates.sum() > len(self.df) * 0.05 else 'MEDIUM'
            }
        
        # 3. Outliers Analysis (per column)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_details = {}
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR)))
            if outliers.sum() > 0:
                outlier_details[col] = {
                    'count': int(outliers.sum()),
                    'percentage': round((outliers.sum() / len(self.df)) * 100, 2),
                    'min_outlier': float(self.df[col][outliers].min()),
                    'max_outlier': float(self.df[col][outliers].max()),
                    'normal_range': f"[{Q1 - 1.5 * IQR:.2f}, {Q3 + 1.5 * IQR:.2f}]"
                }
        
        if outlier_details:
            issues['outliers'] = {
                'total_outliers': sum(d['count'] for d in outlier_details.values()),
                'affected_columns': outlier_details,
                'severity': 'MEDIUM'
            }
        
        # 4. Data Type Issues
        dtype_issues = {}
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Check if numeric values stored as strings
                try:
                    pd.to_numeric(self.df[col].dropna(), errors='raise')
                    dtype_issues[col] = {
                        'issue': 'Numeric data stored as text',
                        'recommendation': 'Convert to numeric type'
                    }
                except:
                    pass
        
        if dtype_issues:
            issues['dtype_issues'] = {
                'affected_columns': dtype_issues,
                'severity': 'LOW'
            }
        
        # 5. Scale/Normalization Check
        scale_issues = {}
        for col in numeric_cols:
            mean = self.df[col].mean()
            std = self.df[col].std()
            if std > 0:
                cv = std / abs(mean) if mean != 0 else float('inf')
                if cv > 1.0 or self.df[col].max() > 1000:
                    scale_issues[col] = {
                        'mean': float(mean),
                        'std': float(std),
                        'min': float(self.df[col].min()),
                        'max': float(self.df[col].max()),
                        'cv': float(cv) if cv != float('inf') else 'inf'
                    }
        
        if scale_issues:
            issues['scale_issues'] = {
                'affected_columns': scale_issues,
                'severity': 'LOW',
                'recommendation': 'Normalization recommended for ML models'
            }
        
        return issues
    
    def _get_ai_explanation(self, issue_type, details):
        """Get AI explanation for specific issue"""
        try:
            client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
            
            prompt = self._build_ai_prompt(issue_type, details)
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data science expert explaining preprocessing decisions to professors. Be technical, concise, and justify every recommendation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"AI explanation error: {e}")
            return self._get_fallback_explanation(issue_type, details)
    
    def _build_ai_prompt(self, issue_type, details):
        """Build specific prompt for each issue type"""
        if issue_type == 'missing_values':
            return f"""Explain why missing values are problematic and recommend solution:
- Total missing: {details['total_missing']}
- Affected columns: {len(details['affected_columns'])}
- Severity: {details['severity']}

Explain: 1) WHY this is a problem, 2) WHAT solution to apply, 3) HOW it improves quality."""

        elif issue_type == 'duplicates':
            return f"""Explain why duplicate rows are problematic:
- Duplicate count: {details['count']} ({details['percentage']}%)
- Severity: {details['severity']}

Explain: 1) WHY duplicates harm analysis, 2) WHAT to do, 3) IMPACT on results."""

        elif issue_type == 'outliers':
            return f"""Explain outlier handling strategy:
- Total outliers: {details['total_outliers']}
- Affected columns: {len(details['affected_columns'])}

Explain: 1) WHY outliers matter, 2) Capping vs removal, 3) WHEN to keep outliers."""

        elif issue_type == 'scale_issues':
            return f"""Explain why feature scaling is needed:
- Columns with different scales: {len(details['affected_columns'])}

Explain: 1) WHY scaling matters for ML, 2) Min-Max vs Standardization, 3) WHEN to apply."""

        else:
            return f"Explain the {issue_type} issue and recommend solution."
    
    def _get_fallback_explanation(self, issue_type, details):
        """Fallback explanations if AI unavailable"""
        explanations = {
            'missing_values': f"Missing values detected in {len(details.get('affected_columns', {}))} columns. Missing data can bias statistical analyses and cause ML model errors. Imputation strategy: median for numeric, mode for categorical.",
            'duplicates': f"Found {details.get('count', 0)} duplicate rows. Duplicates artificially inflate sample size and can skew statistical measures. Recommendation: Remove duplicates to ensure data integrity.",
            'outliers': f"Detected {details.get('total_outliers', 0)} outliers. Extreme values can distort statistical measures and ML model training. Strategy: Cap outliers using IQR method to preserve distribution while removing extremes.",
            'scale_issues': f"Features have different scales. Unscaled features can dominate distance-based algorithms. Recommendation: Apply Min-Max normalization to scale all features to [0,1] range."
        }
        return explanations.get(issue_type, "Issue detected. Applying standard preprocessing.")
    
    def apply_intelligent_preprocessing(self):
        """Apply transformations with detailed logging"""
        if not self.analysis_report:
            self.analyze_with_ai()
        
        issues = self.analysis_report['issues_detected']
        
        # Apply transformations in order of severity
        if 'missing_values' in issues:
            self._handle_missing_values(issues['missing_values'])
        
        if 'duplicates' in issues:
            self._remove_duplicates(issues['duplicates'])
        
        if 'outliers' in issues:
            self._handle_outliers(issues['outliers'])
        
        if 'dtype_issues' in issues:
            self._fix_dtypes(issues['dtype_issues'])
        
        if 'scale_issues' in issues:
            self._normalize_features(issues['scale_issues'])
        
        return self.df
    
    def _handle_missing_values(self, details):
        """Handle missing values with detailed logging"""
        for col, info in details['affected_columns'].items():
            before = self.df[col].isnull().sum()
            
            if pd.api.types.is_numeric_dtype(self.df[col]):
                strategy = 'median'
                fill_value = self.df[col].median()
                self.df[col] = self.df[col].fillna(fill_value)
            else:
                strategy = 'mode'
                fill_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                self.df[col] = self.df[col].fillna(fill_value)
            
            after = self.df[col].isnull().sum()
            
            self.transformation_log.append({
                'step': len(self.transformation_log) + 1,
                'operation': 'Missing Value Imputation',
                'column': col,
                'strategy': strategy,
                'fill_value': str(fill_value),
                'before': int(before),
                'after': int(after),
                'filled': int(before - after),
                'reason': f"Filled {before} missing values ({info['percentage']}%) using {strategy}"
            })
    
    def _remove_duplicates(self, details):
        """Remove duplicates with logging"""
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)
        
        self.transformation_log.append({
            'step': len(self.transformation_log) + 1,
            'operation': 'Duplicate Removal',
            'before_rows': before,
            'after_rows': after,
            'removed': before - after,
            'reason': f"Removed {before - after} duplicate rows ({details['percentage']}%)"
        })
    
    def _handle_outliers(self, details):
        """Handle outliers with detailed logging"""
        for col, info in details['affected_columns'].items():
            before_outliers = info['count']
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            self.df[col] = self.df[col].clip(lower=lower, upper=upper)
            
            self.transformation_log.append({
                'step': len(self.transformation_log) + 1,
                'operation': 'Outlier Capping',
                'column': col,
                'method': 'IQR (1.5x)',
                'capped_values': before_outliers,
                'lower_bound': float(lower),
                'upper_bound': float(upper),
                'reason': f"Capped {before_outliers} outliers ({info['percentage']}%) to range [{lower:.2f}, {upper:.2f}]"
            })
    
    def _fix_dtypes(self, details):
        """Fix data type issues"""
        for col, info in details['affected_columns'].items():
            try:
                self.df[col] = pd.to_numeric(self.df[col])
                self.transformation_log.append({
                    'step': len(self.transformation_log) + 1,
                    'operation': 'Data Type Conversion',
                    'column': col,
                    'from_type': 'object',
                    'to_type': 'numeric',
                    'reason': info['recommendation']
                })
            except:
                pass
    
    def _normalize_features(self, details):
        """Normalize features with logging"""
        cols = list(details['affected_columns'].keys())
        scaler = MinMaxScaler()
        
        before_stats = {col: {'min': float(self.df[col].min()), 'max': float(self.df[col].max())} 
                       for col in cols}
        
        self.df[cols] = scaler.fit_transform(self.df[cols])
        
        after_stats = {col: {'min': float(self.df[col].min()), 'max': float(self.df[col].max())} 
                      for col in cols}
        
        self.transformation_log.append({
            'step': len(self.transformation_log) + 1,
            'operation': 'Feature Normalization',
            'method': 'Min-Max Scaling',
            'columns': cols,
            'before_range': before_stats,
            'after_range': after_stats,
            'reason': f"Scaled {len(cols)} features to [0,1] range for ML compatibility"
        })
    
    def get_comprehensive_report(self):
        """Generate complete transparency report"""
        return {
            'analysis': self.analysis_report,
            'transformations': self.transformation_log,
            'ai_insights': self.ai_insights,
            'summary': {
                'total_issues_found': len(self.analysis_report.get('issues_detected', {})),
                'total_transformations': len(self.transformation_log),
                'original_shape': self.original_df.shape,
                'final_shape': self.df.shape,
                'data_quality_improved': len(self.transformation_log) > 0
            }
        }
