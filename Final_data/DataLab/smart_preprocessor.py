"""
AI-Powered Smart Preprocessing System
Only applies transformations that are actually needed
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import logging
import os
from openai import OpenAI

logger = logging.getLogger(__name__)

class SmartPreprocessor:
    """Intelligent preprocessing that only fixes actual issues"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.issues = {}
        self.transformations = []
        self.ai_recommendations = ""
        
    def analyze_issues(self):
        """Detect actual data quality issues"""
        issues = {}
        
        # 1. Check missing values
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            issues['missing_values'] = {
                'columns': missing[missing > 0].to_dict(),
                'total': int(missing.sum()),
                'percentage': round((missing.sum() / (self.df.shape[0] * self.df.shape[1])) * 100, 2)
            }
        
        # 2. Check duplicates
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            issues['duplicates'] = {
                'count': int(duplicates),
                'percentage': round((duplicates / len(self.df)) * 100, 2)
            }
        
        # 3. Check outliers
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_cols = {}
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df[col] < (Q1 - 1.5 * IQR)) | (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                outlier_cols[col] = int(outliers)
        
        if outlier_cols:
            issues['outliers'] = {
                'columns': outlier_cols,
                'total': sum(outlier_cols.values())
            }
        
        # 4. Check if normalization needed (high variance)
        needs_normalization = []
        for col in numeric_cols:
            if self.df[col].std() > 0:
                cv = self.df[col].std() / abs(self.df[col].mean()) if self.df[col].mean() != 0 else 0
                if cv > 1.0:  # High coefficient of variation
                    needs_normalization.append(col)
        
        if needs_normalization:
            issues['high_variance'] = {
                'columns': needs_normalization,
                'reason': 'Features have different scales'
            }
        
        self.issues = issues
        return issues
    
    def get_ai_recommendations(self):
        """Use GPT-3.5 to recommend preprocessing steps"""
        if not self.issues:
            return "No data quality issues detected. Dataset is clean."
        
        try:
            client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
            
            prompt = f"""Analyze this dataset and recommend ONLY necessary preprocessing steps:

Dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns

Issues Found:
{self._format_issues_for_ai()}

For each issue, explain:
1. WHY it's a problem
2. WHAT transformation to apply
3. HOW it improves data quality

Be concise and technical. Focus only on issues that actually exist."""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a data science expert. Recommend only necessary preprocessing steps."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.7
            )
            
            self.ai_recommendations = response.choices[0].message.content
            return self.ai_recommendations
            
        except Exception as e:
            logger.error(f"AI recommendation error: {e}")
            return f"AI analysis unavailable. Detected issues: {', '.join(self.issues.keys())}"
    
    def _format_issues_for_ai(self):
        """Format issues for AI prompt"""
        formatted = []
        for issue_type, details in self.issues.items():
            if issue_type == 'missing_values':
                formatted.append(f"- Missing values: {details['total']} ({details['percentage']}%)")
            elif issue_type == 'duplicates':
                formatted.append(f"- Duplicates: {details['count']} rows ({details['percentage']}%)")
            elif issue_type == 'outliers':
                formatted.append(f"- Outliers: {details['total']} values in {len(details['columns'])} columns")
            elif issue_type == 'high_variance':
                formatted.append(f"- High variance: {len(details['columns'])} columns need scaling")
        return '\n'.join(formatted)
    
    def apply_smart_cleaning(self):
        """Apply only necessary transformations"""
        if not self.issues:
            return self.df
        
        # 1. Handle missing values (only if they exist)
        if 'missing_values' in self.issues:
            for col in self.issues['missing_values']['columns'].keys():
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown')
            
            self.transformations.append({
                'type': 'imputation',
                'reason': f"Filled {self.issues['missing_values']['total']} missing values",
                'columns': list(self.issues['missing_values']['columns'].keys())
            })
        
        # 2. Remove duplicates (only if they exist)
        if 'duplicates' in self.issues:
            before = len(self.df)
            self.df = self.df.drop_duplicates()
            removed = before - len(self.df)
            
            self.transformations.append({
                'type': 'duplicate_removal',
                'reason': f"Removed {removed} duplicate rows",
                'count': removed
            })
        
        # 3. Handle outliers (only if they exist)
        if 'outliers' in self.issues:
            for col in self.issues['outliers']['columns'].keys():
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                self.df[col] = self.df[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
            
            self.transformations.append({
                'type': 'outlier_capping',
                'reason': f"Capped {self.issues['outliers']['total']} outliers",
                'columns': list(self.issues['outliers']['columns'].keys())
            })
        
        # 4. Normalize (only if high variance detected)
        if 'high_variance' in self.issues:
            scaler = MinMaxScaler()
            cols = self.issues['high_variance']['columns']
            self.df[cols] = scaler.fit_transform(self.df[cols])
            
            self.transformations.append({
                'type': 'normalization',
                'reason': 'Scaled features with high variance to [0,1]',
                'columns': cols
            })
        
        return self.df
    
    def get_summary(self):
        """Get preprocessing summary"""
        return {
            'issues_found': self.issues,
            'transformations_applied': self.transformations,
            'ai_explanation': self.ai_recommendations
        }
