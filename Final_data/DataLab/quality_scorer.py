"""
Robust Data Quality Scoring System
Based on ISO 25012 Data Quality Dimensions
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple

class DataQualityScorer:
    """
    Industry-standard data quality assessment based on ISO 25012
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.total_cells = df.shape[0] * df.shape[1]
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.text_cols = df.select_dtypes(include=['object']).columns
        
    def calculate_completeness(self) -> Tuple[float, Dict]:
        """Completeness: Degree to which data is not missing (STRICT)"""
        missing = self.df.isnull().sum().sum()
        missing_pct = (missing / self.total_cells * 100) if self.total_cells > 0 else 0
        
        # Strict penalty: 2x multiplier for missing data
        penalty = missing_pct * 2
        score = max(0, 100 - penalty)
        
        return score, {
            'missing_count': int(missing),
            'missing_percentage': round(missing_pct, 2),
            'complete_cells': int(self.total_cells - missing)
        }
    
    def calculate_uniqueness(self) -> Tuple[float, Dict]:
        """Uniqueness: Degree to which data is free from duplicates (STRICT)"""
        duplicates = self.df.duplicated().sum()
        dup_pct = (duplicates / len(self.df) * 100) if len(self.df) > 0 else 0
        
        # Strict penalty: 1.5x multiplier for duplicates
        penalty = dup_pct * 1.5
        score = max(0, 100 - penalty)
        
        return score, {
            'duplicate_count': int(duplicates),
            'duplicate_percentage': round(dup_pct, 2),
            'unique_rows': int(len(self.df) - duplicates)
        }
    
    def calculate_consistency(self) -> Tuple[float, Dict]:
        """Consistency: Degree to which data follows patterns and constraints (STRICT)"""
        # Stricter outlier detection: 1.2 IQR instead of 1.5
        outlier_count = 0
        for col in self.numeric_cols:
            if len(self.df[col].dropna()) > 0:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((self.df[col] < (Q1 - 1.2 * IQR)) | (self.df[col] > (Q3 + 1.2 * IQR))).sum()
                outlier_count += outliers
        
        # Check for infinite values (critical issue)
        inf_count = sum(np.isinf(self.df[col]).sum() for col in self.numeric_cols)
        
        # Check for mixed data types
        mixed_types = 0
        for col in self.text_cols:
            unique_types = self.df[col].dropna().apply(type).nunique()
            if unique_types > 1:
                mixed_types += 1
        
        # Check for high variance (unstable data)
        high_variance = 0
        for col in self.numeric_cols:
            if len(self.df[col].dropna()) > 0:
                cv = self.df[col].std() / self.df[col].mean() if self.df[col].mean() != 0 else 0
                if cv > 1.0:  # Coefficient of variation > 100%
                    high_variance += 1
        
        total_numeric_cells = len(self.numeric_cols) * len(self.df) if len(self.numeric_cols) > 0 else 1
        issues = outlier_count + (inf_count * 3) + (mixed_types * 5) + (high_variance * 2)
        
        # Strict penalty: 1.5x multiplier
        penalty = (issues / total_numeric_cells * 100) * 1.5
        score = max(0, 100 - penalty)
        
        details = {
            'outlier_count': int(outlier_count),
            'infinite_count': int(inf_count),
            'mixed_type_columns': int(mixed_types),
            'high_variance_columns': int(high_variance),
            'total_issues': int(issues)
        }
        
        return score, details
    
    def calculate_validity(self) -> Tuple[float, Dict]:
        """Validity: Degree to which data conforms to defined formats (STRICT)"""
        # Check for empty strings (critical)
        empty_strings = sum((self.df[col] == '').sum() + (self.df[col].str.strip() == '').sum() 
                           for col in self.text_cols if col in self.df.columns)
        
        # Check for negative values in suspicious columns
        negative_issues = 0
        for col in self.numeric_cols:
            if any(keyword in col.lower() for keyword in ['age', 'count', 'quantity', 'price', 'amount', 'size']):
                negative_issues += (self.df[col] < 0).sum()
        
        # Check for data type mismatches
        dtype_issues = 0
        for col in self.text_cols:
            try:
                pd.to_numeric(self.df[col].dropna(), errors='raise')
                dtype_issues += 1
            except:
                pass
        
        # Check for special characters in numeric-looking text
        special_char_issues = 0
        for col in self.text_cols:
            special_char_issues += self.df[col].str.contains(r'[^a-zA-Z0-9\s]', na=False).sum()
        
        # Weighted issues (some are more critical)
        issues = (empty_strings * 2) + (negative_issues * 3) + (dtype_issues * 2) + special_char_issues
        
        # Strict penalty: 2x multiplier
        penalty = (issues / self.total_cells * 100) * 2 if self.total_cells > 0 else 0
        score = max(0, 100 - penalty)
        
        details = {
            'empty_strings': int(empty_strings),
            'negative_value_issues': int(negative_issues),
            'dtype_mismatches': int(dtype_issues),
            'special_char_issues': int(special_char_issues),
            'total_issues': int(issues)
        }
        
        return score, details
    
    def calculate_accuracy(self) -> Tuple[float, Dict]:
        """Accuracy: Degree to which data correctly represents real-world values (STRICT)"""
        # Stricter: Values beyond 2.5 standard deviations (98.8% rule)
        extreme_values = 0
        for col in self.numeric_cols:
            if len(self.df[col].dropna()) > 0:
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std > 0:
                    extreme = ((self.df[col] - mean).abs() > 2.5 * std).sum()
                    extreme_values += extreme
        
        # Check for placeholder values (expanded list)
        placeholder_count = 0
        placeholders = [0, 999, -999, 9999, -9999, 99999, -99999, 1111, 2222, 3333]
        for col in self.numeric_cols:
            placeholder_count += self.df[col].isin(placeholders).sum()
        
        # Check for repeated values (suspicious patterns)
        repeated_values = 0
        for col in self.numeric_cols:
            if len(self.df[col].dropna()) > 0:
                most_common_pct = (self.df[col].value_counts().iloc[0] / len(self.df)) * 100
                if most_common_pct > 50:  # More than 50% same value
                    repeated_values += 1
        
        # Check for zero values in suspicious columns
        zero_issues = 0
        for col in self.numeric_cols:
            if any(keyword in col.lower() for keyword in ['price', 'amount', 'salary', 'income']):
                zero_issues += (self.df[col] == 0).sum()
        
        # Weighted issues
        issues = (extreme_values * 1.5) + (placeholder_count * 2) + (repeated_values * 3) + zero_issues
        
        total_numeric_cells = len(self.numeric_cols) * len(self.df) if len(self.numeric_cols) > 0 else 1
        
        # Strict penalty: 2x multiplier
        penalty = (issues / total_numeric_cells * 100) * 2
        score = max(0, 100 - penalty)
        
        details = {
            'extreme_values': int(extreme_values),
            'placeholder_values': int(placeholder_count),
            'repeated_value_columns': int(repeated_values),
            'zero_value_issues': int(zero_issues),
            'total_issues': int(issues)
        }
        
        return score, details
    
    def calculate_overall_score(self) -> Dict:
        """Calculate weighted overall quality score"""
        
        # Calculate individual dimension scores
        completeness_score, completeness_details = self.calculate_completeness()
        uniqueness_score, uniqueness_details = self.calculate_uniqueness()
        consistency_score, consistency_details = self.calculate_consistency()
        validity_score, validity_details = self.calculate_validity()
        accuracy_score, accuracy_details = self.calculate_accuracy()
        
        # Stricter weights - emphasize critical dimensions
        weights = {
            'completeness': 0.35,    # Increased - most critical
            'uniqueness': 0.20,
            'consistency': 0.20,
            'validity': 0.15,
            'accuracy': 0.10         # Decreased - less weight
        }
        
        overall_score = (
            completeness_score * weights['completeness'] +
            uniqueness_score * weights['uniqueness'] +
            consistency_score * weights['consistency'] +
            validity_score * weights['validity'] +
            accuracy_score * weights['accuracy']
        )
        
        return {
            'overall_score': round(overall_score, 1),
            'dimensions': {
                'completeness': {
                    'score': round(completeness_score, 1),
                    'weight': weights['completeness'],
                    'details': completeness_details
                },
                'uniqueness': {
                    'score': round(uniqueness_score, 1),
                    'weight': weights['uniqueness'],
                    'details': uniqueness_details
                },
                'consistency': {
                    'score': round(consistency_score, 1),
                    'weight': weights['consistency'],
                    'details': consistency_details
                },
                'validity': {
                    'score': round(validity_score, 1),
                    'weight': weights['validity'],
                    'details': validity_details
                },
                'accuracy': {
                    'score': round(accuracy_score, 1),
                    'weight': weights['accuracy'],
                    'details': accuracy_details
                }
            }
        }

def calculate_robust_quality_score(df: pd.DataFrame) -> Dict:
    """
    Main function to calculate robust quality score
    """
    scorer = DataQualityScorer(df)
    return scorer.calculate_overall_score()
