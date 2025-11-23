"""
Advanced Data Preprocessing with Normalization, Standardization, and More
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import logging

logger = logging.getLogger(__name__)

class AdvancedPreprocessor:
    """Enhanced preprocessing with normalization, standardization, and more"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.transformations_applied = []
        self.scalers = {}
        self.encoders = {}
    
    def handle_missing_values(self, columns, strategy='mean'):
        """Handle missing values with multiple strategies"""
        for col in columns:
            if col not in self.df.columns:
                continue
            
            before_missing = self.df[col].isnull().sum()
            
            if pd.api.types.is_numeric_dtype(self.df[col]):
                if strategy == 'mean':
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                elif strategy == 'median':
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                elif strategy == 'mode':
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 0)
                elif strategy == 'forward_fill':
                    self.df[col] = self.df[col].fillna(method='ffill')
                elif strategy == 'backward_fill':
                    self.df[col] = self.df[col].fillna(method='bfill')
            else:
                # Categorical
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown')
            
            after_missing = self.df[col].isnull().sum()
            
            self.transformations_applied.append({
                'type': 'imputation',
                'column': col,
                'strategy': strategy,
                'filled': int(before_missing - after_missing)
            })
    
    def normalize(self, columns, method='minmax'):
        """Normalize numeric columns"""
        for col in columns:
            if col not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            
            if method == 'minmax':
                scaler = MinMaxScaler()
            elif method == 'standard':
                scaler = StandardScaler()
            elif method == 'robust':
                scaler = RobustScaler()
            else:
                continue
            
            self.df[col] = scaler.fit_transform(self.df[[col]])
            self.scalers[col] = scaler
            
            self.transformations_applied.append({
                'type': 'normalization',
                'column': col,
                'method': method
            })
    
    def standardize(self, columns):
        """Standardize numeric columns (z-score)"""
        self.normalize(columns, method='standard')
    
    def remove_outliers(self, columns, method='iqr', threshold=1.5):
        """Remove or cap outliers"""
        for col in columns:
            if col not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            
            before_count = len(self.df)
            
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                self.df[col] = self.df[col].clip(lower=lower, upper=upper)
            elif method == 'zscore':
                mean = self.df[col].mean()
                std = self.df[col].std()
                self.df[col] = self.df[col].clip(lower=mean - threshold * std, upper=mean + threshold * std)
            
            after_count = len(self.df)
            
            self.transformations_applied.append({
                'type': 'outlier_removal',
                'column': col,
                'method': method,
                'capped': int(before_count - after_count)
            })
    
    def encode_categorical(self, columns, method='onehot'):
        """Encode categorical variables"""
        for col in columns:
            if col not in self.df.columns:
                continue
            
            if method == 'label':
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le
            elif method == 'onehot':
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df.drop(col, axis=1), dummies], axis=1)
            
            self.transformations_applied.append({
                'type': 'encoding',
                'column': col,
                'method': method
            })
    
    def remove_duplicates(self):
        """Remove duplicate rows"""
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        after = len(self.df)
        
        self.transformations_applied.append({
            'type': 'duplicate_removal',
            'removed': int(before - after)
        })
    
    def log_transform(self, columns):
        """Apply log transformation"""
        for col in columns:
            if col not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            
            min_val = self.df[col].min()
            const = 1 if min_val >= 0 else abs(min_val) + 1
            self.df[col] = np.log1p(self.df[col] + const)
            
            self.transformations_applied.append({
                'type': 'log_transform',
                'column': col
            })
    
    def get_transformed_data(self):
        """Return transformed dataframe"""
        return self.df
    
    def get_transformation_summary(self):
        """Return summary of all transformations"""
        return self.transformations_applied
