import pandas as pd
import numpy as np
import logging
from scipy import stats

def apply_capping(df):
    """Apply capping method to handle outliers using IQR method"""
    try:
        df_capped = df.copy()
        numeric_columns = df_capped.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df_capped[col].notna().sum() > 0:
                Q1 = df_capped[col].quantile(0.25)
                Q3 = df_capped[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers
                df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
        
        logging.info(f"Applied capping to {len(numeric_columns)} numerical columns")
        return df_capped
        
    except Exception as e:
        logging.error(f"Error applying capping: {e}")
        return df

def apply_transformation(df):
    """Apply log transformation to handle outliers"""
    try:
        df_transformed = df.copy()
        numeric_columns = df_transformed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df_transformed[col].notna().sum() > 0:
                # Check if column has positive values for log transformation
                if df_transformed[col].min() > 0:
                    # Apply log transformation
                    df_transformed[col] = np.log1p(df_transformed[col])
                else:
                    # Apply square root transformation for non-positive values
                    df_transformed[col] = np.sqrt(df_transformed[col] - df_transformed[col].min() + 1)
        
        logging.info(f"Applied transformation to {len(numeric_columns)} numerical columns")
        return df_transformed
        
    except Exception as e:
        logging.error(f"Error applying transformation: {e}")
        return df

def apply_deletion(df):
    """Apply deletion method to remove outliers using IQR method"""
    try:
        df_cleaned = df.copy()
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        
        # Create mask for outliers
        outlier_mask = pd.Series([False] * len(df_cleaned), index=df_cleaned.index)
        
        for col in numeric_columns:
            if df_cleaned[col].notna().sum() > 0:
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Mark outliers
                col_outliers = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
                outlier_mask = outlier_mask | col_outliers
        
        # Remove outliers
        df_cleaned = df_cleaned[~outlier_mask]
        
        original_rows = len(df)
        remaining_rows = len(df_cleaned)
        removed_rows = original_rows - remaining_rows
        
        logging.info(f"Removed {removed_rows} rows ({removed_rows/original_rows*100:.1f}%) containing outliers")
        return df_cleaned
        
    except Exception as e:
        logging.error(f"Error applying deletion: {e}")
        return df

def detect_outliers(df, method='iqr'):
    """Detect outliers in the dataset using specified method"""
    try:
        outlier_info = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].notna().sum() > 0:
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = df[col][(df[col] < lower_bound) | (df[col] > upper_bound)]
                    
                elif method == 'zscore':
                    z_scores = np.abs(stats.zscore(df[col].dropna()))
                    outliers = df[col][z_scores > 3]
                
                outlier_info[col] = {
                    'count': len(outliers),
                    'percentage': (len(outliers) / len(df[col].dropna())) * 100,
                    'values': outliers.tolist()
                }
        
        return outlier_info
        
    except Exception as e:
        logging.error(f"Error detecting outliers: {e}")
        return {}