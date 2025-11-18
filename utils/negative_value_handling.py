"""
Negative Value Handling Utilities for Air Quality Data Analysis
Provides functions to detect and handle negative values in datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def check_negative_values(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check for negative values in the dataset and return statistics.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing negative value information
    """
    try:
        # Only check numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        negative_stats = {}
        total_negative_values = 0
        has_negatives = False
        
        for col in numeric_columns:
            negative_mask = df[col] < 0
            negative_count = negative_mask.sum()
            
            if negative_count > 0:
                has_negatives = True
                total_negative_values += negative_count
                negative_stats[col] = {
                    'count': int(negative_count),
                    'percentage': float(negative_count / len(df) * 100),
                    'min_value': float(df[col].min()),
                    'negative_indices': df[negative_mask].index.tolist()
                }
        
        result = {
            'success': True,
            'has_negatives': has_negatives,
            'total_negative_values': int(total_negative_values),
            'negative_stats': negative_stats,
            'total_rows': len(df),
            'numeric_columns': len(numeric_columns)
        }
        
        logger.info(f"Negative value check completed. Found {total_negative_values} negative values across {len(negative_stats)} columns")
        return result
        
    except Exception as e:
        logger.error(f"Error checking negative values: {str(e)}")
        return {
            'success': False,
            'error': f"Failed to check negative values: {str(e)}",
            'has_negatives': False,
            'total_negative_values': 0,
            'negative_stats': {}
        }

def handle_negative_values_interpolation(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle negative values using linear interpolation.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (processed DataFrame, processing info)
    """
    try:
        df_processed = df.copy()
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        
        processing_info = {
            'method': 'interpolation',
            'columns_processed': [],
            'values_interpolated': 0
        }
        
        for col in numeric_columns:
            negative_mask = df_processed[col] < 0
            negative_count = negative_mask.sum()
            
            if negative_count > 0:
                # Replace negative values with NaN, then interpolate
                df_processed.loc[negative_mask, col] = np.nan
                df_processed[col] = df_processed[col].interpolate(method='linear')
                
                # Fill any remaining NaN values with column mean
                if df_processed[col].isna().any():
                    df_processed[col].fillna(df_processed[col].mean(), inplace=True)
                
                processing_info['columns_processed'].append(col)
                processing_info['values_interpolated'] += int(negative_count)
        
        logger.info(f"Linear interpolation completed. Processed {processing_info['values_interpolated']} negative values across {len(processing_info['columns_processed'])} columns")
        
        return df_processed, processing_info
        
    except Exception as e:
        logger.error(f"Error in linear interpolation: {str(e)}")
        raise Exception(f"Failed to apply linear interpolation: {str(e)}")

def handle_negative_values_deletion(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Handle negative values by deleting rows containing them.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (processed DataFrame, processing info)
    """
    try:
        df_processed = df.copy()
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        
        # Find rows with any negative values in numeric columns
        negative_rows = pd.Series([False] * len(df_processed), index=df_processed.index)
        
        columns_with_negatives = []
        total_negative_values = 0
        
        for col in numeric_columns:
            negative_mask = df_processed[col] < 0
            negative_count = negative_mask.sum()
            
            if negative_count > 0:
                negative_rows |= negative_mask
                columns_with_negatives.append(col)
                total_negative_values += int(negative_count)
        
        # Remove rows with negative values
        rows_before = len(df_processed)
        df_processed = df_processed[~negative_rows]
        rows_after = len(df_processed)
        rows_deleted = rows_before - rows_after
        
        processing_info = {
            'method': 'deletion',
            'columns_processed': columns_with_negatives,
            'rows_deleted': int(rows_deleted),
            'rows_remaining': int(rows_after),
            'original_rows': int(rows_before),
            'negative_values_removed': int(total_negative_values)
        }
        
        logger.info(f"Row deletion completed. Removed {rows_deleted} rows containing {total_negative_values} negative values")
        
        return df_processed, processing_info
        
    except Exception as e:
        logger.error(f"Error in row deletion: {str(e)}")
        raise Exception(f"Failed to delete rows with negative values: {str(e)}")

def apply_negative_value_treatment(df: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply the specified negative value treatment method.
    
    Args:
        df: Input DataFrame
        method: Treatment method ('interpolation', 'deletion', 'none')
        
    Returns:
        Tuple of (processed DataFrame, processing info)
    """
    if method == 'interpolation':
        return handle_negative_values_interpolation(df)
    elif method == 'deletion':
        return handle_negative_values_deletion(df)
    elif method == 'none':
        return df.copy(), {
            'method': 'none',
            'message': 'No negative value treatment applied'
        }
    else:
        raise ValueError(f"Unknown negative value treatment method: {method}")