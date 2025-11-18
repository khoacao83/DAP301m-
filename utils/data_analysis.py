import pandas as pd
import numpy as np
import logging
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

class WeatherDataAnalyzer:
    def __init__(self):
        self.data = None
        self.analysis_results = {}
        
    def analyze_dataset(self, file_path_or_dataframe):
        """Perform comprehensive analysis of a weather dataset"""
        try:
            # Load the dataset - support both file path and DataFrame
            if isinstance(file_path_or_dataframe, str):
                self.data = pd.read_csv(file_path_or_dataframe)
            elif isinstance(file_path_or_dataframe, pd.DataFrame):
                self.data = file_path_or_dataframe
            else:
                raise ValueError("Input must be a file path string or pandas DataFrame")
                
            logging.info(f"Loaded dataset with shape: {self.data.shape}")
            
            # Basic dataset information
            basic_info = self._get_basic_info()
            
            # Data types and null values
            dtype_info = self._get_dtype_info()
            
            # Summary statistics
            summary_stats = self._get_summary_statistics()
            
            # Dataset duration in days
            dataset_days = self._get_dataset_duration()
            
            # Generate visualizations
            visualizations = self._generate_visualizations()
            
            # Generate box plot data for outlier analysis
            box_plot_data = self._generate_box_plot_data()
            
            self.analysis_results = {
                'shape': basic_info['shape'],
                'dataset_days': dataset_days,
                'dtypes': dtype_info,
                'summary_stats': summary_stats,
                'missing_values': basic_info['missing_values'],
                'duplicate_rows': basic_info['duplicate_rows'],
                'visualizations': visualizations,
                'box_plot_data': box_plot_data
            }
            
            return self.analysis_results
            
        except Exception as e:
            logging.error(f"Error analyzing dataset: {str(e)}")
            raise Exception(f"Dataset analysis failed: {str(e)}")
    
    def _get_basic_info(self):
        """Get basic information about the dataset"""
        try:
            missing_values = self.data.isnull().sum().to_dict()
            # Convert numpy int64 to regular int for JSON serialization
            missing_values = {k: int(v) for k, v in missing_values.items()}
            
            info = {
                'shape': [int(self.data.shape[0]), int(self.data.shape[1])],
                'missing_values': missing_values,
                'duplicate_rows': int(self.data.duplicated().sum())
            }
            return info
        except Exception as e:
            logging.error(f"Error getting basic info: {str(e)}")
            return {'shape': [0, 0], 'missing_values': {}, 'duplicate_rows': 0}
    
    def _get_dtype_info(self):
        """Get data type information for each column"""
        try:
            dtype_info = []
            for col in self.data.columns:
                dtype_info.append({
                    'column': str(col),
                    'dtype': str(self.data[col].dtype),
                    'non_null_count': int(self.data[col].count()),
                    'null_count': int(self.data[col].isnull().sum()),
                    'unique_values': int(self.data[col].nunique())
                })
            return dtype_info
        except Exception as e:
            logging.error(f"Error getting dtype info: {str(e)}")
            return []
    
    def _get_summary_statistics(self):
        """Get summary statistics for numerical columns"""
        try:
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            summary_stats = {}
            
            for col in numeric_columns:
                try:
                    stats = self.data[col].describe()
                    skew_val = self.data[col].skew()
                    kurt_val = self.data[col].kurtosis()
                    
                    summary_stats[str(col)] = {
                        'count': float(stats['count']),
                        'mean': float(stats['mean']),
                        'std': float(stats['std']),
                        'min': float(stats['min']),
                        'q25': float(stats['25%']),
                        'median': float(stats['50%']),
                        'q75': float(stats['75%']),
                        'max': float(stats['max']),
                        'skewness': float(skew_val) if not pd.isna(skew_val) else 0.0,
                        'kurtosis': float(kurt_val) if not pd.isna(kurt_val) else 0.0
                    }
                except Exception as col_error:
                    logging.warning(f"Error processing column {col}: {col_error}")
                    continue
            
            return summary_stats
        except Exception as e:
            logging.error(f"Error getting summary statistics: {str(e)}")
            return {}
    
    def _get_dataset_duration(self):
        """Calculate the number of days covered by the dataset"""
        try:
            # Look for datetime columns
            datetime_columns = []
            for col in self.data.columns:
                if 'datetime' in col.lower() or 'date' in col.lower() or 'time' in col.lower():
                    datetime_columns.append(col)
            
            if not datetime_columns:
                # If no datetime columns found, estimate from row count
                # Assume hourly data (24 rows per day)
                estimated_days = max(1, self.data.shape[0] // 24)
                return estimated_days
            
            # Use the first datetime column
            datetime_col = datetime_columns[0]
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(self.data[datetime_col]):
                self.data[datetime_col] = pd.to_datetime(self.data[datetime_col])
            
            # Calculate the time span
            min_date = self.data[datetime_col].min()
            max_date = self.data[datetime_col].max()
            
            if pd.isna(min_date) or pd.isna(max_date):
                # Fallback to row count estimation
                estimated_days = max(1, self.data.shape[0] // 24)
                return estimated_days
            
            # Calculate days difference
            days_difference = (max_date - min_date).days + 1  # +1 to include both start and end dates
            return max(1, days_difference)
            
        except Exception as e:
            logging.error(f"Error calculating dataset duration: {str(e)}")
            # Fallback to row count estimation
            estimated_days = max(1, self.data.shape[0] // 24)
            return estimated_days
    
    def _generate_box_plot_data(self):
        """Generate box plot data for outlier analysis"""
        try:
            from utils.visualization import create_box_plots
            
            # Create box plots for outlier detection
            box_plot_result = create_box_plots(self.data)
            
            # Calculate outlier statistics
            outlier_stats = self._calculate_outlier_stats()
            
            return {
                'plot': box_plot_result[0] if box_plot_result else None,
                'outlier_stats': outlier_stats
            }
        except Exception as e:
            logging.error(f"Error generating box plot data: {str(e)}")
            return {'plot': None, 'outlier_stats': {}}
    
    def _calculate_outlier_stats(self):
        """Calculate outlier statistics using IQR method"""
        try:
            numeric_data = self.data.select_dtypes(include=[np.number])
            outlier_stats = {}
            
            for col in numeric_data.columns:
                if numeric_data[col].notna().sum() > 0:
                    Q1 = numeric_data[col].quantile(0.25)
                    Q3 = numeric_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = numeric_data[col][(numeric_data[col] < lower_bound) | (numeric_data[col] > upper_bound)]
                    
                    outlier_stats[col] = {
                        'count': len(outliers),
                        'percentage': (len(outliers) / len(numeric_data[col].dropna())) * 100,
                        'lower_bound': float(lower_bound),
                        'upper_bound': float(upper_bound),
                        'min_outlier': float(outliers.min()) if len(outliers) > 0 else None,
                        'max_outlier': float(outliers.max()) if len(outliers) > 0 else None
                    }
            
            return outlier_stats
        except Exception as e:
            logging.error(f"Error calculating outlier stats: {str(e)}")
            return {}
    
    def _generate_visualizations(self):
        """Generate comprehensive visualizations for the dataset"""
        from utils.visualization import create_advanced_visualizations
        
        try:
            # Generate advanced visualizations using matplotlib
            visualizations = create_advanced_visualizations(self.data)
            return visualizations
        except Exception as e:
            logging.error(f"Error generating visualizations: {str(e)}")
            return {
                'histograms': [],
                'time_series': [],
                'correlation_heatmap': None,
                'box_plots': [],
                'categorical_analysis': []
            }
    
    def _create_histograms(self):
        """Create histogram plots for numerical columns"""
        histograms = []
        
        try:
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns[:6]:  # Limit to first 6 columns to avoid overcrowding
                if self.data[col].notna().sum() > 0:  # Only plot if there's non-null data
                    fig = go.Figure()
                    fig.add_trace(
                        go.Histogram(
                            x=self.data[col].dropna(),
                            name=col,
                            nbinsx=30,
                            opacity=0.7
                        )
                    )
                    
                    fig.update_layout(
                        title=f'Distribution of {col}',
                        xaxis_title=col,
                        yaxis_title='Frequency',
                        height=300,
                        showlegend=False
                    )
                    
                    histograms.append({
                        'data': fig.to_dict()['data'],
                        'layout': fig.to_dict()['layout']
                    })
        
        except Exception as e:
            logging.error(f"Error creating histograms: {str(e)}")
        
        return histograms
    
    def _create_time_series(self):
        """Create time series plots if date columns exist"""
        time_series = []
        
        try:
            # Try to identify date columns
            date_columns = []
            for col in self.data.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        pd.to_datetime(self.data[col])
                        date_columns.append(col)
                    except:
                        continue
            
            if not date_columns:
                return time_series
            
            date_col = date_columns[0]  # Use first date column found
            
            # Convert to datetime
            self.data[date_col] = pd.to_datetime(self.data[date_col])
            
            # Get numerical columns for plotting
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns[:4]:  # Limit to 4 time series
                if self.data[col].notna().sum() > 0:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=self.data[date_col],
                            y=self.data[col],
                            mode='lines+markers',
                            name=col,
                            line=dict(width=2),
                            marker=dict(size=4)
                        )
                    )
                    
                    fig.update_layout(
                        title=f'{col} Over Time',
                        xaxis_title='Date',
                        yaxis_title=col,
                        height=350,
                        showlegend=False
                    )
                    
                    time_series.append({
                        'data': fig.to_dict()['data'],
                        'layout': fig.to_dict()['layout']
                    })
        
        except Exception as e:
            logging.error(f"Error creating time series: {str(e)}")
        
        return time_series
    
    def _create_correlation_heatmap(self):
        """Create correlation heatmap for numerical columns"""
        try:
            numeric_data = self.data.select_dtypes(include=[np.number])
            
            if numeric_data.shape[1] < 2:
                return None
            
            # Calculate correlation matrix
            correlation_matrix = numeric_data.corr()
            
            # Create heatmap
            fig = go.Figure(
                data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(correlation_matrix.values, 2),
                    texttemplate='%{text}',
                    textfont={'size': 10},
                    hoverongaps=False
                )
            )
            
            fig.update_layout(
                title='Correlation Heatmap',
                height=max(400, len(correlation_matrix.columns) * 40),
                xaxis={'side': 'bottom'},
                yaxis={'autorange': 'reversed'}
            )
            
            return {
                'data': fig.to_dict()['data'],
                'layout': fig.to_dict()['layout']
            }
        
        except Exception as e:
            logging.error(f"Error creating correlation heatmap: {str(e)}")
            return None
    
    def _create_box_plots(self):
        """Create box plots for outlier detection"""
        box_plots = []
        
        try:
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns[:6]:  # Limit to 6 box plots
                if self.data[col].notna().sum() > 0:
                    fig = go.Figure()
                    fig.add_trace(
                        go.Box(
                            y=self.data[col].dropna(),
                            name=col,
                            boxpoints='outliers',
                            marker_color='lightblue',
                            line_color='darkblue'
                        )
                    )
                    
                    fig.update_layout(
                        title=f'Box Plot - {col}',
                        yaxis_title=col,
                        height=300,
                        showlegend=False
                    )
                    
                    box_plots.append({
                        'data': fig.to_dict()['data'],
                        'layout': fig.to_dict()['layout']
                    })
        
        except Exception as e:
            logging.error(f"Error creating box plots: {str(e)}")
        
        return box_plots
    
    def _create_categorical_analysis(self):
        """Create analysis for categorical columns"""
        categorical_plots = []
        
        try:
            categorical_columns = self.data.select_dtypes(include=['object']).columns
            
            for col in categorical_columns[:4]:  # Limit to 4 categorical columns
                if self.data[col].notna().sum() > 0:
                    # Get value counts
                    value_counts = self.data[col].value_counts().head(10)  # Top 10 categories
                    
                    if len(value_counts) > 1:
                        fig = go.Figure()
                        fig.add_trace(
                            go.Bar(
                                x=value_counts.index,
                                y=value_counts.values,
                                name=col,
                                marker_color='lightcoral'
                            )
                        )
                        
                        fig.update_layout(
                            title=f'Distribution of {col}',
                            xaxis_title=col,
                            yaxis_title='Count',
                            height=350,
                            showlegend=False
                        )
                        
                        categorical_plots.append({
                            'data': fig.to_dict()['data'],
                            'layout': fig.to_dict()['layout']
                        })
        
        except Exception as e:
            logging.error(f"Error creating categorical analysis: {str(e)}")
        
        return categorical_plots
    
    def get_column_profile(self, column_name):
        """Get detailed profile for a specific column"""
        try:
            if column_name not in self.data.columns:
                raise ValueError(f"Column '{column_name}' not found in dataset")
            
            col_data = self.data[column_name]
            
            profile = {
                'name': column_name,
                'dtype': str(col_data.dtype),
                'count': len(col_data),
                'non_null_count': col_data.count(),
                'null_count': col_data.isnull().sum(),
                'null_percentage': (col_data.isnull().sum() / len(col_data)) * 100,
                'unique_count': col_data.nunique(),
                'unique_percentage': (col_data.nunique() / col_data.count()) * 100 if col_data.count() > 0 else 0
            }
            
            if col_data.dtype in ['int64', 'float64']:
                # Numerical column
                profile.update({
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'q25': col_data.quantile(0.25),
                    'q75': col_data.quantile(0.75),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis()
                })
            else:
                # Categorical column
                value_counts = col_data.value_counts()
                profile.update({
                    'most_frequent': value_counts.index[0] if not value_counts.empty else None,
                    'most_frequent_count': value_counts.iloc[0] if not value_counts.empty else 0,
                    'least_frequent': value_counts.index[-1] if not value_counts.empty else None,
                    'least_frequent_count': value_counts.iloc[-1] if not value_counts.empty else 0
                })
            
            return profile
        
        except Exception as e:
            logging.error(f"Error creating column profile: {str(e)}")
            raise Exception(f"Column profiling failed: {str(e)}")
    
    def detect_outliers(self, column_name, method='iqr'):
        """Detect outliers in a numerical column"""
        try:
            if column_name not in self.data.columns:
                raise ValueError(f"Column '{column_name}' not found in dataset")
            
            col_data = self.data[column_name].dropna()
            
            if col_data.dtype not in ['int64', 'float64']:
                raise ValueError(f"Column '{column_name}' is not numerical")
            
            outliers = []
            
            if method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            elif method == 'zscore':
                z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
                outliers = col_data[z_scores > 3]
            
            return {
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(col_data)) * 100,
                'outlier_values': outliers.tolist()[:20]  # Return first 20 outliers
            }
        
        except Exception as e:
            logging.error(f"Error detecting outliers: {str(e)}")
            raise Exception(f"Outlier detection failed: {str(e)}")

# Global instance
data_analyzer = WeatherDataAnalyzer()

def analyze_dataset(file_path):
    """Analyze a dataset and return comprehensive results"""
    return data_analyzer.analyze_dataset(file_path)

def generate_visualizations(data):
    """Generate visualizations for given data"""
    try:
        # Save data temporarily and analyze
        temp_path = '/tmp/temp_data.csv'
        if isinstance(data, pd.DataFrame):
            data.to_csv(temp_path, index=False)
        else:
            # Assume data is already a file path
            temp_path = data
        
        analyzer = WeatherDataAnalyzer()
        analyzer.data = pd.read_csv(temp_path)
        return analyzer._generate_visualizations()
    
    except Exception as e:
        logging.error(f"Error generating visualizations: {str(e)}")
        return {}

def get_column_profile(file_path, column_name):
    """Get detailed profile for a specific column"""
    analyzer = WeatherDataAnalyzer()
    analyzer.data = pd.read_csv(file_path)
    return analyzer.get_column_profile(column_name)

def detect_outliers(file_path, column_name, method='iqr'):
    """Detect outliers in a column"""
    analyzer = WeatherDataAnalyzer()
    analyzer.data = pd.read_csv(file_path)
    return analyzer.detect_outliers(column_name, method)

def get_data_quality_report(file_path):
    """Generate a comprehensive data quality report"""
    try:
        analyzer = WeatherDataAnalyzer()
        data = pd.read_csv(file_path)
        analyzer.data = data
        
        # Basic quality metrics
        total_rows = len(data)
        total_cols = len(data.columns)
        missing_data = data.isnull().sum().sum()
        duplicate_rows = data.duplicated().sum()
        
        # Column-wise quality
        column_quality = []
        for col in data.columns:
            col_quality = {
                'column': col,
                'completeness': ((data[col].count() / total_rows) * 100) if total_rows > 0 else 0,
                'uniqueness': ((data[col].nunique() / data[col].count()) * 100) if data[col].count() > 0 else 0,
                'data_type': str(data[col].dtype)
            }
            column_quality.append(col_quality)
        
        quality_report = {
            'overall_quality': {
                'total_rows': total_rows,
                'total_columns': total_cols,
                'missing_data_points': missing_data,
                'missing_data_percentage': (missing_data / (total_rows * total_cols) * 100) if total_rows * total_cols > 0 else 0,
                'duplicate_rows': duplicate_rows,
                'duplicate_percentage': (duplicate_rows / total_rows * 100) if total_rows > 0 else 0
            },
            'column_quality': column_quality
        }
        
        return quality_report
    
    except Exception as e:
        logging.error(f"Error generating data quality report: {str(e)}")
        raise Exception(f"Data quality report generation failed: {str(e)}")
