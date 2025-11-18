import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
import base64
import io
from datetime import datetime
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')
from pywaffle import Waffle

def safe_json_serialize(obj):
    """Convert numpy arrays and other non-serializable objects to JSON-safe formats"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [safe_json_serialize(item) for item in obj]
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    else:
        return obj

def create_actual_vs_predicted_line_plot(y_test, y_pred_test, model_name):
    """Create line plot showing actual vs predicted values"""
    try:
        # Convert to lists for JSON serialization
        y_test = [float(x) for x in y_test]
        y_pred_test = [float(x) for x in y_pred_test]
        
        # Sort data by actual values for better line visualization
        sorted_data = sorted(zip(y_test, y_pred_test), key=lambda x: x[0])
        y_test_sorted = [x[0] for x in sorted_data]
        y_pred_sorted = [x[1] for x in sorted_data]
        
        fig = go.Figure()
        
        # Actual values line
        fig.add_trace(
            go.Scatter(
                x=list(range(len(y_test_sorted))),
                y=y_test_sorted,
                mode='lines',
                name='Actual Values',
                line=dict(color='blue', width=2)
            )
        )
        
        # Predicted values line
        fig.add_trace(
            go.Scatter(
                x=list(range(len(y_pred_sorted))),
                y=y_pred_sorted,
                mode='lines',
                name='Predicted Values',
                line=dict(color='red', width=2)
            )
        )
        
        fig.update_layout(
            title=f'{model_name} - Actual vs Predicted Values (Line Plot)',
            height=400,
            xaxis_title='Sample Index (sorted by actual value)',
            yaxis_title='Values'
        )
        
        return {
            'data': safe_json_serialize(fig.to_dict()['data']),
            'layout': safe_json_serialize(fig.to_dict()['layout'])
        }
        
    except Exception as e:
        logging.error(f"Error creating line plot: {str(e)}")
        return None

def create_learning_curve(shared_splits, models, model_name):
    """Create learning curve visualization"""
    try:
        if not shared_splits:
            return None
            
        # Get shared splits
        X_train = shared_splits['X_train_scaled']
        y_train = shared_splits['y_train']
        X_test = shared_splits['X_test_scaled']
        y_test = shared_splits['y_test']
        
        # Get model
        model = models[model_name]
        
        # Create training set size increments
        n_samples = len(X_train)
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes = [int(size * n_samples) for size in train_sizes]
        
        train_scores = []
        test_scores = []
        
        for size in train_sizes:
            # Use subset of training data
            X_subset = X_train[:size]
            y_subset = y_train[:size]
            
            # Train model on subset
            model_copy = models[model_name].__class__(**models[model_name].get_params())
            model_copy.fit(X_subset, y_subset)
            
            # Evaluate on training subset
            train_pred = model_copy.predict(X_subset)
            train_score = r2_score(y_subset, train_pred)
            train_scores.append(train_score)
            
            # Evaluate on test set
            test_pred = model_copy.predict(X_test)
            test_score = r2_score(y_test, test_pred)
            test_scores.append(test_score)
        
        # Create learning curve plot
        fig = go.Figure()
        
        # Training scores
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=train_scores,
                mode='lines+markers',
                name='Training Score',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            )
        )
        
        # Test scores
        fig.add_trace(
            go.Scatter(
                x=train_sizes,
                y=test_scores,
                mode='lines+markers',
                name='Test Score',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            )
        )
        
        fig.update_layout(
            title=f'{model_name} - Learning Curve',
            xaxis_title='Training Set Size',
            yaxis_title='R² Score',
            height=400
        )
        
        return {
            'data': safe_json_serialize(fig.to_dict()['data']),
            'layout': safe_json_serialize(fig.to_dict()['layout'])
        }
        
    except Exception as e:
        logging.error(f"Error creating learning curve: {str(e)}")
        return None

def create_residuals_plot(y_test, y_pred_test, model_name):
    """Create residuals plot"""
    try:
        # Convert to lists for JSON serialization
        y_test = [float(x) for x in y_test]
        y_pred_test = [float(x) for x in y_pred_test]
        
        # Calculate residuals
        residuals_test = [actual - pred for actual, pred in zip(y_test, y_pred_test)]
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=y_pred_test, y=residuals_test,
                mode='markers',
                name='Test Residuals',
                marker=dict(color='red', opacity=0.6)
            )
        )
        
        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black")
        
        fig.update_layout(
            title=f'{model_name} - Residuals Plot',
            xaxis_title='Predicted Values',
            yaxis_title='Residuals',
            height=400
        )
        
        return {
            'data': safe_json_serialize(fig.to_dict()['data']),
            'layout': safe_json_serialize(fig.to_dict()['layout'])
        }
        
    except Exception as e:
        logging.error(f"Error creating residuals plot: {str(e)}")
        return None

def create_model_visualizations(y_test, y_pred_test, feature_names, is_classification, model_name, shared_splits, models):
    """Create all visualizations for model performance"""
    visualizations = []
    
    try:
        # 1. Actual vs Predicted Line Plot
        line_plot = create_actual_vs_predicted_line_plot(y_test, y_pred_test, model_name)
        if line_plot:
            visualizations.append(line_plot)
        
        # 2. Learning Curve
        learning_curve = create_learning_curve(shared_splits, models, model_name)
        if learning_curve:
            visualizations.append(learning_curve)
        
        # 3. Residuals Plot
        residuals_plot = create_residuals_plot(y_test, y_pred_test, model_name)
        if residuals_plot:
            visualizations.append(residuals_plot)
            
    except Exception as e:
        logging.error(f"Error creating visualizations: {str(e)}")
    
    return visualizations

def create_prediction_visualizations(y_actual, y_pred, feature_names, model_name=None):
    """Create visualizations for predictions"""
    visualizations = []
    
    try:
        # Actual vs Predicted
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=y_actual, y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(color='blue', opacity=0.6)
            )
        )
        
        # Perfect prediction line
        min_val = min(min(y_actual), min(y_pred))
        max_val = max(max(y_actual), max(y_pred))
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        
        title = f'{model_name} - Model Predictions vs Actual Values' if model_name else 'Model Predictions vs Actual Values'
        fig.update_layout(
            title=title,
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            height=400
        )
        
        visualizations.append({
            'data': safe_json_serialize(fig.to_dict()['data']),
            'layout': safe_json_serialize(fig.to_dict()['layout'])
        })
        
    except Exception as e:
        logging.error(f"Error creating prediction visualizations: {str(e)}")
    
    return visualizations


def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML"""
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=300, 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close(fig)
        
        # Convert to base64
        plot_b64 = base64.b64encode(plot_data).decode('utf-8')
        return f"data:image/png;base64,{plot_b64}"
    except Exception as e:
        logging.error(f"Error converting plot to base64: {e}")
        plt.close(fig)
        return None


def create_advanced_visualizations(data):
    """Create comprehensive advanced visualizations using matplotlib and seaborn"""
    try:
        # Set style for white background plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        visualizations = {
            'correlation_heatmap': create_correlation_heatmap(data),
            'word_cloud': create_word_cloud(data),
            'distribution_plots': create_distribution_plots(data),
            'aqi_pie_chart': create_aqi_pie_chart(data),
            'aqi_waffle_chart': create_aqi_waffle_chart(data),
            'time_series_plots': create_time_series_plots(data),
            'pairwise_relationships': create_pairwise_plots(data)
        }
        
        return visualizations
        
    except Exception as e:
        logging.error(f"Error creating advanced visualizations: {e}")
        return {
            'correlation_heatmap': None,
            'word_cloud': None,
            'distribution_plots': [],
            'box_plots': [],
            'aqi_pie_chart': None,
            'aqi_waffle_chart': None,
            'time_series_plots': [],
            'pairwise_relationships': None
        }


def create_correlation_heatmap(data):
    """Create correlation heatmap for numerical columns"""
    try:
        # Get numerical columns only
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty or len(numeric_data.columns) < 2:
            return None
            
        # Calculate correlation matrix
        correlation_matrix = numeric_data.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap without mask (show full matrix)
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', 
                   center=0, square=True, fmt='.2f', 
                   cbar_kws={'shrink': 0.8},
                   linewidths=0.5, linecolor='white',
                   ax=ax)
        
        ax.set_title('Correlation Heatmap of Features', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Set axis labels
        ax.set_xlabel('Features', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        return plot_to_base64(fig)
        
    except Exception as e:
        logging.error(f"Error creating correlation heatmap: {e}")
        return None


def create_word_cloud(data):
    """Create word cloud from feature names"""
    try:
        # Get column names (features)
        feature_names = list(data.columns)
        
        # Remove non-meaningful columns
        excluded_cols = ['datetime', 'date', 'time', 'timestamp', 'index', 'id']
        feature_names = [col for col in feature_names if col.lower() not in excluded_cols]
        
        if len(feature_names) < 2:
            return None
        
        # Create text from feature names (repeat based on importance/correlation)
        text_data = []
        numeric_data = data.select_dtypes(include=[np.number])
        
        for col in feature_names:
            if col in numeric_data.columns:
                # Add feature name multiple times based on its variance (importance indicator)
                try:
                    variance = numeric_data[col].var()
                    if not np.isnan(variance) and variance > 0:
                        # Normalize variance and use as frequency
                        freq = max(1, int(variance / numeric_data.var().mean() * 5))
                        text_data.extend([col.replace('_', ' ').title()] * freq)
                    else:
                        text_data.append(col.replace('_', ' ').title())
                except:
                    text_data.append(col.replace('_', ' ').title())
            else:
                text_data.append(col.replace('_', ' ').title())
        
        # Create word cloud
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white',
                             colormap='viridis',
                             max_words=50,
                             relative_scaling=0.5,
                             random_state=42).generate(' '.join(text_data))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('Feature Names Word Cloud', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        return plot_to_base64(fig)
        
    except Exception as e:
        logging.error(f"Error creating word cloud: {e}")
        return None


def create_distribution_plots(data):
    """Create distribution plots for all numerical columns (histograms only)"""
    try:
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return []
            
        plots = []
        
        # Limit to first 8 columns to avoid too many plots
        columns = numeric_data.columns[:8]
        
        for i, col in enumerate(columns):
            if numeric_data[col].notna().sum() > 10:  # Need at least 10 data points
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Histogram with KDE
                ax.hist(numeric_data[col].dropna(), bins=30, alpha=0.7, 
                       color='skyblue', edgecolor='black', density=True)
                
                # Add KDE curve
                try:
                    from scipy import stats
                    kde_data = numeric_data[col].dropna()
                    if len(kde_data) > 1:
                        density = stats.gaussian_kde(kde_data)
                        xs = np.linspace(kde_data.min(), kde_data.max(), 200)
                        ax.plot(xs, density(xs), 'r-', linewidth=2, label='KDE')
                        ax.legend()
                except:
                    pass
                
                ax.set_title(f'Distribution of {col}', fontsize=14, fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel('Density')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                plot_b64 = plot_to_base64(fig)
                if plot_b64:
                    plots.append({
                        'column': col,
                        'plot': plot_b64,
                        'type': 'distribution'
                    })
        
        return plots
        
    except Exception as e:
        logging.error(f"Error creating distribution plots: {e}")
        return []


def create_box_plots(data):
    """Create comprehensive box plots for numerical data"""
    try:
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return []
            
        plots = []
        
        # Create a combined box plot for all numerical columns
        if len(numeric_data.columns) > 1:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Normalize data for better comparison
            normalized_data = numeric_data.copy()
            for col in numeric_data.columns:
                if numeric_data[col].std() > 0:
                    normalized_data[col] = (numeric_data[col] - numeric_data[col].mean()) / numeric_data[col].std()
            
            bp = ax.boxplot([normalized_data[col].dropna() for col in normalized_data.columns], 
                           labels=normalized_data.columns, patch_artist=True)
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title('Normalized Box Plots of All Numerical Features', 
                        fontsize=16, fontweight='bold')
            ax.set_ylabel('Normalized Values (Z-score)')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plot_b64 = plot_to_base64(fig)
            if plot_b64:
                plots.append({
                    'type': 'combined_boxplot',
                    'plot': plot_b64,
                    'title': 'Combined Box Plot Analysis'
                })
        
        return plots
        
    except Exception as e:
        logging.error(f"Error creating box plots: {e}")
        return []





def create_time_series_plots(data):
    """Create time series plots if datetime columns exist"""
    try:
        plots = []
        
        # Look for datetime columns
        datetime_columns = []
        for col in data.columns:
            if 'datetime' in col.lower() or 'date' in col.lower() or 'time' in col.lower():
                datetime_columns.append(col)
        
        if not datetime_columns:
            return plots
            
        datetime_col = datetime_columns[0]
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data[datetime_col]):
            data[datetime_col] = pd.to_datetime(data[datetime_col])
        
        # Get numerical columns for time series
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return plots
            
        # Create time series plots for key pollutants
        key_pollutants = ['pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2']
        available_pollutants = [col for col in key_pollutants if col in numeric_data.columns]
        
        if available_pollutants:
            # Sample data if too large
            sample_size = min(1000, len(data))
            if len(data) > sample_size:
                sample_data = data.sample(n=sample_size).sort_values(datetime_col)
            else:
                sample_data = data.sort_values(datetime_col)
            
            fig, axes = plt.subplots(len(available_pollutants), 1, 
                                   figsize=(14, 4*len(available_pollutants)))
            
            if len(available_pollutants) == 1:
                axes = [axes]
            
            for i, pollutant in enumerate(available_pollutants):
                axes[i].plot(sample_data[datetime_col], sample_data[pollutant], 
                           linewidth=1, alpha=0.8, color=plt.cm.Set1(i))
                axes[i].set_title(f'{pollutant.upper()} Time Series', 
                                fontsize=14, fontweight='bold')
                axes[i].set_ylabel(pollutant.upper())
                axes[i].grid(True, alpha=0.3)
                
                # Add rolling average
                if len(sample_data) > 24:
                    window = min(24, len(sample_data) // 4)
                    rolling_avg = sample_data[pollutant].rolling(window=window).mean()
                    axes[i].plot(sample_data[datetime_col], rolling_avg, 
                               color='red', linewidth=2, alpha=0.7,
                               label=f'{window}-point Moving Average')
                    axes[i].legend()
            
            plt.suptitle('Air Quality Time Series Analysis', fontsize=18, fontweight='bold')
            plt.tight_layout()
            
            plot_b64 = plot_to_base64(fig)
            if plot_b64:
                plots.append({
                    'type': 'time_series',
                    'plot': plot_b64,
                    'title': 'Air Quality Time Series'
                })
        
        return plots
        
    except Exception as e:
        logging.error(f"Error creating time series plots: {e}")
        return []


def create_pairwise_plots(data):
    """Create pairwise relationship plots for key variables"""
    try:
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty or len(numeric_data.columns) < 2:
            return None
            
        # Select key pollutants for pairwise analysis
        key_pollutants = ['pm2_5', 'pm10', 'co', 'no2', 'o3', 'so2']
        available_pollutants = [col for col in key_pollutants if col in numeric_data.columns]
        
        if len(available_pollutants) < 2:
            # Use first 4 numerical columns if pollutants not available
            available_pollutants = numeric_data.columns[:4].tolist()
        
        if len(available_pollutants) < 2:
            return None
            
        # Sample data if too large
        sample_size = min(1000, len(data))
        if len(data) > sample_size:
            sample_data = data[available_pollutants].sample(n=sample_size)
        else:
            sample_data = data[available_pollutants]
        
        # Create pairwise scatter plots
        fig, axes = plt.subplots(len(available_pollutants), len(available_pollutants), 
                               figsize=(12, 12))
        
        for i, col1 in enumerate(available_pollutants):
            for j, col2 in enumerate(available_pollutants):
                if i == j:
                    # Diagonal: histogram
                    axes[i, j].hist(sample_data[col1].dropna(), bins=20, alpha=0.7, 
                                  color=plt.cm.Set1(i))
                    axes[i, j].set_title(f'{col1}')
                else:
                    # Off-diagonal: scatter plot
                    valid_data = sample_data[[col1, col2]].dropna()
                    if len(valid_data) > 0:
                        axes[i, j].scatter(valid_data[col2], valid_data[col1], 
                                         alpha=0.5, s=20, color=plt.cm.Set1(i))
                        
                        # Add trend line
                        z = np.polyfit(valid_data[col2], valid_data[col1], 1)
                        p = np.poly1d(z)
                        axes[i, j].plot(valid_data[col2], p(valid_data[col2]), 
                                      "r--", alpha=0.8, linewidth=1)
                
                axes[i, j].grid(True, alpha=0.3)
                
                if i == len(available_pollutants) - 1:
                    axes[i, j].set_xlabel(col2)
                if j == 0:
                    axes[i, j].set_ylabel(col1)
        
        plt.suptitle('Pairwise Relationships Between Air Quality Variables', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return plot_to_base64(fig)
        
    except Exception as e:
        logging.error(f"Error creating pairwise plots: {e}")
        return None

def create_box_plots(data):
    """Create box plots for outlier detection"""
    try:
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) == 0:
            return None
        
        # Create box plots for all numeric columns
        n_cols = min(3, len(numeric_data.columns))
        n_rows = (len(numeric_data.columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.patch.set_facecolor('white')
        
        # Handle single subplot case
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Flatten axes for easy iteration
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for i, col in enumerate(numeric_data.columns):
            if i < len(axes_flat):
                ax = axes_flat[i]
                
                # Create box plot
                bp = ax.boxplot(numeric_data[col].dropna(), 
                              patch_artist=True, 
                              boxprops=dict(facecolor='lightblue', alpha=0.7),
                              medianprops=dict(color='red', linewidth=2),
                              whiskerprops=dict(color='blue'),
                              capprops=dict(color='blue'),
                              flierprops=dict(marker='o', markerfacecolor='red', markersize=5, alpha=0.7))
                
                ax.set_title(f'{col} - Outlier Detection', fontsize=12, fontweight='bold', pad=20)
                ax.set_ylabel('Values', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_facecolor('white')
        
        # Hide unused subplots
        for i in range(len(numeric_data.columns), len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        plot_data = {
            'title': 'Box Plot Analysis - Outlier Detection',
            'plot': f'data:image/png;base64,{image_base64}'
        }
        
        return [plot_data]
        
    except Exception as e:
        logging.error(f"Error creating box plots: {e}")
        return None

def create_area_plot(data, features):
    """Create area plot for selected features"""
    try:
        # Set style for white background plots
        plt.style.use('default')
        
        # Prepare data for plotting
        plot_data = data[features].copy()
        
        # Create index for x-axis (use row numbers if no datetime column)
        x_values = range(len(plot_data))
        
        # Check if there's a datetime column for better x-axis
        datetime_cols = []
        for col in data.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'datetime']):
                try:
                    pd.to_datetime(data[col])
                    datetime_cols.append(col)
                except:
                    pass
        
        if datetime_cols:
            # Use the first datetime column found
            datetime_col = datetime_cols[0]
            try:
                x_values = pd.to_datetime(data[datetime_col])
            except:
                x_values = range(len(plot_data))
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Prepare data for stackplot
        data_arrays = []
        labels = []
        for feature in features:
            values = plot_data[feature].fillna(0)
            data_arrays.append(values)
            labels.append(feature)
        
        # Define professional colors similar to the reference image
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        # Create stackplot with better styling
        ax.stackplot(x_values, *data_arrays, labels=labels, alpha=0.9, colors=colors[:len(features)])
        
        # Customize the plot with professional styling
        ax.set_title(f'Weekly Average Pollutant Composition', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date' if datetime_cols else 'Data Points', fontsize=12)
        ax.set_ylabel('Concentration (μg/m³)', fontsize=12)
        
        # Add legend with better positioning
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=10)
        
        # Format x-axis if datetime
        if datetime_cols and len(x_values) > 1:
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            fig.autofmt_xdate()
        else:
            ax.tick_params(axis='x', labelsize=10)
        
        ax.tick_params(axis='y', labelsize=10)
        
        # Add subtle grid
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Set background colors and styling
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Add border around the plot
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')
        
        plt.tight_layout()
        
        plot_base64 = plot_to_base64(fig)
        
        if plot_base64:
            return {
                'plot': plot_base64,
                'title': f'Weekly Average Pollutant Composition',
                'description': f'Stacked area chart showing the composition and trends of {len(features)} selected pollutants over time with professional styling.'
            }
        else:
            return None
            
    except Exception as e:
        logging.error(f"Error creating area plot: {e}")
        return None

def get_aqi_category(aqi):
    """Map AQI numeric values to category names"""
    if aqi == 1:
        return 'Good'
    elif aqi == 2:
        return 'Moderate'
    elif aqi == 3:
        return 'Unhealthy for Sensitive'
    elif aqi == 4:
        return 'Unhealthy'
    elif aqi == 5:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

def create_aqi_pie_chart(data):
    """Create pie chart showing AQI category distribution"""
    try:
        # Check if AQI column exists
        if 'aqi' not in data.columns:
            return None
            
        # Map AQI values to categories
        data_copy = data.copy()
        data_copy['aqi_category'] = data_copy['aqi'].apply(get_aqi_category)
        aqi_counts = data_copy['aqi_category'].value_counts()
        
        # Define AQI colors
        aqi_colors = {
            'Good': '#00e400',
            'Moderate': '#ffff00',
            'Unhealthy for Sensitive': '#ff7e00',
            'Unhealthy': '#ff0000',
            'Very Unhealthy': '#8f3f97',
            'Hazardous': '#7e0023'
        }
        
        # Filter colors for categories present in the data
        present_categories = aqi_counts.index
        filtered_colors = [aqi_colors[cat] for cat in present_categories]
        
        # Create the pie chart
        fig, ax = plt.subplots(figsize=(10, 8))
        
        wedges, texts, autotexts = ax.pie(
            aqi_counts.values, 
            labels=aqi_counts.index, 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=filtered_colors,
            textprops={'fontsize': 12}
        )
        
        # Enhance text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('AQI Category Distribution - Pie Chart', fontsize=16, fontweight='bold', pad=20)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Set background
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        
        plot_base64 = plot_to_base64(fig)
        
        if plot_base64:
            return {
                'plot': plot_base64,
                'title': 'AQI Category Distribution - Pie Chart',
                'description': f'Pie chart showing the distribution of {len(aqi_counts)} AQI categories across {len(data)} data points.'
            }
        else:
            return None
            
    except Exception as e:
        logging.error(f"Error creating AQI pie chart: {e}")
        return None

def create_aqi_waffle_chart(data):
    """Create waffle chart showing AQI category distribution"""
    try:
        # Check if AQI column exists
        if 'aqi' not in data.columns:
            return None
            
        # Map AQI values to categories
        data_copy = data.copy()
        data_copy['aqi_category'] = data_copy['aqi'].apply(get_aqi_category)
        aqi_counts = data_copy['aqi_category'].value_counts()
        
        # Define AQI colors
        aqi_colors = {
            'Good': '#00e400',
            'Moderate': '#ffff00',
            'Unhealthy for Sensitive': '#ff7e00',
            'Unhealthy': '#ff0000',
            'Very Unhealthy': '#8f3f97',
            'Hazardous': '#7e0023'
        }
        
        # Filter colors for categories present in the data
        present_categories = aqi_counts.index
        filtered_colors = [aqi_colors[cat] for cat in present_categories]
        
        # Convert counts to percentages for the waffle chart
        waffle_data = (aqi_counts / len(data_copy) * 100).round().astype(int)
        waffle_data = waffle_data[waffle_data > 0]  # Remove zero values
        
        # Create the waffle chart
        fig = plt.figure(
            FigureClass=Waffle,
            rows=10,
            columns=10,
            values=waffle_data.values,
            labels=list(waffle_data.index),
            colors=[aqi_colors[cat] for cat in waffle_data.index],
            legend={'loc': 'upper left', 'bbox_to_anchor': (1.1, 1)},
            title={'label': 'AQI Category Distribution - Waffle Chart', 'loc': 'center', 'fontsize': 16},
            figsize=(12, 8)
        )
        
        # Set background
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        
        plot_base64 = plot_to_base64(fig)
        
        if plot_base64:
            return {
                'plot': plot_base64,
                'title': 'AQI Category Distribution - Waffle Chart',
                'description': f'Waffle chart showing the proportion of {len(waffle_data)} AQI categories, where each square represents 1% of the data.'
            }
        else:
            return None
            
    except Exception as e:
        logging.error(f"Error creating AQI waffle chart: {e}")
        return None