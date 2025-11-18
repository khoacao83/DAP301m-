import os
import pandas as pd
import numpy as np
import joblib
import logging
import base64
from io import BytesIO
from datetime import datetime
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
# Machine learning models for air quality prediction

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

class WeatherMLModels:
    def __init__(self):
        self.regression_models = {
            'Random Forest': RandomForestRegressor(
                bootstrap=True,
                max_depth=8,
                max_features='sqrt',
                min_samples_leaf=2,
                min_samples_split=5,
                n_estimators=50,  # Reduced from 500 to prevent timeout/memory issues
                random_state=42,
                n_jobs=1  # Single thread to avoid resource conflicts
            ),
            'KNN': KNeighborsRegressor(
                n_neighbors=30,
                p=1,
                weights='distance'
            ),
            'Gradient Boost': GradientBoostingRegressor(
                learning_rate=0.1,
                max_depth=3,
                n_estimators=50,  # Reduced from 100 to prevent memory issues
                subsample=0.8,
                random_state=42
            )
        }
        self.classification_models = {
            'Random Forest': RandomForestClassifier(
                bootstrap=True,
                max_depth=8,
                max_features='sqrt',
                min_samples_leaf=2,
                min_samples_split=5,
                n_estimators=50,
                random_state=42,
                n_jobs=1
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=30,
                p=1,
                weights='distance'
            ),
            'Gradient Boost': GradientBoostingClassifier(
                learning_rate=0.1,
                max_depth=3,
                n_estimators=50,
                subsample=0.8,
                random_state=42
            )
        }
        self.models = self.regression_models  # Default to regression
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.trained_model = None
        self.model_name = None
        self.feature_names = None
        self.target_name = None
        self.model_metrics = {}
        self.problem_type = 'regression'  # Default problem type
        
        # Shared data splits for consistent training across models
        self.shared_splits = None
        self.current_dataset_path = None
        self.current_features = None
        self.current_target = None
    
    def create_shared_splits(self, dataset_path, input_features, output_feature):
        """Create shared data splits that all models will use"""
        try:
            # Load dataset
            df = pd.read_csv(dataset_path)
            
            # Prepare data
            X, y = self.prepare_data(df, input_features, output_feature)
            
            # Split data for time series (no shuffle to maintain temporal order)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            # Reset and fit scaler on training data
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Store shared splits
            self.shared_splits = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled
            }
            
            # Store current configuration
            self.current_dataset_path = dataset_path
            self.current_features = input_features
            self.current_target = output_feature
            
            return True
            
        except Exception as e:
            logging.error(f"Error creating shared splits: {str(e)}")
            return False
    
    def get_or_create_shared_splits(self, dataset_path, input_features, output_feature):
        """Get existing shared splits or create new ones if configuration changed"""
        # Check if we need to create new splits
        if (self.shared_splits is None or 
            self.current_dataset_path != dataset_path or
            self.current_features != input_features or
            self.current_target != output_feature):
            
            if not self.create_shared_splits(dataset_path, input_features, output_feature):
                raise ValueError("Failed to create shared data splits")
        
        return self.shared_splits
        
    def prepare_data(self, df, input_features, output_feature):
        """Prepare data for training/prediction"""
        try:
            # Make a copy of the dataframe
            data = df.copy()
            
            # Handle missing values
            data = data.dropna(subset=input_features + [output_feature])
            
            if data.empty:
                raise ValueError("No valid data remaining after removing missing values")
            
            # Prepare features and target
            X = data[input_features].copy()
            y = data[output_feature].copy()
            
            # Handle categorical variables in features
            categorical_features = X.select_dtypes(include=['object']).columns
            for feature in categorical_features:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    X[feature] = self.label_encoders[feature].fit_transform(X[feature])
                else:
                    # For prediction, use existing encoder
                    try:
                        X[feature] = self.label_encoders[feature].transform(X[feature])
                    except ValueError:
                        # Handle unseen categories
                        X[feature] = X[feature].map(
                            lambda x: 0 if x not in self.label_encoders[feature].classes_ else 
                            self.label_encoders[feature].transform([x])[0]
                        )
            
            # Handle target variable based on problem type
            if self.problem_type == 'classification':
                # Convert continuous target to categorical classes for classification
                if pd.api.types.is_numeric_dtype(y):
                    # Create bins for continuous variables
                    n_bins = min(5, len(y.unique()))  # Max 5 classes, or number of unique values
                    if n_bins < 2:
                        n_bins = 2
                    
                    try:
                        # Use quantile-based binning for better class balance
                        y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
                        
                        # Store binning information for later use
                        if 'target_bins' not in self.label_encoders:
                            # Get the bin edges for reference
                            _, bin_edges = pd.qcut(y, q=n_bins, labels=False, duplicates='drop', retbins=True)
                            self.label_encoders['target_bins'] = bin_edges
                        
                        y = pd.Series(y_binned)
                        n_unique_classes = len(y.unique())
                        logging.info(f"Converted continuous target to {n_unique_classes} classification classes")
                    except Exception as binning_error:
                        logging.warning(f"Quantile binning failed: {binning_error}, using equal-width binning")
                        # Fallback to equal-width binning
                        y_binned = pd.cut(y, bins=n_bins, labels=False)
                        y = pd.Series(y_binned)
                        n_unique_classes = len(y.unique())
                        logging.info(f"Used equal-width binning, created {n_unique_classes} classes")
                elif y.dtype == 'object':
                    # Handle categorical target
                    if 'target' not in self.label_encoders:
                        self.label_encoders['target'] = LabelEncoder()
                        y = pd.Series(self.label_encoders['target'].fit_transform(y))
                    else:
                        try:
                            y = pd.Series(self.label_encoders['target'].transform(y))
                        except ValueError:
                            # Handle unseen categories in target
                            y = y.map(
                                lambda x: 0 if x not in self.label_encoders['target'].classes_ else 
                                self.label_encoders['target'].transform([x])[0]
                            )
            else:
                # For regression, handle only categorical targets
                if y.dtype == 'object':
                    if 'target' not in self.label_encoders:
                        self.label_encoders['target'] = LabelEncoder()
                        y = pd.Series(self.label_encoders['target'].fit_transform(y))
                    else:
                        try:
                            y = pd.Series(self.label_encoders['target'].transform(y))
                        except ValueError:
                            # Handle unseen categories in target
                            y = y.map(
                                lambda x: 0 if x not in self.label_encoders['target'].classes_ else 
                                self.label_encoders['target'].transform([x])[0]
                            )
            
            return X, y
            
        except Exception as e:
            logging.error(f"Error preparing data: {str(e)}")
            raise Exception(f"Data preparation failed: {str(e)}")
    
    def train_model(self, dataset_path, model_name, input_features, output_feature, problem_type='regression', epochs=100):
        """Train a machine learning model"""
        try:
            # Set problem type and select appropriate models
            self.problem_type = problem_type
            if problem_type == 'classification':
                self.models = self.classification_models
            else:
                self.models = self.regression_models
            
            # Load dataset
            df = pd.read_csv(dataset_path)
            logging.info(f"Loaded dataset with shape: {df.shape}")
            
            # Validate inputs
            if model_name not in self.models:
                raise ValueError(f"Unknown model: {model_name}")
            
            missing_features = set(input_features + [output_feature]) - set(df.columns)
            if missing_features:
                raise ValueError(f"Missing columns in dataset: {missing_features}")
            
            # Get or create shared data splits
            splits = self.get_or_create_shared_splits(dataset_path, input_features, output_feature)
            
            # Extract splits
            X_train = splits['X_train']
            X_test = splits['X_test']
            y_train = splits['y_train']
            y_test = splits['y_test']
            X_train_scaled = splits['X_train_scaled']
            X_test_scaled = splits['X_test_scaled']
            
            if len(X_train) < 5:
                raise ValueError("Insufficient data for training (need at least 5 samples)")
            
            # Train model
            model = self.models[model_name]
            
            # Set epochs for ensemble methods that support it
            if hasattr(model, 'n_estimators') and epochs != 100:
                model.set_params(n_estimators=min(epochs, 1000))
            
            model.fit(X_train_scaled, y_train)
            
            # Make predictions only on test set
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate metrics based on problem type
            if problem_type == 'classification':
                # Ensure predictions are integers for classification
                y_pred_test_int = np.round(y_pred_test).astype(int)
                y_test_int = y_test.astype(int)
                
                accuracy = accuracy_score(y_test_int, y_pred_test_int)
                
                # Calculate regression-like metrics for comparison (using original numeric predictions)
                mae = mean_absolute_error(y_test_int, y_pred_test_int)
                rmse = np.sqrt(mean_squared_error(y_test_int, y_pred_test_int))
                
                # R² score for classification (measures how well we predict class labels)
                try:
                    r2 = r2_score(y_test_int, y_pred_test_int)
                except:
                    r2 = 0.0  # Fallback if R² calculation fails for classification
                
                metrics = {
                    'Problem Type': 'Classification',
                    'Test Accuracy': float(accuracy),
                    'Test MAE': float(mae),
                    'Test RMSE': float(rmse),
                    'Test R²': float(r2),
                    'Training Samples': int(len(X_train)),
                    'Test Samples': int(len(X_test)),
                    'Number of Classes': int(len(np.unique(y_test_int)))
                }
            else:
                metrics = {
                    'Problem Type': 'Regression',
                    'Test RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
                    'Test MAE': float(mean_absolute_error(y_test, y_pred_test)),
                    'Test R²': float(r2_score(y_test, y_pred_test)),
                    'Training Samples': int(len(X_train)),
                    'Test Samples': int(len(X_test))
                }
            
            # Store model info
            self.trained_model = model
            self.model_name = model_name
            self.feature_names = input_features
            self.target_name = output_feature
            self.model_metrics = metrics
            
            # Generate interactive visualizations
            try:
                # Use appropriate data types for visualization
                if problem_type == 'classification':
                    vis_y_test = y_test_int
                    vis_y_pred = y_pred_test_int
                else:
                    vis_y_test = y_test
                    vis_y_pred = y_pred_test
                    
                visualizations = self.create_interactive_visualizations(
                    vis_y_test, vis_y_pred, model_name, input_features, problem_type
                )
            except Exception as viz_error:
                logging.warning(f"Visualization creation failed: {str(viz_error)}")
                visualizations = []  # Continue without visualizations
            
            # Save model
            model_filename = f"{model_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            model_path = os.path.join('trained_model', model_filename)
            
            model_data = {
                'model': model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_names': input_features,
                'target_name': output_feature,
                'model_name': model_name,
                'metrics': metrics
            }
            
            joblib.dump(model_data, model_path)
            logging.info(f"Model saved to: {model_path}")
            
            result = {
                'success': True,
                'model_name': str(model_name),
                'metrics': metrics,  # Already converted to native types above
                'visualizations': visualizations,  # HTML strings, should be safe
                'model_path': str(model_path),
                'feature_names': [str(f) for f in input_features],  # Convert to string list
                'target_name': str(output_feature)
            }
            
            # Apply safe serialization to the entire result
            return safe_json_serialize(result)
            
        except Exception as e:
            logging.error(f"Error training model: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def load_pretrained_model(self, dataset_path, model_name, input_features, output_feature):
        """Load or create a pre-trained model"""
        try:
            # Look for existing model file
            model_files = [f for f in os.listdir('trained_model') 
                          if f.startswith(model_name.lower().replace(' ', '_')) and f.endswith('.joblib')]
            
            if model_files:
                # Load most recent model
                model_files.sort(reverse=True)
                model_path = os.path.join('trained_model', model_files[0])
                
                try:
                    model_data = joblib.load(model_path)
                    
                    # Validate model compatibility
                    if (set(model_data['feature_names']) == set(input_features) and 
                        model_data['target_name'] == output_feature):
                        
                        self.trained_model = model_data['model']
                        self.scaler = model_data['scaler']
                        self.label_encoders = model_data['label_encoders']
                        self.model_name = model_data['model_name']
                        self.feature_names = model_data['feature_names']
                        self.target_name = model_data['target_name']
                        self.model_metrics = model_data['metrics']
                        
                        # Create visualizations with current data
                        df = pd.read_csv(dataset_path)
                        X, y = self.prepare_data(df, input_features, output_feature)
                        X_scaled = self.scaler.transform(X)
                        y_pred = self.trained_model.predict(X_scaled)
                        
                        visualizations = create_prediction_visualizations(y, y_pred, input_features, model_name)
                        
                        return {
                            'success': True,
                            'model_name': model_name,
                            'metrics': self.model_metrics,
                            'visualizations': visualizations,
                            'message': f'Loaded existing model: {model_files[0]}'
                        }
                
                except Exception as load_error:
                    logging.warning(f"Failed to load existing model: {load_error}")
            
            # Train new model if no compatible model found
            logging.info("No compatible pre-trained model found, training new model...")
            return self.train_model(dataset_path, model_name, input_features, output_feature)
            
        except Exception as e:
            logging.error(f"Error loading pre-trained model: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def make_prediction(self, input_data):
        """Make a prediction using the trained model"""
        try:
            if not self.trained_model:
                raise ValueError("No trained model available")
            
            # Convert input data to DataFrame
            df_input = pd.DataFrame([input_data])
            
            # Ensure all required features are present
            missing_features = set(self.feature_names) - set(df_input.columns)
            if missing_features:
                raise ValueError(f"Missing input features: {missing_features}")
            
            # Reorder columns to match training data
            df_input = df_input[self.feature_names]
            
            # Handle categorical variables
            for feature in df_input.columns:
                if feature in self.label_encoders and df_input[feature].dtype == 'object':
                    try:
                        df_input[feature] = self.label_encoders[feature].transform(df_input[feature])
                    except ValueError:
                        # Handle unseen categories
                        df_input[feature] = 0
            
            # Scale features
            X_scaled = self.scaler.transform(df_input)
            
            # Make prediction
            prediction = self.trained_model.predict(X_scaled)[0]
            
            # Decode prediction if target was categorical
            if 'target' in self.label_encoders:
                try:
                    prediction = self.label_encoders['target'].inverse_transform([int(prediction)])[0]
                except (ValueError, IndexError):
                    # If prediction is out of range, return as is
                    pass
            
            return {
                'success': True,
                'prediction': prediction,
                'input_features': input_data,
                'model_name': self.model_name
            }
            
        except Exception as e:
            logging.error(f"Error making prediction: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_interactive_visualizations(self, y_test, y_pred, model_name, input_features, problem_type='regression'):
        """Create matplotlib-based visualizations for model evaluation"""
        visualizations = []
        
        if problem_type == 'classification':
            try:
                # 1. Confusion Matrix Heatmap
                confusion_img = self.create_confusion_matrix_heatmap(y_test, y_pred, model_name)
                if confusion_img:
                    visualizations.append({
                        'type': 'confusion_matrix',
                        'title': 'Confusion Matrix Heatmap',
                        'image': confusion_img
                    })
            except Exception as e:
                logging.warning(f"Confusion matrix creation failed: {e}")
            
            try:
                # 2. Feature Importance Plot
                feature_importance_img = self.create_feature_importance_plot(model_name, input_features)
                if feature_importance_img:
                    visualizations.append({
                        'type': 'feature_importance',
                        'title': 'Feature Importance Plot',
                        'image': feature_importance_img
                    })
            except Exception as e:
                logging.warning(f"Feature importance plot failed: {e}")
                
            try:
                # 3. Class Distribution Bar Chart
                class_dist_img = self.create_class_distribution_chart(y_test, y_pred, model_name)
                if class_dist_img:
                    visualizations.append({
                        'type': 'class_distribution',
                        'title': 'Class Distribution Bar Chart',
                        'image': class_dist_img
                    })
            except Exception as e:
                logging.warning(f"Class distribution chart failed: {e}")
                
        else:  # regression
            try:
                # 1. Time Series Plot (Predicted vs Actual)
                time_series_img = self.create_time_series_plot(y_test, y_pred, model_name)
                if time_series_img:
                    visualizations.append({
                        'type': 'time_series',
                        'title': 'Actual vs Predicted Time Series',
                        'image': time_series_img
                    })
            except Exception as e:
                logging.warning(f"Time series plot failed: {e}")
            
            try:
                # 2. Residuals Analysis (Distribution and Q-Q only)
                residuals_img = self.create_residuals_plot(y_test, y_pred, model_name)
                if residuals_img:
                    visualizations.append({
                        'type': 'residuals',
                        'title': 'Residuals Analysis',
                        'image': residuals_img
                    })
            except Exception as e:
                logging.warning(f"Residuals plot failed: {e}")
        
        return visualizations
    

    def create_time_series_plot(self, y_test, y_pred, model_name):
        """Create time series plot showing actual vs predicted values"""
        try:
            # Calculate R²
            r2 = r2_score(y_test, y_pred)
            
            # Create matplotlib figure
            plt.figure(figsize=(12, 6))
            
            # Create time index
            time_index = range(len(y_test))
            
            # Plot actual and predicted lines
            plt.plot(time_index, y_test, 'b-', linewidth=2, label='Actual', alpha=0.8)
            plt.plot(time_index, y_pred, 'r-', linewidth=2, label='Predicted', alpha=0.8)
            
            plt.title(f'Actual vs Predicted Time Series - {model_name}\nR² = {r2:.3f}')
            plt.xlabel('Time Index')
            plt.ylabel('Values')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Convert to base64 image with reduced DPI
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logging.error(f"Error creating time series plot: {e}")
            return None
    
    def create_residuals_plot(self, y_test, y_pred, model_name):
        """Create simplified residuals analysis plot using matplotlib"""
        try:
            residuals = y_test - y_pred
            
            # Create matplotlib figure with 2 subplots only
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'Residuals Analysis - {model_name}', fontsize=16)
            
            # 1. Residuals Distribution (Histogram)
            ax1.hist(residuals, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
            ax1.set_xlabel('Residuals')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Residuals Distribution')
            ax1.grid(True, alpha=0.3)
            
            # 2. Q-Q Plot
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=ax2)
            ax2.set_title('Q-Q Plot (Normality Test)')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to base64 image with lower DPI to reduce memory usage
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logging.error(f"Error creating residuals plot: {e}")
            return None
    
    def create_confusion_matrix_heatmap(self, y_test, y_pred, model_name):
        """Create confusion matrix heatmap for classification"""
        try:
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Create matplotlib figure
            plt.figure(figsize=(8, 6))
            
            # Create heatmap
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       square=True, cbar_kws={'shrink': 0.8})
            
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            # Convert to base64 image
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logging.error(f"Error creating confusion matrix: {e}")
            return None
    
    def create_feature_importance_plot(self, model_name, input_features):
        """Create feature importance plot for tree-based models"""
        try:
            if not self.trained_model:
                return None
                
            # Get feature importance (only for tree-based models)
            if hasattr(self.trained_model, 'feature_importances_'):
                importances = self.trained_model.feature_importances_
                
                # Create matplotlib figure
                plt.figure(figsize=(10, 6))
                
                # Sort features by importance
                indices = np.argsort(importances)[::-1]
                
                # Create bar plot
                plt.bar(range(len(importances)), importances[indices])
                plt.xticks(range(len(importances)), [input_features[i] for i in indices], rotation=45)
                plt.title(f'Feature Importance - {model_name}')
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.tight_layout()
                
                # Convert to base64 image
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                image_data = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                return f"data:image/png;base64,{image_data}"
            else:
                # For models without feature_importances_ (like KNN), create a placeholder
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, f'Feature importance not available for {model_name}', 
                        ha='center', va='center', fontsize=14)
                plt.title(f'Feature Importance - {model_name}')
                plt.axis('off')
                
                # Convert to base64 image
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                image_data = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logging.error(f"Error creating feature importance plot: {e}")
            return None
    
    def create_class_distribution_chart(self, y_test, y_pred, model_name):
        """Create class distribution bar chart comparing actual vs predicted"""
        try:
            # For AQI data, ensure we use all 5 classes (1-5) in correct order
            aqi_classes = [1, 2, 3, 4, 5]
            
            # Count actual and predicted class distributions
            actual_counts = pd.Series(y_test).value_counts().reindex(aqi_classes, fill_value=0)
            pred_counts = pd.Series(y_pred).value_counts().reindex(aqi_classes, fill_value=0)
            
            # Create matplotlib figure with white background
            plt.figure(figsize=(12, 7))
            plt.style.use('default')
            
            # Bar positions
            x = np.arange(len(aqi_classes))
            width = 0.35
            
            # Define colors for better visualization
            actual_color = '#2E86AB'
            pred_color = '#A23B72'
            
            # Create bars
            plt.bar(x - width/2, actual_counts.values, width, label='Actual', 
                   alpha=0.8, color=actual_color, edgecolor='white', linewidth=0.5)
            plt.bar(x + width/2, pred_counts.values, width, label='Predicted', 
                   alpha=0.8, color=pred_color, edgecolor='white', linewidth=0.5)
            
            plt.title(f'AQI Class Distribution - {model_name}', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('AQI Values', fontsize=14, fontweight='bold')
            plt.ylabel('Count', fontsize=14, fontweight='bold')
            plt.xticks(x, aqi_classes, fontsize=12)
            plt.yticks(fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Set background colors
            plt.gca().set_facecolor('white')
            plt.gcf().patch.set_facecolor('white')
            
            # Convert to base64 image
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_data}"
            
        except Exception as e:
            logging.error(f"Error creating class distribution chart: {e}")
            return None
    

    


# Global instance
ml_models = WeatherMLModels()

def get_available_models():
    """Get list of available ML models"""
    return list(ml_models.models.keys())

def train_model(dataset_path, model_name, input_features, output_feature, problem_type='regression', epochs=100):
    """Train a machine learning model"""
    return ml_models.train_model(dataset_path, model_name, input_features, output_feature, problem_type, epochs)

def load_model(dataset_path, model_name, input_features, output_feature):
    """Load or create a pre-trained model"""
    return ml_models.load_pretrained_model(dataset_path, model_name, input_features, output_feature)

def make_prediction(input_data):
    """Make a prediction using the trained model"""
    return ml_models.make_prediction(input_data)

def get_model_info():
    """Get information about the currently loaded model"""
    if ml_models.trained_model:
        return {
            'model_name': ml_models.model_name,
            'feature_names': ml_models.feature_names,
            'target_name': ml_models.target_name,
            'metrics': ml_models.model_metrics
        }
    return None

def export_model(export_path=None):
    """Export the current model"""
    if not ml_models.trained_model:
        raise ValueError("No trained model to export")
    
    if not export_path:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_path = f"trained_model/exported_{ml_models.model_name.lower().replace(' ', '_')}_{timestamp}.joblib"
    
    model_data = {
        'model': ml_models.trained_model,
        'scaler': ml_models.scaler,
        'label_encoders': ml_models.label_encoders,
        'feature_names': ml_models.feature_names,
        'target_name': ml_models.target_name,
        'model_name': ml_models.model_name,
        'metrics': ml_models.model_metrics,
        'export_timestamp': datetime.now().isoformat()
    }
    
    joblib.dump(model_data, export_path)
    return export_path
