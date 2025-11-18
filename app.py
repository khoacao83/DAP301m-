import os
import logging
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
import pandas as pd
import json
from dotenv import load_dotenv
from utils.weather_api import get_current_weather, update_api_credentials
from utils.data_analysis import analyze_dataset, generate_visualizations
from utils.ml_models import train_model, load_model, make_prediction, get_available_models
from utils.negative_value_handling import check_negative_values, apply_negative_value_treatment
import joblib
from datetime import datetime
from ibm_watson import AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_cloud_sdk_core import ApiException
from utils.gemini_chatbot import send_message_to_gemini, get_gemini_config

# Load environment variables
load_dotenv()
print("Environment variables loaded from .env file")

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__, static_folder='build', static_url_path='')
app.secret_key = os.environ.get("SESSION_SECRET", "your-secret-key-here")

# Initialize IBM Watson Assistant (conditionally)
assistant = None
session_holder = []  # Store session_id in memory

try:
    assistant_apikey = os.getenv('WATSONX_API_KEY')
    assistant_url = os.getenv('WATSONX_SERVICE_URL')
    environment_id = os.getenv('WATSONX_ENVIRONMENT_ID')

    if assistant_apikey and assistant_url and environment_id:
        assistant_auth = IAMAuthenticator(assistant_apikey)
        assistant = AssistantV2(version='2024-08-25', authenticator=assistant_auth)
        assistant.set_service_url(assistant_url)
        logging.info("Watson Assistant initialized successfully")
    else:
        missing_vars = []
        if not assistant_apikey:
            missing_vars.append('WATSONX_API_KEY')
        if not assistant_url:
            missing_vars.append('WATSONX_SERVICE_URL')
        if not environment_id:
            missing_vars.append('WATSONX_ENVIRONMENT_ID')
        logging.warning(f"Watson Assistant credentials incomplete. Missing: {', '.join(missing_vars)}")
        assistant = None
except Exception as e:
    logging.error(f"Failed to initialize Watson Assistant: {e}")
    assistant = None

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('trained_model', exist_ok=True)

# Global dataframe storage for dual dataframe system
dataframe_storage = {
    'original': {},  # Store original dataframes by key
    'modified': {}   # Store modified dataframes by key
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_dataframe_key(dataset_type, location=None, days=None):
    """Generate a unique key for dataframe storage based on dataset type and parameters"""
    if dataset_type == 'search':
        return f"search_{location}_{days}"
    elif dataset_type == 'upload':
        return "upload"
    else:
        return "default"

def get_current_dataframe(dataset_type, data=None):
    """Get the current dataframe to use - modified if exists, otherwise original"""
    if data is None:
        data = {}
    
    key = get_dataframe_key(dataset_type, data.get('location'), data.get('days'))
    
    # Return modified dataframe if it exists, otherwise return original
    if key in dataframe_storage['modified']:
        return dataframe_storage['modified'][key]
    elif key in dataframe_storage['original']:
        return dataframe_storage['original'][key]
    else:
        return None

def store_original_dataframe(dataset_type, df, location=None, days=None):
    """Store the original dataframe"""
    key = get_dataframe_key(dataset_type, location, days)
    dataframe_storage['original'][key] = df.copy()
    # Remove any existing modified version when new original is stored
    if key in dataframe_storage['modified']:
        del dataframe_storage['modified'][key]

def store_modified_dataframe(dataset_type, df, location=None, days=None):
    """Store the modified dataframe"""
    key = get_dataframe_key(dataset_type, location, days)
    dataframe_storage['modified'][key] = df.copy()

@app.route('/')
def home():
    """Serve React frontend"""
    # Check if we're in development mode and local_index.html exists
    if app.debug and os.path.exists('local_index.html'):
        logging.info("Serving local development version")
        return send_from_directory('.', 'local_index.html')

    # Fallback to build version
    if os.path.exists('build/index.html'):
        return send_from_directory('build', 'index.html')
    else:
        # If no build exists, serve local version as fallback
        if os.path.exists('local_index.html'):
            return send_from_directory('.', 'local_index.html')
        else:
            return jsonify({
                'error': 'Frontend not found',
                'message': 'Please build the React frontend or ensure local_index.html exists'
            }), 404

@app.route('/notebook')
def notebook():
    """Serve the Jupyter notebook HTML"""
    return send_from_directory('static', 'notebook.html')

# Catch all route for React Router
@app.route('/<path:path>')
def catch_all(path):
    """Serve React app for client-side routing"""
    if path.startswith('api/'):
        return jsonify({'error': 'API endpoint not found'}), 404

    # Serve the same HTML file as the home route
    if app.debug and os.path.exists('local_index.html'):
        return send_from_directory('.', 'local_index.html')

    if os.path.exists('build/index.html'):
        return send_from_directory('build', 'index.html')
    else:
        if os.path.exists('local_index.html'):
            return send_from_directory('.', 'local_index.html')
        else:
            return jsonify({'error': 'Frontend not found'}), 404

@app.route('/api/weather/current')
def api_current_weather():
    """API endpoint for current weather data"""
    try:
        weather_data = get_current_weather("Ho Chi Minh City")
        return jsonify(weather_data)
    except Exception as e:
        logging.error(f"Error fetching weather data: {str(e)}")
        return jsonify({'error': f'Error fetching weather data: {str(e)}'}), 500

@app.route('/upload-dataset', methods=['POST'])
def upload_dataset():
    """Handle dataset upload and analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(str(file.filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Store the uploaded file path in session
            session['uploaded_dataset'] = filepath

            # Import pandas first
            import pandas as pd
            
            # Validate dataset has required air quality columns
            from utils.air_quality_data_fetch import AirQualityDataFetcher
            validator = AirQualityDataFetcher()
            df = pd.read_csv(filepath)
            is_valid, message = validator.validate_dataset(df)
            
            if not is_valid:
                return jsonify({
                    'success': False,
                    'error': f'Invalid dataset: {message}'
                }), 400

            # Analyze the dataset using the new analyzer
            from utils.data_analysis import WeatherDataAnalyzer
            analyzer = WeatherDataAnalyzer()
            analysis_result = analyzer.analyze_dataset(filepath)
            
            # Get dataset columns for feature selection
            columns = df.columns.tolist()

            return jsonify({
                'success': True,
                'analysis': analysis_result,
                'filename': filename,
                'columns': columns
            })
        else:
            return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400

    except Exception as e:
        logging.error(f"Error uploading dataset: {str(e)}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/analyze-default-dataset')
def analyze_default_dataset_legacy():
    """Legacy endpoint for default dataset analysis"""
    try:
        default_dataset = 'HCM_WEATHER_DEFAULT.csv'
        if not os.path.exists(default_dataset):
            return jsonify({'error': 'Default dataset not found'}), 404

        # Store the default dataset path in session
        session['uploaded_dataset'] = default_dataset

        analysis_result = analyze_dataset(default_dataset)
        return jsonify({
            'success': True,
            'analysis': analysis_result,
            'filename': 'HCM_WEATHER_DEFAULT.csv'
        })

    except Exception as e:
        logging.error(f"Error analyzing default dataset: {str(e)}")
        return jsonify({'error': f'Error analyzing dataset: {str(e)}'}), 500

@app.route('/weather-forecast')
def weather_forecast():
    """Weather forecast page with ML models"""
    available_models = get_available_models()
    return render_template('weather_forecast.html', available_models=available_models)

@app.route('/get-dataset-columns')
def get_dataset_columns():
    """Get columns from the current dataset"""
    try:
        dataset_path = session.get('uploaded_dataset')
        if not dataset_path or not os.path.exists(dataset_path):
            return jsonify({'error': 'No dataset available. Please upload a dataset first.'}), 400

        df = pd.read_csv(dataset_path)
        columns = df.columns.tolist()
        return jsonify({'columns': columns})

    except Exception as e:
        logging.error(f"Error getting dataset columns: {str(e)}")
        return jsonify({'error': f'Error reading dataset: {str(e)}'}), 500

@app.route('/train-model', methods=['POST'])
def train_model_route():
    """Train a new ML model"""
    try:
        data = request.get_json()
        dataset_path = session.get('uploaded_dataset')

        if not dataset_path or not os.path.exists(dataset_path):
            return jsonify({'error': 'No dataset available. Please upload a dataset first.'}), 400

        model_name = data.get('model_name')
        input_features = data.get('input_features', [])
        output_feature = data.get('output_feature')
        problem_type = data.get('problem_type', 'regression')  # Default to regression
        epochs = data.get('epochs', 100)

        if not all([model_name, input_features, output_feature]):
            return jsonify({'error': 'Missing required parameters'}), 400

        # Train the model
        result = train_model(dataset_path, model_name, input_features, output_feature, problem_type, epochs)

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error training model: {str(e)}")
        return jsonify({'error': f'Error training model: {str(e)}'}), 500

@app.route('/use-pretrained-model', methods=['POST'])
def use_pretrained_model():
    """Use a pre-trained model for prediction"""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        input_features = data.get('input_features', [])
        output_feature = data.get('output_feature')

        dataset_path = session.get('uploaded_dataset', 'HCM_WEATHER_DEFAULT.csv')

        if not os.path.exists(dataset_path):
            return jsonify({'error': 'Dataset not found'}), 400

        # Load or train the model
        result = load_model(dataset_path, model_name, input_features, output_feature)

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error using pre-trained model: {str(e)}")
        return jsonify({'error': f'Error using model: {str(e)}'}), 500

@app.route('/make-prediction', methods=['POST'])
def make_prediction_route():
    """Make a prediction using the trained model"""
    try:
        data = request.get_json()
        input_data = data.get('input_data', {})

        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400

        # Make prediction
        result = make_prediction(input_data)

        return jsonify(result)

    except Exception as e:
        logging.error(f"Error making prediction: {str(e)}")
        return jsonify({'error': f'Error making prediction: {str(e)}'}), 500

@app.route('/api-settings')
def api_settings():
    """API settings page"""
    return render_template('api_settings.html')

@app.route('/update-api-settings', methods=['POST'])
def update_api_settings():
    """Update API settings"""
    try:
        data = request.get_json()

        # Update environment variables
        update_api_credentials(data)

        flash('API settings updated successfully!', 'success')
        return jsonify({'success': True, 'message': 'API settings updated successfully!'})

    except Exception as e:
        logging.error(f"Error updating API settings: {str(e)}")
        return jsonify({'error': f'Error updating settings: {str(e)}'}), 500

@app.route('/api/message', methods=['POST'])
def assistant_message():
    """Handle chatbot messages for both Watson and Gemini"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text in request'}), 400

        text = data['text'].strip()
        chatbot_type = data.get('chatbot_type', 'watson')  # Default to Watson
        conversation_history = data.get('conversation_history', [])
        
        # Route to appropriate chatbot
        if chatbot_type == 'gemini':
            # Handle Gemini chatbot
            result = send_message_to_gemini(text, conversation_history)
            if result.get('success'):
                return jsonify([{"text": result['response']}])
            else:
                return jsonify({'error': result.get('error', 'Gemini error occurred')}), 500
        
        else:
            # Handle Watson Assistant (existing logic)
            if not assistant:
                return jsonify({'error': 'Watson Assistant not initialized.'}), 500

            # Try to get session ID from the list
            session_id = session_holder[0] if session_holder else None

            def send_message_with_session(session_id):
                environment_id = os.getenv("WATSONX_ENVIRONMENT_ID")
                return assistant.message(
                    assistant_id=environment_id,
                    environment_id=environment_id,
                    session_id=session_id,
                    input={"message_type": "text", "text": text}
                ).get_result()

            try:
                # Try with existing session
                if session_id:
                    response = send_message_with_session(session_id)
                else:
                    raise KeyError

            except (ApiException, KeyError) as e:
                # If session invalid or missing, create new and retry
                if isinstance(e, ApiException) and e.code != 404:
                    raise  # Only retry on expired session

                # Create new session
                environment_id = os.getenv("WATSONX_ENVIRONMENT_ID")
                new_session_id = assistant.create_session(assistant_id=environment_id).get_result()["session_id"]

                # Store it (overwrite the list)
                session_holder.clear()
                session_holder.append(new_session_id)

                # Retry message
                response = send_message_with_session(new_session_id)

            return jsonify(response["output"]["generic"])

    except Exception as e:
        logging.error(f"Error in assistant_message: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/chatbot-config')
def chatbot_config():
    """Get chatbot configuration for both Watson and Gemini"""
    # Watson Assistant configuration
    assistant_apikey = os.getenv('WATSONX_API_KEY', '')
    assistant_url = os.getenv('WATSONX_SERVICE_URL', '')
    environment_id = os.getenv('WATSONX_ENVIRONMENT_ID', '')
    
    watson_has_credentials = bool(assistant_apikey) and bool(assistant_url) and bool(environment_id)
    watson_is_initialized = assistant is not None and watson_has_credentials
    
    # Gemini configuration
    gemini_config = get_gemini_config()

    config = {
        'chatbots': {
            'watson': {
                'name': 'Watson Assistant',
                'initialized': watson_is_initialized,
                'configured': watson_has_credentials,
                'status': 'ready' if watson_is_initialized else 'not_configured',
                'message': 'Watson Assistant is ready' if watson_is_initialized else 'Watson Assistant not configured'
            },
            'gemini': {
                'name': 'Gemini',
                'initialized': gemini_config['initialized'],
                'configured': gemini_config['configured'],
                'status': 'ready' if gemini_config['initialized'] else 'not_configured',
                'message': 'Gemini is ready' if gemini_config['initialized'] else 'Gemini not configured'
            }
        },
        'default_chatbot': 'watson'  # Default selection
    }

    logging.info(f"Chatbot config requested - Watson: {watson_is_initialized}, Gemini: {gemini_config['initialized']}")

    # Add CORS headers for local development
    response = jsonify(config)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/api/debug/env')
def debug_env():
    """Debug endpoint to check environment variables (for development only)"""
    if not app.debug:
        return jsonify({'error': 'Debug endpoint only available in development mode'}), 403

    return jsonify({
        'WATSONX_API_KEY': 'SET' if os.getenv('WATSONX_API_KEY') else 'NOT SET',
        'WATSONX_SERVICE_URL': 'SET' if os.getenv('WATSONX_SERVICE_URL') else 'NOT SET', 
        'WATSONX_ENVIRONMENT_ID': 'SET' if os.getenv('WATSONX_ENVIRONMENT_ID') else 'NOT SET',
        'assistant_initialized': assistant is not None,
        'debug_mode': app.debug
    })

# Data Analysis API Endpoints
@app.route('/api/data-analysis/default')
def analyze_default_dataset():
    """Analyze the default air quality dataset"""
    try:
        from utils.data_analysis import WeatherDataAnalyzer
        
        # Use the new air quality dataset
        default_path = 'uploads/default/HCMC_Air_Quality_capped.csv'
        
        if not os.path.exists(default_path):
            return jsonify({
                'success': False,
                'error': 'Default air quality dataset not found'
            }), 404
        
        # Analyze the dataset
        analyzer = WeatherDataAnalyzer()
        results = analyzer.analyze_dataset(default_path)
        
        # Also check for negative values
        from utils.negative_value_handling import check_negative_values as check_neg_vals
        df = pd.read_csv(default_path)
        negative_values_info = check_neg_vals(df)
        
        # Add negative values info to results
        results['negative_values'] = negative_values_info
        
        return jsonify({
            'success': True,
            'analysis': results
        })
        
    except Exception as e:
        logging.error(f"Error analyzing default dataset: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/data-analysis/search', methods=['POST'])
def analyze_search_dataset():
    """Analyze air quality data for a searched location"""
    try:
        from utils.data_analysis import WeatherDataAnalyzer
        from utils.air_quality_data_fetch import AirQualityDataFetcher
        
        data = request.get_json()
        location = data.get('location', '').strip()
        days = data.get('days', 30)
        
        if not location:
            return jsonify({
                'success': False,
                'error': 'Location is required'
            }), 400
        
        if not isinstance(days, int) or days < 1:
            return jsonify({
                'success': False,
                'error': 'Days must be a positive number'
            }), 400
        
        # Fetch air quality data for the location
        fetcher = AirQualityDataFetcher()
        df = fetcher.fetch_air_quality_data(location, days=days)
        
        if df is None:
            return jsonify({
                'success': False,
                'error': f'Could not fetch air quality data for location: {location}'
            }), 400
        
        # Analyze the dataset
        analyzer = WeatherDataAnalyzer()
        results = analyzer.analyze_dataset(df)
        
        # Also check for negative values
        from utils.negative_value_handling import check_negative_values as check_neg_vals
        negative_values_info = check_neg_vals(df)
        
        # Add negative values info to results
        results['negative_values'] = negative_values_info
        
        return jsonify({
            'success': True,
            'analysis': results
        })
        
    except Exception as e:
        logging.error(f"Error analyzing search dataset: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/data-analysis/remove-columns', methods=['POST'])
def remove_columns_from_dataset():
    """Remove specified columns from the current dataset"""
    try:
        import pandas as pd
        from utils.data_analysis import WeatherDataAnalyzer
        from utils.air_quality_data_fetch import AirQualityDataFetcher
        
        data = request.get_json()
        columns_to_remove = data.get('columns_to_remove', [])
        dataset_type = data.get('dataset_type', 'default')  # default, search, or upload
        
        if not columns_to_remove:
            return jsonify({
                'success': False,
                'error': 'No columns specified for removal'
            }), 400
        
        # Get the current dataset using dual dataframe system
        df = get_current_dataset(dataset_type, data)
        
        if df is None:
            return jsonify({
                'success': False,
                'error': 'No dataset available'
            }), 400
        
        # Remove specified columns (but keep required air quality columns)
        validator = AirQualityDataFetcher()
        required_columns = validator.required_columns + ['datetime', 'aqi']
        
        # Filter out required columns from removal list
        safe_to_remove = [col for col in columns_to_remove if col not in required_columns]
        cannot_remove = [col for col in columns_to_remove if col in required_columns]
        
        if cannot_remove:
            return jsonify({
                'success': False,
                'error': f'Cannot remove required columns: {", ".join(cannot_remove)}'
            }), 400
        
        # Remove columns from dataframe
        df_modified = df.drop(columns=safe_to_remove, errors='ignore')
        
        # Store the modified dataframe
        if dataset_type == 'search':
            location = data.get('location', 'Ho Chi Minh City')
            days = data.get('days', 30)
            store_modified_dataframe('search', df_modified, location, days)
        else:
            store_modified_dataframe(dataset_type, df_modified)
        
        # Re-analyze the updated dataset
        analyzer = WeatherDataAnalyzer()
        results = analyzer.analyze_dataset(df_modified)
        
        # Also check for negative values
        from utils.negative_value_handling import check_negative_values as check_neg_vals
        negative_values_info = check_neg_vals(df_modified)
        
        # Add negative values info to results
        results['negative_values'] = negative_values_info
        
        return jsonify({
            'success': True,
            'analysis': results,
            'removed_columns': safe_to_remove
        })
        
    except Exception as e:
        logging.error(f"Error removing columns: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/outlier-handling/capping', methods=['POST'])
def handle_outliers_capping():
    """Handle outliers using capping method"""
    try:
        from utils.outlier_handling import apply_capping
        from utils.data_analysis import WeatherDataAnalyzer
        
        data = request.get_json()
        dataset_type = data.get('dataset_type', 'default')
        
        # Get the current dataset
        df = get_current_dataset(dataset_type, data)
        if df is None:
            return jsonify({'success': False, 'error': 'No dataset available'}), 400
        
        # Apply capping
        df_capped = apply_capping(df)
        
        # Store the modified dataframe
        if dataset_type == 'search':
            location = data.get('location', 'Ho Chi Minh City')
            days = data.get('days', 30)
            store_modified_dataframe('search', df_capped, location, days)
        else:
            store_modified_dataframe(dataset_type, df_capped)
        
        # Re-analyze the dataset
        analyzer = WeatherDataAnalyzer()
        results = analyzer.analyze_dataset(df_capped)
        
        return jsonify({
            'success': True,
            'analysis': results,
            'method': 'capping'
        })
        
    except Exception as e:
        logging.error(f"Error applying capping: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/api/outlier-handling/deletion', methods=['POST'])
def handle_outliers_deletion():
    """Handle outliers using deletion method"""
    try:
        from utils.outlier_handling import apply_deletion
        from utils.data_analysis import WeatherDataAnalyzer
        
        data = request.get_json()
        dataset_type = data.get('dataset_type', 'default')
        
        # Get the current dataset
        df = get_current_dataset(dataset_type, data)
        if df is None:
            return jsonify({'success': False, 'error': 'No dataset available'}), 400
        
        # Apply deletion
        df_cleaned = apply_deletion(df)
        
        # Store the modified dataframe
        if dataset_type == 'search':
            location = data.get('location', 'Ho Chi Minh City')
            days = data.get('days', 30)
            store_modified_dataframe('search', df_cleaned, location, days)
        else:
            store_modified_dataframe(dataset_type, df_cleaned)
        
        # Re-analyze the dataset
        analyzer = WeatherDataAnalyzer()
        results = analyzer.analyze_dataset(df_cleaned)
        
        return jsonify({
            'success': True,
            'analysis': results,
            'method': 'deletion'
        })
        
    except Exception as e:
        logging.error(f"Error applying deletion: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/check-negative-values', methods=['POST'])
def check_negative_values_api():
    """Check if dataset contains negative values"""
    try:
        data = request.get_json()
        dataset_type = data.get('dataset_type', 'default')
        
        # Get the current dataset
        df = get_current_dataset(dataset_type, data)
        if df is None:
            return jsonify({'success': False, 'error': 'No dataset available'}), 400
        
        # Check for negative values using the imported function
        from utils.negative_value_handling import check_negative_values as check_neg_vals
        result = check_neg_vals(df)
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Error checking negative values: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/handle-negative-values', methods=['POST'])
def handle_negative_values_api():
    """Handle negative values using specified method"""
    try:
        from utils.data_analysis import WeatherDataAnalyzer
        
        data = request.get_json()
        method = data.get('method', 'none')
        dataset_type = data.get('dataset_type', 'default')
        
        # Get the current dataset
        df = get_current_dataset(dataset_type, data)
        if df is None:
            return jsonify({'success': False, 'error': 'No dataset available'}), 400
        
        # Apply negative value treatment
        df_processed, processing_info = apply_negative_value_treatment(df, method)
        
        # Only store modified dataframe if method is not 'none'
        if method != 'none':
            if dataset_type == 'search':
                location = data.get('location', 'Ho Chi Minh City')
                days = data.get('days', 30)
                store_modified_dataframe('search', df_processed, location, days)
            else:
                store_modified_dataframe(dataset_type, df_processed)
        
        # Re-analyze the dataset
        analyzer = WeatherDataAnalyzer()
        results = analyzer.analyze_dataset(df_processed)
        
        return jsonify({
            'success': True,
            'analysis': results,
            'method': method,
            'processing_info': processing_info
        })
        
    except Exception as e:
        logging.error(f"Error handling negative values: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500



def get_current_dataset(dataset_type, data):
    """Helper function to get current dataset - loads from source if not in storage"""
    import pandas as pd
    
    # First try to get from storage (dual dataframe system)
    df = get_current_dataframe(dataset_type, data)
    if df is not None:
        return df
    
    # If not in storage, load from source and store as original
    if dataset_type == 'default':
        default_path = 'uploads/default/HCMC_Air_Quality_capped.csv'
        if os.path.exists(default_path):
            df = pd.read_csv(default_path)
            store_original_dataframe('default', df)
            return df
    elif dataset_type == 'upload':
        dataset_path = session.get('uploaded_dataset')
        if dataset_path and os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            store_original_dataframe('upload', df)
            return df
    elif dataset_type == 'search':
        from utils.air_quality_data_fetch import AirQualityDataFetcher
        location = data.get('location', 'Ho Chi Minh City')
        days = data.get('days', 30)
        
        fetcher = AirQualityDataFetcher()
        df = fetcher.fetch_air_quality_data(location, days=days)
        if df is not None:
            store_original_dataframe('search', df, location, days)
        return df
    
    return None

@app.route('/api/area-plot', methods=['POST'])
def generate_area_plot():
    """Generate area plot for selected features"""
    try:
        from utils.visualization import create_area_plot
        
        data = request.get_json()
        features = data.get('features', [])
        dataset_type = data.get('dataset_type', 'default')
        outlier_handling = data.get('outlier_handling')
        
        if not features:
            return jsonify({
                'success': False,
                'error': 'No features specified'
            }), 400
        
        # Get the current dataset
        df = get_current_dataset(dataset_type, data)
        if df is None:
            return jsonify({
                'success': False,
                'error': 'No dataset available'
            }), 400
        
        # Apply outlier handling if specified
        if outlier_handling and outlier_handling != 'none':
            if outlier_handling == 'capping':
                from utils.outlier_handling import apply_capping
                df = apply_capping(df)

            elif outlier_handling == 'deletion':
                from utils.outlier_handling import apply_deletion
                df = apply_deletion(df)
        
        # Check if features exist in dataset
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            return jsonify({
                'success': False,
                'error': f'Features not found in dataset: {", ".join(missing_features)}'
            }), 400
        
        # Generate area plot
        plot_data = create_area_plot(df, features)
        
        if plot_data is None:
            return jsonify({
                'success': False,
                'error': 'Failed to generate area plot'
            }), 500
        
        return jsonify({
            'success': True,
            'plot_data': plot_data
        })
        
    except Exception as e:
        logging.error(f"Error generating area plot: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/export-modified-dataset', methods=['POST'])
def export_modified_dataset():
    """Export modified dataset as CSV file"""
    try:
        from flask import make_response
        import io
        from datetime import datetime
        
        data = request.get_json()
        dataset_type = data.get('dataset_type', 'default')
        
        # Get the current dataset (modified if available, otherwise original)
        df = get_current_dataset(dataset_type, data)
        if df is None:
            return jsonify({'success': False, 'error': 'No dataset available'}), 400
        
        # Generate filename based on dataset type and current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if dataset_type == 'search':
            location = data.get('location', 'Unknown_Location').replace(' ', '_')
            days = data.get('days', 30)
            filename = f"{location}_AirQuality_{days}days_modified_{timestamp}.csv"
        elif dataset_type == 'upload':
            filename = f"uploaded_dataset_modified_{timestamp}.csv"
        else:
            filename = f"default_dataset_modified_{timestamp}.csv"
        
        # Convert DataFrame to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        csv_data = output.getvalue()
        output.close()
        
        # Create response
        response = make_response(csv_data)
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        response.headers["Content-type"] = "text/csv"
        
        return response
        
    except Exception as e:
        logging.error(f"Error exporting modified dataset: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/export-raw-dataset', methods=['POST'])
def export_raw_dataset():
    """Export raw dataset as CSV file"""
    try:
        from flask import make_response
        import io
        
        data = request.get_json()
        dataset_type = data.get('dataset_type', 'default')
        
        # Get the current dataset
        df = get_current_dataset(dataset_type, data)
        if df is None:
            return jsonify({
                'success': False,
                'error': 'No dataset available for export'
            }), 400
        
        # Convert DataFrame to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        csv_data = output.getvalue()
        output.close()
        
        # Create response with CSV data
        response = make_response(csv_data)
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = 'attachment; filename=raw_dataset.csv'
        
        return response
        
    except Exception as e:
        logging.error(f"Error exporting raw dataset: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/export-model', methods=['POST'])
def export_model_api():
    """Export trained ML model with download"""
    try:
        from utils.ml_models import export_model, get_model_info
        from flask import send_file
        import tempfile
        import shutil
        
        # Check if there's a trained model
        model_info = get_model_info()
        if not model_info:
            return jsonify({
                'success': False,
                'error': 'No trained model available for export'
            }), 400
        
        # Export the model
        model_path = export_model()
        
        # Copy to temporary location for download
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, os.path.basename(model_path))
        shutil.copy2(model_path, temp_file)
        
        return send_file(
            temp_file,
            as_attachment=True,
            download_name=f"{model_info['model_name'].lower().replace(' ', '_')}_model.joblib",
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        logging.error(f"Error exporting model: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500



@app.route('/api/upload-models', methods=['POST'])
def upload_models():
    """Handle multiple model file uploads"""
    try:
        if 'models[]' not in request.files:
            return jsonify({'success': False, 'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('models[]')
        if not files:
            return jsonify({'success': False, 'error': 'No files selected'}), 400
        
        uploaded_paths = {}
        upload_dir = 'uploads/models'
        os.makedirs(upload_dir, exist_ok=True)
        
        for file in files:
            if file.filename == '':
                continue
                
            # Check file extension
            if not file.filename.lower().endswith(('.joblib', '.pkl')):
                return jsonify({
                    'success': False, 
                    'error': f'Invalid file type: {file.filename}. Only .joblib and .pkl files are allowed.'
                }), 400
            
            # Save file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{filename}"
            file_path = os.path.join(upload_dir, unique_filename)
            
            file.save(file_path)
            uploaded_paths[file.filename] = file_path
            
            logging.info(f"Model uploaded: {file_path}")
        
        return jsonify({
            'success': True,
            'message': f'Successfully uploaded {len(uploaded_paths)} model(s)',
            'paths': uploaded_paths
        })
        
    except Exception as e:
        logging.error(f"Error uploading models: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/model-inference', methods=['POST'])
def run_model_inference():
    """Run inference on selected model for specified duration"""
    try:
        data = request.get_json()
        model_selection = data.get('model_selection')
        duration = data.get('duration', 1)
        
        if not model_selection:
            return jsonify({'success': False, 'error': 'Model selection is required'}), 400
        
        # Parse model selection (format: "trained:model_name" or "uploaded:filename")
        selection_parts = model_selection.split(':', 1)
        if len(selection_parts) != 2:
            return jsonify({'success': False, 'error': 'Invalid model selection format'}), 400
        
        model_type, model_identifier = selection_parts
        
        # Load the appropriate model
        if model_type == 'trained':
            # Load from trained_model directory
            model_files = [f for f in os.listdir('trained_model') if f.startswith(model_identifier.lower()) and f.endswith('.joblib')]
            if not model_files:
                return jsonify({'success': False, 'error': f'Trained model {model_identifier} not found'}), 400
            
            model_path = os.path.join('trained_model', model_files[0])
            
        elif model_type == 'uploaded':
            # Load from uploads/models directory
            upload_dir = 'uploads/models'
            uploaded_files = [f for f in os.listdir(upload_dir) if f.endswith(model_identifier)]
            if not uploaded_files:
                return jsonify({'success': False, 'error': f'Uploaded model {model_identifier} not found'}), 400
            
            model_path = os.path.join(upload_dir, uploaded_files[0])
            
        else:
            return jsonify({'success': False, 'error': 'Invalid model type'}), 400
        
        # Load the model
        try:
            import joblib
            model = joblib.load(model_path)
            logging.info(f"Model loaded from: {model_path}")
        except Exception as e:
            return jsonify({'success': False, 'error': f'Failed to load model: {str(e)}'}), 400
        
        # Generate sample input data for predictions
        # Use the default dataset to get feature structure
        import numpy as np
        default_dataset_path = 'HCMC_Air_Quality_capped.csv'
        if os.path.exists(default_dataset_path):
            df = pd.read_csv(default_dataset_path)
            
            # Remove target columns and prepare features
            feature_cols = [col for col in df.columns if col not in ['aqi', 'datetime', 'date', 'time']]
            if len(feature_cols) == 0:
                return jsonify({'success': False, 'error': 'No valid feature columns found'}), 400
            
            # Generate predictions for the specified duration
            predictions = []
            latest_features = df[feature_cols].iloc[-1:].values  # Use latest data point as base
            
            for day in range(duration):
                try:
                    # Add some variation to simulate time progression
                    input_features = latest_features.copy()
                    if day > 0:
                        # Add small random variations for future predictions
                        noise = np.random.normal(0, 0.01, input_features.shape)
                        input_features = input_features + noise
                    
                    pred = model.predict(input_features)[0]
                    predictions.append(float(pred))
                    
                except Exception as e:
                    logging.warning(f"Prediction failed for day {day}: {e}")
                    predictions.append(None)
        
        else:
            return jsonify({'success': False, 'error': 'Default dataset not found for feature structure'}), 400
        
        return jsonify({
            'success': True,
            'model_name': model_identifier,
            'duration': duration,
            'predictions': predictions,
            'feature_count': len(feature_cols)
        })
        
    except Exception as e:
        logging.error(f"Error in model inference: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
