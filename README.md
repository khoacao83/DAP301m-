# Weather Dashboard

A comprehensive Flask-based weather dashboard with React frontend featuring machine learning forecasting, data analysis, and IBM Watson chatbot integration.

## 🌟 Features

- **Real-time Weather Data**: AccuWeather API integration for accurate current conditions
- **Machine Learning Forecasting**: Scikit-learn models for weather prediction
- **Data Analysis**: CSV upload and comprehensive weather data visualization
- **IBM Watson Chatbot**: Integrated AI assistant for weather queries
- **Interactive UI**: Modern React frontend with ShadCN UI components
- **4 Main Tabs**: Home, Weather Data, Weather Forecast, API Settings

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [API Configuration](#api-configuration)
- [Development](#development)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)
- [Deployment](#deployment)

## 🚀 Quick Start

### 1. Installation

```bash
# Install Python dependencies
uv sync

# Install Node.js dependencies
npm install
```

### 2. Environment Configuration

Create a `.env` file in the root directory:

```bash
# Weather API Configuration
ACCUWEATHER_API_KEY=Wrl3dIGqBKZpDpvbvATG7Q852oGPlhpV
OPENWEATHER_API_KEY=8589b6a9dacaa1c18c226c0bc4e295a5

# Watsonx Assistant Configuration
WATSONX_API_KEY=YOUR_API_KEY
WATSONX_SERVICE_URL=YOUR_URL
WATSONX_ASSISTANT_ID=YOUR_ASSISTANT_ID
WATSONX_ENVIRONMENT_ID=YOUR_ENVIRONMENT_ID

# Flask Configuration
SESSION_SECRET=YOUR_SESSION_SECRET_KEY

# Gemini Configuration
GEMINI_API_KEY=YOUR_API_KEY
```

### 3. Run the Application

```bash
# Start the Flask backend server
gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app

# Or for development
python main.py
```

The application will be available at `http://localhost:5000`

## 📁 Project Structure

```
weather-dashboard/
├── app.py                 # Flask backend application
├── main.py               # Application entry point
├── pyproject.toml        # Python dependencies
├── package.json          # Node.js dependencies
├── build/
│   └── index.html        # Compiled React frontend
├── src/                  # React source components
│   ├── components/       # React UI components
│   ├── App.tsx          # Main React app
│   └── types/           # TypeScript definitions
├── static/              # Static assets
│   └── notebook.html    # Jupyter notebook HTML
├── utils/               # Python utilities
│   ├── data_analysis.py # Data analysis functions
│   ├── ml_models.py     # Machine learning models
│   └── weather_api.py   # Weather API integration
├── uploads/             # File upload directory
├── trained_model/       # ML model storage
└── HCM_WEATHER_DEFAULT.csv # Default dataset
```

## 📦 Dependencies

### Python Dependencies (Backend)

#### Core Framework
- **flask** >=3.1.1 - Web framework
- **werkzeug** >=3.1.3 - WSGI utilities
- **gunicorn** >=23.0.0 - Production WSGI server

#### Data Science & Machine Learning
- **pandas** >=2.3.0 - Data manipulation and analysis
- **numpy** >=2.3.0 - Numerical computing
- **scikit-learn** >=1.7.0 - Machine learning algorithms
- **joblib** >=1.5.1 - Model serialization
- **plotly** >=6.1.2 - Interactive charts and graphs

#### APIs & HTTP
- **requests** >=2.32.3 - HTTP library
- **python-dotenv** >=1.1.0 - Environment variable management

#### IBM Watson Assistant
- **ibm-watson** >=10.0.0 - Watson services SDK
- **ibm-cloud-sdk-core** >=3.24.2 - IBM Cloud SDK core

### Node.js Dependencies (Frontend)

#### Core React Framework
- **react** ^19.1.0 - UI library
- **react-dom** ^19.1.0 - React DOM rendering
- **react-scripts** ^5.0.1 - Create React App scripts
- **typescript** ^4.9.5 - TypeScript support

#### UI Components & Styling
- **@radix-ui/react-slot** ^1.2.3 - Slot primitive
- **@radix-ui/react-tabs** ^1.1.12 - Tab components
- **tailwindcss** ^4.1.11 - CSS framework
- **lucide-react** ^0.525.0 - Icon library
- **recharts** ^3.0.2 - React chart library

#### Utilities
- **axios** ^1.10.0 - HTTP client
- **date-fns** ^4.1.0 - Date utility library

### Environment Requirements
- **Python**: >=3.11
- **Node.js**: >=16.0.0
- **npm**: >=8.0.0

## 🔧 API Configuration

### AccuWeather API Setup

#### Getting Your API Key

1. **Create an AccuWeather Account**
   - Go to [AccuWeather Developer Portal](https://developer.accuweather.com/)
   - Click "Register" and fill in your information
   - Company name can be personal use

2. **Create an App**
   - Click "My Apps" → "Add a new App"
   - App Name: Weather Dashboard
   - Package Name: com.weather.dashboard
   - Product: LocationIQ (free tier)

3. **Get Your API Key**
   - Copy the 32-character API key
   - Add to your `.env` file as `ACCUWEATHER_API_KEY`

#### AccuWeather Features

**Free Tier (50 calls/day):**
- Current conditions
- 5-day daily forecasts
- Location search
- Weather alerts

**Paid Tiers:**
- Higher rate limits (1000+ calls/day)
- Hourly forecasts
- Historical weather data
- Severe weather alerts

### IBM Watson Assistant Setup

#### Getting Your Credentials

1. **Create IBM Cloud Account**
   - Go to [IBM Cloud](https://cloud.ibm.com/)
   - Create a free account

2. **Create Watson Assistant Service**
   - Navigate to AI/Watson services
   - Create a Watson Assistant instance
   - Get your API key, service URL, and environment ID

3. **Add to Environment Variables**
   ```bash
   ASSISTANT_APIKEY=your_watson_api_key
   ASSISTANT_URL=your_watson_service_url
   ENVIRONMENT_ID=your_watson_environment_id
   ```

## 💻 Development

### Backend Development (Python/Flask)
- Main application: `app.py`
- API endpoints serve both data and React frontend
- Watson Assistant integration handles chatbot functionality
- ML models for weather prediction and data analysis

### Frontend Development (React/TypeScript)
- Source code in `src/` directory
- Built application served from `build/index.html`
- Modern React with TypeScript and Tailwind CSS
- ShadCN UI components for consistent design

### Key Features by Tab

1. **Home Tab**: Current weather conditions with Apple-style cards
2. **Weather Data Tab**: CSV upload, analysis, and embedded Jupyter notebook
3. **Weather Forecast Tab**: ML-based predictions and forecasting
4. **API Settings Tab**: Configuration for AccuWeather and Watson
5. **Watson Chatbot**: Floating chat interface for weather queries

## 🔗 API Endpoints

### Weather API
- `GET /api/current-weather` - Current weather data
- `GET /api/weather-forecast` - Weather forecast

### Data Analysis
- `POST /api/upload-dataset` - Upload and analyze CSV data
- `GET /api/analyze-default-dataset` - Analyze default dataset
- `GET /api/get-dataset-columns` - Get available columns

### Machine Learning
- `POST /api/train-model` - Train new ML model
- `POST /api/use-pretrained-model` - Load existing model
- `POST /api/make-prediction` - Make weather predictions

### Watson Assistant
- `POST /api/message` - Send message to chatbot
- `GET /chatbot-config` - Get chatbot configuration

### Static Files
- `GET /notebook` - Serve embedded Jupyter notebook

## 🔍 Troubleshooting

### Common Issues

#### 1. AccuWeather API Issues

**"API key not configured"**
```bash
# Check your .env file
cat .env | grep ACCUWEATHER_API_KEY
# Should show: ACCUWEATHER_API_KEY=your_key_here
```

**"Rate limit exceeded"**
- You've exceeded 50 calls per day (free tier)
- Wait until next day or upgrade your plan
- Check for unnecessary repeated calls

**"Location not found"**
- Try different city name formats
- Use "City, State" for US cities
- Use "City, Country" for international cities
- Examples: "New York, NY", "London, UK", "Ho Chi Minh City, Vietnam"

#### 2. Watson Assistant Issues

**"Watson Assistant not configured"**

1. **Verify Backend Configuration**
   ```bash
   curl http://localhost:5000/chatbot-config
   # Should return: {"initialized": true, "configured": true, "status": "ready"}
   ```

2. **Check .env File Format**
   ```bash
   # Correct format (no spaces around =)
   ASSISTANT_APIKEY=your_watson_api_key_here
   ASSISTANT_URL=your_watson_service_url_here
   ENVIRONMENT_ID=your_watson_environment_id_here
   ```

3. **Restart Application**
   ```bash
   # Stop server (Ctrl+C), then restart
   python main.py
   ```

4. **Test Credentials**
   ```python
   # Save as test_watson.py
   import os
   from dotenv import load_dotenv
   from ibm_watson import AssistantV2
   from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
   
   load_dotenv()
   
   api_key = os.getenv('ASSISTANT_APIKEY')
   service_url = os.getenv('ASSISTANT_URL')
   environment_id = os.getenv('ENVIRONMENT_ID')
   
   print(f"API Key: {'SET' if api_key else 'NOT SET'}")
   print(f"Service URL: {'SET' if service_url else 'NOT SET'}")
   print(f"Environment ID: {'SET' if environment_id else 'NOT SET'}")
   ```

#### 3. Frontend Issues

**"Frontend not loading"**
- Ensure `build/index.html` exists
- Check browser console for JavaScript errors
- Clear browser cache or use incognito mode

**"Data upload fails"**
- Check file permissions in `uploads/` directory
- Ensure file is valid CSV format
- Check file size limits

**"ML models not working"**
- Verify `trained_model/` directory exists
- Check if model files are corrupted
- Ensure sufficient memory for model training

### Testing Your Setup

#### Test AccuWeather API
```bash
# Replace YOUR_API_KEY with your actual key
curl "http://dataservice.accuweather.com/locations/v1/cities/search?apikey=YOUR_API_KEY&q=New%20York"
```

#### Test Watson Assistant
```bash
curl http://localhost:5000/chatbot-config
```

#### Test Application
```bash
curl -I http://localhost:5000
# Should return HTTP/1.1 200 OK
```


## 📝 Recent Updates

- **2025-07-03**: Enhanced Data Tab with embedded Jupyter notebook
- **2025-07-03**: Improved AccuWeather API error handling for rate limits
- **2025-07-03**: Complete migration from OpenWeather to AccuWeather API
- **2025-07-02**: Fixed Watson Assistant configuration issues
- **2024-12-29**: Complete frontend rewrite with React and ShadCN UI


## 📄 License

This project is licensed under the MIT License.


For AccuWeather API issues, contact AccuWeather support.
For Watson Assistant issues, check IBM Cloud documentation.