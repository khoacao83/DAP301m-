import requests
import os
import logging
import math
from datetime import datetime
from dotenv import load_dotenv

class AccuWeatherAPI:
    def __init__(self):
        # Ensure .env is loaded when the class is instantiated
        load_dotenv()
        self.api_key = os.getenv('ACCUWEATHER_API_KEY')
        self.base_url = 'http://dataservice.accuweather.com'
        
    def get_location_key(self, query):
        """Get location key for a city"""
        url = f'{self.base_url}/locations/v1/cities/search'
        params = {
            'apikey': self.api_key,
            'q': query
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            # Check for AccuWeather API errors (rate limits, etc.)
            if 'Code' in data and 'Message' in data:
                if data.get('Code') == 'ServiceUnavailable':
                    raise Exception("The allowed number of requests has been exceeded.")
                else:
                    raise Exception(f"AccuWeather API Error: {data['Message']}")
            
            if data and len(data) > 0:
                return data[0]['Key'], data[0]
            else:
                raise Exception(f"Location '{query}' not found")
                
        except requests.RequestException as e:
            logging.error(f"Error fetching location data: {e}")
            raise Exception(f"Failed to get location data: {str(e)}")
    
    def get_current_weather(self, city="Ho Chi Minh City"):
        """Get current weather for a city"""
        if not self.api_key:
            raise Exception("AccuWeather API key not configured. Please set ACCUWEATHER_API_KEY environment variable.")
        
        try:
            # Get location key first
            location_key, location_data = self.get_location_key(city)
            
            # Get current conditions
            url = f'{self.base_url}/currentconditions/v1/{location_key}'
            params = {
                'apikey': self.api_key,
                'details': 'true'
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            # Check for AccuWeather API errors (rate limits, etc.)
            if 'Code' in data and 'Message' in data:
                error_msg = f"AccuWeather API Error: {data['Message']}"
                if data.get('Code') == 'ServiceUnavailable':
                    error_msg = "The allowed number of requests has been exceeded."
                logging.error(f"AccuWeather API error: {data}")
                return {'error': error_msg, 'Code': data.get('Code')}
            
            if not data or not isinstance(data, list):
                raise Exception("No current conditions data available")
            
            # Format the data for frontend consumption
            return self._format_current_weather_data(data[0], location_data)
            
        except requests.RequestException as e:
            logging.error(f"Network error fetching current weather: {e}")
            return {'error': f'Network error: {str(e)}'}
        except Exception as e:
            logging.error(f"Error fetching current weather: {e}")
            return {'error': str(e)}
    
    def get_forecast(self, city="Ho Chi Minh City", days=5):
        """Get weather forecast for a city"""
        if not self.api_key:
            raise Exception("AccuWeather API key not configured. Please set ACCUWEATHER_API_KEY environment variable.")
        
        try:
            # Get location key first
            location_key, location_data = self.get_location_key(city)
            
            # Get forecast data
            url = f'{self.base_url}/forecasts/v1/daily/5day/{location_key}'
            params = {
                'apikey': self.api_key,
                'details': 'true',
                'metric': 'true'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data or 'DailyForecasts' not in data:
                raise Exception("No forecast data available")
            
            # Format the data for frontend consumption
            return self._format_forecast_data(data, location_data)
            
        except Exception as e:
            logging.error(f"Error fetching forecast: {e}")
            return {'error': str(e)}
    
    def _format_current_weather_data(self, weather_data, location_data):
        """Format AccuWeather current conditions data for frontend consumption"""
        try:
            # Extract location information
            location = location_data.get('LocalizedName', 'Unknown')
            country = location_data.get('Country', {}).get('LocalizedName', 'Unknown')
            
            # Extract weather information
            temperature = weather_data.get('Temperature', {}).get('Metric', {}).get('Value', 0)
            feels_like = weather_data.get('RealFeelTemperature', {}).get('Metric', {}).get('Value', temperature)
            humidity = weather_data.get('RelativeHumidity', 0)
            pressure = weather_data.get('Pressure', {}).get('Metric', {}).get('Value', 0)
            wind_speed = weather_data.get('Wind', {}).get('Speed', {}).get('Metric', {}).get('Value', 0)
            wind_direction = weather_data.get('Wind', {}).get('Direction', {}).get('Degrees', 0)
            visibility = weather_data.get('Visibility', {}).get('Metric', {}).get('Value', 0)
            uv_index = weather_data.get('UVIndex', 0)
            
            # Weather description
            description = weather_data.get('WeatherText', 'Unknown')
            
            # Weather icon mapping
            weather_icon = weather_data.get('WeatherIcon', 1)
            icon = self._map_accuweather_icon(weather_icon)
            
            # Last updated time
            last_updated = weather_data.get('LocalObservationDateTime', datetime.now().isoformat())
            
            # Calculate AQI estimate based on available data
            # This is a simplified estimation - in production, use dedicated AQI API
            aqi_estimate = self._estimate_aqi(visibility, humidity, pressure)
            
            # Extract dew point from API response
            dew_point = weather_data.get('DewPoint', {}).get('Metric', {}).get('Value', 0)
            
            return {
                'location': location,
                'country': country,
                'temperature': round(temperature, 1),
                'feels_like': round(feels_like, 1),
                'humidity': humidity,
                'pressure': round(pressure, 1),
                'wind_speed': round(wind_speed * 0.277778, 1),  # Convert km/h to m/s
                'wind_direction': wind_direction,
                'visibility': round(visibility, 1),
                'description': description,
                'icon': icon,
                'uv_index': uv_index,
                'aqi_index': aqi_estimate,
                'dew_point': math.floor(dew_point),
                'last_updated': last_updated
            }
            
        except Exception as e:
            logging.error(f"Error formatting weather data: {e}")
            return {'error': f'Error formatting weather data: {str(e)}'}
    
    def _format_forecast_data(self, forecast_data, location_data):
        """Format AccuWeather forecast data for frontend consumption"""
        try:
            location = location_data.get('LocalizedName', 'Unknown')
            country = location_data.get('Country', {}).get('LocalizedName', 'Unknown')
            
            forecasts = []
            for day in forecast_data.get('DailyForecasts', []):
                # Extract temperature data
                min_temp = day.get('Temperature', {}).get('Minimum', {}).get('Value', 0)
                max_temp = day.get('Temperature', {}).get('Maximum', {}).get('Value', 0)
                avg_temp = (min_temp + max_temp) / 2
                
                # Extract other weather data
                humidity = day.get('Day', {}).get('RelativeHumidity', {}).get('Average', 0)
                wind_speed = day.get('Day', {}).get('Wind', {}).get('Speed', {}).get('Value', 0)
                
                # Weather description and icon
                description = day.get('Day', {}).get('IconPhrase', 'Unknown')
                weather_icon = day.get('Day', {}).get('Icon', 1)
                icon = self._map_accuweather_icon(weather_icon)
                
                # Date
                date = day.get('Date', datetime.now().isoformat())
                
                forecasts.append({
                    'datetime': date,
                    'temperature': round(avg_temp, 1),
                    'feels_like': round(avg_temp, 1),  # AccuWeather doesn't provide feels_like for forecasts
                    'humidity': humidity,
                    'pressure': 1013,  # Default value as AccuWeather forecast doesn't always include pressure
                    'wind_speed': round(wind_speed * 0.277778, 1),  # Convert km/h to m/s
                    'description': description,
                    'icon': icon,
                    'min_temp': round(min_temp, 1),
                    'max_temp': round(max_temp, 1)
                })
            
            return {
                'location': location,
                'country': country,
                'forecasts': forecasts
            }
            
        except Exception as e:
            logging.error(f"Error formatting forecast data: {e}")
            return {'error': f'Error formatting forecast data: {str(e)}'}
    
    def _estimate_aqi(self, visibility, humidity, pressure):
        """Estimate AQI based on available weather parameters"""
        # This is a simplified estimation - in production, use dedicated AQI API
        # Base AQI on visibility and humidity
        aqi = 50  # Start with moderate AQI
        
        # Adjust based on visibility (lower visibility = higher AQI)
        if visibility < 5:
            aqi += 40
        elif visibility < 10:
            aqi += 20
        elif visibility < 15:
            aqi += 10
        
        # Adjust based on humidity (very high humidity can indicate poor air quality)
        if humidity > 85:
            aqi += 15
        elif humidity > 70:
            aqi += 10
        
        # Adjust based on pressure (extreme pressure can affect air quality)
        if pressure < 1000 or pressure > 1020:
            aqi += 5
        
        # Keep AQI in reasonable range (0-500)
        return min(max(aqi, 0), 500)
    

    def _map_accuweather_icon(self, icon_number):
        """Map AccuWeather icon numbers to weather condition strings"""
        icon_mapping = {
            1: 'clear-day',      # Sunny
            2: 'clear-day',      # Mostly Sunny
            3: 'partly-cloudy-day',  # Partly Sunny
            4: 'partly-cloudy-day',  # Intermittent Clouds
            5: 'partly-cloudy-day',  # Hazy Sunshine
            6: 'cloudy',         # Mostly Cloudy
            7: 'cloudy',         # Cloudy
            8: 'cloudy',         # Dreary (Overcast)
            11: 'fog',           # Fog
            12: 'rain',          # Showers
            13: 'partly-cloudy-day',  # Mostly Cloudy w/ Showers
            14: 'partly-cloudy-day',  # Partly Sunny w/ Showers
            15: 'thunderstorm',  # T-Storms
            16: 'partly-cloudy-day',  # Mostly Cloudy w/ T-Storms
            17: 'partly-cloudy-day',  # Partly Sunny w/ T-Storms
            18: 'rain',          # Rain
            19: 'snow',          # Flurries
            20: 'partly-cloudy-day',  # Mostly Cloudy w/ Flurries
            21: 'partly-cloudy-day',  # Partly Sunny w/ Flurries
            22: 'snow',          # Snow
            23: 'partly-cloudy-day',  # Mostly Cloudy w/ Snow
            24: 'snow',          # Ice
            25: 'sleet',         # Sleet
            26: 'rain',          # Freezing Rain
            29: 'sleet',         # Rain and Snow
            30: 'hot',           # Hot
            31: 'cold',          # Cold
            32: 'wind',          # Windy
            33: 'clear-night',   # Clear (Night)
            34: 'clear-night',   # Mostly Clear (Night)
            35: 'partly-cloudy-night',  # Partly Cloudy (Night)
            36: 'partly-cloudy-night',  # Intermittent Clouds (Night)
            37: 'partly-cloudy-night',  # Hazy Moonlight (Night)
            38: 'cloudy',        # Mostly Cloudy (Night)
            39: 'rain',          # Partly Cloudy w/ Showers (Night)
            40: 'rain',          # Mostly Cloudy w/ Showers (Night)
            41: 'thunderstorm',  # Partly Cloudy w/ T-Storms (Night)
            42: 'thunderstorm',  # Mostly Cloudy w/ T-Storms (Night)
            43: 'snow',          # Mostly Cloudy w/ Flurries (Night)
            44: 'snow'           # Mostly Cloudy w/ Snow (Night)
        }
        
        return icon_mapping.get(icon_number, 'clear-day')


# Initialize the API instance
weather_api = AccuWeatherAPI()

# Convenience functions for backward compatibility
def get_current_weather(city="Ho Chi Minh City"):
    """Get current weather for a city"""
    return weather_api.get_current_weather(city)

def get_weather_forecast(city="Ho Chi Minh City", days=5):
    """Get weather forecast for a city"""
    return weather_api.get_forecast(city, days)

def update_api_credentials(credentials):
    """Update API credentials in environment variables"""
    try:
        if 'accuweather_api_key' in credentials:
            os.environ['ACCUWEATHER_API_KEY'] = credentials['accuweather_api_key']
            # Reinitialize the API with new credentials
            global weather_api
            weather_api = AccuWeatherAPI()
            return {'success': True, 'message': 'AccuWeather API credentials updated successfully'}
        else:
            return {'success': False, 'message': 'AccuWeather API key not provided'}
    except Exception as e:
        logging.error(f"Error updating API credentials: {e}")
        return {'success': False, 'message': f'Error updating credentials: {str(e)}'}

def test_api_connection():
    """Test API connection"""
    try:
        result = get_current_weather("New York")
        if 'error' in result:
            return {'success': False, 'message': result['error']}
        else:
            return {'success': True, 'message': 'AccuWeather API connection successful'}
    except Exception as e:
        logging.error(f"API connection test failed: {e}")
        return {'success': False, 'message': f'API connection test failed: {str(e)}'}