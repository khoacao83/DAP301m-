import os
import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

class AirQualityDataFetcher:
    def __init__(self):
        self.api_key = os.environ.get('OPENWEATHER_API_KEY')
        self.geocoding_url = "http://api.openweathermap.org/geo/1.0/direct"
        self.air_pollution_history_url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
        self.required_columns = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
        
    def fetch_air_quality_data(self, location, days):
        """Fetch air quality data for a specific location and time period"""
        try:
            # Get coordinates for the location
            coordinates = self._get_coordinates(location)
            if not coordinates:
                logging.error(f"Could not get coordinates for location: {location}")
                return None
            
            lat, lon = coordinates
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # Convert to Unix timestamps
            start_unix = int(start_time.timestamp())
            end_unix = int(end_time.timestamp())
            
            # Fetch historical air pollution data
            historical_data = self._fetch_historical_air_pollution(lat, lon, start_unix, end_unix)
            
            if historical_data:
                return self._process_api_response(historical_data, location)
            else:
                logging.error("Failed to fetch air quality data from OpenWeather API")
                return None
                
        except Exception as e:
            logging.error(f"Error fetching air quality data: {e}")
            return None
    
    def _get_coordinates(self, location):
        """Get latitude and longitude for a location using OpenWeather Geocoding API"""
        try:
            params = {
                'q': location,
                'limit': 1,
                'appid': self.api_key
            }
            
            response = requests.get(self.geocoding_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data:
                return data[0]['lat'], data[0]['lon']
            return None
            
        except Exception as e:
            logging.error(f"Error getting coordinates for {location}: {e}")
            return None
    
    def _fetch_historical_air_pollution(self, lat, lon, start_unix, end_unix):
        """Fetch historical air pollution data from OpenWeather API"""
        try:
            params = {
                'lat': lat,
                'lon': lon,
                'start': start_unix,
                'end': end_unix,
                'appid': self.api_key
            }
            
            logging.info(f"Fetching historical data from {start_unix} to {end_unix}")
            response = requests.get(self.air_pollution_history_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            logging.info(f"Successfully fetched {len(data.get('list', []))} historical data points")
            return data
            
        except Exception as e:
            logging.error(f"Error fetching historical air pollution data: {e}")
            logging.error(f"Response status: {response.status_code if 'response' in locals() else 'No response'}")
            if 'response' in locals():
                logging.error(f"Response text: {response.text}")
            return None
    
    def _process_api_response(self, api_data, location):
        """Process OpenWeather API response into DataFrame without any modifications"""
        try:
            if 'list' not in api_data:
                logging.error("Invalid API response format")
                return None
            
            # Extract data from API response without any modifications
            rows = []
            for entry in api_data['list']:
                dt_timestamp = entry['dt']
                dt_datetime = datetime.fromtimestamp(dt_timestamp)
                
                # Get AQI from main section (OpenWeather's 1-5 scale)
                aqi = entry.get('main', {}).get('aqi', None)
                
                # Get pollutant concentrations from components section
                components = entry.get('components', {})
                
                row = {
                    'datetime': dt_datetime,
                    'aqi': aqi,
                    'co': components.get('co', None),
                    'no': components.get('no', None),
                    'no2': components.get('no2', None),
                    'o3': components.get('o3', None),
                    'so2': components.get('so2', None),
                    'pm2_5': components.get('pm2_5', None),
                    'pm10': components.get('pm10', None),
                    'nh3': components.get('nh3', None),
                    'location': location
                }
                rows.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(rows)
            
            # Sort by datetime
            df = df.sort_values('datetime').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Error processing API response: {e}")
            return None
    
    def validate_dataset(self, df):
        """Validate that the dataset contains all required columns"""
        if df is None:
            return False, "Dataset is None"
        
        missing_columns = []
        for col in self.required_columns:
            if col not in df.columns:
                missing_columns.append(col)
        
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        return True, "Dataset validation passed"