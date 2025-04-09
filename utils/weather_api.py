import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def get_weather_data(location, start_date, end_date):
    """
    Fetch weather data for a given location and date range
    
    Parameters:
    -----------
    location : str
        Location (city name)
    start_date : str or datetime
        Start date for weather data
    end_date : str or datetime
        End date for weather data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing weather data by date
    """
    # Convert dates to datetime if they're strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Calculate date range
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Here we would typically call a weather API to get real data
    # For this example, we'll generate synthetic data based on the location and date range
    
    # In a real application, we would use a weather API like OpenWeatherMap:
    # api_key = os.getenv("OPENWEATHERMAP_API_KEY", "default_key")
    # And make API calls for each date in the range
    
    weather_data = []
    
    # Generate synthetic weather data for each date
    for date in date_range:
        # Generate weather data based on typical patterns for the time of year
        temperature, precipitation, weather_condition = generate_synthetic_weather(location, date)
        
        weather_data.append({
            'Date': date,
            'Location': location,
            'Temperature': temperature,
            'Precipitation': precipitation,
            'Weather_Condition': weather_condition
        })
    
    # Convert to DataFrame
    weather_df = pd.DataFrame(weather_data)
    
    return weather_df

def generate_synthetic_weather(location, date):
    """
    Generate synthetic weather data based on location and date
    
    Parameters:
    -----------
    location : str
        Location (city name)
    date : datetime
        Date for which to generate weather
    
    Returns:
    --------
    tuple
        (temperature, precipitation, weather_condition)
    """
    # Base values depend on location (very simplified)
    location_lower = location.lower()
    
    # Temperature baseline based on simplified location matching
    if any(city in location_lower for city in ['new york', 'boston', 'chicago']):
        temp_baseline = 10  # Northeastern/Midwestern US
    elif any(city in location_lower for city in ['los angeles', 'san francisco', 'phoenix']):
        temp_baseline = 18  # Western US
    elif any(city in location_lower for city in ['miami', 'houston', 'atlanta']):
        temp_baseline = 22  # Southern US
    elif any(city in location_lower for city in ['london', 'paris', 'berlin']):
        temp_baseline = 12  # Western Europe
    else:
        temp_baseline = 15  # Default
    
    # Month adjustments
    month = date.month
    
    # Northern hemisphere seasonal adjustments
    if 1 <= month <= 2 or month == 12:  # Winter
        temp_adjustment = -10
        precip_base = 5
        condition_weights = {
            'Sunny': 0.2, 'Cloudy': 0.3, 'Rainy': 0.2, 'Snowy': 0.3
        }
    elif 3 <= month <= 5:  # Spring
        temp_adjustment = 0
        precip_base = 4
        condition_weights = {
            'Sunny': 0.4, 'Cloudy': 0.3, 'Rainy': 0.3, 'Snowy': 0.0
        }
    elif 6 <= month <= 8:  # Summer
        temp_adjustment = 10
        precip_base = 3
        condition_weights = {
            'Sunny': 0.6, 'Cloudy': 0.2, 'Rainy': 0.2, 'Snowy': 0.0
        }
    else:  # Fall
        temp_adjustment = 0
        precip_base = 4
        condition_weights = {
            'Sunny': 0.3, 'Cloudy': 0.4, 'Rainy': 0.3, 'Snowy': 0.0
        }
    
    # Calculate temperature with some random variation
    temperature = temp_baseline + temp_adjustment + random.normalvariate(0, 3)
    
    # Generate precipitation (rainfall/snowfall)
    # Higher chance of precipitation on certain days based on random seed
    random.seed(int(date.strftime('%Y%m%d')))
    precip_chance = random.random()
    
    if precip_chance < 0.4:  # 40% chance of precipitation
        precipitation = random.uniform(0, precip_base * 2)
    else:
        precipitation = 0
    
    # Determine weather condition
    if precipitation == 0:
        # No precipitation, so either Sunny or Cloudy
        if random.random() < 0.7:  # 70% chance of sunny when no precipitation
            weather_condition = 'Sunny'
        else:
            weather_condition = 'Cloudy'
    else:
        # Precipitation, so either Rainy or Snowy
        if temperature < 2:  # Below 2°C, it's likely to snow
            weather_condition = 'Snowy'
        else:
            weather_condition = 'Rainy'
    
    # Add extreme weather occasionally
    if random.random() < 0.05:  # 5% chance of extreme weather
        if temperature > 25:
            weather_condition = 'Stormy'
        elif temperature < 0:
            weather_condition = 'Blizzard'
    
    return temperature, precipitation, weather_condition
