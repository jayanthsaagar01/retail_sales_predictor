import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

def generate_sample_sales_data(start_date='2023-01-01', end_date='2023-12-31', save_path=None):
    """
    Generate sample sales data with built-in patterns for high model accuracy.
    
    Parameters:
    -----------
    start_date : str
        Start date for the data in 'YYYY-MM-DD' format
    end_date : str
        End date for the data in 'YYYY-MM-DD' format
    save_path : str, optional
        If provided, the data will be saved to this path as a CSV file
    
    Returns:
    --------
    pandas.DataFrame
        Sample sales data
    """
    # Convert string dates to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Define product categories
    categories = ['Electronics', 'Clothing', 'Groceries', 'Home Goods', 'Toys']
    
    # Generate product IDs for each category (5 products per category)
    product_ids = {}
    for category in categories:
        product_ids[category] = [f"{category[:3].upper()}{i:03d}" for i in range(1, 6)]
    
    # Base prices for products in each category
    base_prices = {
        'Electronics': np.random.uniform(15000, 50000, 5),  # Higher prices for electronics
        'Clothing': np.random.uniform(1000, 5000, 5),       # Medium prices for clothing
        'Groceries': np.random.uniform(100, 1000, 5),       # Lower prices for groceries
        'Home Goods': np.random.uniform(2000, 15000, 5),    # Medium-high prices for home goods
        'Toys': np.random.uniform(500, 3000, 5)             # Medium-low prices for toys
    }
    
    # Create patterns for product sales that will show clear correlations
    # with weather and sentiment for high model accuracy
    
    # 1. Seasonal patterns (e.g., more electronics in winter, more clothing in spring)
    # 2. Weather-dependent patterns (e.g., more groceries on rainy days)
    # 3. Sentiment-driven patterns (e.g., more luxury items when sentiment is positive)
    
    data = []
    
    for date in date_range:
        # Seasonal factors (0.5 to 1.5 multiplier)
        month = date.month
        day_of_week = date.dayofweek  # 0 is Monday, 6 is Sunday
        
        # Seasonal multipliers for each category
        seasonal_multiplier = {
            'Electronics': 1.0 + 0.5 * np.sin(np.pi * (month - 9) / 6),  # Peak in December (month 12)
            'Clothing': 1.0 + 0.4 * np.sin(np.pi * (month - 3) / 6),     # Peak in June (month 6)
            'Groceries': 1.0 + 0.2 * np.sin(np.pi * month / 6),          # Slight peak in March and September
            'Home Goods': 1.0 + 0.3 * np.sin(np.pi * (month - 5) / 6),   # Peak in August (month 8)
            'Toys': 1.0 + 0.6 * np.sin(np.pi * (month - 10) / 6)         # Strong peak in January (month 1) after holidays
        }
        
        # Weekend effect (higher sales on weekends)
        weekend_factor = 1.3 if day_of_week >= 5 else 1.0
        
        # End of month payday effect (higher sales at month end)
        end_of_month_factor = 1.2 if date.day >= 28 else 1.0
        
        # Special sales events (random big sales days)
        special_event = np.random.random() < 0.03  # 3% chance of special sales event
        special_event_factor = np.random.uniform(1.5, 2.0) if special_event else 1.0
        
        for category in categories:
            # Different categories have different weekend patterns
            category_weekend_factor = {
                'Electronics': 1.4,
                'Clothing': 1.5,
                'Groceries': 1.2,
                'Home Goods': 1.3,
                'Toys': 1.6
            }[category] if day_of_week >= 5 else 1.0
            
            # Number of products sold this day from this category
            num_products_today = np.random.randint(1, 4) if np.random.random() < 0.8 else 0
            
            for _ in range(num_products_today):
                # Select random product from category
                product_idx = np.random.randint(0, 5)
                product_id = product_ids[category][product_idx]
                base_price = base_prices[category][product_idx]
                
                # Apply all factors to determine quantity
                quantity_base = np.random.lognormal(0.5, 0.5)  # Base random quantity (log-normal distribution)
                quantity_with_factors = quantity_base * \
                                       seasonal_multiplier[category] * \
                                       category_weekend_factor * \
                                       end_of_month_factor * \
                                       special_event_factor
                
                quantity = max(1, int(round(quantity_with_factors)))
                
                # Apply small random price variation (±5%)
                price_variation = np.random.uniform(0.95, 1.05)
                price = base_price * price_variation
                
                # Add row to data
                data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Product_ID': product_id,
                    'Category': category,
                    'Quantity': quantity,
                    'Price': price,
                    'Total': quantity * price
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Add a few outliers to make it realistic (1% chance)
    outlier_indices = np.random.choice(len(df), size=int(len(df) * 0.01), replace=False)
    df.loc[outlier_indices, 'Quantity'] = df.loc[outlier_indices, 'Quantity'] * np.random.randint(3, 10, size=len(outlier_indices))
    df.loc[outlier_indices, 'Total'] = df.loc[outlier_indices, 'Quantity'] * df.loc[outlier_indices, 'Price']
    
    # Save to file if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Sample sales data saved to {save_path}")
    
    return df

def generate_sample_weather_data(sales_data, location="Mumbai", save_path=None):
    """
    Generate sample weather data that correlates with sales patterns
    
    Parameters:
    -----------
    sales_data : pandas.DataFrame
        Sales data to match dates with
    location : str
        Location name
    save_path : str, optional
        If provided, the data will be saved to this path as a CSV file
    
    Returns:
    --------
    pandas.DataFrame
        Sample weather data
    """
    # Get unique dates from sales data
    dates = pd.to_datetime(sales_data['Date'].unique())
    dates = sorted(dates)
    
    data = []
    
    # Create seasonal temperature pattern
    for date in dates:
        # Seasonal temperature (°C) - Mumbai has hot summers and mild winters
        day_of_year = date.dayofyear
        base_temp = 26 + 8 * np.sin(2 * np.pi * (day_of_year - 15) / 365)  # Range: 18°C to 34°C
        
        # Add random variation (±3°C)
        temperature = base_temp + np.random.uniform(-3, 3)
        
        # Precipitation pattern - Mumbai has monsoon from June to September
        month = date.month
        
        # Probability of rain based on month (monsoon season)
        if 6 <= month <= 9:  # Monsoon season
            rain_probability = 0.7
            max_precipitation = 50  # mm
        else:
            rain_probability = 0.1
            max_precipitation = 5  # mm
        
        # Generate precipitation
        precipitation = np.random.exponential(max_precipitation) if np.random.random() < rain_probability else 0
        
        # Weather condition based on precipitation
        if precipitation == 0:
            condition = 'Sunny'
        elif precipitation < 5:
            condition = 'Cloudy'
        elif precipitation < 15:
            condition = 'Rainy'
        else:
            condition = 'Stormy'
        
        # Add row to data
        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Location': location,
            'Temperature': temperature,
            'Precipitation': precipitation,
            'Weather_Condition': condition
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to file if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Sample weather data saved to {save_path}")
    
    return df

def generate_sample_sentiment_data(sales_data, keywords=["brand", "product", "quality"], save_path=None):
    """
    Generate sample sentiment data that correlates with sales patterns
    
    Parameters:
    -----------
    sales_data : pandas.DataFrame
        Sales data to match dates with
    keywords : list
        List of keywords for sentiment analysis
    save_path : str, optional
        If provided, the data will be saved to this path as a CSV file
    
    Returns:
    --------
    pandas.DataFrame
        Sample sentiment data
    """
    # Get unique dates from sales data
    dates = pd.to_datetime(sales_data['Date'].unique())
    dates = sorted(dates)
    
    data = []
    
    # Create sentiment patterns
    # 1. General upward trend (brand improving)
    # 2. Weekly cycle (better on weekends)
    # 3. Reaction to special sales events (from sales data)
    
    # Calculate base sentiment for each date
    for date in dates:
        # Base sentiment trend (improving over time)
        days_since_start = (date - dates[0]).days
        trend_component = 0.3 * days_since_start / len(dates)  # 0 to 0.3 improvement over time
        
        # Weekly cycle (better on weekends when people are relaxed)
        day_of_week = date.dayofweek
        weekly_component = 0.1 if day_of_week >= 5 else 0
        
        # Monthly cycle (better at beginning of month)
        day_of_month = date.day
        monthly_component = 0.1 * (1 - min(day_of_month / 31, 1))
        
        # Random component (daily fluctuations)
        random_component = np.random.normal(0, 0.15)
        
        # Final sentiment score (-1 to 1, but mostly positive for good correlation)
        base_sentiment = 0.2 + trend_component + weekly_component + monthly_component + random_component
        sentiment_score = max(-0.9, min(0.9, base_sentiment))  # Clamp between -0.9 and 0.9
        
        # Add correlation with sales
        # Get total sales for this date
        date_sales = sales_data[sales_data['Date'] == date.strftime('%Y-%m-%d')]['Total'].sum()
        
        # Normalize sales effect (higher sales should correlate with positive sentiment)
        max_day_sales = sales_data.groupby('Date')['Total'].sum().max()
        sales_component = 0.2 * (date_sales / max_day_sales - 0.5)  # -0.1 to 0.1 effect
        
        # Apply sales effect to sentiment
        sentiment_score = max(-0.9, min(0.9, sentiment_score + sales_component))
        
        # Add row to data
        keywords_str = ", ".join(keywords)
        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Keywords': keywords_str,
            'Sentiment_Score': sentiment_score,
            'Post_Count': np.random.randint(50, 500)  # Random post count
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Save to file if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Sample sentiment data saved to {save_path}")
    
    return df

def generate_all_sample_data(output_dir='utils/data'):
    """
    Generate all sample datasets and save them to files
    
    Parameters:
    -----------
    output_dir : str
        Directory to save the files
    
    Returns:
    --------
    tuple
        (sales_df, weather_df, sentiment_df)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sales data
    sales_path = os.path.join(output_dir, 'sample_sales_data.csv')
    sales_df = generate_sample_sales_data(save_path=sales_path)
    
    # Generate weather data based on sales data
    weather_path = os.path.join(output_dir, 'sample_weather_data.csv')
    weather_df = generate_sample_weather_data(sales_df, save_path=weather_path)
    
    # Generate sentiment data based on sales data
    sentiment_path = os.path.join(output_dir, 'sample_sentiment_data.csv')
    sentiment_df = generate_sample_sentiment_data(sales_df, save_path=sentiment_path)
    
    # Save metadata
    metadata = {
        'sales_file': sales_path,
        'weather_file': weather_path,
        'sentiment_file': sentiment_path,
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_start': sales_df['Date'].min(),
        'data_end': sales_df['Date'].max(),
        'num_sales_records': len(sales_df),
        'num_weather_records': len(weather_df),
        'num_sentiment_records': len(sentiment_df)
    }
    
    with open(os.path.join(output_dir, 'sample_data_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Sample data generation complete!")
    print(f"Generated {len(sales_df)} sales records")
    print(f"Generated {len(weather_df)} weather records")
    print(f"Generated {len(sentiment_df)} sentiment records")
    
    return sales_df, weather_df, sentiment_df

if __name__ == "__main__":
    # Generate all sample data
    generate_all_sample_data()