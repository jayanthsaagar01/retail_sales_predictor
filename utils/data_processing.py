import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def process_sales_data(sales_df):
    """
    Process raw sales data to prepare for analysis and modeling
    
    Parameters:
    -----------
    sales_df : pandas.DataFrame
        Raw sales data DataFrame
    
    Returns:
    --------
    pandas.DataFrame
        Processed sales data
    """
    # Make a copy to avoid modifying the original
    df = sales_df.copy()
    
    # Ensure Date column is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate total sales
    if 'Total_Sales' not in df.columns:
        if 'Price' in df.columns and 'Quantity' in df.columns:
            df['Total_Sales'] = df['Price'] * df['Quantity']
        else:
            # If we don't have price and quantity, assume Total_Sales is directly provided
            if 'Sales' in df.columns:
                df.rename(columns={'Sales': 'Total_Sales'}, inplace=True)
            else:
                raise ValueError("Sales data must contain either Price and Quantity columns, or a Sales column")
    
    # Ensure we have a Category column
    if 'Category' not in df.columns:
        if 'Product_Category' in df.columns:
            df.rename(columns={'Product_Category': 'Category'}, inplace=True)
        elif 'Product_ID' in df.columns:
            # If we have Product_ID but no category, create dummy categories
            df['Category'] = df['Product_ID'].astype(str).apply(lambda x: f"Category_{x[0]}")
        else:
            # If we don't have any category information, create a single category
            df['Category'] = 'General'
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Aggregate data by date and category
    aggregated_df = df.groupby(['Date', 'Category']).agg({
        'Total_Sales': 'sum',
        'Quantity': 'sum' if 'Quantity' in df.columns else lambda x: np.nan
    }).reset_index()
    
    # Calculate moving averages for sales (7-day window)
    for category in aggregated_df['Category'].unique():
        category_data = aggregated_df[aggregated_df['Category'] == category].copy()
        
        # Check if there's enough data for a 7-day rolling average
        if len(category_data) >= 7:
            category_data.loc[:, 'Sales_7D_MA'] = category_data['Total_Sales'].rolling(7, min_periods=1).mean()
            
            # Update the original dataframe
            aggregated_df.loc[aggregated_df['Category'] == category, 'Sales_7D_MA'] = category_data['Sales_7D_MA'].values
        else:
            # If not enough data, use whatever we have
            aggregated_df.loc[aggregated_df['Category'] == category, 'Sales_7D_MA'] = category_data['Total_Sales'].rolling(
                min(len(category_data), 7), min_periods=1).mean().values
    
    return aggregated_df

def combine_datasets(sales_df, weather_df, sentiment_df):
    """
    Combine sales, weather, and sentiment data into a single dataset
    
    Parameters:
    -----------
    sales_df : pandas.DataFrame
        Processed sales data
    weather_df : pandas.DataFrame
        Weather data
    sentiment_df : pandas.DataFrame
        Sentiment data
    
    Returns:
    --------
    pandas.DataFrame
        Combined dataset
    """
    # Make copies to avoid modifying originals
    sales = sales_df.copy()
    weather = weather_df.copy()
    sentiment = sentiment_df.copy()
    
    # Ensure date columns are datetime
    sales['Date'] = pd.to_datetime(sales['Date'])
    weather['Date'] = pd.to_datetime(weather['Date'])
    sentiment['Date'] = pd.to_datetime(sentiment['Date'])
    
    # Merge sales and weather data
    combined_df = pd.merge(sales, weather, on='Date', how='left')
    
    # Merge with sentiment data
    combined_df = pd.merge(combined_df, sentiment, on='Date', how='left')
    
    # Fill missing values
    for col in combined_df.columns:
        if combined_df[col].dtype == 'float64' or combined_df[col].dtype == 'int64':
            combined_df[col] = combined_df[col].fillna(combined_df[col].mean())
        else:
            # Handle mode() differently to avoid list/scalar error
            mode_value = "Unknown"
            mode_series = combined_df[col].mode()
            if not mode_series.empty:
                mode_value = mode_series.iloc[0]
            # Ensure mode_value is a scalar, not a list or series
            if isinstance(mode_value, (list, pd.Series)):
                mode_value = mode_value[0] if len(mode_value) > 0 else "Unknown"
            combined_df[col] = combined_df[col].fillna(mode_value)
    
    # Add day of week feature
    combined_df['DayOfWeek'] = combined_df['Date'].dt.dayofweek
    combined_df['IsWeekend'] = combined_df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Add month and season
    combined_df['Month'] = combined_df['Date'].dt.month
    combined_df['Season'] = combined_df['Month'].apply(get_season)
    
    # Create weather condition categories if needed
    if 'Weather_Condition' not in combined_df.columns and 'WeatherCondition' in combined_df.columns:
        combined_df.rename(columns={'WeatherCondition': 'Weather_Condition'}, inplace=True)
    
    if 'Weather_Condition' in combined_df.columns:
        # Simplify weather conditions to broader categories if needed
        if combined_df['Weather_Condition'].nunique() > 10:
            combined_df['Weather_Condition'] = combined_df['Weather_Condition'].apply(simplify_weather_condition)
    
    return combined_df

def get_season(month):
    """
    Convert month to season
    
    Parameters:
    -----------
    month : int
        Month (1-12)
    
    Returns:
    --------
    str
        Season (Spring, Summer, Fall, Winter)
    """
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:  # month in [12, 1, 2]
        return 'Winter'

def simplify_weather_condition(condition):
    """
    Simplify detailed weather conditions into broader categories
    
    Parameters:
    -----------
    condition : str
        Detailed weather condition
    
    Returns:
    --------
    str
        Simplified weather condition
    """
    condition = str(condition).lower()
    
    if any(word in condition for word in ['sun', 'clear']):
        return 'Sunny'
    elif any(word in condition for word in ['rain', 'drizzle', 'shower']):
        return 'Rainy'
    elif any(word in condition for word in ['cloud', 'overcast']):
        return 'Cloudy'
    elif any(word in condition for word in ['snow', 'flurries', 'blizzard']):
        return 'Snowy'
    elif any(word in condition for word in ['storm', 'thunder', 'lightning']):
        return 'Stormy'
    elif any(word in condition for word in ['fog', 'mist', 'haze']):
        return 'Foggy'
    else:
        return 'Other'
