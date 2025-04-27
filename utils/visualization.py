import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def plot_sales_trend(data):
    """
    Plot sales trends over time by category

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing sales data

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by date and category
    grouped_data = data.groupby(['Date', 'Category'])['Total_Sales'].sum().reset_index()

    # Pivot to have categories as columns
    pivot_data = grouped_data.pivot(index='Date', columns='Category', values='Total_Sales')

    # Plot each category
    pivot_data.plot(ax=ax, marker='o', markersize=4, linewidth=2)

    ax.set_title('Sales Trends by Product Category', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Total Sales (₹)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(title='Category')

    plt.tight_layout()

    return fig

def plot_correlation_heatmap(data):
    """
    Plot correlation heatmap between numerical variables

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing combined data

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    # Select numerical columns
    numerical_data = data.select_dtypes(include=['float64', 'int64'])

    # Drop date-related columns
    if 'DayOfWeek' in numerical_data.columns:
        numerical_data = numerical_data.drop(['DayOfWeek'], axis=1)

    # Calculate correlation matrix
    corr_matrix = numerical_data.corr()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create custom colormap (blue to white to red)
    colors = ['#2271B5', '#FFFFFF', '#D55E00']
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

    # Create heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True,
        cmap=cmap,
        vmin=-1, vmax=1,
        linewidths=0.5,
        ax=ax,
        fmt='.2f',
        annot_kws={"size": 8}
    )

    ax.set_title('Correlation Between Variables', fontsize=16)

    plt.tight_layout()

    return fig

def plot_weather_impact(data, weather_metric='Temperature'):
    """
    Plot impact of weather on sales

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing combined data
    weather_metric : str
        Weather metric to analyze ('Temperature', 'Precipitation', 'Weather_Condition')

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    if weather_metric == 'Temperature':
        # Group data by temperature ranges
        data['Temp_Range'] = pd.cut(
            data['Temperature'], 
            bins=[-20, 0, 10, 20, 30, 50],
            labels=['Below 0°C', '0-10°C', '10-20°C', '20-30°C', 'Above 30°C']
        )

        # Calculate average sales by temperature range and category
        grouped_data = data.groupby(['Temp_Range', 'Category'])['Total_Sales'].mean().reset_index()

        # Pivot data
        pivot_data = grouped_data.pivot(index='Temp_Range', columns='Category', values='Total_Sales')

        # Plot
        pivot_data.plot(kind='bar', ax=ax)

        ax.set_title('Average Sales by Temperature Range', fontsize=16)
        ax.set_xlabel('Temperature Range', fontsize=12)
        ax.set_ylabel('Average Sales (₹)', fontsize=12)

    elif weather_metric == 'Precipitation':
        # Group data by precipitation ranges
        data['Precip_Range'] = pd.cut(
            data['Precipitation'] if 'Precipitation' in data.columns else data['Rainfall'],
            bins=[-0.1, 0, 5, 10, 20, 100],
            labels=['None', '0-5mm', '5-10mm', '10-20mm', 'Above 20mm']
        )

        # Calculate average sales by precipitation range and category
        grouped_data = data.groupby(['Precip_Range', 'Category'])['Total_Sales'].mean().reset_index()

        # Pivot data
        pivot_data = grouped_data.pivot(index='Precip_Range', columns='Category', values='Total_Sales')

        # Plot
        pivot_data.plot(kind='bar', ax=ax)

        ax.set_title('Average Sales by Precipitation', fontsize=16)
        ax.set_xlabel('Precipitation Range', fontsize=12)
        ax.set_ylabel('Average Sales (₹)', fontsize=12)

    elif weather_metric == 'Weather_Condition':
        # Calculate average sales by weather condition and category
        grouped_data = data.groupby(['Weather_Condition', 'Category'])['Total_Sales'].mean().reset_index()

        # Pivot data
        pivot_data = grouped_data.pivot(index='Weather_Condition', columns='Category', values='Total_Sales')

        # Plot
        pivot_data.plot(kind='bar', ax=ax)

        ax.set_title('Average Sales by Weather Condition', fontsize=16)
        ax.set_xlabel('Weather Condition', fontsize=12)
        ax.set_ylabel('Average Sales (₹)', fontsize=12)

    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(title='Category')

    plt.tight_layout()

    return fig

def plot_sentiment_impact(data):
    """
    Plot impact of social media sentiment on sales

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing combined data

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group data by sentiment score ranges
    data['Sentiment_Range'] = pd.cut(
        data['Sentiment_Score'],
        bins=[-1.1, -0.5, 0, 0.5, 1.1],
        labels=['Very Negative', 'Negative', 'Positive', 'Very Positive']
    )

    # Calculate average sales by sentiment range and category
    grouped_data = data.groupby(['Sentiment_Range', 'Category'])['Total_Sales'].mean().reset_index()

    # Pivot data
    pivot_data = grouped_data.pivot(index='Sentiment_Range', columns='Category', values='Total_Sales')

    # Plot
    pivot_data.plot(kind='bar', ax=ax)

    ax.set_title('Average Sales by Sentiment Score', fontsize=16)
    ax.set_xlabel('Sentiment Range', fontsize=12)
    ax.set_ylabel('Average Sales (₹)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(title='Category')

    plt.tight_layout()

    return fig

def plot_sales_forecast(historical_data, predicted_data):
    """
    Plot historical sales with forecasted values

    Parameters:
    -----------
    historical_data : pandas.DataFrame
        DataFrame containing historical sales data
    predicted_data : pandas.DataFrame
        DataFrame containing predicted sales data

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group historical data by date
    historical_grouped = historical_data.groupby('Date')['Total_Sales'].sum().reset_index()

    # Group predicted data by date
    predicted_grouped = predicted_data.groupby('Date')['Predicted_Sales'].sum().reset_index()

    # Plot historical data
    ax.plot(
        historical_grouped['Date'], 
        historical_grouped['Total_Sales'],
        marker='o',
        markersize=4,
        linewidth=2,
        label='Historical Sales'
    )

    # Plot predicted data
    ax.plot(
        predicted_grouped['Date'],
        predicted_grouped['Predicted_Sales'],
        marker='s',
        markersize=4,
        linewidth=2,
        linestyle='--',
        color='red',
        label='Predicted Sales'
    )

    # Add shaded area to distinguish historical from predicted
    min_y = min(historical_grouped['Total_Sales'].min(), predicted_grouped['Predicted_Sales'].min()) * 0.9
    max_y = max(historical_grouped['Total_Sales'].max(), predicted_grouped['Predicted_Sales'].max()) * 1.1

    last_historical_date = historical_grouped['Date'].max()

    ax.axvspan(last_historical_date, predicted_grouped['Date'].max(), alpha=0.1, color='gray')
    ax.axvline(last_historical_date, linestyle=':', color='gray')

    ax.text(
        last_historical_date, 
        max_y * 0.95, 
        'Forecast →', 
        ha='right', 
        fontsize=10, 
        color='gray'
    )

    ax.set_title('Sales Forecast', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Total Sales (₹)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    return fig