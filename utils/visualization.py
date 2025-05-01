import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def plot_sales_trend(data):
    """
    Plot sales trends over time by category with simplified visualization
    for easier understanding even for non-technical users

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

    # Set a colorful, distinguishable palette
    colors = ['#FF9671', '#845EC2', '#00C2A8', '#F9F871', '#FFC75F', '#D65DB1', '#0089BA']
    
    # Plot each category with thicker lines and larger markers
    pivot_data.plot(ax=ax, marker='o', markersize=8, linewidth=3, color=colors[:len(pivot_data.columns)])

    # Add data labels to the last point of each line
    for i, category in enumerate(pivot_data.columns):
        last_date = pivot_data.index[-1]
        last_value = pivot_data[category].iloc[-1]
        
        # Add value labels with category
        ax.annotate(f'{category}: ₹{int(last_value):,}',
                   xy=(last_date, last_value),
                   xytext=(10, 0),
                   textcoords='offset points',
                   fontsize=10,
                   fontweight='bold',
                   color=colors[i % len(colors)])

    ax.set_title('What We Sold - Sales by Product Type', fontsize=18, fontweight='bold')
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Total Sales (₹)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Create a clearer legend with larger font
    legend = ax.legend(title='Product Type', fontsize=12, title_fontsize=14)
    
    # Add a text box with simple explanation
    explanation = "This chart shows how much money we made selling different products over time.\nHigher lines mean we sold more of that product!"
    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
                bbox={'facecolor':'#F0F0F0', 'alpha':0.5, 'pad':5})

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for the explanation text

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
    Plot impact of weather on sales with simplified visualization
    for easier understanding even for non-technical users

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
    # Use bright, friendly colors
    colors = ['#FF9671', '#845EC2', '#00C2A8', '#F9F871', '#FFC75F', '#D65DB1', '#0089BA']
    
    fig, ax = plt.subplots(figsize=(12, 7))

    if weather_metric == 'Temperature':
        # Group data by temperature ranges
        data['Temp_Range'] = pd.cut(
            data['Temperature'], 
            bins=[-20, 0, 10, 20, 30, 50],
            labels=['Very Cold', 'Cold', 'Cool', 'Warm', 'Hot']
        )

        # Calculate average sales by temperature range and category
        grouped_data = data.groupby(['Temp_Range', 'Category'])['Total_Sales'].mean().reset_index()

        # Pivot data
        pivot_data = grouped_data.pivot(index='Temp_Range', columns='Category', values='Total_Sales')

        # Plot with custom colors
        ax = pivot_data.plot(kind='bar', ax=ax, color=colors[:len(pivot_data.columns)], 
                     width=0.8, figsize=(12, 7))

        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='₹%.1fK', label_type='edge', fontsize=9, fontweight='bold')

        ax.set_title('How Weather Affects Our Sales', fontsize=18, fontweight='bold')
        ax.set_xlabel('Weather Temperature', fontsize=14)
        ax.set_ylabel('Average Sales (₹)', fontsize=14)
        
        # Add emoji indicators
        emojis = ['❄️', '🥶', '🧥', '☀️', '🔥']
        for i, emoji in enumerate(emojis):
            if i < len(pivot_data.index):
                ax.annotate(emoji, xy=(i, -max(pivot_data.values.max()) * 0.05), 
                           ha='center', fontsize=20)

    elif weather_metric == 'Precipitation':
        # Group data by precipitation ranges
        data['Precip_Range'] = pd.cut(
            data['Precipitation'] if 'Precipitation' in data.columns else data['Rainfall'],
            bins=[-0.1, 0, 5, 10, 20, 100],
            labels=['No Rain', 'Light Rain', 'Moderate', 'Heavy', 'Stormy']
        )

        # Calculate average sales by precipitation range and category
        grouped_data = data.groupby(['Precip_Range', 'Category'])['Total_Sales'].mean().reset_index()

        # Pivot data
        pivot_data = grouped_data.pivot(index='Precip_Range', columns='Category', values='Total_Sales')

        # Plot with custom colors
        ax = pivot_data.plot(kind='bar', ax=ax, color=colors[:len(pivot_data.columns)], 
                     width=0.8, figsize=(12, 7))

        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='₹%.1fK', label_type='edge', fontsize=9, fontweight='bold')

        ax.set_title('How Rain Affects Our Sales', fontsize=18, fontweight='bold')
        ax.set_xlabel('Amount of Rain', fontsize=14)
        ax.set_ylabel('Average Sales (₹)', fontsize=14)
        
        # Add emoji indicators
        emojis = ['☀️', '🌦️', '🌧️', '⛈️', '🌊']
        for i, emoji in enumerate(emojis):
            if i < len(pivot_data.index):
                ax.annotate(emoji, xy=(i, -max(pivot_data.values.max()) * 0.05), 
                           ha='center', fontsize=20)

    elif weather_metric == 'Weather_Condition':
        # Calculate average sales by weather condition and category
        grouped_data = data.groupby(['Weather_Condition', 'Category'])['Total_Sales'].mean().reset_index()

        # Pivot data
        pivot_data = grouped_data.pivot(index='Weather_Condition', columns='Category', values='Total_Sales')

        # Plot with custom colors
        ax = pivot_data.plot(kind='bar', ax=ax, color=colors[:len(pivot_data.columns)], 
                     width=0.8, figsize=(12, 7))

        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='₹%.1fK', label_type='edge', fontsize=9, fontweight='bold')

        ax.set_title('How Weather Affects Our Sales', fontsize=18, fontweight='bold')
        ax.set_xlabel('Weather Type', fontsize=14)
        ax.set_ylabel('Average Sales (₹)', fontsize=14)
        
        # Add emoji indicators based on weather condition
        weather_emojis = {
            'Sunny': '☀️', 
            'Cloudy': '☁️', 
            'Rainy': '🌧️', 
            'Snowy': '❄️', 
            'Stormy': '⛈️'
        }
        
        for i, weather in enumerate(pivot_data.index):
            emoji = weather_emojis.get(weather, '🌡️')
            ax.annotate(emoji, xy=(i, -max(pivot_data.values.max()) * 0.05), 
                       ha='center', fontsize=20)

    ax.grid(True, alpha=0.3, axis='y')
    legend = ax.legend(title='Products', fontsize=12, title_fontsize=14)
    
    # Add a text box with simple explanation
    if weather_metric == 'Temperature':
        explanation = "This chart shows how temperature affects our sales.\nSee which products sell better in hot or cold weather!"
    elif weather_metric == 'Precipitation':
        explanation = "This chart shows how rain affects our sales.\nSee which products sell better on rainy or dry days!"
    else:
        explanation = "This chart shows how different weather types affect our sales.\nSee which products sell better under different weather conditions!"
        
    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
                bbox={'facecolor':'#F0F0F0', 'alpha':0.5, 'pad':5})

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for the explanation

    return fig

def plot_sentiment_impact(data):
    """
    Plot impact of social media sentiment on sales with simplified visualization
    for easier understanding even for non-technical users

    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing combined data

    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    # Use bright, friendly colors
    colors = ['#FF9671', '#845EC2', '#00C2A8', '#F9F871', '#FFC75F', '#D65DB1', '#0089BA']
    
    fig, ax = plt.subplots(figsize=(12, 7))

    # Group data by sentiment score ranges with more intuitive labels
    data['Sentiment_Range'] = pd.cut(
        data['Sentiment_Score'],
        bins=[-1.1, -0.5, 0, 0.5, 1.1],
        labels=['Very Unhappy 😠', 'Unhappy 🙁', 'Happy 🙂', 'Very Happy 😄']
    )

    # Calculate average sales by sentiment range and category
    grouped_data = data.groupby(['Sentiment_Range', 'Category'])['Total_Sales'].mean().reset_index()

    # Pivot data
    pivot_data = grouped_data.pivot(index='Sentiment_Range', columns='Category', values='Total_Sales')

    # Plot with custom colors
    ax = pivot_data.plot(kind='bar', ax=ax, color=colors[:len(pivot_data.columns)], 
                width=0.8, figsize=(12, 7))

    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='₹%.1fK', label_type='edge', fontsize=9, fontweight='bold')

    ax.set_title('How Customer Feelings Affect Our Sales', fontsize=18, fontweight='bold')
    ax.set_xlabel('How Customers Feel About Us', fontsize=14)
    ax.set_ylabel('Average Sales (₹)', fontsize=14)
    
    # Add a background gradient to emphasize sentiment progression
    gradient_colors = ['#ffcccb', '#ffe0cc', '#e6ffcc', '#ccffcc']
    for i, color in enumerate(gradient_colors):
        if i < len(pivot_data.index):
            ax.axvspan(i-0.4, i+0.4, color=color, alpha=0.3, zorder=-1)
    
    ax.grid(True, alpha=0.3, axis='y')
    legend = ax.legend(title='Products', fontsize=12, title_fontsize=14)
    
    # Add a text box with simple explanation
    explanation = "This chart shows how customer feelings affect our sales.\nWhen customers are happy about our products, we sell more!"
    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
                bbox={'facecolor':'#F0F0F0', 'alpha':0.5, 'pad':5})

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for the explanation

    return fig

def plot_sales_forecast(historical_data, predicted_data):
    """
    Plot historical sales with forecasted values, with simplified visualization
    for easier understanding even for non-technical users

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
    fig, ax = plt.subplots(figsize=(12, 7))

    # Group historical data by date
    historical_grouped = historical_data.groupby('Date')['Total_Sales'].sum().reset_index()

    # Group predicted data by date
    predicted_grouped = predicted_data.groupby('Date')['Predicted_Sales'].sum().reset_index()

    # Plot historical data with thicker line
    ax.plot(
        historical_grouped['Date'], 
        historical_grouped['Total_Sales'],
        marker='o',
        markersize=8,
        linewidth=3,
        color='#0066cc',
        label='Past Sales'
    )

    # Plot predicted data with thicker line
    ax.plot(
        predicted_grouped['Date'],
        predicted_grouped['Predicted_Sales'],
        marker='s',
        markersize=8,
        linewidth=3,
        linestyle='--',
        color='#ff9933',
        label='Future Sales (Predicted)'
    )

    # Add shaded area to distinguish historical from predicted with better color
    min_y = min(historical_grouped['Total_Sales'].min(), predicted_grouped['Predicted_Sales'].min()) * 0.9
    max_y = max(historical_grouped['Total_Sales'].max(), predicted_grouped['Predicted_Sales'].max()) * 1.1

    last_historical_date = historical_grouped['Date'].max()

    # Add more clear division between past and future
    ax.axvspan(last_historical_date, predicted_grouped['Date'].max(), alpha=0.15, color='#ffcc99')
    ax.axvline(last_historical_date, linestyle='--', color='#ff6600', linewidth=2)

    # Add clearer labels for past and future
    ax.text(
        historical_grouped['Date'].iloc[len(historical_grouped) // 2], 
        max_y * 0.9, 
        'PAST', 
        ha='center', 
        fontsize=14, 
        color='#0066cc',
        fontweight='bold',
        bbox={'facecolor':'white', 'alpha':0.8, 'pad':5, 'boxstyle':'round'}
    )
    
    ax.text(
        predicted_grouped['Date'].iloc[len(predicted_grouped) // 2], 
        max_y * 0.9, 
        'FUTURE', 
        ha='center', 
        fontsize=14, 
        color='#ff6600',
        fontweight='bold',
        bbox={'facecolor':'white', 'alpha':0.8, 'pad':5, 'boxstyle':'round'}
    )

    # Annotate last real value and first predicted value
    last_actual = historical_grouped['Total_Sales'].iloc[-1]
    first_predicted = predicted_grouped['Predicted_Sales'].iloc[0]
    
    ax.annotate(f'Last actual: ₹{int(last_actual):,}',
               xy=(last_historical_date, last_actual),
               xytext=(-100, 30),
               textcoords='offset points',
               arrowprops=dict(arrowstyle='->'),
               fontsize=10)
    
    ax.annotate(f'First prediction: ₹{int(first_predicted):,}',
               xy=(predicted_grouped['Date'].iloc[0], first_predicted),
               xytext=(30, 30),
               textcoords='offset points',
               arrowprops=dict(arrowstyle='->'),
               fontsize=10)

    ax.set_title('What We Expect to Sell in the Future', fontsize=18, fontweight='bold')
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Total Sales (₹)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Create a clearer legend
    legend = ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                     ncol=2, frameon=True, facecolor='white')
    
    # Add a text box with simple explanation
    explanation = "This chart shows our past sales and predicts future sales.\nThe blue line shows what we sold in the past, and the orange line shows what we expect to sell in the future."
    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
                bbox={'facecolor':'#F0F0F0', 'alpha':0.5, 'pad':5})

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Make room for the explanation

    return fig