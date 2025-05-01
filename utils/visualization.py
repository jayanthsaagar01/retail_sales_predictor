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
        grouped_data = data.groupby(['Temp_Range', 'Category'], observed=True)['Total_Sales'].mean().reset_index()

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
        
        # Check if we have data to display emojis
        if not pivot_data.empty and len(pivot_data.index) > 0:
            y_min = ax.get_ylim()[0]
            for i, emoji in enumerate(emojis):
                if i < len(pivot_data.index):
                    ax.annotate(emoji, xy=(i, y_min * 0.9), 
                               ha='center', fontsize=20)

    elif weather_metric == 'Precipitation':
        # Group data by precipitation ranges
        data['Precip_Range'] = pd.cut(
            data['Precipitation'] if 'Precipitation' in data.columns else data['Rainfall'],
            bins=[-0.1, 0, 5, 10, 20, 100],
            labels=['No Rain', 'Light Rain', 'Moderate', 'Heavy', 'Stormy']
        )

        # Calculate average sales by precipitation range and category
        grouped_data = data.groupby(['Precip_Range', 'Category'], observed=True)['Total_Sales'].mean().reset_index()

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
        
        # Check if we have data to display emojis
        if not pivot_data.empty and len(pivot_data.index) > 0:
            y_min = ax.get_ylim()[0]
            for i, emoji in enumerate(emojis):
                if i < len(pivot_data.index):
                    ax.annotate(emoji, xy=(i, y_min * 0.9), 
                               ha='center', fontsize=20)

    elif weather_metric == 'Weather_Condition':
        # Calculate average sales by weather condition and category
        grouped_data = data.groupby(['Weather_Condition', 'Category'], observed=True)['Total_Sales'].mean().reset_index()

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
        
        # Check if we have data to display emojis
        if not pivot_data.empty and len(pivot_data.index) > 0:
            y_min = ax.get_ylim()[0]
            for i, weather in enumerate(pivot_data.index):
                emoji = weather_emojis.get(weather, '🌡️')
                ax.annotate(emoji, xy=(i, y_min * 0.9), 
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
    grouped_data = data.groupby(['Sentiment_Range', 'Category'], observed=True)['Total_Sales'].mean().reset_index()

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
    
    # Group by date and calculate total sales
    historical = historical_data.copy()
    historical['Date'] = pd.to_datetime(historical['Date'])
    historical_grouped = historical.groupby('Date')['Total_Sales'].sum().reset_index()
    
    # Prepare prediction data
    predictions = predicted_data.copy()
    predictions['Date'] = pd.to_datetime(predictions['Date'])
    predictions_grouped = predictions.groupby('Date')['Predicted_Sales'].sum().reset_index()
    
    # Plot historical data in blue with gradient shading underneath
    ax.plot(historical_grouped['Date'], historical_grouped['Total_Sales'], 
            marker='o', color='#3498db', linestyle='-', linewidth=3, markersize=8,
            label='Historical Sales 📊')
    
    # Add fill below historical data line
    ax.fill_between(historical_grouped['Date'], 0, historical_grouped['Total_Sales'], 
                     color='#3498db', alpha=0.2)
    
    # Plot predicted data in red with gradient shading underneath
    ax.plot(predictions_grouped['Date'], predictions_grouped['Predicted_Sales'], 
            marker='s', color='#e74c3c', linestyle='--', linewidth=3, markersize=8,
            label='Predicted Sales 🔮')
    
    # Add fill below prediction data line
    ax.fill_between(predictions_grouped['Date'], 0, predictions_grouped['Predicted_Sales'], 
                     color='#e74c3c', alpha=0.2)
    
    # Add a vertical line separating historical from predicted data
    cutoff_date = historical_grouped['Date'].max()
    ax.axvline(x=cutoff_date, color='gray', linestyle='--', linewidth=2)
    
    # Add labels and a legend
    ax.set_title('Sales Forecast: What Happened vs What Will Happen', fontsize=18, fontweight='bold')
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Total Sales (₹)', fontsize=14)
    
    # Format y-axis as currency with ₹ symbol
    import matplotlib.ticker as mticker
    def rupee_format(x, pos):
        return f'₹{int(x):,}'
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(rupee_format))
    
    # Improve grid and background for readability
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_facecolor('#f8f9fa')
    
    # Determine chart min/max for annotation positioning
    min_y = min(
        historical_grouped['Total_Sales'].min() if not historical_grouped.empty else 0, 
        predictions_grouped['Predicted_Sales'].min() if not predictions_grouped.empty else 0
    ) * 0.8
    
    max_y = max(
        historical_grouped['Total_Sales'].max() if not historical_grouped.empty else 0, 
        predictions_grouped['Predicted_Sales'].max() if not predictions_grouped.empty else 0
    ) * 1.1
    
    # Add annotations for past and future
    if len(historical_grouped) > 1:
        ax.text(
            historical_grouped['Date'].iloc[len(historical_grouped) // 2], 
            max_y * 0.9, 
            'What happened in the past', 
            ha='center', 
            fontsize=12, 
            color='#3498db', 
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", fc='white', ec='#3498db', alpha=0.8)
        )
    
    if len(predictions_grouped) > 1:
        ax.text(
            predictions_grouped['Date'].iloc[len(predictions_grouped) // 2], 
            max_y * 0.9, 
            'What will happen in the future', 
            ha='center', 
            fontsize=12, 
            color='#e74c3c', 
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", fc='white', ec='#e74c3c', alpha=0.8)
        )
    
    # Add "Today" annotation at the cutoff point
    ax.annotate('Today', 
                xy=(cutoff_date, min_y * 1.2), 
                xytext=(cutoff_date, min_y * 1.2),
                fontsize=10, ha='center', color='gray', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gray', alpha=0.8))
    
    # Annotate last real value and first predicted value
    if not historical_grouped.empty and len(historical_grouped) > 0:
        last_actual = historical_grouped['Total_Sales'].iloc[-1]
        ax.annotate(f'Last actual: ₹{int(last_actual):,}',
                   xy=(cutoff_date, last_actual),
                   xytext=(-100, 30),
                   textcoords='offset points',
                   arrowprops=dict(arrowstyle='->'),
                   fontsize=10)
    
    if not predictions_grouped.empty and len(predictions_grouped) > 0:
        first_predicted = predictions_grouped['Predicted_Sales'].iloc[0]
        ax.annotate(f'First prediction: ₹{int(first_predicted):,}',
                   xy=(predictions_grouped['Date'].iloc[0], first_predicted),
                   xytext=(30, 30),
                   textcoords='offset points',
                   arrowprops=dict(arrowstyle='->'),
                   fontsize=10)
    
    # Create a clearer legend with shadow and rounded corners
    legend = ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                     ncol=2, frameon=True, facecolor='white')
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('lightgray')
    
    # Add a text box with simple explanation
    explanation = "This chart shows what we sold before and what we expect to sell next.\nBlue line = past sales, Red line = future sales we expect."
    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
                bbox={'facecolor':'#F0F0F0', 'alpha':0.5, 'pad':5})
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Make room for the explanation

    return fig