import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import re
import string
from textblob import TextBlob

def analyze_sentiment(keywords, start_date, end_date):
    """
    Analyze sentiment from social media data
    
    Parameters:
    -----------
    keywords : list
        List of keywords to search for
    start_date : str or datetime
        Start date for sentiment analysis
    end_date : str or datetime
        End date for sentiment analysis
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing sentiment data by date
    """
    # Convert dates to datetime if they're strings
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Calculate date range
    date_range = pd.date_range(start=start_date, end=end_date)
    
    # Here we would typically call a social media API to get real data
    # For this example, we'll generate synthetic data based on the provided keywords
    
    # Since we can't access real-time social media data, we'll simulate it
    # In a real application, this would be replaced with API calls
    
    sentiment_data = []
    
    # Generate synthetic sentiment data for each date
    for date in date_range:
        # Calculate sentiment scores
        # In a real application, this would be based on actual social media posts
        
        # Generate synthetic positive and negative post counts
        positive_count = generate_synthetic_count(date, is_positive=True)
        negative_count = generate_synthetic_count(date, is_positive=False)
        neutral_count = generate_synthetic_count(date, is_neutral=True)
        
        # Calculate sentiment score (-1 to 1)
        total_count = positive_count + negative_count + neutral_count
        if total_count > 0:
            sentiment_score = (positive_count - negative_count) / total_count
        else:
            sentiment_score = 0
        
        # Add seasonal and weekly patterns
        sentiment_score = add_time_patterns(sentiment_score, date)
        
        # Generate some example posts (for demonstration purposes)
        example_posts = generate_example_posts(keywords, sentiment_score, 3)
        
        sentiment_data.append({
            'Date': date,
            'Positive_Count': positive_count,
            'Negative_Count': negative_count,
            'Neutral_Count': neutral_count,
            'Total_Count': total_count,
            'Sentiment_Score': sentiment_score,
            'Example_Posts': example_posts
        })
    
    # Convert to DataFrame
    sentiment_df = pd.DataFrame(sentiment_data)
    
    return sentiment_df

def generate_synthetic_count(date, is_positive=False, is_neutral=False):
    """
    Generate synthetic post counts
    
    Parameters:
    -----------
    date : datetime
        Date for which to generate counts
    is_positive : bool
        Whether to generate positive counts
    is_neutral : bool
        Whether to generate neutral counts
    
    Returns:
    --------
    int
        Synthetic post count
    """
    # Base count
    base_count = 50
    
    # Add day of week pattern
    day_of_week = date.dayofweek
    if day_of_week >= 5:  # Weekend
        base_count += 20
    else:  # Weekday
        base_count += 10
    
    # Add monthly pattern
    month = date.month
    if month in [11, 12]:  # Holiday season
        base_count += 30
    elif month in [6, 7, 8]:  # Summer
        base_count += 15
    
    # Add random noise
    noise = random.normalvariate(0, 10)
    count = max(0, base_count + noise)
    
    # Adjust based on sentiment type
    if is_positive:
        # More positive sentiment during holidays and weekends
        if month in [11, 12] or day_of_week >= 5:
            count *= 1.2
    elif is_neutral:
        # Neutral sentiment is generally consistent
        count *= 0.8
    else:  # Negative
        # More negative sentiment during bad weather months
        if month in [1, 2, 3]:
            count *= 1.1
    
    return int(count)

def add_time_patterns(sentiment_score, date):
    """
    Add time-based patterns to sentiment score
    
    Parameters:
    -----------
    sentiment_score : float
        Base sentiment score
    date : datetime
        Date for the sentiment
    
    Returns:
    --------
    float
        Adjusted sentiment score
    """
    # Add day of week pattern
    day_of_week = date.dayofweek
    if day_of_week >= 5:  # Weekend
        sentiment_score += 0.1
    else:  # Weekday
        # Mondays tend to be more negative
        if day_of_week == 0:
            sentiment_score -= 0.1
    
    # Add monthly/seasonal pattern
    month = date.month
    if month in [11, 12]:  # Holiday season
        sentiment_score += 0.15
    elif month in [1, 2]:  # Winter blues
        sentiment_score -= 0.05
    elif month in [4, 5]:  # Spring
        sentiment_score += 0.08
    
    # Ensure score is within -1 to 1 range
    sentiment_score = max(-1, min(1, sentiment_score))
    
    return sentiment_score

def generate_example_posts(keywords, sentiment_score, count=3):
    """
    Generate example social media posts based on keywords and sentiment
    
    Parameters:
    -----------
    keywords : list
        List of keywords to include
    sentiment_score : float
        Sentiment score (-1 to 1)
    count : int
        Number of posts to generate
    
    Returns:
    --------
    list
        List of synthetic posts
    """
    positive_templates = [
        "I love shopping at {store}! Their {product} is amazing!",
        "Just bought a new {product} and I'm so happy with it!",
        "The customer service at {store} is outstanding. Highly recommend!",
        "Great experience with {store} today. Will definitely return!",
        "These {product} are the best I've ever used. Five stars!"
    ]
    
    neutral_templates = [
        "Went to {store} today to get a {product}.",
        "Has anyone tried the new {product} at {store}?",
        "Looking for recommendations on where to find a good {product}.",
        "Comparing prices on {product} between different stores.",
        "The {store} was quite busy today."
    ]
    
    negative_templates = [
        "Disappointed with my purchase of {product} from {store}.",
        "The quality of {product} at {store} has really gone downhill.",
        "Had a poor experience at {store} today. Won't be returning.",
        "The {product} I bought is already broken. Waste of money.",
        "Customer service at {store} was terrible. Still waiting for my refund!"
    ]
    
    # Determine which templates to use based on sentiment score
    if sentiment_score > 0.3:
        primary_templates = positive_templates
        secondary_templates = neutral_templates
        primary_weight = 0.7
    elif sentiment_score < -0.3:
        primary_templates = negative_templates
        secondary_templates = neutral_templates
        primary_weight = 0.7
    else:
        primary_templates = neutral_templates
        if sentiment_score > 0:
            secondary_templates = positive_templates
        else:
            secondary_templates = negative_templates
        primary_weight = 0.6
    
    posts = []
    store_names = ["RetailCo", "ShopMart", "StyleHub", "ValueStore", "TrendyShop"]
    
    for _ in range(count):
        # Randomly choose between primary and secondary templates
        if random.random() < primary_weight:
            template = random.choice(primary_templates)
        else:
            template = random.choice(secondary_templates)
        
        # Choose a random store name and keyword for product
        store = random.choice(store_names)
        product = random.choice(keywords)
        
        # Fill in the template
        post = template.format(store=store, product=product)
        
        posts.append(post)
    
    return posts

def analyze_text_sentiment(text):
    """
    Analyze sentiment of a text string
    
    Parameters:
    -----------
    text : str
        Text to analyze
    
    Returns:
    --------
    float
        Sentiment score (-1 to 1)
    """
    # Clean text
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    
    # Use TextBlob for sentiment analysis
    analysis = TextBlob(text)
    
    # Return polarity score (-1 to 1)
    return analysis.sentiment.polarity
