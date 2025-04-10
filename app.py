import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import base64
from io import StringIO
import json
import requests

# Import custom modules
from utils.data_processing import process_sales_data, combine_datasets
from utils.visualization import (
    plot_sales_trend, 
    plot_correlation_heatmap, 
    plot_weather_impact, 
    plot_sentiment_impact,
    plot_sales_forecast
)
from utils.model import train_model, predict_sales
from utils.sentiment_analysis import analyze_sentiment
from utils.weather_api import get_weather_data
from utils.firebase_config import (
    firebase_authenticate,
    firebase_register_user,
    save_to_firestore,
    get_from_firestore,
    save_model_to_firebase,
    load_model_from_firebase
)
from utils.database import (
    init_db, 
    save_sales_data, 
    save_weather_data, 
    save_sentiment_data,
    load_sales_data, 
    load_weather_data, 
    load_sentiment_data,
    save_user, 
    save_model_metadata, 
    get_user_models, 
    has_data
)

# Set page configuration
st.set_page_config(
    page_title="Sales Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define session state variables
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'sales_data' not in st.session_state:
    st.session_state.sales_data = None
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
if 'sentiment_data' not in st.session_state:
    st.session_state.sentiment_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'combined_data' not in st.session_state:
    st.session_state.combined_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Authentication function using Firebase
def authenticate(email, password):
    """
    Authenticate a user using Firebase Authentication
    
    Parameters:
    -----------
    email : str
        User email
    password : str
        User password
    
    Returns:
    --------
    bool
        True if authentication successful, False otherwise
    """
    # Use Firebase authentication
    user = firebase_authenticate(email, password)
    
    if user:
        # Store user information in session state
        st.session_state.user_id = user['localId']
        st.session_state.user_token = user['idToken']
        st.session_state.user_email = email
        return True
    
    # If Firebase authentication fails or not configured, fall back to simple validation
    # This is for development purposes only and should be removed in production
    if "BYPASS_AUTH" in st.secrets and st.secrets["BYPASS_AUTH"] and email and password:
        st.session_state.user_id = "test_user"
        st.session_state.user_token = "test_token"
        st.session_state.user_email = email
        return True
        
    return False

# Login page
def login_page():
    st.title("📊 Sales Prediction Dashboard")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image("https://images.unsplash.com/photo-1542744173-05336fcc7ad4", 
                 caption="Data Visualization Dashboard", use_column_width=True)
        
    with col2:
        # Create tabs for login and registration
        login_tab, register_tab = st.tabs(["Login", "Register"])
        
        with login_tab:
            st.subheader("Login")
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login"):
                if authenticate(email, password):
                    st.session_state.authenticated = True
                    st.session_state.username = email.split('@')[0]  # Use first part of email as username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid email or password")
        
        with register_tab:
            st.subheader("Register")
            email = st.text_input("Email", key="register_email")
            password = st.text_input("Password", type="password", key="register_password")
            confirm_password = st.text_input("Confirm Password", type="password")
            display_name = st.text_input("Display Name (optional)")
            
            if st.button("Register"):
                if password != confirm_password:
                    st.error("Passwords don't match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    # Register user with Firebase
                    user = firebase_register_user(email, password, display_name)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.username = display_name if display_name else email.split('@')[0]
                        st.session_state.user_id = user['localId']
                        st.session_state.user_token = user['idToken']
                        st.session_state.user_email = email
                        
                        st.success("Registration successful! You are now logged in.")
                        
                        # Save user data to Firestore
                        user_data = {
                            "email": email,
                            "display_name": display_name if display_name else email.split('@')[0],
                            "created_at": datetime.now().isoformat(),
                            "last_login": datetime.now().isoformat()
                        }
                        save_to_firestore("users", user['localId'], user_data)
                        
                        st.rerun()

# Main application
def main_app():
    # Sidebar for navigation
    if 'user_id' in st.session_state:
        welcome_name = st.session_state.username
        st.sidebar.title(f"Welcome, {welcome_name}")
        
        # User account info
        with st.sidebar.expander("Account Info"):
            st.write(f"Email: {st.session_state.get('user_email', 'Not available')}")
            st.write(f"User ID: {st.session_state.user_id}")
    else:
        st.sidebar.title(f"Welcome, {st.session_state.username}")
    
    # Log out button
    if st.sidebar.button("Log Out"):
        for key in ['authenticated', 'username', 'user_id', 'user_token', 'user_email']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    # Main navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Home", "Data Upload", "Data Visualization", "Sales Prediction", "My Models", "About"]
    )
    
    if page == "Home":
        home_page()
    elif page == "Data Upload":
        data_upload_page()
    elif page == "Data Visualization":
        data_visualization_page()
    elif page == "Sales Prediction":
        sales_prediction_page()
    elif page == "My Models":
        my_models_page()
    elif page == "About":
        about_page()

# Home page
def home_page():
    st.title("📊 Sales Prediction Dashboard")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Application Overview
        
        This application helps you analyze the impact of weather conditions and social media sentiment on your retail sales.
        
        ### Key Features:
        - **Upload sales data** in CSV format
        - **Fetch weather data** based on location and date range
        - **Analyze social media sentiment** related to your products
        - **Visualize correlations** between sales, weather, and sentiment
        - **Predict future sales** using machine learning models
        
        ### Getting Started:
        1. Navigate to the **Data Upload** page
        2. Upload your sales data CSV file
        3. Provide location information for weather data
        4. Enter keywords for social media sentiment analysis
        5. Explore visualizations and predictions
        """)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71", 
                 caption="Data Analysis Dashboard", use_column_width=True)
    
    # Display dashboard images in grid
    st.subheader("Dashboard Previews")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.image("https://images.unsplash.com/photo-1504868584819-f8e8b4b6d7e3", 
                 caption="Sales Analysis", use_column_width=True)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1560472354-b33ff0c44a43", 
                 caption="Performance Metrics", use_column_width=True)
    
    with col3:
        st.image("https://images.unsplash.com/photo-1559526324-4b87b5e36e44", 
                 caption="Sales Prediction", use_column_width=True)
    
    with col4:
        st.image("https://images.unsplash.com/photo-1491438590914-bc09fcaaf77a", 
                 caption="Sentiment Analysis", use_column_width=True)

# Data Upload page
def data_upload_page():
    st.title("Data Upload")
    
    # Sales data upload
    st.subheader("Upload Sales Data")
    st.info("Please upload a CSV file containing your sales data with columns: Date, Product_ID, Category, Quantity, Price")
    
    sales_data_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if sales_data_file is not None:
        try:
            # Read sales data
            data = pd.read_csv(sales_data_file)
            st.success("Sales data uploaded successfully!")
            
            # Display sample data
            st.subheader("Sample Data")
            st.dataframe(data.head())
            
            # Process sales data
            processed_data = process_sales_data(data)
            st.session_state.sales_data = processed_data
            
            # Display summary statistics
            st.subheader("Summary Statistics")
            st.dataframe(processed_data.describe())
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
    
    # Weather data parameters
    st.subheader("Weather Data Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        location = st.text_input("Location (City)", "New York")
        
    with col2:
        days_back = st.slider("Historical days to fetch", 30, 365, 90)
    
    # Social media sentiment parameters
    st.subheader("Social Media Sentiment Parameters")
    
    product_keywords = st.text_input("Product Keywords (comma separated)", "retail, shopping, store")
    
    # Fetch data button
    if st.button("Fetch Weather & Sentiment Data"):
        if st.session_state.sales_data is not None:
            with st.spinner("Fetching weather data..."):
                # Get date range from sales data
                start_date = st.session_state.sales_data['Date'].min()
                end_date = st.session_state.sales_data['Date'].max()
                
                # Get weather data
                weather_data = get_weather_data(location, start_date, end_date)
                st.session_state.weather_data = weather_data
                
                st.success("Weather data fetched successfully!")
                
                # Display sample weather data
                st.subheader("Sample Weather Data")
                st.dataframe(weather_data.head())
            
            with st.spinner("Analyzing social media sentiment..."):
                # Get sentiment data
                keywords = [k.strip() for k in product_keywords.split(",")]
                sentiment_data = analyze_sentiment(keywords, start_date, end_date)
                st.session_state.sentiment_data = sentiment_data
                
                st.success("Sentiment analysis completed!")
                
                # Display sample sentiment data
                st.subheader("Sample Sentiment Data")
                st.dataframe(sentiment_data.head())
            
            # Combine datasets
            with st.spinner("Combining datasets..."):
                combined_data = combine_datasets(
                    st.session_state.sales_data,
                    st.session_state.weather_data,
                    st.session_state.sentiment_data
                )
                st.session_state.combined_data = combined_data
                
                st.success("Data preparation completed!")
                
                # Display combined data
                st.subheader("Combined Dataset")
                st.dataframe(combined_data.head())
        else:
            st.warning("Please upload sales data first!")

# Data Visualization page
def data_visualization_page():
    st.title("Data Visualization")
    
    if st.session_state.combined_data is None:
        st.warning("No data available. Please upload and process data first.")
        return
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Sales Trends", 
        "Weather Impact", 
        "Sentiment Impact", 
        "Correlation Analysis"
    ])
    
    with tab1:
        st.subheader("Sales Trends Over Time")
        
        # Filter options
        product_categories = st.session_state.sales_data['Category'].unique()
        selected_categories = st.multiselect(
            "Select Product Categories",
            options=product_categories,
            default=product_categories[:3] if len(product_categories) > 3 else product_categories
        )
        
        if selected_categories:
            # Filter data
            filtered_data = st.session_state.combined_data[
                st.session_state.combined_data['Category'].isin(selected_categories)
            ]
            
            # Plot sales trend
            fig = plot_sales_trend(filtered_data)
            st.pyplot(fig)
            
            # Display key statistics
            st.subheader("Key Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_sales = filtered_data['Total_Sales'].sum()
                st.metric("Total Sales", f"${total_sales:,.2f}")
            
            with col2:
                avg_sales = filtered_data['Total_Sales'].mean()
                st.metric("Average Daily Sales", f"${avg_sales:,.2f}")
            
            with col3:
                growth = filtered_data.groupby('Date')['Total_Sales'].sum().pct_change().mean() * 100
                st.metric("Average Daily Growth", f"{growth:.2f}%")
        else:
            st.warning("Please select at least one product category.")
    
    with tab2:
        st.subheader("Weather Impact on Sales")
        
        # Weather visualization options
        weather_metric = st.selectbox(
            "Select Weather Metric",
            options=["Temperature", "Precipitation", "Weather_Condition"]
        )
        
        # Plot weather impact
        fig = plot_weather_impact(st.session_state.combined_data, weather_metric)
        st.pyplot(fig)
        
        # Display weather images/icons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.image("https://images.unsplash.com/photo-1531789694268-03cfe5989f89", 
                     caption="Sunny", use_column_width=True)
        
        with col2:
            st.image("https://images.unsplash.com/photo-1697525994215-0fadc6c2c593", 
                     caption="Rainy", use_column_width=True)
        
        with col3:
            st.image("https://images.unsplash.com/photo-1640704124529-3921034f491e", 
                     caption="Snowy", use_column_width=True)
        
        with col4:
            st.image("https://images.unsplash.com/photo-1605028262919-f90a57bf6812", 
                     caption="Cloudy", use_column_width=True)
    
    with tab3:
        st.subheader("Social Media Sentiment Impact")
        
        # Sentiment visualization
        fig = plot_sentiment_impact(st.session_state.combined_data)
        st.pyplot(fig)
        
        # Display sentiment images
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.image("https://images.unsplash.com/photo-1491438590914-bc09fcaaf77a", 
                     caption="Positive Sentiment", use_column_width=True)
        
        with col2:
            st.image("https://images.unsplash.com/photo-1455849318743-b2233052fcff", 
                     caption="Neutral Sentiment", use_column_width=True)
        
        with col3:
            st.image("https://images.unsplash.com/photo-1496449903678-68ddcb189a24", 
                     caption="Negative Sentiment", use_column_width=True)
    
    with tab4:
        st.subheader("Correlation Analysis")
        
        # Plot correlation heatmap
        fig = plot_correlation_heatmap(st.session_state.combined_data)
        st.pyplot(fig)
        
        # Display insights
        st.subheader("Key Insights")
        st.markdown("""
        - **Temperature Correlation:** Shows the relationship between temperature and sales
        - **Sentiment Score Impact:** Illustrates how positive or negative sentiment affects sales
        - **Weather Conditions:** Displays the correlation between different weather conditions and product categories
        - **Product Categories:** Highlights which product categories are most affected by weather and sentiment
        """)

# Sales Prediction page
def sales_prediction_page():
    st.title("Sales Prediction")
    
    if st.session_state.combined_data is None:
        st.warning("No data available. Please upload and process data first.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Train Prediction Model")
        
        # Model parameters
        st.markdown("### Model Parameters")
        
        model_type = st.selectbox(
            "Select Model Type",
            options=["Linear Regression", "Random Forest", "XGBoost"]
        )
        
        test_size = st.slider("Test Data Size (%)", 10, 40, 20) / 100
        
        # Train model button
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Train model
                model, X_train, X_test, y_train, y_test, metrics = train_model(
                    st.session_state.combined_data,
                    model_type,
                    test_size
                )
                
                st.session_state.model = {
                    'model': model,
                    'model_type': model_type,
                    'metrics': metrics,
                    'X_test': X_test,
                    'y_test': y_test
                }
                
                # Save model metadata to Firestore if user is authenticated
                if 'user_id' in st.session_state:
                    model_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # Save model to Firebase Storage
                    model_url = save_model_to_firebase(model, model_name, st.session_state.user_id)
                    
                    if model_url:
                        st.success(f"Model trained and saved successfully! You can access it in My Models.")
                    else:
                        st.success("Model trained successfully! (Model could not be saved to cloud storage)")
                else:
                    st.success("Model trained successfully!")
                
                # Display model metrics
                st.subheader("Model Performance Metrics")
                
                metrics_df = pd.DataFrame({
                    'Metric': ['R² Score', 'Mean Absolute Error', 'Mean Squared Error'],
                    'Value': [
                        metrics['r2'],
                        metrics['mae'],
                        metrics['mse']
                    ]
                })
                
                st.dataframe(metrics_df)
    
    with col2:
        st.image("https://images.unsplash.com/photo-1559526324-4b87b5e36e44", 
                 caption="Sales Prediction", use_column_width=True)
    
    st.subheader("Make Predictions")
    
    if st.session_state.model is not None:
        # Prediction parameters
        st.markdown("### Prediction Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            prediction_days = st.slider("Days to Predict", 1, 30, 7)
            temperature = st.slider("Average Temperature (°C)", -10, 40, 20)
        
        with col2:
            weather_condition = st.selectbox(
                "Weather Condition",
                options=["Sunny", "Rainy", "Cloudy", "Snowy", "Stormy"]
            )
            sentiment_score = st.slider("Sentiment Score", -1.0, 1.0, 0.2, 0.1)
        
        if st.button("Generate Prediction"):
            with st.spinner("Generating predictions..."):
                # Get the last date from our data
                last_date = pd.to_datetime(st.session_state.combined_data['Date'].max())
                
                # Create prediction data
                prediction_data = []
                
                for i in range(1, prediction_days + 1):
                    prediction_date = last_date + timedelta(days=i)
                    for category in st.session_state.combined_data['Category'].unique():
                        prediction_data.append({
                            'Date': prediction_date,
                            'Category': category,
                            'Temperature': temperature,
                            'Weather_Condition': weather_condition,
                            'Sentiment_Score': sentiment_score
                        })
                
                prediction_df = pd.DataFrame(prediction_data)
                
                # Make predictions
                predictions = predict_sales(
                    st.session_state.model['model'],
                    prediction_df,
                    st.session_state.combined_data
                )
                
                st.session_state.predictions = predictions
                
                st.success("Predictions generated successfully!")
                
                # Display predictions
                st.subheader("Sales Predictions")
                st.dataframe(predictions)
                
                # Plot predictions
                fig = plot_sales_forecast(
                    st.session_state.combined_data, 
                    predictions
                )
                st.pyplot(fig)
                
                # Display sales prediction images
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.image("https://images.unsplash.com/photo-1559526324-4b87b5e36e44", 
                             caption="Sales Forecast", use_column_width=True)
                
                with col2:
                    st.image("https://images.unsplash.com/photo-1527264935190-1401c51b5bbc", 
                             caption="Business Growth", use_column_width=True)
                
                with col3:
                    st.image("https://images.unsplash.com/photo-1576748872293-f4972ceda096", 
                             caption="Retail Trends", use_column_width=True)
    else:
        st.info("Please train a model first.")

# My Models page
def my_models_page():
    st.title("My Models")
    
    if 'user_id' not in st.session_state:
        st.warning("Please log in to view your models")
        return
        
    st.subheader("Saved Models")
    
    # Get user's models from Firestore
    user_models = get_from_firestore("models", query=("user_id", "==", st.session_state.user_id))
    
    if not user_models:
        st.info("You haven't saved any models yet. Train a model in the Sales Prediction page.")
        return
    
    # Display models in a table
    models_data = []
    for model in user_models:
        # Extract model data
        model_name = model.get("name", "Unknown")
        created_at = model.get("created_at", "Unknown date")
        if isinstance(created_at, str):
            created_at_display = created_at
        else:
            # Handle Firestore timestamp
            try:
                created_at_display = created_at.strftime("%Y-%m-%d %H:%M:%S")
            except:
                created_at_display = str(created_at)
                
        models_data.append({
            "Model Name": model_name,
            "Created At": created_at_display,
            "Type": model_name.split('_')[0] if '_' in model_name else "Unknown"
        })
    
    # Convert to DataFrame for display
    if models_data:
        models_df = pd.DataFrame(models_data)
        st.dataframe(models_df)
        
        # Model selection for loading
        selected_model = st.selectbox(
            "Select a model to load", 
            [model["Model Name"] for model in models_data]
        )
        
        if st.button("Load Model"):
            with st.spinner("Loading model..."):
                # Load model from Firebase Storage
                model = load_model_from_firebase(selected_model, st.session_state.user_id)
                
                if model:
                    st.session_state.model = {
                        'model': model,
                        'model_type': selected_model.split('_')[0] if '_' in selected_model else "Unknown",
                        'metrics': None,  # We don't have metrics for loaded models
                        'X_test': None,
                        'y_test': None
                    }
                    
                    st.success(f"Model '{selected_model}' loaded successfully!")
                    st.info("You can now use this model in the Sales Prediction page.")
                else:
                    st.error("Failed to load the model. Please try again.")

# About page
def about_page():
    st.title("About")
    
    st.markdown("""
    ## Sales Prediction Application
    
    This application helps retailers predict sales by analyzing the impact of weather conditions and social media sentiment on consumer behavior.
    
    ### How It Works
    
    1. **Data Integration**: The application combines sales data, weather information, and social media sentiment to create a comprehensive dataset.
    
    2. **Exploratory Analysis**: Through various visualizations, users can explore the relationships between sales, weather conditions, and social media sentiment.
    
    3. **Predictive Modeling**: Machine learning models are trained on the integrated data to predict future sales based on forecasted weather and estimated sentiment.
    
    ### Data Sources
    
    - **Sales Data**: Uploaded by the user in CSV format
    - **Weather Data**: Retrieved from weather APIs based on location and date
    - **Social Media Sentiment**: Analyzed from social media posts related to specified keywords
    
    ### Technologies Used
    
    - **Frontend**: Streamlit
    - **Data Processing**: Pandas, NumPy
    - **Machine Learning**: Scikit-learn
    - **Data Visualization**: Matplotlib, Plotly
    - **NLP**: NLTK for sentiment analysis
    
    ### Contact Information
    
    For any questions or suggestions, please contact support@salesprediction.com
    """)

# Main application flow
if st.session_state.authenticated:
    main_app()
else:
    login_page()
