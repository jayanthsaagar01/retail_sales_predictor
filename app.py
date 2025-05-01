import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

# Import only required modules
from utils.data_processing import process_sales_data, combine_datasets
from utils.visualization import (
    plot_sales_trend, 
    plot_correlation_heatmap, 
    plot_weather_impact, 
    plot_sentiment_impact,
    plot_sales_forecast,
    plot_feature_importance
)
from utils.model import train_model, predict_sales
from utils.sentiment_analysis import analyze_sentiment
from utils.weather_api import get_weather_data
from utils.database import (
    init_db, 
    save_sales_data, 
    save_weather_data, 
    save_sentiment_data,
    load_sales_data, 
    load_weather_data, 
    load_sentiment_data,
    save_model_metadata, 
    get_user_models, 
    load_model,
    has_data,
    authenticate_user,
    register_user
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

# Authentication function using local database (replace Firebase)
def authenticate(email, password):
    """
    Authenticate a user using local database.  Replace with your actual authentication logic.
    """
    return authenticate_user(email, password)


# Login page
def login_page():
    # Initialize database tables
    init_db()
    st.title("📊 Retail Forecaster")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image("https://images.unsplash.com/photo-1542744173-05336fcc7ad4", 
                 caption="Data Visualization Dashboard", use_container_width=True)

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
                    # Placeholder for user registration in local database
                    # Replace with your database registration logic
                    if register_user(email, password, display_name):
                        st.session_state.authenticated = True
                        st.session_state.username = display_name if display_name else email.split('@')[0]
                        st.session_state.user_id = "test_user" # Replace with actual ID from database
                        st.session_state.user_email = email
                        st.success("Registration successful! You are now logged in.")
                        st.rerun()
                    else:
                        st.error("Registration failed. Please try again.")

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
    st.title("📊 Retail Forecaster")

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
                 caption="Data Analysis Dashboard", use_container_width=True)

    # Display dashboard images in grid
    st.subheader("Dashboard Previews")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.image("https://images.unsplash.com/photo-1504868584819-f8e8b4b6d7e3", 
                 caption="Sales Analysis", use_container_width=True)

    with col2:
        st.image("https://images.unsplash.com/photo-1560472354-b33ff0c44a43", 
                 caption="Performance Metrics", use_container_width=True)

    with col3:
        st.image("https://images.unsplash.com/photo-1559526324-4b87b5e36e44", 
                 caption="Sales Prediction", use_container_width=True)

    with col4:
        st.image("https://images.unsplash.com/photo-1491438590914-bc09fcaaf77a", 
                 caption="Sentiment Analysis", use_container_width=True)

# Data Upload page
def data_upload_page():
    st.title("Data Upload")

    # Sales data upload
    st.subheader("Upload Sales Data")
    
    with st.expander("📋 Data Format Requirements (Important)", expanded=True):
        st.markdown("""
        ### Required Data Format for Accurate Predictions
        
        Your CSV file must contain these columns (column names are case-sensitive):
        
        | Column | Format | Description |
        |--------|--------|-------------|
        | **Date** | YYYY-MM-DD | Transaction date (required format) |
        | **Product_ID** | Text/Number | Unique product identifier |
        | **Category** | Text | Product category (e.g., Electronics, Clothing) |
        | **Quantity** | Number | Units sold |
        | **Price** | Number | Price per unit in ₹ |
        
        ### Additional Guidelines:
        - **Date column is critical** and must be in YYYY-MM-DD format
        - Ensure no missing values in Date, Category, Quantity and Price columns
        - Categories should be consistent (same spelling and capitalization)
        - Price should be in Indian Rupees (₹)
        - Include at least 30 days of data for meaningful predictions
        - For best results, upload data with at least 90 days of history
        
        ### Sample Format:
        ```
        Date,Product_ID,Category,Quantity,Price
        2023-01-01,P001,Electronics,5,15000
        2023-01-01,P002,Clothing,10,1200
        2023-01-02,P001,Electronics,3,15000
        ```
        """)
    
    st.info("Please upload a CSV file containing your sales data with the required columns")

    sales_data_file = st.file_uploader("Choose a CSV file", type="csv")

    if sales_data_file is not None:
        try:
            # Read sales data
            data = pd.read_csv(sales_data_file)
            st.success("Sales data uploaded successfully!")

            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(data.head())

            # Process sales data
            processed_data = process_sales_data(data)
            st.session_state.sales_data = processed_data

            # Display summary statistics
            st.subheader("Summary Statistics")
            st.dataframe(processed_data.describe())

            # Save to database
            with st.spinner("Saving data to database..."):
                user_id = st.session_state.get('user_id', None)
                if save_sales_data(processed_data, user_id):
                    st.success("Sales data saved to database successfully!")
                else:
                    st.warning("Data is available in memory but could not be saved to database.")

        except Exception as e:
            st.error(f"Error processing file: {e}")

    # Weather data parameters
    st.subheader("Weather Data Parameters")
    
    with st.expander("🌦️ Weather Data Guidelines", expanded=True):
        st.markdown("""
        ### Weather Data Requirements
        
        For accurate weather-related insights, follow these guidelines:
        
        - **Enter your actual business location** for relevant weather data
        - Indian cities (e.g., Mumbai, Delhi, Bangalore) have comprehensive weather data
        - Weather data will be automatically matched to your sales data dates
        - For best results, we'll collect data on:
          - Daily average temperature
          - Weather conditions (sunny, rainy, etc.)
          - Any extreme weather events
        
        ### How Weather Data Improves Predictions
        - Temperature trends show seasonal buying patterns
        - Weather conditions often influence foot traffic and product interest
        - Helps identify weather-sensitive product categories
        """)

    col1, col2 = st.columns(2)

    with col1:
        location = st.text_input("Location (City)", "Mumbai")

    with col2:
        days_back = st.slider("Historical days to fetch", 30, 365, 90)

    # Social media sentiment parameters
    st.subheader("Social Media Sentiment Parameters")
    
    with st.expander("📱 Sentiment Analysis Guidelines", expanded=True):
        st.markdown("""
        ### Sentiment Analysis Requirements
        
        Sentiment analysis examines public opinion about your products or brand:
        
        - **Enter product-specific keywords** for most relevant results 
          (e.g., "smartphone, mobile phone, iPhone" for an electronics store)
        - **Include your brand name** for brand sentiment analysis
        - **Add product categories** to analyze sentiment by category
        - Separate keywords with commas
        - More specific keywords = more relevant sentiment data
        
        ### How Sentiment Data Improves Predictions
        - Identifies how public perception affects sales
        - Shows correlation between social buzz and buying patterns
        - Helps anticipate demand spikes from positive sentiment
        - Acts as early warning system for potential sales drops
        """)

    product_keywords = st.text_input("Product Keywords (comma separated)", "clothing, fashion, apparel, ethnic wear, festive")

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

                # Save weather data to database
                user_id = st.session_state.get('user_id', None)
                if save_weather_data(weather_data, location, user_id):
                    st.success("Weather data fetched and saved to database successfully!")
                else:
                    st.success("Weather data fetched successfully!")

                # Display weather data preview
                st.subheader("Weather Data Preview")
                st.dataframe(weather_data.head())

            with st.spinner("Analyzing social media sentiment..."):
                # Get sentiment data
                keywords = [k.strip() for k in product_keywords.split(",")]
                sentiment_data = analyze_sentiment(keywords, start_date, end_date)
                st.session_state.sentiment_data = sentiment_data

                # Save sentiment data to database
                user_id = st.session_state.get('user_id', None)
                if save_sentiment_data(sentiment_data, keywords, user_id):
                    st.success("Sentiment analysis completed and saved to database!")
                else:
                    st.success("Sentiment analysis completed!")

                # Display sentiment data preview
                st.subheader("Sentiment Data Preview")
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
                st.metric("Total Sales", f"₹{total_sales:,.2f}")

            with col2:
                avg_sales = filtered_data['Total_Sales'].mean()
                st.metric("Average Daily Sales", f"₹{avg_sales:,.2f}")

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
                     caption="Sunny", use_container_width=True)

        with col2:
            st.image("https://images.unsplash.com/photo-1697525994215-0fadc6c2c593", 
                     caption="Rainy", use_container_width=True)

        with col3:
            st.image("https://images.unsplash.com/photo-1640704124529-3921034f491e", 
                     caption="Snowy", use_container_width=True)

        with col4:
            st.image("https://images.unsplash.com/photo-1605028262919-f90a57bf6812", 
                     caption="Cloudy", use_container_width=True)

    with tab3:
        st.subheader("Social Media Sentiment Impact")

        # Sentiment visualization
        fig = plot_sentiment_impact(st.session_state.combined_data)
        st.pyplot(fig)

        # Display sentiment images
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image("https://images.unsplash.com/photo-1491438590914-bc09fcaaf77a", 
                     caption="Positive Sentiment", use_container_width=True)

        with col2:
            st.image("https://images.unsplash.com/photo-1455849318743-b2233052fcff", 
                     caption="Neutral Sentiment", use_container_width=True)

        with col3:
            st.image("https://images.unsplash.com/photo-1496449903678-68ddcb189a24", 
                     caption="Negative Sentiment", use_container_width=True)

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
        st.subheader("Predict Sales")
        
        # Simple instructions for users
        st.markdown("""
        ### Automatic Ensemble Model
        
        Our system uses a powerful ensemble model that combines multiple algorithms for maximum accuracy:
        
        - Random Forest for capturing non-linear patterns
        - XGBoost for high-performance prediction
        - CatBoost for optimizing categorical features
        
        This combined approach produces more reliable predictions than any single model.
        """)
        
        # Hidden test size, always using 20%
        test_size = 0.2
        
        # Train model button with clearer label
        if st.button("Generate Sales Predictions"):
            with st.spinner("Training model..."):
                # Always use the Ensemble model
                model_type = "Ensemble"
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

                # Save model metadata if user is authenticated
                if 'user_id' in st.session_state:
                    model_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                    # Save model metadata to database
                    if save_model_metadata(model, model_name, st.session_state.user_id, metrics):
                        st.success(f"Model trained and saved successfully! You can access it in My Models.")
                    else:
                        st.success("Model trained successfully! (Model could not be saved to database)")
                else:
                    st.success("Model trained successfully!")

                # Just show a success message - no detailed metrics
                st.success("🚀 Model trained successfully! You can now generate predictions below.")
                
                # Add a simple help section
                with st.expander("📋 Tips for Better Predictions"):
                    st.markdown("""
                    ### Tips for Better Predictions
                    
                    For the most accurate sales predictions:
                    
                    - Upload at least 6-12 months of daily sales data
                    - Include complete data for all categories
                    - Use precise location information for weather data
                    - Include brand-specific keywords for sentiment analysis
                    - Note any special promotions or events in your data
                    """)
                
                # Show feature importance visualization
                st.subheader("What Factors Affect Sales the Most?")
                st.write("This chart shows which factors have the biggest impact on your sales predictions.")
                
                # Get feature names from the model
                if 'X_train' in st.session_state.model:
                    feature_names = st.session_state.model['X_train'].columns.tolist()
                    feature_importance_fig = plot_feature_importance(
                        st.session_state.model['model'], 
                        feature_names
                    )
                    
                    if feature_importance_fig:
                        st.pyplot(feature_importance_fig)
                    else:
                        st.info("Feature importance visualization is not available for this model type.")

    with col2:
        st.image("https://images.unsplash.com/photo-1559526324-4b87b5e36e44", 
                 caption="Sales Prediction", use_container_width=True)

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

                # Display predictions with Indian Rupees (₹) formatting
                st.subheader("Sales Predictions (in ₹)")
                
                # Format the predictions with Indian Rupee symbol and thousand separators
                display_predictions = predictions.copy()
                if 'Predicted_Sales' in display_predictions.columns:
                    display_predictions['Predicted_Sales'] = display_predictions['Predicted_Sales'].apply(
                        lambda x: f"₹{x:,.2f}" if pd.notna(x) else "N/A"
                    )
                
                # Add confidence intervals if available
                if 'Lower_Bound' in display_predictions.columns and 'Upper_Bound' in display_predictions.columns:
                    display_predictions['Confidence Range'] = display_predictions.apply(
                        lambda row: f"₹{row['Lower_Bound']:,.2f} - ₹{row['Upper_Bound']:,.2f}" 
                        if pd.notna(row['Lower_Bound']) and pd.notna(row['Upper_Bound']) else "N/A", 
                        axis=1
                    )
                
                # Add day type column if not already present
                if 'Day_Type' not in display_predictions.columns and 'Date' in display_predictions.columns:
                    display_predictions['Day_Type'] = pd.to_datetime(display_predictions['Date']).dt.dayofweek.apply(
                        lambda x: 'Weekend' if x >= 5 else 'Weekday'
                    )
                
                # Show with helpful styling
                st.dataframe(display_predictions.style.highlight_max(subset=['Predicted_Sales'], color='lightgreen'))

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
                             caption="Sales Forecast", use_container_width=True)

                with col2:
                    st.image("https://images.unsplash.com/photo-1527264935190-1401c51b5bbc", 
                             caption="Business Growth", use_container_width=True)

                with col3:
                    st.image("https://images.unsplash.com/photo-1576748872293-f4972ceda096", 
                             caption="Retail Trends", use_container_width=True)
    else:
        st.info("Please train a model first.")

# My Models page
def my_models_page():
    st.title("My Models")

    if 'user_id' not in st.session_state:
        st.warning("Please log in to view your models")
        return

    st.subheader("Saved Models")

    # Get models from database
    user_models = get_user_models(st.session_state.user_id) or []

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
            # Handle database timestamp (adapt as needed for your database)
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
                # Load model from local database
                model = load_model(selected_model, st.session_state.user_id)

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
    ## Retail Forecaster

    This advanced application helps retailers predict sales by analyzing the impact of weather conditions and social media sentiment on consumer behavior.

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
    - **Data Visualization**: Matplotlib, Seaborn
    - **NLP**: TextBlob for sentiment analysis
    - **Database**: SQLite with SQLAlchemy for local persistent storage
    - **Authentication**: Local authentication system
    - **APIs**: OpenWeatherMap for weather data

    ### Contact Information

    For any questions or suggestions, please contact support@salesprediction.com
    """)

# Main application flow
if st.session_state.authenticated:
    # Initialize local database
    main_app()
else:
    login_page()