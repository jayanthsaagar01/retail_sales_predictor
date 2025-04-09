import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
from datetime import datetime, timedelta

def train_model(combined_data, model_type="Random Forest", test_size=0.2):
    """
    Train a predictive model for sales
    
    Parameters:
    -----------
    combined_data : pandas.DataFrame
        DataFrame containing combined sales, weather, and sentiment data
    model_type : str
        Type of model to train ('Linear Regression', 'Random Forest', or 'XGBoost')
    test_size : float
        Proportion of data to use for testing
    
    Returns:
    --------
    tuple
        (trained_model, X_train, X_test, y_train, y_test, performance_metrics)
    """
    # Make a copy of the data
    df = combined_data.copy()
    
    # Prepare features and target
    # Drop unnecessary columns
    X = df.drop(['Total_Sales', 'Date'], axis=1)
    if 'Sales_7D_MA' in X.columns:
        X = X.drop(['Sales_7D_MA'], axis=1)
    if 'Quantity' in X.columns:
        X = X.drop(['Quantity'], axis=1)
    
    y = df['Total_Sales']
    
    # Identify categorical and numerical columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    # Select model
    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "XGBoost":
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        except ImportError:
            # Fall back to Random Forest if XGBoost is not available
            model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred)
    }
    
    return pipeline, X_train, X_test, y_train, y_test, metrics

def predict_sales(model, prediction_data, historical_data):
    """
    Generate sales predictions using the trained model
    
    Parameters:
    -----------
    model : sklearn.pipeline.Pipeline
        Trained model pipeline
    prediction_data : pandas.DataFrame
        Data for which to generate predictions
    historical_data : pandas.DataFrame
        Historical data used for reference
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing predictions
    """
    # Make a copy of the prediction data
    pred_df = prediction_data.copy()
    
    # Ensure the prediction data has the same columns as the training data
    # Add any missing columns that were in the training data
    for column in model.feature_names_in_:
        if column not in pred_df.columns and column not in ['Total_Sales', 'Date']:
            # If it's a categorical column, fill with the most common value
            if column in historical_data.select_dtypes(include=['object', 'category']).columns:
                pred_df[column] = historical_data[column].mode()[0]
            # If it's a numerical column, fill with the mean
            else:
                pred_df[column] = historical_data[column].mean()
    
    # Generate predictions
    predictions = model.predict(pred_df)
    
    # Add predictions to the dataframe
    pred_df['Predicted_Sales'] = predictions
    
    # Ensure predictions are positive
    pred_df['Predicted_Sales'] = pred_df['Predicted_Sales'].apply(lambda x: max(0, x))
    
    return pred_df[['Date', 'Category', 'Predicted_Sales']]
