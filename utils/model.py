import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
from datetime import datetime, timedelta
import os

def preprocess_data(df):
    """Clean and preprocess data before model training"""
    # Convert lists to strings if present
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)
    return df

def train_model(combined_data, model_type="Random Forest", test_size=0.2):
    """Train a predictive model for sales"""
    # Make a copy of the data
    df = combined_data.copy()

    # Preprocess data
    df = preprocess_data(df)
    
    # Validate model type
    valid_models = ["Random Forest", "XGBoost", "ARIMA"]
    if model_type not in valid_models:
        raise ValueError(f"Model type must be one of {valid_models}")

    if model_type == "ARIMA":
        # Import pmdarima only when needed
        from pmdarima import auto_arima
        # For ARIMA, we only need the time series data
        time_series = df.groupby('Date')['Total_Sales'].sum().sort_index()
        model = auto_arima(time_series,
                          start_p=1, start_q=1,
                          max_p=3, max_q=3,
                          m=7,  # Weekly seasonality
                          start_P=0, seasonal=True,
                          d=1, D=1,
                          trace=True,
                          error_action='ignore',
                          suppress_warnings=True,
                          stepwise=True)

        # Split data for evaluation
        train_size = int(len(time_series) * (1 - test_size))
        train = time_series[:train_size]
        test = time_series[train_size:]

        # Fit model
        model.fit(train)

        # Make predictions
        predictions = model.predict(n_periods=len(test))

        metrics = {
            'r2': r2_score(test, predictions),
            'mae': mean_absolute_error(test, predictions),
            'mse': mean_squared_error(test, predictions)
        }

        return model, train, test, None, predictions, metrics

    else:
        # Prepare features and target
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
        if model_type == "XGBoost":
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        else:  # Random Forest
            model = RandomForestRegressor(n_estimators=100, random_state=42)

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
    """Generate sales predictions using the trained model"""
    if isinstance(model, Pipeline):
        # For Random Forest and XGBoost
        pred_df = prediction_data.copy()
        pred_df = preprocess_data(pred_df)
        
        missing_cols = set(model.feature_names_in_) - set(pred_df.columns)
        for col in missing_cols:
            if col in historical_data.columns:
                if historical_data[col].dtype == 'object':
                    # Repeat the mode value for each row in pred_df
                    pred_df[col] = [historical_data[col].mode()[0]] * len(pred_df)
                else:
                    # Repeat the mean value for each row in pred_df
                    pred_df[col] = [historical_data[col].mean()] * len(pred_df)
            else:
                pred_df[col] = [0] * len(pred_df)  # Fill with zeros for all rows


        predictions = model.predict(pred_df)
    else:
        # For ARIMA
        predictions = model.predict(n_periods=len(prediction_data))

    pred_df = prediction_data.copy()
    pred_df['Predicted_Sales'] = predictions
    pred_df['Predicted_Sales'] = pred_df['Predicted_Sales'].apply(lambda x: max(0, x))

    return pred_df[['Date', 'Category', 'Predicted_Sales']]