import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
# Make sure to install CatBoost with: pip install catboost
from catboost import CatBoostRegressor
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
    valid_models = ["Random Forest", "XGBoost", "CatBoost", "Linear Regression", "SVR"]
    if model_type not in valid_models:
        raise ValueError(f"Model type must be one of {valid_models}")

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

    # Create enhanced preprocessing pipeline with improved feature handling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
        ('scaler', RobustScaler())  # More robust to outliers than StandardScaler
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'  # Include any other columns
    )

    # Select model based on user choice
    if model_type == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features='sqrt',
            bootstrap=True,
            random_state=42
        )
    elif model_type == "XGBoost":
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42
        )
    elif model_type == "CatBoost":
        model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.03,
            depth=6,
            loss_function='RMSE',
            verbose=False,
            random_seed=42
        )
    elif model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "SVR":
        model = SVR(
            kernel='rbf',
            C=100,
            epsilon=0.1,
            gamma='scale'
        )
        
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

    # Calculate all the requested metrics
    metrics = {
        'r2': r2_score(y_test, y_pred),  # Proportion of variance explained (1 is perfect)
        'mae': mean_absolute_error(y_test, y_pred),  # Average absolute difference 
        'mse': mean_squared_error(y_test, y_pred),  # Penalizes large errors more heavily
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),  # Root Mean Squared Error
        'mape': np.mean(np.abs((y_test - y_pred) / np.maximum(1e-10, np.abs(y_test)))) * 100  # Mean Absolute Percentage Error
    }

    return pipeline, X_train, X_test, y_train, y_test, metrics

def predict_sales(model, prediction_data, historical_data):
    """Generate sales predictions using the trained model"""
    # We're using a sklearn Pipeline for all models now
    pred_df = prediction_data.copy()
    
    # Ensure date is in the correct format
    pred_df['Date'] = pd.to_datetime(pred_df['Date']).dt.date
    
    # Preprocess weather condition to ensure it's a string
    if 'Weather_Condition' in pred_df.columns:
        pred_df['Weather_Condition'] = pred_df['Weather_Condition'].astype(str)
    
    # Preprocess data
    pred_df = preprocess_data(pred_df)
    
    try:
        # Get required columns from the model
        model_features = []
        if hasattr(model, 'feature_names_in_'):
            model_features = model.feature_names_in_
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps['preprocessor'], 'get_feature_names_out'):
            model_features = model.named_steps['preprocessor'].get_feature_names_out()
        
        missing_cols = set(model_features) - set(pred_df.columns)
        
        # Fill missing columns with appropriate values
        for col in missing_cols:
            if col in historical_data.columns:
                if pd.api.types.is_categorical_dtype(historical_data[col]):
                    mode_val = str(historical_data[col].mode().iloc[0])
                    pred_df[col] = mode_val
                elif historical_data[col].dtype == 'object':
                    mode_val = str(historical_data[col].mode().iloc[0])
                    pred_df[col] = mode_val
                else:
                    mean_val = float(historical_data[col].astype(float).mean())
                    pred_df[col] = mean_val
            else:
                pred_df[col] = 0
    except Exception as e:
        # If error occurs during feature extraction, log and continue
        print(f"Warning: Error processing model features: {e}")
        # We'll proceed with what we have
    
    # Make predictions
    predictions = model.predict(pred_df)
    
    # Create results DataFrame
    results_df = prediction_data.copy()
    results_df['Predicted_Sales'] = predictions
    
    # Ensure no negative sales predictions
    results_df['Predicted_Sales'] = results_df['Predicted_Sales'].apply(lambda x: max(0, x))
    
    return results_df[['Date', 'Category', 'Predicted_Sales']]