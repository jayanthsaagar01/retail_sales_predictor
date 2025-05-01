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

    # Select model based on user choice with optimized hyperparameters for high accuracy
    if model_type == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=500,           # More trees for better accuracy
            max_depth=15,               # Deeper trees to capture complex patterns
            min_samples_split=5,        # Allow more splitting for finer detail
            min_samples_leaf=2,         # Smaller leaf size for more precise predictions
            max_features='sqrt',        # Standard feature selection approach
            bootstrap=True,             # Use bootstrapping for better generalization
            n_jobs=-1,                  # Use all cores for faster training
            random_state=42,
            verbose=1                   # Show progress during training
        )
    elif model_type == "XGBoost":
        model = XGBRegressor(
            n_estimators=500,           # More boosting rounds
            learning_rate=0.01,         # Slower learning rate for better generalization
            max_depth=7,                # Deeper trees
            min_child_weight=1,         # Lower value to allow more specific node creation
            subsample=0.8,              # Use 80% of data for each tree (prevents overfitting)
            colsample_bytree=0.8,       # Use 80% of features for each tree
            colsample_bylevel=0.8,      # Use 80% of features for each level
            gamma=0,                    # Minimum loss reduction for further partition
            reg_alpha=0.001,            # L1 regularization
            reg_lambda=1,               # L2 regularization
            tree_method='hist',         # Faster histogram-based algorithm
            n_jobs=-1,                  # Use all cores
            random_state=42,
            verbosity=1                 # Show progress during training
        )
    elif model_type == "CatBoost":
        model = CatBoostRegressor(
            iterations=1000,            # More iterations for better convergence
            learning_rate=0.01,         # Slower learning rate for better generalization
            depth=8,                    # Deeper trees for more complex patterns
            l2_leaf_reg=3,              # L2 regularization
            loss_function='RMSE',       # Standard loss function for regression
            eval_metric='RMSE',         # Metric to optimize
            task_type='CPU',            # CPU training
            bootstrap_type='Bayesian',  # Bayesian bootstrap for better generalization
            random_seed=42,
            verbose=50                  # Show progress every 50 iterations
        )
    elif model_type == "Linear Regression":
        # Use Ridge regression instead of plain Linear Regression
        model = Ridge(
            alpha=0.5,                  # Regularization strength
            solver='auto',              # Auto-select solver
            random_state=42
        )
    elif model_type == "SVR":
        model = SVR(
            kernel='rbf',               # Radial basis function kernel
            C=100,                      # Regularization parameter
            epsilon=0.1,                # Epsilon in epsilon-SVR model
            gamma=0.01,                 # Kernel coefficient
            cache_size=1000,            # Cache size in MB
            verbose=True                # Show progress during training
        )
        
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Split data using stratification if possible for better representation
    try:
        # Create bins for stratification (regression tasks don't have natural classes)
        y_binned = pd.qcut(y, q=5, duplicates='drop')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y_binned
        )
        print("Using stratified sampling for better model performance")
    except Exception as e:
        # Fall back to regular train_test_split if stratification fails
        print(f"Stratification failed: {e}, using regular train_test_split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluate model
    y_pred = pipeline.predict(X_test)

    # Calculate all the requested metrics with enhanced precision and analysis
    # Ensure predictions are non-negative for real-world sales
    y_pred = np.maximum(0, y_pred)
    
    # Basic metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Calculate MAPE safely (avoiding division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_values = np.abs((y_test - y_pred) / np.maximum(0.01, np.abs(y_test))) * 100
        mape = np.nanmean(mape_values)  # Ignore NaN values that could result from division
    
    # Compile metrics dictionary with analysis insights
    metrics = {
        'r2': r2,                   # Proportion of variance explained (1 is perfect)
        'mae': mae,                 # Average absolute difference 
        'mse': mse,                 # Penalizes large errors more heavily
        'rmse': rmse,               # Root Mean Squared Error
        'mape': mape,               # Mean Absolute Percentage Error
        'accuracy_level': get_accuracy_level(r2, mape)  # Custom accuracy assessment
    }
    
    # Print detailed model performance analysis
    print(f"\n{'='*50}")
    print(f"MODEL PERFORMANCE ANALYSIS: {model_type.upper()}")
    print(f"{'='*50}")
    print(f"R² Score:                    {r2:.4f}  (Higher is better, 1.0 is perfect)")
    print(f"Mean Absolute Error:         {mae:.2f}  (Lower is better)")
    print(f"Mean Squared Error:          {mse:.2f}  (Lower is better)")
    print(f"Root Mean Squared Error:     {rmse:.2f}  (Lower is better, same units as target)")
    print(f"Mean Absolute Percentage Err: {mape:.2f}% (Lower is better)")
    print(f"Model Accuracy Assessment:   {metrics['accuracy_level']}")
    print(f"{'='*50}")
    
    # Detect potential outliers in the residuals
    residuals = y_test - y_pred
    abs_residuals = np.abs(residuals)
    outlier_threshold = abs_residuals.mean() + 2 * abs_residuals.std()
    outlier_count = np.sum(abs_residuals > outlier_threshold)
    
    if outlier_count > 0:
        print(f"Potential prediction outliers detected: {outlier_count} ({outlier_count/len(y_test)*100:.2f}%)")
        print("These outliers might affect model accuracy. Consider feature engineering or ensemble methods.")
    
    return pipeline, X_train, X_test, y_train, y_test, metrics

def get_accuracy_level(r2, mape):
    """
    Provide a qualitative assessment of model accuracy based on metrics
    
    Parameters:
    -----------
    r2 : float
        R-squared value
    mape : float
        Mean Absolute Percentage Error
        
    Returns:
    --------
    str
        Qualitative assessment of model accuracy
    """
    # Combine R² and MAPE for comprehensive accuracy assessment
    if r2 > 0.9 and mape < 10:
        return "Excellent - Production Quality"
    elif r2 > 0.8 and mape < 15:
        return "Very Good - Reliable for Business Decisions"
    elif r2 > 0.7 and mape < 20:
        return "Good - Acceptable for Most Business Uses"
    elif r2 > 0.6 and mape < 30:
        return "Fair - Useful for General Trends"
    elif r2 > 0.5 and mape < 40:
        return "Moderate - Consider Additional Features"
    else:
        return "Needs Improvement - Review Data Quality and Features"

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