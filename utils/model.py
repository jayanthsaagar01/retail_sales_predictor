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
    """Clean and preprocess data before model training with advanced feature engineering"""
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # Convert lists to strings if present
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].apply(lambda x: str(x) if isinstance(x, list) else x)
    
    # Ensure Date is in datetime format
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Extract time-based features
        data['Year'] = data['Date'].dt.year
        data['Month'] = data['Date'].dt.month
        data['Day'] = data['Date'].dt.day
        data['DayOfWeek'] = data['Date'].dt.dayofweek
        data['Quarter'] = data['Date'].dt.quarter
        data['DayOfYear'] = data['Date'].dt.dayofyear
        data['WeekOfYear'] = data['Date'].dt.isocalendar().week
        
        # Create weekend indicator (0 for weekday, 1 for weekend)
        data['IsWeekend'] = data['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Create month start/end indicators
        data['IsMonthStart'] = data['Date'].dt.is_month_start.astype(int)
        data['IsMonthEnd'] = data['Date'].dt.is_month_end.astype(int)
        
        # Create season indicator (1:Winter, 2:Spring, 3:Summer, 4:Fall)
        data['Season'] = data['Month'].apply(lambda month: 
                                            1 if month in [12, 1, 2] else
                                            2 if month in [3, 4, 5] else
                                            3 if month in [6, 7, 8] else 4)
        
        # Create Indian festival indicators (approximate dates for common festivals)
        data['IsDiwali'] = ((data['Month'] == 10) & (data['Day'] >= 20) & (data['Day'] <= 30)).astype(int)
        data['IsHoli'] = ((data['Month'] == 3) & (data['Day'] >= 10) & (data['Day'] <= 20)).astype(int)
        data['IsNavratri'] = (((data['Month'] == 9) | (data['Month'] == 10)) & 
                             (data['Day'] >= 25 if data['Month'] == 9 else data['Day'] <= 15)).astype(int)
        
        # Create end-of-financial-year indicator (March in India)
        data['IsFinancialYearEnd'] = ((data['Month'] == 3) & (data['Day'] >= 25)).astype(int)
        
    # Handle weather data if available
    if 'Temperature' in data.columns:
        # Create temperature bins for easier pattern recognition
        data['Temp_Bin'] = pd.cut(data['Temperature'], 
                                 bins=[-100, 0, 15, 25, 35, 100], 
                                 labels=['Freezing', 'Cold', 'Mild', 'Warm', 'Hot'])
        
        # Create weather interaction features
        if 'Weather_Condition' in data.columns:
            # Weather-Day interactions
            data['Weather_Weekend'] = data.apply(
                lambda row: f"{row['Weather_Condition']}_{row['IsWeekend']}", axis=1)
            
            # Weather-Season interactions
            data['Weather_Season'] = data.apply(
                lambda row: f"{row['Weather_Condition']}_{row['Season']}", axis=1)
            
            # Temperature impact intensity - customize this based on your domain knowledge
            # Hot conditions likely reduce footfall for certain categories
            rain_conditions = ['Rain', 'Drizzle', 'Thunderstorm', 'Rainy', 'Stormy']
            snow_conditions = ['Snow', 'Blizzard', 'Snowy']
            
            data['Is_Rainy'] = data['Weather_Condition'].apply(
                lambda x: 1 if any(cond in x for cond in rain_conditions) else 0)
            data['Is_Snowy'] = data['Weather_Condition'].apply(
                lambda x: 1 if any(cond in x for cond in snow_conditions) else 0)
            
    # Handle sentiment data if available
    if 'Sentiment_Score' in data.columns:
        # Create sentiment bins for better pattern recognition
        data['Sentiment_Bin'] = pd.cut(data['Sentiment_Score'], 
                                      bins=[-1, -0.5, -0.1, 0.1, 0.5, 1], 
                                      labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])
        
        # Sentiment impact lag feature (impact of sentiment on next day/week sales)
        if 'Date' in data.columns:
            # Sort by date to calculate accurate lags
            data = data.sort_values('Date')
            data['Sentiment_Lag_1'] = data['Sentiment_Score'].shift(1)
            data['Sentiment_Lag_7'] = data['Sentiment_Score'].shift(7)
            
    # Create aggregated sale features if available
    if 'Total_Sales' in data.columns:
        # Sort by date
        if 'Date' in data.columns:
            data = data.sort_values('Date')
            
            # Create lag features
            data['Sales_Lag_1'] = data.groupby(['Category'])['Total_Sales'].shift(1)
            data['Sales_Lag_7'] = data.groupby(['Category'])['Total_Sales'].shift(7)
            data['Sales_Lag_30'] = data.groupby(['Category'])['Total_Sales'].shift(30)
            
            # Create moving averages
            data['Sales_MA_7'] = data.groupby(['Category'])['Total_Sales'].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean())
            data['Sales_MA_30'] = data.groupby(['Category'])['Total_Sales'].transform(
                lambda x: x.rolling(window=30, min_periods=1).mean())
            
            # Create expanding mean (cumulative average)
            data['Sales_Expanding_Mean'] = data.groupby(['Category'])['Total_Sales'].transform(
                lambda x: x.expanding().mean())
            
            # Create volatility features
            data['Sales_Volatility_7'] = data.groupby(['Category'])['Total_Sales'].transform(
                lambda x: x.rolling(window=7, min_periods=1).std())
            
    # Handle outliers in sales using IQR method if sales data is available
    if 'Total_Sales' in data.columns:
        # Calculate IQR and bounds for each category
        for category in data['Category'].unique():
            category_data = data[data['Category'] == category]
            Q1 = category_data['Total_Sales'].quantile(0.25)
            Q3 = category_data['Total_Sales'].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds with wider range (3*IQR) to avoid excessive capping
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            # Apply winsorization (capping) instead of removing outliers
            data.loc[(data['Category'] == category) & (data['Total_Sales'] < lower_bound), 'Total_Sales'] = lower_bound
            data.loc[(data['Category'] == category) & (data['Total_Sales'] > upper_bound), 'Total_Sales'] = upper_bound
    
    # Fill missing values with more sophisticated methods
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        # Use forward fill and backward fill for time-series data first
        data[col] = data[col].ffill().bfill()
        
        # If still missing, use median
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].median())
    
    # For categorical features, use mode
    for col in data.select_dtypes(include=['object', 'category']).columns:
        if col != 'Date':  # Skip date column
            data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'unknown')
    
    # Create polynomial features for key numerical variables to capture non-linear relationships
    # Don't create too many to avoid overfitting
    if 'Temperature' in data.columns and 'Sentiment_Score' in data.columns:
        data['Temp_Sentiment_Interaction'] = data['Temperature'] * data['Sentiment_Score']
        data['Temp_Squared'] = data['Temperature'] ** 2  # Quadratic temperature effect

    # Drop the original Date column at the end since we've extracted all useful features
    if 'Date' in data.columns and data.shape[1] > 20:  # Only drop if we have enough other features
        data = data.drop('Date', axis=1)
        
    return data

def train_model(combined_data, model_type="Random Forest", test_size=0.2):
    """Train a predictive model for sales"""
    # Make a copy of the data
    df = combined_data.copy()

    # Preprocess data
    df = preprocess_data(df)
    
    # Validate model type
    valid_models = ["Random Forest", "XGBoost", "CatBoost", "Linear Regression", "SVR", "Ensemble"]
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
    elif model_type == "Ensemble":
        # Create a stacked ensemble model using VotingRegressor for maximum accuracy
        # This combines multiple top models with different strengths
        from sklearn.ensemble import VotingRegressor
        
        # Create base models with different strengths
        rf_model = RandomForestRegressor(
            n_estimators=250, 
            max_depth=10,
            min_samples_split=5,
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        )
        
        xgb_model = XGBRegressor(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            verbosity=1
        )
        
        catboost_model = CatBoostRegressor(
            iterations=250,
            learning_rate=0.03,
            depth=6,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42,
            verbose=50
        )
        
        # Create voting ensemble
        print("Creating ensemble model combining Random Forest, XGBoost, and CatBoost...")
        print("This may take longer to train but will provide higher accuracy")
        
        model = VotingRegressor(
            estimators=[
                ('rf', rf_model),
                ('xgb', xgb_model),
                ('catboost', catboost_model)
            ],
            weights=[1, 1.2, 1.1]  # Give slightly more weight to XGBoost and CatBoost
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
    """Generate sales predictions using the trained model with advanced handling of features"""
    # Make a deep copy to avoid modifying the original data
    pred_df = prediction_data.copy()
    
    # Ensure date is in the correct format
    pred_df['Date'] = pd.to_datetime(pred_df['Date'])
    
    # Create a proper prediction window by adding required features
    # Many features in our preprocessing rely on historical data
    # We'll combine historical and prediction data to calculate proper features
    
    # First prepare historical data in the same format
    hist_df = historical_data.copy()
    if 'Date' in hist_df.columns:
        hist_df['Date'] = pd.to_datetime(hist_df['Date'])
    
    # Check if we have overlapping data by identifying common columns
    common_cols = list(set(pred_df.columns).intersection(set(hist_df.columns)))
    
    # Create time series continuity for better predictions
    if 'Date' in common_cols and 'Category' in common_cols:
        # Sort historical data by date to ensure proper feature construction
        hist_df = hist_df.sort_values('Date')
        
        # Get the latest date from historical data and earliest from prediction data
        hist_max_date = hist_df['Date'].max()
        pred_min_date = pred_df['Date'].min()
        
        # Only combine continuous periods (no gaps)
        if (pred_min_date - hist_max_date).days <= 7:  # Allow up to a week gap
            # Use only relevant historical data - last 60 days to avoid computational issues
            recent_cutoff = pred_min_date - pd.Timedelta(days=60)
            recent_hist = hist_df[hist_df['Date'] >= recent_cutoff]
            
            # Combine datasets for proper time series feature creation
            combined_df = pd.concat([recent_hist, pred_df], ignore_index=True)
            combined_df = combined_df.sort_values('Date')
            
            # Apply preprocessing to combined dataset
            processed_df = preprocess_data(combined_df)
            
            # Split back to get only the prediction period
            mask = processed_df.index.isin(combined_df[combined_df['Date'] >= pred_min_date].index)
            pred_processed = processed_df[mask].copy()
            
            # Handle case where Date column was dropped in preprocessing
            if 'Date' not in pred_processed.columns:
                # Add back the Date column from original prediction data
                date_mapping = dict(zip(combined_df.index, combined_df['Date']))
                pred_processed_indices = pred_processed.index
                pred_processed['Date'] = [date_mapping[idx] for idx in pred_processed_indices]
        else:
            # If there's a gap, process prediction data separately
            pred_processed = preprocess_data(pred_df)
    else:
        # No common time-based columns, process prediction data separately
        pred_processed = preprocess_data(pred_df)
    
    # Preprocess categorical variables
    categorical_cols = pred_processed.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if col != 'Date':  # Skip date column
            pred_processed[col] = pred_processed[col].astype(str)
    
    # Try to get required columns from the model
    try:
        # Get required columns from the model (will vary by model type)
        if hasattr(model, 'feature_names_in_'):
            model_features = model.feature_names_in_
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps['preprocessor'], 'get_feature_names_out'):
            model_features = model.named_steps['preprocessor'].get_feature_names_out()
        else:
            # For complex pipelines or ensembles, we may not be able to extract feature names
            model_features = []
            
        # Handle missing columns if we know what they are
        if model_features:
            missing_cols = set(model_features) - set(pred_processed.columns)
            
            # Fill missing columns with appropriate values
            for col in missing_cols:
                if col in hist_df.columns:
                    if pd.api.types.is_categorical_dtype(hist_df[col]):
                        mode_val = str(hist_df[col].mode().iloc[0])
                        pred_processed[col] = mode_val
                    elif hist_df[col].dtype == 'object':
                        mode_val = str(hist_df[col].mode().iloc[0])
                        pred_processed[col] = mode_val
                    else:
                        mean_val = float(hist_df[col].astype(float).mean())
                        pred_processed[col] = mean_val
                else:
                    pred_processed[col] = 0  # Default value for unknown columns
    except Exception as e:
        # If error occurs during feature extraction, log and continue
        print(f"Warning: Error processing model features: {e}")
    
    # Check if we're using a pipeline or ensemble
    if hasattr(model, 'named_steps') and 'model' in model.named_steps:
        # It's a pipeline, get the actual model
        estimator = model.named_steps['model']
        is_ensemble = isinstance(estimator, (RandomForestRegressor, XGBRegressor, CatBoostRegressor))
    else:
        # Direct model
        is_ensemble = isinstance(model, (RandomForestRegressor, XGBRegressor, CatBoostRegressor))
    
    # Make predictions
    try:
        # Apply model to get predictions
        predictions = model.predict(pred_processed)
        
        # Post-process predictions
        # For ensemble models, we can get prediction intervals (confidence bounds)
        if is_ensemble and 'Category' in pred_df.columns:
            # Prepare final results with additional context
            results_df = pred_df[['Date', 'Category']].copy()
            results_df['Predicted_Sales'] = predictions
            
            # Add prediction confidence intervals for ensemble models if possible
            try:
                # This works for RandomForest and similar models
                if hasattr(model, 'estimators_') or (hasattr(model, 'named_steps') and 
                                                    hasattr(model.named_steps['model'], 'estimators_')):
                    # Get the estimator
                    if hasattr(model, 'estimators_'):
                        estimators = model.estimators_
                    else:
                        estimators = model.named_steps['model'].estimators_
                    
                    # Get predictions from each estimator
                    individual_preds = np.array([estimator.predict(pred_processed) 
                                               for estimator in estimators])
                    
                    # Calculate confidence intervals
                    lower_bound = np.percentile(individual_preds, 5, axis=0)
                    upper_bound = np.percentile(individual_preds, 95, axis=0)
                    
                    # Add to results
                    results_df['Lower_Bound'] = np.maximum(0, lower_bound)  # No negative sales
                    results_df['Upper_Bound'] = upper_bound
                    
                    # Add prediction volatility (uncertainty measure)
                    results_df['Prediction_Volatility'] = np.std(individual_preds, axis=0)
            except Exception as e:
                # If confidence intervals fail, continue without them
                print(f"Warning: Could not calculate prediction intervals: {e}")
        else:
            # For non-ensemble models, just return the predictions
            results_df = pred_df[['Date', 'Category']].copy()
            results_df['Predicted_Sales'] = predictions
        
        # Apply post-processing to predictions
        # Ensure no negative sales predictions
        results_df['Predicted_Sales'] = results_df['Predicted_Sales'].apply(lambda x: max(0, x))
        
        # Add weekday/weekend flag for easier interpretation
        if pd.api.types.is_datetime64_dtype(results_df['Date']):
            results_df['Day_Type'] = results_df['Date'].dt.dayofweek.apply(
                lambda x: 'Weekend' if x >= 5 else 'Weekday')
        
        # Sort by date and category for consistent output
        results_df = results_df.sort_values(['Date', 'Category'])
        
        # Format date for output
        results_df['Date'] = results_df['Date'].dt.date
        
        return results_df
    
    except Exception as e:
        # Fallback if prediction fails
        print(f"Error during prediction: {e}")
        # Create basic results with NaN predictions
        results_df = pred_df[['Date', 'Category']].copy()
        results_df['Predicted_Sales'] = np.nan
        results_df['Date'] = pd.to_datetime(results_df['Date']).dt.date
        return results_df