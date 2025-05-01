@echo off
:: run_app.bat - Script to run the Sales Prediction Application on Windows

echo Checking for virtual environment...
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    pip install --upgrade pip wheel setuptools
    
    echo Installing required packages...
    pip install joblib==1.3.2
    pip install matplotlib==3.8.2
    pip install numpy==1.26.2
    pip install pandas==2.1.3
    pip install pmdarima==2.0.4
    pip install psycopg2==2.9.9
    pip install requests==2.31.0
    pip install scikit-learn==1.3.2
    pip install seaborn==0.13.0
    pip install sqlalchemy==2.0.23
    pip install streamlit==1.28.2
    pip install textblob==0.17.1
    pip install xgboost==2.0.2
    pip install catboost==1.2.2
) else (
    call venv\Scripts\activate
)

:: Check if OPENWEATHERMAP_API_KEY is set
if "%OPENWEATHERMAP_API_KEY%"=="" (
    echo Warning: OPENWEATHERMAP_API_KEY environment variable is not set.
    echo Weather data functionality may be limited.
    echo You can set it by running: $env:OPENWEATHERMAP_API_KEY="your_api_key"
)

:: Run the application
echo Starting Sales Prediction Application...
streamlit run app.py

:: Keep the window open after Streamlit exits
pause