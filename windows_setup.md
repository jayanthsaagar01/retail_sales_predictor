# Setting Up the Retail Forecaster App on Windows

This guide will help you set up and run the Retail Forecaster Application on Windows.

## Requirements

- Windows 10 or 11
- Python 3.9+ (preferably Python 3.11)
- Command Prompt or PowerShell access

## Step 1: Create a Virtual Environment

First, create a virtual environment to isolate the project dependencies:

```powershell
# Navigate to the project directory
cd path\to\project

# Create a Python virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

## Step 2: Install Required Packages

Install the required packages:

```powershell
# Upgrade pip first
pip install --upgrade pip wheel setuptools

# Install the required packages
pip install -r requirements_windows.txt
```

Or install each package individually:

```powershell
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
```

## Step 3: Set Up Environment Variables

If you plan to use the weather API functionality, you'll need to set the OpenWeatherMap API key:

In PowerShell:
```powershell
# Set OpenWeatherMap API key for current session
$env:OPENWEATHERMAP_API_KEY="your_api_key_here"
```

Or for permanent setting, in Command Prompt as Administrator:
```cmd
setx OPENWEATHERMAP_API_KEY "your_api_key_here"
```

## Step 4: Run the Application

Now you can run the Streamlit application:

```powershell
# Ensure you're in the project directory and the virtual environment is activated
streamlit run app.py
```

The application should now be running, and you can access it in your browser at http://localhost:8501

## Step 5: Create a Convenient Run Script

To make it easier to run the app in the future, create a batch script named `run_app.bat`:

```batch
@echo off
:: run_app.bat - Script to run the Retail Forecaster Application on Windows

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
echo Starting Retail Forecaster Application...
streamlit run app.py

:: Deactivate the virtual environment when done
deactivate
```

Now you can run the application by double-clicking the `run_app.bat` file or running it from the command line.

## Troubleshooting Tips

### 1. Package Installation Issues

If you encounter issues with package installation:

```powershell
# Try installing with --no-cache-dir option
pip install --no-cache-dir -r requirements_windows.txt
```

### 2. Database Issues

If the SQLite database doesn't work correctly:

```powershell
# Reset the database
Remove-Item retail_forecaster.db
```

The application will regenerate the database tables on the next start.

### 3. Streamlit Port Conflicts

If port 8501 is already in use:

```powershell
# Specify a different port
streamlit run app.py --server.port 8502
```

## Using Jupyter Notebook

To view the Jupyter notebook explaining the project:

```powershell
# Install Jupyter
pip install jupyter

# Run the notebook server
jupyter notebook
```

Then open `project_explanation.ipynb` from the Jupyter interface.