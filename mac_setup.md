# Setting Up the Sales Prediction App on Mac (Apple Silicon)

This guide will help you set up and run the Sales Prediction Application on your Mac with Apple Silicon (M1/M2/M3).

## Requirements

- macOS 11.0 or higher
- Python 3.9+ (preferably Python 3.11)
- Terminal access
- Git (optional, for cloning the repository)

## Step 1: Create a Virtual Environment

First, create a virtual environment to isolate the project dependencies:

```bash
# Navigate to the project directory
cd path/to/project

# Create a Python virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

## Step 2: Install Required Packages

Install the required packages optimized for Apple Silicon:

```bash
# Upgrade pip first
pip install --upgrade pip wheel setuptools

# Install the required packages
pip install joblib==1.3.2
pip install matplotlib==3.8.2
pip install numpy==1.26.2
pip install pandas==2.1.3
pip install pmdarima==2.0.4
pip install psycopg2-binary==2.9.9
pip install requests==2.31.0
pip install scikit-learn==1.3.2
pip install seaborn==0.13.0
pip install sqlalchemy==2.0.23
pip install streamlit==1.28.2
pip install textblob==0.17.1
pip install xgboost==2.0.2
pip install catboost==1.2.2
```

Alternatively, you can save the following to a file named `requirements_mac.txt` and install them all at once:

```
joblib==1.3.2
matplotlib==3.8.2
numpy==1.26.2
pandas==2.1.3
pmdarima==2.0.4
psycopg2-binary==2.9.9
requests==2.31.0
scikit-learn==1.3.2
seaborn==0.13.0
sqlalchemy==2.0.23
streamlit==1.28.2
textblob==0.17.1
xgboost==2.0.2
catboost==1.2.2
```

Then install using:

```bash
pip install -r requirements_mac.txt
```

## Step 3: Set Up Environment Variables

If you plan to use the weather API functionality, you'll need to set the OpenWeatherMap API key:

```bash
# Set OpenWeatherMap API key
export OPENWEATHERMAP_API_KEY="your_api_key_here"
```

To make this permanent, add it to your `~/.zshrc` or `~/.bash_profile` file.

## Step 4: Run the Application

Now you can run the Streamlit application:

```bash
# Ensure you're in the project directory and the virtual environment is activated
streamlit run app.py
```

The application should now be running, and you can access it in your browser at http://localhost:8501

## Step 5: Create a Convenient Run Script

To make it easier to run the app in the future, create a shell script named `run_app.sh`:

```bash
#!/bin/bash
# run_app.sh - Script to run the Sales Prediction Application

# Check if virtual environment exists, if not create one
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip wheel setuptools
    pip install joblib matplotlib numpy pandas pmdarima psycopg2-binary requests scikit-learn seaborn sqlalchemy streamlit textblob xgboost catboost
else
    source venv/bin/activate
fi

# Run the application
streamlit run app.py

# Deactivate the virtual environment when done
deactivate
```

Make the script executable:

```bash
chmod +x run_app.sh
```

Now you can run the application with a simple command:

```bash
./run_app.sh
```

## Troubleshooting Tips

### 1. Silicon-specific Package Issues

If you encounter issues with any packages, try installing them with conda:

```bash
# Install miniconda for Apple Silicon
brew install --cask miniconda

# Create and activate conda environment
conda create -n sales_prediction python=3.11
conda activate sales_prediction

# Install packages using conda
conda install -c conda-forge joblib matplotlib numpy pandas scikit-learn seaborn sqlalchemy streamlit xgboost
```

### 2. Database Issues

If the SQLite database doesn't work correctly:

```bash
# Reset the database
rm sales_prediction.db
```

The application will regenerate the database tables on the next start.

### 3. Visualization Problems

If you encounter visualization issues:

```bash
# Install additional dependencies
pip install pycairo
```

## Using Jupyter Notebook

To view the Jupyter notebook explaining the project:

```bash
# Install Jupyter
pip install jupyter

# Run the notebook server
jupyter notebook
```

Then open `project_explanation.ipynb` from the Jupyter interface.