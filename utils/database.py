import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey, Table, MetaData, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
import streamlit as st

# Get database URL from environment variable
DATABASE_URL = os.environ.get("DATABASE_URL")

# Create SQLAlchemy engine and session
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# Define database models
class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    firebase_id = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    display_name = Column(String)
    created_at = Column(Date, default=datetime.datetime.now)
    
class Product(Base):
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True)
    product_id = Column(String, unique=True, nullable=False)
    category = Column(String, nullable=False)
    name = Column(String)
    price = Column(Float)
    
class Sales(Base):
    __tablename__ = 'sales'
    
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    product_id = Column(String, ForeignKey('products.product_id'), nullable=False)
    quantity = Column(Integer, nullable=False)
    total_sales = Column(Float, nullable=False)
    
class Weather(Base):
    __tablename__ = 'weather'
    
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False, unique=True)
    location = Column(String, nullable=False)
    temperature = Column(Float)
    precipitation = Column(Float)
    weather_condition = Column(String)
    
class Sentiment(Base):
    __tablename__ = 'sentiment'
    
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False, unique=True)
    keywords = Column(String, nullable=False)
    sentiment_score = Column(Float, nullable=False)
    positive_count = Column(Integer)
    negative_count = Column(Integer)
    neutral_count = Column(Integer)
    
class Model(Base):
    __tablename__ = 'models'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    user_id = Column(String, ForeignKey('users.firebase_id'), nullable=False)
    type = Column(String, nullable=False)
    metrics = Column(String)  # JSON string of metrics
    storage_path = Column(String)  # Path in Firebase Storage
    created_at = Column(Date, default=datetime.datetime.now)

def init_db():
    """Create all tables in the database"""
    Base.metadata.create_all(engine)
    
def get_session():
    """Get a new database session"""
    return Session()

def save_sales_data(sales_df):
    """
    Save sales data to the database
    
    Parameters:
    -----------
    sales_df : pandas.DataFrame
        DataFrame containing sales data
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        session = get_session()
        
        # Extract unique products
        products_df = sales_df[['Product_ID', 'Category', 'Price']].drop_duplicates()
        products_df = products_df.rename(columns={
            'Product_ID': 'product_id',
            'Category': 'category',
            'Price': 'price'
        })
        
        # Add product names (placeholder)
        products_df['name'] = products_df['product_id'].apply(lambda x: f"Product {x}")
        
        # Save products
        products_df.to_sql('products', engine, if_exists='append', index=False, 
                          method='multi', chunksize=1000)
        
        # Prepare sales data
        sales_data = sales_df[['Date', 'Product_ID', 'Quantity', 'Total_Sales']]
        sales_data = sales_data.rename(columns={
            'Date': 'date',
            'Product_ID': 'product_id',
            'Quantity': 'quantity',
            'Total_Sales': 'total_sales'
        })
        
        # Convert date to datetime
        sales_data['date'] = pd.to_datetime(sales_data['date'])
        
        # Save sales data
        sales_data.to_sql('sales', engine, if_exists='append', index=False,
                         method='multi', chunksize=1000)
                         
        session.commit()
        return True
    except Exception as e:
        st.error(f"Error saving sales data: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def save_weather_data(weather_df, location):
    """
    Save weather data to the database
    
    Parameters:
    -----------
    weather_df : pandas.DataFrame
        DataFrame containing weather data
    location : str
        Location for the weather data
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        session = get_session()
        
        # Prepare weather data
        weather_data = weather_df.copy()
        weather_data['location'] = location
        weather_data = weather_data.rename(columns={
            'Date': 'date',
            'Temperature': 'temperature',
            'Precipitation': 'precipitation',
            'Weather_Condition': 'weather_condition'
        })
        
        # Convert date to datetime
        weather_data['date'] = pd.to_datetime(weather_data['date'])
        
        # Save weather data
        weather_data.to_sql('weather', engine, if_exists='append', index=False,
                           method='multi', chunksize=1000)
                           
        session.commit()
        return True
    except Exception as e:
        st.error(f"Error saving weather data: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def save_sentiment_data(sentiment_df, keywords):
    """
    Save sentiment data to the database
    
    Parameters:
    -----------
    sentiment_df : pandas.DataFrame
        DataFrame containing sentiment data
    keywords : list
        List of keywords used for sentiment analysis
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        session = get_session()
        
        # Prepare sentiment data
        sentiment_data = sentiment_df.copy()
        sentiment_data['keywords'] = ','.join(keywords)
        sentiment_data = sentiment_data.rename(columns={
            'Date': 'date',
            'Sentiment_Score': 'sentiment_score',
            'Positive_Count': 'positive_count',
            'Negative_Count': 'negative_count',
            'Neutral_Count': 'neutral_count'
        })
        
        # Convert date to datetime
        sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
        
        # Save sentiment data
        sentiment_data.to_sql('sentiment', engine, if_exists='append', index=False,
                             method='multi', chunksize=1000)
                             
        session.commit()
        return True
    except Exception as e:
        st.error(f"Error saving sentiment data: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def load_sales_data():
    """
    Load sales data from the database
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing sales data
    """
    try:
        # Query to join sales and products tables
        query = """
        SELECT 
            s.date as Date, 
            s.product_id as Product_ID, 
            p.category as Category, 
            s.quantity as Quantity, 
            p.price as Price,
            s.total_sales as Total_Sales
        FROM 
            sales s
        JOIN 
            products p ON s.product_id = p.product_id
        ORDER BY 
            s.date
        """
        
        # Load data into DataFrame
        sales_df = pd.read_sql(query, engine)
        return sales_df
    except Exception as e:
        st.error(f"Error loading sales data: {e}")
        return pd.DataFrame()

def load_weather_data():
    """
    Load weather data from the database
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing weather data
    """
    try:
        # Query weather table
        query = """
        SELECT 
            date as Date, 
            temperature as Temperature, 
            precipitation as Precipitation,
            weather_condition as Weather_Condition
        FROM 
            weather
        ORDER BY 
            date
        """
        
        # Load data into DataFrame
        weather_df = pd.read_sql(query, engine)
        return weather_df
    except Exception as e:
        st.error(f"Error loading weather data: {e}")
        return pd.DataFrame()

def load_sentiment_data():
    """
    Load sentiment data from the database
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing sentiment data
    """
    try:
        # Query sentiment table
        query = """
        SELECT 
            date as Date, 
            sentiment_score as Sentiment_Score,
            positive_count as Positive_Count,
            negative_count as Negative_Count,
            neutral_count as Neutral_Count
        FROM 
            sentiment
        ORDER BY 
            date
        """
        
        # Load data into DataFrame
        sentiment_df = pd.read_sql(query, engine)
        return sentiment_df
    except Exception as e:
        st.error(f"Error loading sentiment data: {e}")
        return pd.DataFrame()

def save_user(firebase_id, email, display_name=None):
    """
    Save user to the database
    
    Parameters:
    -----------
    firebase_id : str
        Firebase user ID
    email : str
        User email
    display_name : str, optional
        User display name
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        session = get_session()
        
        # Check if user already exists
        existing_user = session.query(User).filter_by(firebase_id=firebase_id).first()
        
        if existing_user:
            # Update existing user
            existing_user.email = email
            if display_name:
                existing_user.display_name = display_name
        else:
            # Create new user
            new_user = User(
                firebase_id=firebase_id,
                email=email,
                display_name=display_name
            )
            session.add(new_user)
            
        session.commit()
        return True
    except Exception as e:
        st.error(f"Error saving user: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def save_model_metadata(name, user_id, model_type, metrics=None, storage_path=None):
    """
    Save model metadata to the database
    
    Parameters:
    -----------
    name : str
        Model name
    user_id : str
        Firebase user ID
    model_type : str
        Type of model
    metrics : dict, optional
        Model performance metrics
    storage_path : str, optional
        Path to model in Firebase Storage
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        session = get_session()
        
        # Create new model entry
        new_model = Model(
            name=name,
            user_id=user_id,
            type=model_type,
            metrics=str(metrics) if metrics else None,
            storage_path=storage_path,
            created_at=datetime.datetime.now()
        )
        
        session.add(new_model)
        session.commit()
        return True
    except Exception as e:
        st.error(f"Error saving model metadata: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def get_user_models(user_id):
    """
    Get all models for a user
    
    Parameters:
    -----------
    user_id : str
        Firebase user ID
    
    Returns:
    --------
    list
        List of model dictionaries
    """
    try:
        session = get_session()
        
        # Query models table
        models = session.query(Model).filter_by(user_id=user_id).all()
        
        # Convert to list of dictionaries
        result = []
        for model in models:
            result.append({
                "id": model.id,
                "name": model.name,
                "type": model.type,
                "metrics": model.metrics,
                "storage_path": model.storage_path,
                "created_at": model.created_at
            })
            
        return result
    except Exception as e:
        st.error(f"Error getting user models: {e}")
        return []
    finally:
        session.close()

def has_data():
    """
    Check if the database has any data
    
    Returns:
    --------
    bool
        True if database has data, False otherwise
    """
    try:
        session = get_session()
        
        # Check if sales table has data
        sales_count = session.query(Sales).count()
        
        return sales_count > 0
    except Exception as e:
        st.error(f"Error checking database: {e}")
        return False
    finally:
        session.close()

# Initialize database when module is imported
try:
    init_db()
    print("Database initialized successfully")
except Exception as e:
    print(f"Error initializing database: {e}")