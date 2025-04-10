"""
Firebase Storage Utilities

This module provides functions to store and retrieve data from Firebase Firestore
as an alternative to the PostgreSQL database.
"""

import pandas as pd
import json
import streamlit as st
from datetime import datetime
from .firebase_config import initialize_firebase, initialize_firebase_admin
from firebase_admin import firestore

# Save sales data to Firebase
def save_sales_data_to_firebase(sales_df, user_id=None):
    """
    Save sales data to Firebase
    
    Parameters:
    -----------
    sales_df : pandas.DataFrame
        DataFrame containing sales data
    user_id : str, optional
        User ID who owns the data
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Convert the DataFrame to records (list of dictionaries)
        sales_records = sales_df.to_dict('records')
        
        # Group records by date for better storage efficiency
        sales_by_date = {}
        for record in sales_records:
            # Convert datetime to string format
            date_str = record['Date'].strftime('%Y-%m-%d') if hasattr(record['Date'], 'strftime') else str(record['Date'])
            
            if date_str not in sales_by_date:
                sales_by_date[date_str] = []
            
            # Remove the date from the record to avoid duplication
            record_copy = record.copy()
            record_copy['Date'] = date_str
            sales_by_date[date_str].append(record_copy)
        
        # Initialize Firebase Admin if not already initialized
        initialize_firebase_admin()
        
        # Get Firestore client
        db = firestore.client()
        
        # Create a collection reference
        collection_ref = db.collection('sales')
        
        # Create a batch write to save all records
        batch = db.batch()
        
        # Add all records to the batch
        for date_str, records in sales_by_date.items():
            doc_ref = collection_ref.document(f"{user_id}_{date_str}" if user_id else date_str)
            batch.set(doc_ref, {
                'date': date_str,
                'user_id': user_id,
                'records': records,
                'created_at': firestore.SERVER_TIMESTAMP
            }, merge=True)
        
        # Commit the batch
        batch.commit()
        
        # Also store categories and product metadata separately
        product_data = {}
        for record in sales_records:
            product_id = record.get('Product_ID')
            if product_id and product_id not in product_data:
                product_data[product_id] = {
                    'product_id': product_id,
                    'category': record.get('Category', 'Unknown'),
                    'price': record.get('Price', 0)
                }
        
        # Save product data
        product_batch = db.batch()
        product_ref = db.collection('products')
        
        for product_id, data in product_data.items():
            doc_ref = product_ref.document(str(product_id))
            product_batch.set(doc_ref, data, merge=True)
        
        product_batch.commit()
        
        st.success("Sales data saved to Firebase successfully!")
        return True
    except Exception as e:
        st.error(f"Failed to save sales data to Firebase: {str(e)}")
        return False

# Load sales data from Firebase
def load_sales_data_from_firebase(user_id=None):
    """
    Load sales data from Firebase
    
    Parameters:
    -----------
    user_id : str, optional
        User ID to filter data by
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing sales data
    """
    try:
        # Initialize Firebase Admin if not already initialized
        initialize_firebase_admin()
        
        # Get Firestore client
        db = firestore.client()
        
        # Query the collection
        if user_id:
            query = db.collection('sales').where('user_id', '==', user_id)
        else:
            query = db.collection('sales')
            
        # Get all documents
        docs = query.get()
        
        # Extract records
        all_records = []
        for doc in docs:
            data = doc.to_dict()
            records = data.get('records', [])
            all_records.extend(records)
        
        # Convert to DataFrame
        if all_records:
            df = pd.DataFrame(all_records)
            
            # Convert date strings to datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load sales data from Firebase: {str(e)}")
        return pd.DataFrame()

# Save weather data to Firebase
def save_weather_data_to_firebase(weather_df, location, user_id=None):
    """
    Save weather data to Firebase
    
    Parameters:
    -----------
    weather_df : pandas.DataFrame
        DataFrame containing weather data
    location : str
        Location for the weather data
    user_id : str, optional
        User ID who owns the data
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Convert the DataFrame to records (list of dictionaries)
        weather_records = weather_df.to_dict('records')
        
        # Initialize Firebase Admin if not already initialized
        initialize_firebase_admin()
        
        # Get Firestore client
        db = firestore.client()
        
        # Create a collection reference
        collection_ref = db.collection('weather')
        
        # Create a batch write to save all records
        batch = db.batch()
        
        # Add all records to the batch
        for record in weather_records:
            # Convert datetime to string format
            date_str = record['Date'].strftime('%Y-%m-%d') if hasattr(record['Date'], 'strftime') else str(record['Date'])
            
            doc_ref = collection_ref.document(f"{location}_{date_str}")
            record_copy = record.copy()
            record_copy['Date'] = date_str
            record_copy['location'] = location
            record_copy['user_id'] = user_id
            record_copy['created_at'] = firestore.SERVER_TIMESTAMP
            
            batch.set(doc_ref, record_copy, merge=True)
        
        # Commit the batch
        batch.commit()
        
        st.success("Weather data saved to Firebase successfully!")
        return True
    except Exception as e:
        st.error(f"Failed to save weather data to Firebase: {str(e)}")
        return False

# Load weather data from Firebase
def load_weather_data_from_firebase(location=None, user_id=None):
    """
    Load weather data from Firebase
    
    Parameters:
    -----------
    location : str, optional
        Location to filter data by
    user_id : str, optional
        User ID to filter data by
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing weather data
    """
    try:
        # Initialize Firebase Admin if not already initialized
        initialize_firebase_admin()
        
        # Get Firestore client
        db = firestore.client()
        
        # Query the collection
        query = db.collection('weather')
        
        if location:
            query = query.where('location', '==', location)
        
        if user_id:
            query = query.where('user_id', '==', user_id)
            
        # Get all documents
        docs = query.get()
        
        # Extract records
        records = []
        for doc in docs:
            data = doc.to_dict()
            records.append(data)
        
        # Convert to DataFrame
        if records:
            df = pd.DataFrame(records)
            
            # Convert date strings to datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load weather data from Firebase: {str(e)}")
        return pd.DataFrame()

# Save sentiment data to Firebase
def save_sentiment_data_to_firebase(sentiment_df, keywords, user_id=None):
    """
    Save sentiment data to Firebase
    
    Parameters:
    -----------
    sentiment_df : pandas.DataFrame
        DataFrame containing sentiment data
    keywords : list
        List of keywords used for sentiment analysis
    user_id : str, optional
        User ID who owns the data
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Convert the DataFrame to records (list of dictionaries)
        sentiment_records = sentiment_df.to_dict('records')
        
        # Initialize Firebase Admin if not already initialized
        initialize_firebase_admin()
        
        # Get Firestore client
        db = firestore.client()
        
        # Create a collection reference
        collection_ref = db.collection('sentiment')
        
        # Create a batch write to save all records
        batch = db.batch()
        
        # Add all records to the batch
        for record in sentiment_records:
            # Convert datetime to string format
            date_str = record['Date'].strftime('%Y-%m-%d') if hasattr(record['Date'], 'strftime') else str(record['Date'])
            
            keywords_str = ','.join(keywords) if isinstance(keywords, list) else str(keywords)
            doc_ref = collection_ref.document(f"{keywords_str}_{date_str}")
            
            record_copy = record.copy()
            record_copy['Date'] = date_str
            record_copy['keywords'] = keywords_str
            record_copy['user_id'] = user_id
            record_copy['created_at'] = firestore.SERVER_TIMESTAMP
            
            batch.set(doc_ref, record_copy, merge=True)
        
        # Commit the batch
        batch.commit()
        
        st.success("Sentiment data saved to Firebase successfully!")
        return True
    except Exception as e:
        st.error(f"Failed to save sentiment data to Firebase: {str(e)}")
        return False

# Load sentiment data from Firebase
def load_sentiment_data_from_firebase(keywords=None, user_id=None):
    """
    Load sentiment data from Firebase
    
    Parameters:
    -----------
    keywords : str or list, optional
        Keywords to filter data by
    user_id : str, optional
        User ID to filter data by
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing sentiment data
    """
    try:
        # Initialize Firebase Admin if not already initialized
        initialize_firebase_admin()
        
        # Get Firestore client
        db = firestore.client()
        
        # Query the collection
        query = db.collection('sentiment')
        
        if keywords:
            keywords_str = ','.join(keywords) if isinstance(keywords, list) else str(keywords)
            query = query.where('keywords', '==', keywords_str)
        
        if user_id:
            query = query.where('user_id', '==', user_id)
            
        # Get all documents
        docs = query.get()
        
        # Extract records
        records = []
        for doc in docs:
            data = doc.to_dict()
            records.append(data)
        
        # Convert to DataFrame
        if records:
            df = pd.DataFrame(records)
            
            # Convert date strings to datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load sentiment data from Firebase: {str(e)}")
        return pd.DataFrame()

# Helper function to check if Firebase has data
def has_firebase_data(user_id=None):
    """
    Check if Firebase has any data
    
    Parameters:
    -----------
    user_id : str, optional
        User ID to filter data by
    
    Returns:
    --------
    bool
        True if Firebase has data, False otherwise
    """
    try:
        # Initialize Firebase Admin if not already initialized
        initialize_firebase_admin()
        
        # Get Firestore client
        db = firestore.client()
        
        # Query the sales collection
        if user_id:
            query = db.collection('sales').where('user_id', '==', user_id).limit(1)
        else:
            query = db.collection('sales').limit(1)
            
        # Get the documents
        docs = query.get()
        
        # Return True if there are any documents
        return len(list(docs)) > 0
    except Exception as e:
        st.error(f"Failed to check if Firebase has data: {str(e)}")
        return False