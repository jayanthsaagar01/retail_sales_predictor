import firebase_admin
from firebase_admin import credentials, firestore, auth
import pyrebase
import os
import json
import streamlit as st

# Initialize Firebase Admin SDK (for server-side operations)
def initialize_firebase_admin():
    """
    Initialize Firebase Admin SDK for server-side operations
    
    Returns:
    --------
    firebase_admin.App
        Firebase Admin app instance
    """
    # Check if Firebase is already initialized
    if not firebase_admin._apps:
        # Use credentials from environment or secrets
        if 'FIREBASE_CREDENTIALS' in st.secrets:
            # Load from Streamlit secrets (JSON string)
            cred_info = json.loads(st.secrets["FIREBASE_CREDENTIALS"])
            cred = credentials.Certificate(cred_info)
            firebase_admin.initialize_app(cred)
        else:
            # Default initialization with no credentials (limited functionality)
            firebase_admin.initialize_app()
            
    return firebase_admin.get_app()

# Initialize Pyrebase (for client-side operations)
def initialize_firebase():
    """
    Initialize Firebase with Pyrebase for client operations
    
    Returns:
    --------
    pyrebase.Firebase
        Firebase instance
    """
    # Firebase configuration
    # In production, these should be stored in environment variables or Streamlit secrets
    config = {
        "apiKey": st.secrets.get("FIREBASE_API_KEY", ""),
        "authDomain": st.secrets.get("FIREBASE_AUTH_DOMAIN", ""),
        "databaseURL": st.secrets.get("FIREBASE_DATABASE_URL", ""),
        "projectId": st.secrets.get("FIREBASE_PROJECT_ID", ""),
        "storageBucket": st.secrets.get("FIREBASE_STORAGE_BUCKET", ""),
        "messagingSenderId": st.secrets.get("FIREBASE_MESSAGING_SENDER_ID", ""),
        "appId": st.secrets.get("FIREBASE_APP_ID", ""),
        "measurementId": st.secrets.get("FIREBASE_MEASUREMENT_ID", "")
    }
    
    return pyrebase.initialize_app(config)

# Firebase Authentication
def firebase_authenticate(email, password):
    """
    Authenticate a user with Firebase Authentication
    
    Parameters:
    -----------
    email : str
        User email
    password : str
        User password
    
    Returns:
    --------
    dict or None
        User information if authentication is successful, None otherwise
    """
    try:
        firebase = initialize_firebase()
        auth = firebase.auth()
        user = auth.sign_in_with_email_and_password(email, password)
        return user
    except Exception as e:
        st.error(f"Authentication failed: {str(e)}")
        return None

# Firebase User Registration
def firebase_register_user(email, password, display_name=None):
    """
    Register a new user with Firebase Authentication
    
    Parameters:
    -----------
    email : str
        User email
    password : str
        User password
    display_name : str, optional
        User display name
    
    Returns:
    --------
    dict or None
        User information if registration is successful, None otherwise
    """
    try:
        firebase = initialize_firebase()
        auth = firebase.auth()
        user = auth.create_user_with_email_and_password(email, password)
        
        # Set display name if provided
        if display_name:
            admin_auth = firebase_admin.auth
            admin_auth.update_user(user['localId'], display_name=display_name)
            
        return user
    except Exception as e:
        st.error(f"Registration failed: {str(e)}")
        return None

# Save data to Firestore
def save_to_firestore(collection, document_id, data):
    """
    Save data to Firestore
    
    Parameters:
    -----------
    collection : str
        Firestore collection name
    document_id : str
        Document ID
    data : dict
        Data to save
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    try:
        # Initialize Firebase Admin if not already initialized
        initialize_firebase_admin()
        
        # Get Firestore client
        db = firestore.client()
        
        # Add or update document
        db.collection(collection).document(document_id).set(data, merge=True)
        
        return True
    except Exception as e:
        st.error(f"Failed to save data: {str(e)}")
        return False

# Get data from Firestore
def get_from_firestore(collection, document_id=None, query=None):
    """
    Get data from Firestore
    
    Parameters:
    -----------
    collection : str
        Firestore collection name
    document_id : str, optional
        Document ID (if None, returns all documents in collection)
    query : tuple, optional
        Query parameters as (field, operator, value)
    
    Returns:
    --------
    dict or list
        Document data or list of documents
    """
    try:
        # Initialize Firebase Admin if not already initialized
        initialize_firebase_admin()
        
        # Get Firestore client
        db = firestore.client()
        
        # Get specific document
        if document_id:
            doc_ref = db.collection(collection).document(document_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict()
            else:
                return None
        
        # Query collection
        elif query:
            field, operator, value = query
            docs = db.collection(collection).where(field, operator, value).stream()
            return [doc.to_dict() for doc in docs]
        
        # Get all documents in collection
        else:
            docs = db.collection(collection).stream()
            return [doc.to_dict() for doc in docs]
            
    except Exception as e:
        st.error(f"Failed to retrieve data: {str(e)}")
        return None

# Save a model to Firebase Storage
def save_model_to_firebase(model, model_name, user_id):
    """
    Save a trained model to Firebase Storage
    
    Parameters:
    -----------
    model : object
        Trained model object
    model_name : str
        Name to identify the model
    user_id : str
        User ID who owns the model
    
    Returns:
    --------
    str or None
        URL to the uploaded model if successful, None otherwise
    """
    try:
        import tempfile
        import joblib
        
        # Initialize Firebase
        firebase = initialize_firebase()
        storage = firebase.storage()
        
        # Create a temporary file to save the model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            # Save model to temporary file
            joblib.dump(model, tmp.name)
            
            # Upload the file
            storage_path = f"models/{user_id}/{model_name}.pkl"
            storage.child(storage_path).put(tmp.name)
            
            # Get the URL
            url = storage.child(storage_path).get_url(None)
            
            # Save metadata to Firestore
            model_meta = {
                "name": model_name,
                "created_at": firestore.SERVER_TIMESTAMP,
                "user_id": user_id,
                "storage_path": storage_path,
                "url": url
            }
            
            save_to_firestore("models", f"{user_id}_{model_name}", model_meta)
            
            return url
            
    except Exception as e:
        st.error(f"Failed to save model: {str(e)}")
        return None

# Load a model from Firebase Storage
def load_model_from_firebase(model_name, user_id):
    """
    Load a model from Firebase Storage
    
    Parameters:
    -----------
    model_name : str
        Name of the model to load
    user_id : str
        User ID who owns the model
    
    Returns:
    --------
    object or None
        Loaded model if successful, None otherwise
    """
    try:
        import tempfile
        import joblib
        
        # Initialize Firebase
        firebase = initialize_firebase()
        storage = firebase.storage()
        
        # Create a temporary file to download the model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            # Download the file
            storage_path = f"models/{user_id}/{model_name}.pkl"
            storage.child(storage_path).download(tmp.name)
            
            # Load model from temporary file
            model = joblib.load(tmp.name)
            
            return model
            
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None