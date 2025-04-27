import datetime
import json
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create database engine
engine = create_engine('sqlite:///sales_prediction.db')
Base = declarative_base()
Session = sessionmaker(bind=engine)

# Define models
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    display_name = Column(String)

class SalesData(Base):
    __tablename__ = 'sales'

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'))
    product_id = Column(String)
    category = Column(String)
    quantity = Column(Integer)
    price = Column(Float)

class WeatherData(Base):
    __tablename__ = 'weather'

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    location = Column(String)
    user_id = Column(Integer, ForeignKey('users.id'))
    temperature = Column(Float)
    condition = Column(String)

class SentimentData(Base):
    __tablename__ = 'sentiment'

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'))
    keywords = Column(String)
    sentiment_score = Column(Float)

class Model(Base):
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id'))
    type = Column(String, nullable=False)
    metrics = Column(String)
    model_data = Column(String)  # Serialized model data
    created_at = Column(Date, default=datetime.datetime.now)

def create_tables():
    """Drop all tables and create them again"""
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

def init_db():
    """Create all tables in the database"""
    create_tables()

def get_session():
    """Get a new database session"""
    return Session()

def authenticate_user(email, password):
    """Authenticate user"""
    session = get_session()
    user = session.query(User).filter_by(email=email, password=password).first()
    session.close()
    return user

def register_user(email, password, display_name=None):
    """Register new user"""
    session = get_session()
    try:
        user = User(email=email, password=password, display_name=display_name)
        session.add(user)
        session.commit()
        return user
    except:
        session.rollback()
        return None
    finally:
        session.close()

def save_sales_data(sales_df, user_id):
    """Save sales data to database"""
    session = get_session()
    try:
        for _, row in sales_df.iterrows():
            sale = SalesData(
                date=row['Date'],
                user_id=user_id,
                product_id=row.get('Product_ID'),
                category=row.get('Category'),
                quantity=row.get('Quantity'),
                price=row.get('Price')
            )
            session.add(sale)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        return False
    finally:
        session.close()

def save_weather_data(weather_df, location, user_id):
    """Save weather data to database"""
    session = get_session()
    try:
        for _, row in weather_df.iterrows():
            weather = WeatherData(
                date=row['Date'],
                location=location,
                user_id=user_id,
                temperature=row.get('Temperature'),
                condition=row.get('Weather_Condition')
            )
            session.add(weather)
        session.commit()
        return True
    except:
        session.rollback()
        return False
    finally:
        session.close()

def save_sentiment_data(sentiment_df, keywords, user_id):
    """Save sentiment data to database"""
    session = get_session()
    try:
        keywords_str = ','.join(keywords) if isinstance(keywords, list) else str(keywords)
        for _, row in sentiment_df.iterrows():
            sentiment = SentimentData(
                date=row['Date'],
                user_id=user_id,
                keywords=keywords_str,
                sentiment_score=row.get('Sentiment_Score')
            )
            session.add(sentiment)
        session.commit()
        return True
    except:
        session.rollback()
        return False
    finally:
        session.close()

def load_sales_data(user_id=None):
    """Load sales data from database"""
    session = get_session()
    try:
        query = session.query(SalesData)
        if user_id:
            query = query.filter_by(user_id=user_id)
        records = query.all()

        data = []
        for record in records:
            data.append({
                'Date': record.date,
                'Product_ID': record.product_id,
                'Category': record.category,
                'Quantity': record.quantity,
                'Price': record.price
            })
        return pd.DataFrame(data)
    finally:
        session.close()

def load_weather_data(location=None, user_id=None):
    """Load weather data from database"""
    session = get_session()
    try:
        query = session.query(WeatherData)
        if location:
            query = query.filter_by(location=location)
        if user_id:
            query = query.filter_by(user_id=user_id)
        records = query.all()

        data = []
        for record in records:
            data.append({
                'Date': record.date,
                'Location': record.location,
                'Temperature': record.temperature,
                'Weather_Condition': record.condition
            })
        return pd.DataFrame(data)
    finally:
        session.close()

def load_sentiment_data(keywords=None, user_id=None):
    """Load sentiment data from database"""
    session = get_session()
    try:
        query = session.query(SentimentData)
        if keywords:
            keywords_str = ','.join(keywords) if isinstance(keywords, list) else str(keywords)
            query = query.filter_by(keywords=keywords_str)
        if user_id:
            query = query.filter_by(user_id=user_id)
        records = query.all()

        data = []
        for record in records:
            data.append({
                'Date': record.date,
                'Keywords': record.keywords,
                'Sentiment_Score': record.sentiment_score
            })
        return pd.DataFrame(data)
    finally:
        session.close()

def has_data(user_id=None):
    """Check if database has data"""
    session = get_session()
    try:
        query = session.query(SalesData)
        if user_id:
            query = query.filter_by(user_id=user_id)
        return query.count() > 0
    finally:
        session.close()

def save_model(model, model_name, user_id, metrics=None):
    """Save model to database"""
    import joblib
    import io
    import base64

    # Serialize model to base64 string
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    model_data = base64.b64encode(buffer.getvalue()).decode()

    session = get_session()
    try:
        model_record = Model(
            name=model_name,
            user_id=user_id,
            type=model_name.split('_')[0],
            metrics=json.dumps(metrics) if metrics else None,
            model_data=model_data
        )
        session.add(model_record)
        session.commit()
        return True
    except:
        session.rollback()
        return False
    finally:
        session.close()

def load_model(model_name, user_id):
    """Load model from database"""
    import joblib
    import io
    import base64

    session = get_session()
    try:
        model_record = session.query(Model).filter_by(
            name=model_name,
            user_id=user_id
        ).first()

        if model_record and model_record.model_data:
            # Deserialize model from base64 string
            buffer = io.BytesIO(base64.b64decode(model_record.model_data))
            return joblib.load(buffer)
        return None
    finally:
        session.close()

def get_user_models(user_id):
    """Get all models for a user"""
    session = get_session()
    try:
        models = session.query(Model).filter_by(user_id=user_id).all()
        return [{
            'name': model.name,
            'type': model.type,
            'created_at': model.created_at,
            'metrics': json.loads(model.metrics) if model.metrics else None
        } for model in models]
    finally:
        session.close()

def save_model_metadata(model, model_name, user_id, metrics):
    """Save model metadata to database"""
    session = get_session()
    try:
        # Serialize model data using joblib
        import joblib
        import io
        import base64

        # Serialize model to base64 string
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        model_data = base64.b64encode(buffer.getvalue()).decode()

        model_record = Model(
            name=model_name,
            user_id=user_id,
            type=model_name.split('_')[0],
            metrics=json.dumps(metrics) if metrics else None,
            model_data=model_data
        )
        session.add(model_record)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        return False
    finally:
        session.close()
