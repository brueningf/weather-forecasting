import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Source Database Configuration (where sensor data comes from)
    SOURCE_DB_HOST = os.getenv('SOURCE_DB_HOST', 'localhost')
    SOURCE_DB_USER = os.getenv('SOURCE_DB_USER', 'user')
    SOURCE_DB_PASSWORD = os.getenv('SOURCE_DB_PASSWORD', 'pass')
    SOURCE_DB_NAME = os.getenv('SOURCE_DB_NAME', 'sensor_db')
    SOURCE_DB_PORT = int(os.getenv('SOURCE_DB_PORT', 3306))
    
    # API Database Configuration (where predictions are stored)
    API_DB_HOST = os.getenv('API_DB_HOST', 'localhost')
    API_DB_USER = os.getenv('API_DB_USER', 'user')
    API_DB_PASSWORD = os.getenv('API_DB_PASSWORD', 'pass')
    API_DB_NAME = os.getenv('API_DB_NAME', 'api_db')
    API_DB_PORT = int(os.getenv('API_DB_PORT', 3306))
    
    # API Configuration
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 8000))
    API_RELOAD = os.getenv('API_RELOAD', 'true').lower() == 'true'
    
    # Model Configuration
    MODEL_PATH = os.getenv('MODEL_PATH', 'model.pth')
    PREDICTION_INTERVAL_MINUTES = int(os.getenv('PREDICTION_INTERVAL_MINUTES', 60))
    
    # Data Export Configuration
    EXPORT_BATCH_SIZE = int(os.getenv('EXPORT_BATCH_SIZE', 1000)) 