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
    API_PORT = int(os.getenv('API_PORT', 5000))
    API_RELOAD = os.getenv('API_RELOAD', 'true').lower() == 'true'
    
    # CORS Configuration
    ENVIRONMENT = os.getenv('ENVIRONMENT', 'development').lower()
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '').split(',') if os.getenv('CORS_ORIGINS') else []
    
    @property
    def allowed_origins(self):
        """Get allowed CORS origins based on environment"""
        if self.ENVIRONMENT == 'production':
            # In production, use specific origins or default to weather.fredesk.com
            if self.CORS_ORIGINS:
                return self.CORS_ORIGINS
            else:
                return ['https://weather.fredesk.com', 'http://weather.fredesk.com']
        else:
            # In development, allow all origins
            return ["*"]
    
    # Model Configuration
    MODEL_PATH = os.getenv('MODEL_PATH', 'model.pth')
    METRICS_PATH = os.getenv('METRICS_PATH', 'metrics.json')
    PREDICTION_INTERVAL_MINUTES = int(os.getenv('PREDICTION_INTERVAL_MINUTES', 10))
    
    # Logging Configuration
    LOGGING_ENABLED = os.getenv('LOGGING_ENABLED', 'true').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'info').lower()  # error, warn, info, debug
    LOG_VERBOSE = os.getenv('LOG_VERBOSE', 'false').lower() == 'true'
    
    # Data Processing Configuration
    EXPORT_BATCH_SIZE = int(os.getenv('EXPORT_BATCH_SIZE', 1000))

# Create a global config instance
config = Config() 