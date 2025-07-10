import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import Config
import logging
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.config = Config()
        self.last_export_time = None
        self.sensor_data_table = "timeseries"
        
        # Create SQLAlchemy engines
        self.source_engine = create_engine(
            f"mysql+mysqlconnector://{self.config.SOURCE_DB_USER}:{self.config.SOURCE_DB_PASSWORD}@{self.config.SOURCE_DB_HOST}:{self.config.SOURCE_DB_PORT}/{self.config.SOURCE_DB_NAME}"
        )
        self.api_engine = create_engine(
            f"mysql+mysqlconnector://{self.config.API_DB_USER}:{self.config.API_DB_PASSWORD}@{self.config.API_DB_HOST}:{self.config.API_DB_PORT}/{self.config.API_DB_NAME}"
        )
    
    def get_source_connection(self):
        """Create connection to source database (where sensor data comes from)"""
        return mysql.connector.connect(
            user=self.config.SOURCE_DB_USER,
            password=self.config.SOURCE_DB_PASSWORD,
            host=self.config.SOURCE_DB_HOST,
            database=self.config.SOURCE_DB_NAME,
            port=self.config.SOURCE_DB_PORT
        )
    
    def get_api_connection(self):
        """Create connection to API database (where predictions are stored)"""
        return mysql.connector.connect(
            user=self.config.API_DB_USER,
            password=self.config.API_DB_PASSWORD,
            host=self.config.API_DB_HOST,
            database=self.config.API_DB_NAME,
            port=self.config.API_DB_PORT
        )
    
    def export_data(self):
        """Export new data from source database since last export"""
        try:
            # Get last export time or use a default
            if self.last_export_time is None:
                # Get the latest timestamp from the source database
                with self.source_engine.connect() as conn:
                    result = conn.execute(text(f"SELECT MAX(timestamp) FROM {self.sensor_data_table}"))
                    row = result.fetchone()
                    if row[0]:
                        self.last_export_time = row[0]
                    else:
                        # If no data exists, use 24 hours ago
                        self.last_export_time = datetime.now() - timedelta(hours=24)
            
            # Query for new data from source database
            query = f"""
                SELECT timestamp, temperature 
                FROM {self.sensor_data_table} 
                WHERE timestamp > %s 
                ORDER BY timestamp
            """
            
            df = pd.read_sql(query, self.source_engine, params=(self.last_export_time,))
            
            if not df.empty:
                # Update last export time to the latest timestamp
                self.last_export_time = df['timestamp'].max()
                logger.info(f"Exported {len(df)} new records from source database")
            else:
                logger.info("No new data to export from source database")
            
            return df
            
        except Exception as e:
            logger.error(f"Error exporting data from source database: {e}")
            return pd.DataFrame()
    
    def preprocess_data(self, df):
        """Preprocess the data for model input"""
        if df.empty:
            return df
        
        try:
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Set timestamp as index and resample to hourly
            df = df.set_index('timestamp').resample('1H').mean()
            
            # Interpolate missing values
            df.interpolate(method='linear', inplace=True)
            
            # Add additional features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            
            # Normalize temperature (you might want to use a scaler here)
            if 'temperature' in df.columns:
                df['temperature_normalized'] = (df['temperature'] - df['temperature'].mean()) / df['temperature'].std()
            
            logger.info(f"Preprocessed data shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return pd.DataFrame()
    
    def save_predictions(self, predictions_df):
        """Save predictions to API database"""
        try:
            conn = self.get_api_connection()
            cursor = conn.cursor()
            
            # Create predictions table if it doesn't exist
            create_table_query = """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    timestamp DATETIME,
                    predicted_temperature FLOAT,
                    confidence FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            cursor.execute(create_table_query)
            
            # Insert predictions
            insert_query = """
                INSERT INTO predictions (timestamp, predicted_temperature, confidence)
                VALUES (%s, %s, %s)
            """
            
            for index, row in predictions_df.iterrows():
                cursor.execute(insert_query, (
                    index,
                    row.get('predicted_temperature', 0),
                    row.get('confidence', 0.8)
                ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Saved {len(predictions_df)} predictions to API database")
            
        except Exception as e:
            logger.error(f"Error saving predictions to API database: {e}")
    
    def get_latest_predictions(self, hours=24):
        """Get latest predictions from API database"""
        try:
            query = """
                SELECT timestamp, predicted_temperature, confidence
                FROM predictions
                WHERE timestamp >= %s
                ORDER BY timestamp DESC
            """
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            df = pd.read_sql(query, self.api_engine, params=(cutoff_time,))
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting latest predictions from API database: {e}")
            return pd.DataFrame()
    
    def get_latest_sensor_data(self, hours=24):
        """Get latest sensor data from source database"""
        try:
            query = f"""
                SELECT timestamp, temperature
                FROM {self.sensor_data_table}
                WHERE timestamp >= %s
                ORDER BY timestamp DESC
                LIMIT 1000
            """
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            df = pd.read_sql(query, self.source_engine, params=(cutoff_time,))
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting latest sensor data from source database: {e}")
            return pd.DataFrame()
    
    def get_api_database_stats(self):
        """Get statistics from API database"""
        try:
            conn = self.get_api_connection()
            cursor = conn.cursor()
            
            # Count prediction records
            cursor.execute("SELECT COUNT(*) FROM predictions")
            prediction_count = cursor.fetchone()[0]
            
            # Get latest prediction timestamp
            cursor.execute("SELECT MAX(timestamp) FROM predictions")
            latest_prediction = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return {
                "prediction_records": prediction_count,
                "latest_prediction": latest_prediction.isoformat() if latest_prediction else None
            }
            
        except Exception as e:
            logger.error(f"Error getting API database stats: {e}")
            return {"prediction_records": 0, "latest_prediction": None}
    
    def get_source_database_stats(self):
        """Get statistics from source database"""
        try:
            conn = self.get_source_connection()
            cursor = conn.cursor()
            
            # Count sensor data records
            cursor.execute(f"SELECT COUNT(*) FROM {self.sensor_data_table}")
            sensor_count = cursor.fetchone()[0]
            
            # Get latest sensor data timestamp
            cursor.execute(f"SELECT MAX(timestamp) FROM {self.sensor_data_table}")
            latest_sensor = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            return {
                "sensor_records": sensor_count,
                "latest_sensor_data": latest_sensor.isoformat() if latest_sensor else None
            }
            
        except Exception as e:
            logger.error(f"Error getting source database stats: {e}")
            return {"sensor_records": 0, "latest_sensor_data": None} 
    
    def reset_export_time(self):
        """Reset the last export time to force a full data export"""
        self.last_export_time = None
        logger.info("Reset last export time - next export will include all available data")
    
    def force_initial_export(self, hours_back=168):  # 168 hours = 1 week
        """Force an initial data export for model training with specified lookback period"""
        try:
            # Calculate the cutoff time
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            # Query for data from the cutoff time
            query = f"""
                SELECT timestamp, temperature 
                FROM {self.sensor_data_table} 
                WHERE timestamp > %s 
                ORDER BY timestamp
            """
            
            df = pd.read_sql(query, self.source_engine, params=(cutoff_time,))
            
            if not df.empty:
                # Set the last export time to the latest timestamp
                self.last_export_time = df['timestamp'].max()
                logger.info(f"Forced initial export: {len(df)} records from last {hours_back} hours")
            else:
                logger.warning(f"No data found in the last {hours_back} hours")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in forced initial export: {e}")
            return pd.DataFrame() 