import mysql.connector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.config = Config()
        self.last_export_time = None
    
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
    
    def get_connection(self):
        """Legacy method - returns API database connection for backward compatibility"""
        return self.get_api_connection()
    
    def export_data(self):
        """Export new data from source database since last export"""
        try:
            conn = self.get_source_connection()
            
            # Get last export time or use a default
            if self.last_export_time is None:
                # Get the latest timestamp from the source database
                cursor = conn.cursor()
                cursor.execute("SELECT MAX(timestamp) FROM sensor_data")
                result = cursor.fetchone()
                if result[0]:
                    self.last_export_time = result[0]
                else:
                    # If no data exists, use 24 hours ago
                    self.last_export_time = datetime.now() - timedelta(hours=24)
                cursor.close()
            
            # Query for new data from source database
            query = """
                SELECT timestamp, temperature 
                FROM sensor_data 
                WHERE timestamp > %s 
                ORDER BY timestamp
            """
            
            df = pd.read_sql(query, conn, params=[self.last_export_time])
            
            if not df.empty:
                # Update last export time to the latest timestamp
                self.last_export_time = df['timestamp'].max()
                logger.info(f"Exported {len(df)} new records from source database")
            else:
                logger.info("No new data to export from source database")
            
            conn.close()
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
            conn = self.get_api_connection()
            
            query = """
                SELECT timestamp, predicted_temperature, confidence
                FROM predictions
                WHERE timestamp >= %s
                ORDER BY timestamp DESC
            """
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            df = pd.read_sql(query, conn, params=[cutoff_time])
            
            conn.close()
            return df
            
        except Exception as e:
            logger.error(f"Error getting latest predictions from API database: {e}")
            return pd.DataFrame()
    
    def get_latest_sensor_data(self, hours=24):
        """Get latest sensor data from source database"""
        try:
            conn = self.get_source_connection()
            
            query = """
                SELECT timestamp, temperature
                FROM sensor_data
                WHERE timestamp >= %s
                ORDER BY timestamp DESC
                LIMIT 1000
            """
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            df = pd.read_sql(query, conn, params=[cutoff_time])
            
            conn.close()
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
            cursor.execute("SELECT COUNT(*) FROM sensor_data")
            sensor_count = cursor.fetchone()[0]
            
            # Get latest sensor data timestamp
            cursor.execute("SELECT MAX(timestamp) FROM sensor_data")
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