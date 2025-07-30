import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.last_export_time = None
    
    def preprocess_data(self, df):
        """Preprocess the data for model input"""
        if df.empty:
            return df
        
        try:
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Localize naive timestamps to UTC, then convert to Peru/Lima
            if df['timestamp'].dt.tz is None or df['timestamp'].dt.tz is pd.NaT:
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            df['timestamp'] = df['timestamp'].dt.tz_convert('America/Lima')
            
            # Set timestamp as index
            df = df.set_index('timestamp')
            
            # Separate numeric and non-numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
            
            # Resample numeric columns with mean (10-minute periods)
            if len(numeric_columns) > 0:
                df_numeric = df[numeric_columns].resample('10min').mean()  # 10min = 10 minutes
            else:
                df_numeric = pd.DataFrame(index=df.resample('10min').mean().index)
            # Always set index to the resampled timestamps
            df_numeric.index.name = 'timestamp'
            df_numeric = df_numeric.dropna()
            
            logger.info(f"Resampled numeric columns: {list(numeric_columns)}")
            logger.info(f"Non-numeric columns found: {list(non_numeric_columns)}")
            # log first row
            if not df_numeric.empty:
                logger.info(f"First row after resampling: {df_numeric.iloc[0].to_dict()}")
            # For non-numeric columns, we can either drop them or handle them differently
            # For now, let's drop them since they're not needed for weather prediction
            if len(non_numeric_columns) > 0:
                logger.info(f"Dropping non-numeric columns for resampling: {list(non_numeric_columns)}")
            return df_numeric
        except Exception as e:
            logger.error(f"Error in preprocess_data: {e}")
            return pd.DataFrame()
    
    def update_export_time(self, df):
        """Update the last export time based on the latest timestamp in the dataframe"""
        if not df.empty and 'timestamp' in df.columns:
            self.last_export_time = df['timestamp'].max()
            logger.info(f"Updated last export time to: {self.last_export_time}")
    
    def get_last_export_time(self):
        """Get the last export time"""
        return self.last_export_time
    
    def set_last_export_time(self, time):
        """Set the last export time"""
        self.last_export_time = time
        logger.info(f"Set last export time to: {self.last_export_time}")
    
    def reset_export_time(self):
        """Reset the last export time to force a full data export"""
        self.last_export_time = None
        logger.info("Reset last export time - next export will include all available data")
    
    def get_default_export_time(self, hours_back=24):
        """Get a default export time for first run"""
        return datetime.now() - timedelta(hours=hours_back) 