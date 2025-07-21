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
            
            # Set timestamp as index
            df = df.set_index('timestamp')
            
            # Separate numeric and non-numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
            
            # Resample numeric columns with mean (10-minute periods)
            if len(numeric_columns) > 0:
                df_numeric = df[numeric_columns].resample('10T').mean()  # 10T = 10 minutes
            else:
                df_numeric = pd.DataFrame(index=df.resample('10T').mean().index)
            
            logger.info(f"Resampled numeric columns: {list(numeric_columns)}")
            logger.info(f"Non-numeric columns found: {list(non_numeric_columns)}")
            # log first row
            if not df_numeric.empty:
                logger.info(f"First row after resampling: {df_numeric.iloc[0].to_dict()}")
            # For non-numeric columns, we can either drop them or handle them differently
            # For now, let's drop them since they're not needed for weather prediction
            if len(non_numeric_columns) > 0:
                logger.info(f"Dropping non-numeric columns for resampling: {list(non_numeric_columns)}")
            
            # Use only numeric columns for the final dataframe
            df = df_numeric
            
            # Interpolate missing values
            df.interpolate(method='linear', inplace=True)
            
            # Add additional features with minute precision
            df['hour'] = df.index.hour
            df['minute'] = df.index.minute
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            
            # Normalize all numeric features
            for column in ['temperature', 'humidity', 'pressure']:
                if column in df.columns:
                    mean_val = df[column].mean()
                    std_val = df[column].std()
                    if std_val > 0:
                        df[f'{column}_normalized'] = (df[column] - mean_val) / std_val
                    else:
                        df[f'{column}_normalized'] = 0
            
            logger.info(f"Preprocessed data shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
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