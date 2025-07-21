import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from database import DatabaseManager
from preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)

class WeatherDataController:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.preprocessor = DataPreprocessor()

    def process_and_store_new_data(self, force_full_export=False, hours_back=168):
        """Process new data and store both raw and preprocessed versions"""
        try:
            if force_full_export:
                logger.info(f"Performing full data export for last {hours_back} hours")
                df = self.initial_data_export(hours_back)
            else:
                df = self._export_data()

            if df.empty:
                logger.info("No data to process")
                return None, None

            processed_df = self.preprocess_raw_data(df)

            if processed_df.empty:
                logger.warning("No data after preprocessing")
                return df, None

            batch_id = self._store_preprocessed_data(processed_df)

            if batch_id:
                logger.info(f"Successfully processed and saved data (batch: {batch_id})")
                return df, processed_df
            else:
                logger.error("Failed to save preprocessed data")
                return df, processed_df
        except Exception as e:
            logger.error(f"Error in process_and_store_new_data: {e}")
            return None, None

    def _export_data(self):
        """Export new data from source database since last export (internal)"""
        try:
            last_export_time = self.preprocessor.get_last_export_time()
            if last_export_time is None:
                last_export_time = self.preprocessor.get_default_export_time()
                logger.info("First run: exporting last 24 hours of data")
            df = self.db_manager.export_data_since(last_export_time)
            if not df.empty:
                self.preprocessor.update_export_time(df)
            return df
        except Exception as e:
            logger.error(f"Error in _export_data: {e}")
            return pd.DataFrame()

    def preprocess_raw_data(self, df):
        """Preprocess the raw data for model input"""
        return self.preprocessor.preprocess_data(df)

    def store_predictions(self, predictions_df):
        """Store predictions to API database"""
        self.db_manager.save_predictions(predictions_df)

    def fetch_recent_sensor_data(self, hours=24, module_id=None):
        """Fetch recent sensor data from source database"""
        return self.db_manager.get_latest_sensor_data(hours, module_id)

    def fetch_api_db_stats(self):
        """Fetch statistics from API database"""
        return self.db_manager.get_api_database_stats()

    def fetch_source_db_stats(self):
        """Fetch statistics from source database"""
        return self.db_manager.get_source_database_stats()

    def reset_export_time(self):
        """Reset the last export time to force a full data export"""
        self.preprocessor.reset_export_time()

    def initial_data_export(self, hours_back=168):
        """Force an initial data export for model training with specified lookback period"""
        try:
            # Calculate the cutoff time based on hours_back
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            df = self.db_manager.export_data_since(cutoff_time)
            if not df.empty:
                self.preprocessor.update_export_time(df)
            return df
        except Exception as e:
            logger.error(f"Error in initial data export: {e}")
            return pd.DataFrame()

    def _store_preprocessed_data(self, df, batch_id=None):
        """Store preprocessed data to database for model training (internal)"""
        return self.db_manager.save_preprocessed_data(df, batch_id)

    def fetch_preprocessed_data(self, hours_back=None, batch_id=None, limit=None):
        """Fetch preprocessed data from database"""
        return self.db_manager.get_preprocessed_data(hours_back, batch_id, limit)

    def fetch_preprocessed_data_stats(self):
        """Fetch statistics about preprocessed data"""
        return self.db_manager.get_preprocessed_data_stats()

    def purge_old_preprocessed_data(self, older_than_days=None):
        """Purge old preprocessed data to manage storage"""
        return self.db_manager.clear_preprocessed_data(older_than_days)

    def fetch_training_data(self, hours_back=168):
        """Fetch preprocessed data for model training"""
        return self.fetch_preprocessed_data(hours_back=hours_back) 
