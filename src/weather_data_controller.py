import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from database import DatabaseManager
from preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)

class WeatherDataController:
    """
    Central controller for weather data operations.
    Handles data export, preprocessing, storage, and retrieval.
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.preprocessor = DataPreprocessor()

    # ============================================================================
    # CORE DATA PROCESSING & STORAGE
    # ============================================================================
    
    def process_and_store_new_data(self, force_full_export=False, hours_back=168):
        """
        Main method to process new data and store both raw and preprocessed versions.
        This is the primary entry point for data processing workflows.
        """
        try:
            if force_full_export:
                logger.info(f"Performing full data export for last {hours_back} hours")
                df = self._export_data_since_cutoff(hours_back)
            else:
                df = self._export_incremental_data()

            if df.empty:
                logger.info("No data to process")
                return None, None

            processed_df = self.preprocessor.preprocess_data(df)

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

    def store_predictions(self, predictions_df):
        """Store predictions to API database"""
        self.db_manager.save_predictions(predictions_df)

    # ============================================================================
    # DATA EXPORT & INCREMENTAL PROCESSING
    # ============================================================================
    
    def _export_incremental_data(self):
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
            logger.error(f"Error in _export_incremental_data: {e}")
            return pd.DataFrame()

    def _export_data_since_cutoff(self, hours_back=168):
        """Force data export for a specific time period (internal)"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            df = self.db_manager.export_data_since(cutoff_time)
            if not df.empty:
                self.preprocessor.update_export_time(df)
            return df
        except Exception as e:
            logger.error(f"Error in _export_data_since_cutoff: {e}")
            return pd.DataFrame()

    def reset_export_time(self):
        """Reset the last export time to force a full data export"""
        self.preprocessor.reset_export_time()

    # ============================================================================
    # CONVENIENCE METHODS FOR COMMON USE CASES
    # ============================================================================
    
    def get_training_data(self, hours_back=168):
        """Get preprocessed data for model training"""
        return self.db_manager.get_preprocessed_data(hours_back=hours_back)

    def get_preprocessed_data_for_evaluation(self, cutoff_time):
        """Get preprocessed data for evaluation purposes"""
        return self.db_manager.get_preprocessed_data(cutoff_time=cutoff_time, window_hours=1)

    # ============================================================================
    # INTERNAL METHODS
    # ============================================================================
    
    def _store_preprocessed_data(self, df, batch_id=None):
        """Store preprocessed data to database for model training (internal)"""
        return self.db_manager.save_preprocessed_data(df, batch_id)

 
