from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import logging
import asyncio
from datetime import datetime

from data_processor import DataProcessor
from model_predictor import ModelPredictor
from config import Config

logger = logging.getLogger(__name__)

class WeatherScheduler:
    def __init__(self):
        self.config = Config()
        self.scheduler = AsyncIOScheduler()
        self.data_processor = DataProcessor()
        self.model_predictor = ModelPredictor()
        self.is_running = False
    
    def process_weather_data(self):
        """Main processing job that runs periodically"""
        try:
            logger.info(f"Starting weather data processing at {datetime.now()}")
            
            # Export new data from database
            df = self.data_processor.export_data()
            
            if df.empty:
                logger.info("No new data to process")
                return
            
            logger.info(f"Exported {len(df)} new records")
            
            # Preprocess the data
            processed_df = self.data_processor.preprocess_data(df)
            
            if processed_df.empty:
                logger.warning("No data after preprocessing")
                return
            
            logger.info(f"Preprocessed data shape: {processed_df.shape}")
            
            # Generate predictions
            predictions_df = self.model_predictor.predict_future(processed_df, hours_ahead=24)
            
            if predictions_df.empty:
                logger.warning("No predictions generated")
                return
            
            # Save predictions to database
            self.data_processor.save_predictions(predictions_df)
            
            logger.info(f"Successfully processed {len(predictions_df)} predictions")
            
        except Exception as e:
            logger.error(f"Error in weather data processing: {e}")
    
    def start(self):
        """Start the scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        try:
            # Add the job to run every N minutes
            self.scheduler.add_job(
                func=self.process_weather_data,
                trigger=IntervalTrigger(minutes=self.config.PREDICTION_INTERVAL_MINUTES),
                id='weather_processing',
                name='Weather Data Processing',
                replace_existing=True
            )
            
            # Start the scheduler
            self.scheduler.start()
            self.is_running = True
            
            logger.info(f"Scheduler started. Processing every {self.config.PREDICTION_INTERVAL_MINUTES} minutes")
            
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
            raise
    
    def stop(self):
        """Stop the scheduler"""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        try:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("Scheduler stopped")
            
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
            raise
    
    def get_status(self):
        """Get scheduler status"""
        return {
            "is_running": self.is_running,
            "jobs": [job.id for job in self.scheduler.get_jobs()],
            "next_run": self.scheduler.get_job('weather_processing').next_run_time.isoformat() if self.scheduler.get_job('weather_processing') else None
        }

# Global scheduler instance
scheduler = WeatherScheduler()

async def start_scheduler():
    """Start the scheduler (for use in main.py)"""
    scheduler.start()

async def stop_scheduler():
    """Stop the scheduler (for use in main.py)"""
    scheduler.stop() 