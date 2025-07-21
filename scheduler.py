from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import logging
import asyncio
from datetime import datetime

from weather_data_controller import WeatherDataController
from model_predictor import ModelPredictor
from config import Config

logger = logging.getLogger(__name__)

class WeatherScheduler:
    def __init__(self, model_predictor=None):
        self.config = Config()
        self.scheduler = AsyncIOScheduler()
        self.data_processor = WeatherDataController()
        self.model_predictor = model_predictor if model_predictor else ModelPredictor()
        self.is_running = False
        self.is_initialized = False  # Track if we've done the initial full export
    
    def initialize_system(self):
        """Initialize the system with full data export and model training"""
        try:
            logger.info("Initializing weather forecasting system...")
            
            # Check if we have preprocessed data
            stats = self.data_processor.fetch_preprocessed_data_stats()
            
            if stats["preprocessed_records"] == 0:
                # First time: do full data export and training
                logger.info("No preprocessed data found. Performing initial setup...")
                self.perform_initial_setup()
            else:
                logger.info(f"Found {stats['preprocessed_records']} existing preprocessed records")
                # Load existing data and retrain if needed
                self.load_and_retrain_model()
            
            self.is_initialized = True
            logger.info("System initialization completed")
            
        except Exception as e:
            logger.error(f"Error in system initialization: {e}")
    
    def perform_initial_setup(self, hours_back=168):
        """Perform initial setup with full data export and model training"""
        try:
            logger.info(f"Performing initial setup with {hours_back} hours of data")
            
            # Process and save data (full export)
            raw_df, processed_df = self.data_processor.process_and_store_new_data(
                force_full_export=True, 
                hours_back=hours_back
            )
            
            if processed_df is None or processed_df.empty:
                logger.error("No data available for initial setup")
                return False
            
            # Train the model
            success = self.model_predictor.train_model(processed_df)
            
            if success:
                logger.info("Initial model training completed successfully")
                return True
            else:
                logger.error("Initial model training failed")
                return False
                
        except Exception as e:
            logger.error(f"Error in initial setup: {e}")
            return False
    
    def load_and_retrain_model(self, hours_back=168):
        """Load existing preprocessed data and retrain model if needed"""
        try:
            logger.info("Loading existing preprocessed data for model training")
            
            # Get preprocessed data
            training_df = self.data_processor.fetch_training_data(hours_back=hours_back)
            
            if training_df.empty:
                logger.warning("No preprocessed data available for training")
                return False
            
            # Check if model needs retraining
            if self.model_predictor.needs_training():
                logger.info("Model needs retraining. Starting training...")
                success = self.model_predictor.train_model(training_df)
                
                if success:
                    logger.info("Model retraining completed successfully")
                    return True
                else:
                    logger.error("Model retraining failed")
                    return False
            else:
                logger.info("Model is up to date, no retraining needed")
                return True
                
        except Exception as e:
            logger.error(f"Error in load_and_retrain_model: {e}")
            return False
    
    def process_weather_data(self):
        """Main processing job that runs periodically"""
        try:
            logger.info(f"Starting weather data processing at {datetime.now()}")
            
            # Initialize system if not done yet
            if not self.is_initialized:
                self.initialize_system()
            
            # Process and save new data
            raw_df, processed_df = self.data_processor.process_and_store_new_data()
            
            if processed_df is None or processed_df.empty:
                logger.info("No new data to process")
                return
            
            logger.info(f"Processed and saved {len(processed_df)} new records")
            
            # Check if we should retrain the model
            if self.should_retrain_model():
                logger.info("Triggering model retraining...")
                self.load_and_retrain_model()
            
            # Generate predictions using current model
            predictions_df = self.model_predictor.predict_future(processed_df, hours_ahead=24)
            
            if predictions_df.empty:
                logger.warning("No predictions generated")
                return
            
            # Save predictions to database
            self.data_processor.store_predictions(predictions_df)
            
            logger.info(f"Successfully processed {len(predictions_df)} predictions")
            
        except Exception as e:
            logger.error(f"Error in weather data processing: {e}")
    
    def should_retrain_model(self):
        """Determine if the model should be retrained based on various factors"""
        try:
            # Get stats about preprocessed data
            stats = self.data_processor.fetch_preprocessed_data_stats()
            
            # Retrain if:
            # 1. Model explicitly needs training
            if self.model_predictor.needs_training():
                return True
            
            # 2. We have accumulated significant new data (e.g., more than 1000 new records)
            # With 10-minute periods, 1000 records = ~7 days of data
            if stats["preprocessed_records"] > 1000:
                # Check if we have enough new data since last training
                # For 10-minute periods, retrain every 24 hours of new data (144 records)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking if model should retrain: {e}")
            return False
    
    def run_once(self):
        self.process_weather_data()
    
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
scheduler = None

def create_scheduler(model_predictor=None):
    """Create a new scheduler instance with optional model predictor"""
    global scheduler
    scheduler = WeatherScheduler(model_predictor)
    return scheduler

async def start_scheduler(model_predictor=None):
    """Start the scheduler (for use in main.py)"""
    global scheduler
    if scheduler is None:
        scheduler = create_scheduler(model_predictor)
    scheduler.start()

async def stop_scheduler():
    """Stop the scheduler (for use in main.py)"""
    global scheduler
    if scheduler:
        scheduler.stop() 