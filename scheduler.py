from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
import logging
import asyncio
from datetime import datetime, timedelta
import pandas as pd
import os

from weather_data_controller import WeatherDataController
from model_predictor import ModelPredictor
from config import Config

logger = logging.getLogger(__name__)

SCHEDULER_STATUS_FILE = 'scheduler_status.flag'

class WeatherScheduler:
    def __init__(self, model_predictor=None):
        try:
            logger.info("Initializing WeatherScheduler...")
            self.config = Config()
            logger.debug("Config loaded successfully")
            
            self.scheduler = AsyncIOScheduler()
            logger.debug("AsyncIOScheduler created successfully")
            
            self.data_processor = WeatherDataController()
            logger.debug("WeatherDataController created successfully")
            
            self.model_predictor = model_predictor if model_predictor else ModelPredictor()
            logger.debug("ModelPredictor created successfully")
            
            self.is_running = False
            self.is_initialized = False  # Track if we've done the initial full export
            
            logger.info("WeatherScheduler initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing WeatherScheduler: {e}")
            import traceback
            logger.error(f"WeatherScheduler init traceback: {traceback.format_exc()}")
            raise
    
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
            self.backfill_predictions()
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
                self.backfill_predictions()
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

    def data_collection_job(self):
        """Data collection job - processes and stores new weather data"""
        try:
            logger.info(f"Starting data collection at {datetime.now()}")
            
            # Process and save new data only
            raw_df, processed_df = self.data_processor.process_and_store_new_data()
            
            if processed_df is None or processed_df.empty:
                logger.info("No new data to process")
                return
            
            logger.info(f"Processed and saved {len(processed_df)} new records")
            
        except Exception as e:
            logger.error(f"Error in data collection: {e}")
    
    def prediction_job(self):
        """Prediction job - fills missing 10-min predictions for the next hour"""
        try:
            logger.info(f"Starting 10-min interval prediction generation at {datetime.now()}")

            if not self.is_initialized:
                logger.info("System not initialized. Initializing now...")
                self.initialize_system()
                if not self.is_initialized:
                    logger.warning("Initialization failed. Skipping prediction.")
                    return

            # 1. Get all existing predictions for the next hour (10-min intervals)
            now = datetime.now().replace(second=0, microsecond=0)
            # Align 'now' to the previous 10-min mark
            minute = (now.minute // 10) * 10
            now_aligned = now.replace(minute=minute)
            future_times = [now_aligned + timedelta(minutes=10 * i) for i in range(1, 7)]  # next 60 minutes, 10-min steps
            existing_preds_df = self.data_processor.db_manager.get_latest_predictions(hours=1)
            existing_pred_times = set(pd.to_datetime(existing_preds_df['timestamp']).dt.floor('10min')) if not existing_preds_df.empty else set()

            # 2. Determine which timestamps are missing
            missing_times = [t for t in future_times if t not in existing_pred_times]
            if not missing_times:
                logger.info("All 10-min predictions for the next hour already exist. Nothing to do.")
                return

            # 3. Get the latest preprocessed data (should be 10-min frequency)
            sequence_length = self.model_predictor.sequence_length
            preprocessed_df = self.data_processor.fetch_preprocessed_data(limit=sequence_length)
            if preprocessed_df is None or preprocessed_df.empty or len(preprocessed_df) < sequence_length:
                logger.error("No preprocessed data available or not enough rows for prediction.")
                return
            preprocessed_df = preprocessed_df.sort_index()
            history_df = preprocessed_df.tail(sequence_length).copy()

            # 4. Generate and save predictions for each missing 10-min timestamp
            for ts in missing_times:
                # Ensure history_df has enough rows
                if len(history_df) < sequence_length:
                    logger.warning(f"Not enough history for prediction at {ts} (have {len(history_df)}, need {sequence_length})")
                    break
                input_df = history_df.copy()
                input_df.index.name = 'timestamp'  # Ensure index is named for downstream code
                # Predict for 1 period, but force the model to use 10-min freq
                pred_df = self.model_predictor.predict(input_df, forecast_periods=1)
                if pred_df.empty:
                    logger.warning(f"No prediction generated for {ts}")
                    break
                pred_temp = float(pred_df.iloc[0]['predicted_temperature'])
                # Use last humidity/pressure (or optionally model them too)
                last_row = history_df.iloc[-1]
                pred_row = {
                    'temperature': pred_temp,
                    'humidity': last_row['humidity'],
                    'pressure': last_row['pressure']
                }
                # Append new prediction to history_df, drop oldest to keep window size
                new_row_df = pd.DataFrame([pred_row], index=[ts])
                history_df = pd.concat([history_df, new_row_df])
                history_df = history_df.tail(sequence_length)
                # Save this prediction immediately
                single_pred_df = pd.DataFrame([{
                    'timestamp': ts,
                    'predicted_temperature': pred_temp,
                    'confidence': 0.8
                }]).set_index('timestamp')
                self.data_processor.store_predictions(single_pred_df)
                logger.info(f"Generated and stored 10-min prediction for {ts}")
            self.backfill_predictions()

        except Exception as e:
            logger.error(f"Error in 10-min prediction job: {e}")
    
    def training_job(self):
        """Training job - only handles model retraining"""
        try:
            logger.info(f"Starting model training job at {datetime.now()}")
            
            # Check if we should retrain the model
            if not self.should_retrain_model():
                logger.info("Model retraining not needed at this time")
                return
            
            logger.info("Model retraining required. Starting training...")
            success = self.load_and_retrain_model()
            
            if success:
                logger.info("Model retraining completed successfully")
            else:
                logger.error("Model retraining failed")
            
        except Exception as e:
            logger.error(f"Error in training job: {e}")
    
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
    
    def run_once_prediction(self):
        """Run prediction job once"""
        self.prediction_job()
        
    def run_once_training(self):
        """Run training job once"""
        self.training_job()
        
    def run_once_data_collection(self):
        """Run data collection job once"""
        self.data_collection_job()
    
    def start(self):
        """Start the scheduler with separate jobs for different concerns"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        try:
            # Data collection job - frequent (every prediction interval)
            self.scheduler.add_job(
                func=self.data_collection_job,
                trigger=IntervalTrigger(minutes=self.config.PREDICTION_INTERVAL_MINUTES),
                id='data_collection',
                name='Weather Data Collection',
                replace_existing=True
            )
            
            # Prediction job - frequent (every prediction interval, offset by 2 minutes after data collection)
            self.scheduler.add_job(
                func=self.prediction_job,
                trigger=IntervalTrigger(minutes=self.config.PREDICTION_INTERVAL_MINUTES), # Run every 10 minutes
                id='prediction_generation',
                name='Weather Prediction Generation',
                replace_existing=True
            )
            
            # Training job - less frequent (daily at 2 AM)
            self.scheduler.add_job(
                func=self.training_job,
                trigger=CronTrigger(hour=2, minute=0),  # Daily at 2 AM
                id='model_training',
                name='Model Training',
                replace_existing=True
            )
            
            # Start the scheduler
            self.scheduler.start()
            self.is_running = True
            
            # Write status to file
            try:
                with open(SCHEDULER_STATUS_FILE, 'w') as f:
                    f.write('running')
            except Exception as e:
                logger.error(f"Failed to write scheduler status file: {e}")

            logger.info(f"Scheduler started with separate jobs:")
            logger.info(f"- Data collection: every {self.config.PREDICTION_INTERVAL_MINUTES} minutes")
            logger.info(f"- Predictions: every {self.config.PREDICTION_INTERVAL_MINUTES} minutes")
            logger.info(f"- Training: daily at 2:00 AM")
            
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
            
            # Write status to file
            try:
                with open(SCHEDULER_STATUS_FILE, 'w') as f:
                    f.write('stopped')
            except Exception as e:
                logger.error(f"Failed to write scheduler status file: {e}")

            logger.info("Scheduler stopped")
            
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
            raise
    
    def get_status(self):
        """Get scheduler status"""
        jobs_info = {}
        for job in self.scheduler.get_jobs():
            jobs_info[job.id] = {
                'name': job.name,
                'next_run': job.next_run_time.isoformat() if job.next_run_time else None
            }
        
        # Add debugging information
        logger.debug(f"Scheduler status - is_running: {self.is_running}, is_initialized: {self.is_initialized}")
        logger.debug(f"APScheduler running: {self.scheduler.running}")
        logger.debug(f"Number of jobs: {len(jobs_info)}")
        
        # Ensure we return the correct running status
        actual_running = self.is_running or self.scheduler.running
        
        # Check file-based status as fallback
        file_status = None
        if os.path.exists(SCHEDULER_STATUS_FILE):
            try:
                with open(SCHEDULER_STATUS_FILE, 'r') as f:
                    file_status = f.read().strip()
            except Exception as e:
                logger.error(f"Failed to read scheduler status file: {e}")
        actual_running = self.is_running or self.scheduler.running or (file_status == 'running')
        
        return {
            "is_running": actual_running,
            "is_initialized": self.is_initialized,
            "jobs": jobs_info,
            "apscheduler_running": self.scheduler.running
        }

    def backfill_predictions(self):
        """Backfill missing predictions from the end of preprocessed data up to 1 hour into the future."""
        try:
            logger.info("Starting backfill of missing predictions...")
            # 1. Get all preprocessed data (actuals)
            preprocessed_df = self.data_processor.fetch_preprocessed_data()
            if preprocessed_df is None or preprocessed_df.empty:
                logger.warning("No preprocessed data available for backfilling predictions.")
                return
            preprocessed_df = preprocessed_df.sort_index()
            # 2. Get all predictions
            all_predictions_df = self.data_processor.db_manager.get_latest_predictions(hours=10000)  # large window
            existing_pred_times = set(pd.to_datetime(all_predictions_df['timestamp']).dt.floor('10min')) if not all_predictions_df.empty else set()
            # 3. Determine the range to fill
            last_actual_time = preprocessed_df.index[-1]
            now = datetime.now().replace(second=0, microsecond=0)
            minute = (now.minute // 10) * 10
            now_aligned = now.replace(minute=minute)
            end_time = now_aligned + timedelta(hours=1)
            # 4. Build all 10-min timestamps from the earliest possible (after enough history) to end_time
            sequence_length = self.model_predictor.sequence_length
            all_times = pd.date_range(
                start=preprocessed_df.index[sequence_length],
                end=end_time,
                freq='10min'
            )
            # 5. For each timestamp, if missing prediction, generate and store
            history_df = preprocessed_df.copy()
            for ts in all_times:
                if ts in existing_pred_times:
                    continue
                # Only predict if we have enough history
                history_window = history_df[history_df.index < ts].tail(sequence_length)
                if len(history_window) < sequence_length:
                    logger.debug(f"Not enough history for prediction at {ts}")
                    continue
                input_df = history_window.copy()
                input_df.index.name = 'timestamp'
                pred_df = self.model_predictor.predict(input_df, forecast_periods=1)
                if pred_df.empty:
                    logger.warning(f"No prediction generated for {ts}")
                    continue
                pred_temp = float(pred_df.iloc[0]['predicted_temperature'])
                last_row = history_window.iloc[-1]
                pred_row = {
                    'temperature': pred_temp,
                    'humidity': last_row['humidity'],
                    'pressure': last_row['pressure']
                }
                # Append new prediction to history_df for future steps
                new_row_df = pd.DataFrame([pred_row], index=[ts])
                history_df = pd.concat([history_df, new_row_df])
                history_df = history_df.sort_index()
                # Save this prediction
                single_pred_df = pd.DataFrame([{
                    'timestamp': ts,
                    'predicted_temperature': pred_temp,
                    'confidence': 0.8
                }]).set_index('timestamp')
                self.data_processor.store_predictions(single_pred_df)
                logger.info(f"Backfilled prediction for {ts}")
            logger.info("Backfill of missing predictions complete.")
        except Exception as e:
            logger.error(f"Error in backfill_predictions: {e}")

# Global scheduler instance
scheduler = None

def create_scheduler(model_predictor=None):
    """Create a new scheduler instance with optional model predictor"""
    global scheduler
    try:
        logger.info("Creating new scheduler instance...")
        scheduler = WeatherScheduler(model_predictor)
        logger.info("Scheduler instance created successfully")
        return scheduler
    except Exception as e:
        logger.error(f"Error creating scheduler instance: {e}")
        import traceback
        logger.error(f"create_scheduler traceback: {traceback.format_exc()}")
        scheduler = None
        raise

async def start_scheduler(model_predictor=None):
    """Start the scheduler (for use in main.py)"""
    global scheduler
    if scheduler is None:
        scheduler = create_scheduler(model_predictor)
    
    # Initialize the system first
    scheduler.initialize_system()
    
    # Start the scheduler
    scheduler.start()
    
    # Verify the scheduler is running
    if scheduler.is_running:
        logger.info("Scheduler started successfully")
    else:
        logger.error("Failed to start scheduler - is_running flag not set")

async def stop_scheduler():
    """Stop the scheduler (for use in main.py)"""
    global scheduler
    if scheduler:
        scheduler.stop() 