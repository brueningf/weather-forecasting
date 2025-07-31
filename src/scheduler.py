from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
import logging
from datetime import datetime, timedelta
import pandas as pd
import os
import numpy as np

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
            stats = self.data_processor.db_manager.get_preprocessed_data_stats()
            
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
            _, processed_df = self.data_processor.process_and_store_new_data(
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
            training_df = self.data_processor.get_training_data(hours_back=hours_back)
            
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
            _, processed_df = self.data_processor.process_and_store_new_data()
            
            if processed_df is None or processed_df.empty:
                logger.info("No new data to process")
                return
            
            logger.info(f"Processed and saved {len(processed_df)} new records")
            
        except Exception as e:
            logger.error(f"Error in data collection: {e}")
    
    def complete_missing_data_with_predictions(self, preprocessed_df, max_predicted_ratio=0.5):
        """
        Complete missing preprocessed data with predicted data, respecting the 50% limit
        
        Args:
            preprocessed_df: DataFrame with preprocessed data
            max_predicted_ratio: Maximum ratio of predicted data allowed (default 0.5 = 50%)
        
        Returns:
            DataFrame with completed data
        """
        try:
            if preprocessed_df.empty:
                logger.warning("No preprocessed data available for completion")
                return preprocessed_df
            
            # Get the latest predictions to use for completion
            predictions_df = self.data_processor.db_manager.get_latest_predictions(hours=48)
            if predictions_df.empty:
                logger.warning("No predictions available for data completion")
                return preprocessed_df
            
            # Convert predictions to the same format as preprocessed data
            pred_df = predictions_df.copy()
            pred_df = pred_df.set_index('timestamp')
            pred_df['temperature'] = pred_df['predicted_temperature']
            pred_df = pred_df[['temperature']]  # Keep only temperature for now
            
            # Find missing timestamps in preprocessed data
            # Look for gaps larger than the prediction interval
            preprocessed_df = preprocessed_df.sort_index()
            expected_times = pd.date_range(
                start=preprocessed_df.index[0],
                end=preprocessed_df.index[-1],
                freq='10min'
            )
            
            missing_times = expected_times.difference(preprocessed_df.index)
            
            if len(missing_times) == 0:
                logger.info("No missing data found in preprocessed data")
                return preprocessed_df
            
            logger.info(f"Found {len(missing_times)} missing timestamps in preprocessed data")
            
            # Calculate how much data we can fill with predictions
            total_data_points = len(preprocessed_df) + len(missing_times)
            max_predicted_points = int(total_data_points * max_predicted_ratio)
            
            # Limit the number of missing points we can fill
            fillable_times = missing_times[:max_predicted_points]
            
            if len(fillable_times) < len(missing_times):
                logger.warning(f"Can only fill {len(fillable_times)} of {len(missing_times)} missing points due to {max_predicted_ratio*100}% limit")
            
            # Fill missing data with predictions
            completed_data = []
            for ts in fillable_times:
                # Find the closest prediction for this timestamp
                if ts in pred_df.index:
                    pred_row = pred_df.loc[ts]
                    # Get humidity and pressure from the last known actual data
                    last_actual = preprocessed_df[preprocessed_df.index < ts].iloc[-1] if len(preprocessed_df[preprocessed_df.index < ts]) > 0 else preprocessed_df.iloc[0]
                    
                    completed_row = {
                        'temperature': pred_row['temperature'],
                        'humidity': last_actual['humidity'],
                        'pressure': last_actual['pressure']
                    }
                    completed_data.append(completed_row)
                else:
                    logger.debug(f"No prediction available for {ts}")
            
            if completed_data:
                # Create DataFrame for completed data
                completed_df = pd.DataFrame(completed_data, index=fillable_times)
                
                # Combine with original data
                result_df = pd.concat([preprocessed_df, completed_df])
                result_df = result_df.sort_index()
                
                logger.info(f"Completed {len(completed_data)} missing data points with predictions")
                return result_df
            
            return preprocessed_df
            
        except Exception as e:
            logger.error(f"Error completing missing data with predictions: {e}")
            return preprocessed_df
    
    def prediction_job(self):
        """Prediction job - fills missing 10-min predictions for the next 48 hours"""
        try:
            logger.info(f"Starting 10-min interval prediction generation at {datetime.now()}")

            if not self.is_initialized:
                logger.info("System not initialized. Initializing now...")
                self.initialize_system()
                if not self.is_initialized:
                    logger.warning("Initialization failed. Skipping prediction.")
                    return

            # 1. Get all existing predictions for the next 48 hours (10-min intervals)
            now = datetime.now().replace(second=0, microsecond=0)
            # Align 'now' to the previous 10-min mark
            minute = (now.minute // 10) * 10
            now_aligned = now.replace(minute=minute)
            
            # Generate timestamps for next 48 hours (288 steps of 10 minutes each)
            future_times = [now_aligned + timedelta(minutes=10 * i) for i in range(1, 289)]  # next 48 hours, 10-min steps
            
            existing_preds_df = self.data_processor.db_manager.get_latest_predictions(hours=48)
            existing_pred_times = set(pd.to_datetime(existing_preds_df['timestamp']).dt.floor('10min')) if not existing_preds_df.empty else set()

            # 2. Determine which timestamps are missing
            missing_times = [t for t in future_times if t not in existing_pred_times]
            if not missing_times:
                logger.info("All 10-min predictions for the next 48 hours already exist. Nothing to do.")
                return

            logger.info(f"Generating predictions for {len(missing_times)} missing timestamps")

            # 3. Get all preprocessed data (should be 10-min frequency)
            preprocessed_df = self.data_processor.db_manager.get_preprocessed_data()
            if preprocessed_df is None or preprocessed_df.empty:
                logger.error("No preprocessed data available for prediction.")
                return
            preprocessed_df = preprocessed_df.sort_index()
            sequence_length = self.model_predictor.sequence_length

            # 4. Check if we need to complete missing preprocessed data with predictions
            # Ensure the latest data is recent enough (within 2x prediction interval)
            latest_data_time = preprocessed_df.index[-1]
            max_age = timedelta(minutes=2 * self.config.PREDICTION_INTERVAL_MINUTES)
            use_predictions_as_history = False
            
            if now - latest_data_time > max_age:
                logger.warning(f"Latest preprocessed data is too old for prediction (latest: {latest_data_time}, now: {now}). Using predictions as history.")
                use_predictions_as_history = True
            else:
                # Complete missing preprocessed data with predictions (max 50%)
                preprocessed_df = self.complete_missing_data_with_predictions(preprocessed_df, max_predicted_ratio=0.5)

            # 5. Build accumulated history like backfill method for better accuracy
            if not use_predictions_as_history:
                # Start with preprocessed data as base history
                history_df = preprocessed_df.copy()
            else:
                # Use predictions as base history
                pred_history_df = self.data_processor.db_manager.get_latest_predictions(hours=24)
                if pred_history_df is None or pred_history_df.empty:
                    logger.warning("No predictions available to use as history. Creating initial predictions...")
                    # Create initial predictions using available preprocessed data
                    if not preprocessed_df.empty:
                        # Use the last few data points to create initial predictions
                        last_data = preprocessed_df.tail(min(10, len(preprocessed_df)))
                        initial_predictions = []
                        
                        # Create predictions for the next few hours using the last known values
                        # Calculate simple trend from last few points
                        if len(last_data) >= 3:
                            recent_temps = last_data['temperature'].tail(3).values
                            temp_trend = (recent_temps[-1] - recent_temps[0]) / 2  # Simple trend
                        else:
                            temp_trend = 0
                        
                        for i in range(1, 13):  # 2 hours of 10-min predictions
                            pred_time = now_aligned + timedelta(minutes=10 * i)
                            last_temp = last_data.iloc[-1]['temperature']
                            last_humidity = last_data.iloc[-1]['humidity']
                            last_pressure = last_data.iloc[-1]['pressure']
                            
                            # Simple prediction: use last known value with trend and small random variation
                            pred_temp = last_temp + (temp_trend * i / 12) + np.random.normal(0, 0.3)  # Trend + small random variation
                            
                            initial_predictions.append({
                                'timestamp': pred_time,
                                'predicted_temperature': pred_temp,
                                'confidence': 0.6  # Lower confidence for initial predictions
                            })
                        
                        # Store initial predictions
                        if initial_predictions:
                            initial_df = pd.DataFrame(initial_predictions).set_index('timestamp')
                            self.data_processor.store_predictions(initial_df)
                            logger.info(f"Created {len(initial_predictions)} initial predictions")
                            
                            # Now get the predictions we just created
                            pred_history_df = self.data_processor.db_manager.get_latest_predictions(hours=24)
                        else:
                            logger.error("Failed to create initial predictions")
                            return
                    else:
                        logger.error("No preprocessed data available for initial predictions")
                        return
                
                pred_history_df = pred_history_df.sort_values('timestamp')
                pred_history_df = pred_history_df.set_index('timestamp')
                
                # Get last known humidity/pressure from preprocessed data
                last_actual = preprocessed_df.iloc[-1] if not preprocessed_df.empty else None
                humidity = last_actual['humidity'] if last_actual is not None and 'humidity' in last_actual else 50.0
                pressure = last_actual['pressure'] if last_actual is not None and 'pressure' in last_actual else 1013.0
                
                # Convert predictions to the format expected by the model
                history_df = pred_history_df.copy()
                history_df['temperature'] = history_df['predicted_temperature']
                history_df['humidity'] = humidity
                history_df['pressure'] = pressure
                history_df = history_df[['temperature', 'humidity', 'pressure']]

            # 6. Generate predictions in batches to handle the large number efficiently
            batch_size = 60  # Process 60 predictions at a time (10 hours)
            predictions_to_store = []
            
            for batch_start in range(0, len(missing_times), batch_size):
                batch_end = min(batch_start + batch_size, len(missing_times))
                batch_times = missing_times[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start//batch_size + 1}: timestamps {batch_start+1}-{batch_end}")
                
                # For each batch, generate predictions using the accumulated history
                for ts in batch_times:
                    # Only predict if we have enough history
                    history_window = history_df[history_df.index < ts].tail(sequence_length)
                    if len(history_window) < sequence_length:
                        logger.warning(f"Not enough history for prediction at {ts} (have {len(history_window)}, need {sequence_length})")
                        continue
                    
                    input_df = history_window.copy()
                    input_df.index.name = 'timestamp'
                    
                    # Generate prediction for this timestamp
                    pred_df = self.model_predictor.predict(input_df, forecast_periods=1)
                    if pred_df.empty:
                        logger.warning(f"No prediction generated for {ts}")
                        continue
                    
                    pred_temp = float(pred_df.iloc[0]['predicted_temperature'])
                    pred_confidence = float(pred_df.iloc[0]['confidence'])
                    last_row = input_df.iloc[-1]
                    pred_row = {
                        'temperature': pred_temp,
                        'humidity': last_row['humidity'],
                        'pressure': last_row['pressure']
                    }
                    
                    # Append new prediction to history_df for future steps
                    new_row_df = pd.DataFrame([pred_row], index=[ts])
                    history_df = pd.concat([history_df, new_row_df])
                    history_df = history_df.sort_index()
                    
                    # Store this prediction
                    predictions_to_store.append({
                        'timestamp': ts,
                        'predicted_temperature': pred_temp,
                        'confidence': pred_confidence
                    })
                
                # Store batch of predictions
                if predictions_to_store:
                    batch_df = pd.DataFrame(predictions_to_store).set_index('timestamp')
                    self.data_processor.store_predictions(batch_df)
                    logger.info(f"Stored {len(predictions_to_store)} predictions for batch")
                    predictions_to_store = []  # Reset for next batch
            
            logger.info(f"Completed prediction generation for {len(missing_times)} timestamps")

        except Exception as e:
            logger.error(f"Error in 10-min prediction job: {e}")
    
    def evaluation_job(self):
        """Evaluation job - evaluates predictions against actual data after 48 hours"""
        try:
            logger.info(f"Starting evaluation job at {datetime.now()}")
            
            # Get predictions from 48 hours ago
            cutoff_time = datetime.now() - timedelta(hours=48)
            
            # Get predictions that were made 48 hours ago
            predictions_df = self.data_processor.db_manager.get_latest_predictions(cutoff_time=cutoff_time, window_hours=1)
            if predictions_df.empty:
                logger.info("No predictions found for evaluation (48 hours ago)")
                return
            
            # Get actual preprocessed data for the same period
            actual_df = self.data_processor.get_preprocessed_data_for_evaluation(cutoff_time)
            if actual_df.empty:
                logger.info("No actual data found for evaluation (48 hours ago)")
                return
            
            # Align predictions and actual data by timestamp
            evaluation_results = self.evaluate_predictions(predictions_df, actual_df)
            
            if evaluation_results:
                # Store evaluation metrics
                self.store_evaluation_metrics(evaluation_results)
                logger.info(f"Evaluation completed. Metrics: {evaluation_results}")
            else:
                logger.warning("No evaluation results generated")
            
        except Exception as e:
            logger.error(f"Error in evaluation job: {e}")
    
    def evaluate_predictions(self, predictions_df, actual_df):
        """Evaluate predictions against actual data"""
        try:
            # Align timestamps
            predictions_df = predictions_df.set_index('timestamp')
            actual_df = actual_df.set_index('timestamp')
            
            # Find common timestamps
            common_times = predictions_df.index.intersection(actual_df.index)
            
            if len(common_times) == 0:
                logger.warning("No common timestamps found for evaluation")
                return None
            
            # Get aligned data
            pred_temps = predictions_df.loc[common_times, 'predicted_temperature']
            actual_temps = actual_df.loc[common_times, 'temperature']
            
            # Calculate metrics
            mae = np.mean(np.abs(pred_temps - actual_temps))
            rmse = np.sqrt(np.mean((pred_temps - actual_temps) ** 2))
            mape = np.mean(np.abs((actual_temps - pred_temps) / actual_temps)) * 100
            
            # Calculate R² score
            ss_res = np.sum((actual_temps - pred_temps) ** 2)
            ss_tot = np.sum((actual_temps - np.mean(actual_temps)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            evaluation_results = {
                'evaluation_timestamp': datetime.now().isoformat(),
                'prediction_timestamp': predictions_df.index[0].isoformat(),
                'data_points': len(common_times),
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'r2': float(r2),
                'min_temp_error': float(np.min(np.abs(pred_temps - actual_temps))),
                'max_temp_error': float(np.max(np.abs(pred_temps - actual_temps))),
                'mean_predicted_temp': float(np.mean(pred_temps)),
                'mean_actual_temp': float(np.mean(actual_temps))
            }
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating predictions: {e}")
            return None
    
    def store_evaluation_metrics(self, metrics):
        """Store evaluation metrics to database"""
        try:
            # Store metrics to database (you can implement this in DatabaseManager)
            # For now, we'll log them
            logger.info(f"Evaluation metrics stored: {metrics}")
            
            # You can also save to a file or database table
            metrics_file = 'evaluation_metrics.json'
            import json
            
            # Load existing metrics if file exists
            existing_metrics = []
            if os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        existing_metrics = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load existing metrics: {e}")
            
            # Add new metrics
            existing_metrics.append(metrics)
            
            # Keep only last 100 evaluations
            if len(existing_metrics) > 100:
                existing_metrics = existing_metrics[-100:]
            
            # Save to file
            with open(metrics_file, 'w') as f:
                json.dump(existing_metrics, f, indent=2)
            
            logger.info(f"Evaluation metrics saved to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error storing evaluation metrics: {e}")
    
    def training_job(self):
        """Training job - only handles model retraining"""
        try:
            logger.info(f"Starting model training job at {datetime.now()}")
            
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
            stats = self.data_processor.db_manager.get_preprocessed_data_stats()
            
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
                trigger=IntervalTrigger(minutes=self.config.PREDICTION_INTERVAL_MINUTES, start_date=(datetime.now() + timedelta(minutes=2))), # Offset by 2 minutes
                id='prediction_generation',
                name='Weather Prediction Generation',
                replace_existing=True
            )
            
            # Evaluation job - runs every hour to check for 48-hour old predictions
            self.scheduler.add_job(
                func=self.evaluation_job,
                trigger=IntervalTrigger(hours=1),
                id='prediction_evaluation',
                name='Prediction Evaluation',
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
            logger.info(f"- Predictions: every {self.config.PREDICTION_INTERVAL_MINUTES} minutes (offset by 2 minutes)")
            logger.info(f"- Evaluation: every hour")
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