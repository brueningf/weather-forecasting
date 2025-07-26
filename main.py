import uvicorn
import asyncio
import logging
from contextlib import asynccontextmanager
import os

from api import app
from scheduler import start_scheduler, stop_scheduler, create_scheduler
from config import Config
from weather_data_controller import WeatherDataController
from model_predictor import ModelPredictor

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global config
config = Config()

async def train_initial_model(model_predictor=None):
    """Train the model on preprocessed data from the database"""
    try:
        logger.info("Starting initial model training...")
        
        # Initialize components
        data_processor = WeatherDataController()
        if model_predictor is None:
            model_predictor = ModelPredictor()
        
        # Check if model needs training
        if not model_predictor.needs_training():
            logger.info("Model is already trained, skipping initial training")
            return True
        
        # Get preprocessed data from database for training
        logger.info("Loading preprocessed data for model training...")
        processed_df = data_processor.fetch_training_data(hours_back=8760)  # 1 year of data
        
        if processed_df.empty:
            logger.warning("No preprocessed data available for training")
            logger.info("Triggering data preprocessing first...")
            # Trigger preprocessing if no data exists
            raw_df, processed_df = data_processor.process_and_store_new_data(
                force_full_export=True, 
                hours_back=8760
            )
            if processed_df is None or processed_df.empty:
                logger.error("Failed to preprocess data for training")
                return False
        
        logger.info(f"Training data shape: {processed_df.shape}")
        
        # Train the model
        success = model_predictor.train_model(processed_df, epochs=50, learning_rate=0.001)
        
        if success:
            logger.info("Initial model training completed successfully")
            return True
        else:
            logger.error("Initial model training failed")
            return False
            
    except Exception as e:
        logger.error(f"Error in initial model training: {e}")
        return False

@asynccontextmanager
async def lifespan(app):
    """Manage application lifespan"""
    # Startup
    logger.info("Starting Weather Forecasting API...")
    try:
        # Initialize components
        data_processor = WeatherDataController()
        model_predictor = ModelPredictor()
        
        # Create scheduler first (this makes it globally available)
        logger.info("Creating scheduler...")
        create_scheduler(model_predictor)
        
        # Train model on startup if needed
        if model_predictor.needs_training():
            await train_initial_model(model_predictor)
        else:
            logger.info("Model is already trained, skipping initial training")
        
        # Start scheduler with the trained model predictor
        logger.info("Starting scheduler...")
        await start_scheduler(model_predictor)
        
        # Run initial prediction
        from scheduler import scheduler
        if scheduler:
            scheduler.run_once_prediction()
            logger.info("Scheduler started successfully")
        else:
            logger.error("Failed to create scheduler")
            
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Weather Forecasting API...")
    try:
        await stop_scheduler()
        logger.info("Scheduler stopped successfully")
    except Exception as e:
        logger.error(f"Error stopping scheduler: {e}")

# Update app with lifespan
app.router.lifespan_context = lifespan

def run_api():
    """Run the FastAPI application"""
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_RELOAD,
        log_level="info"
    )

if __name__ == '__main__':
    run_api()
