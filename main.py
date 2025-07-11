import uvicorn
import asyncio
import logging
from contextlib import asynccontextmanager

from api import app
from scheduler import start_scheduler, stop_scheduler
from config import Config
from data_processor import DataProcessor
from model_predictor import ModelPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global config
config = Config()

async def train_initial_model():
    """Train the model on all available historical data"""
    try:
        logger.info("Starting initial model training...")
        
        # Initialize components
        data_processor = DataProcessor()
        model_predictor = ModelPredictor()
        
        # Check if model needs training
        if not model_predictor.needs_training():
            logger.info("Model is already trained, skipping initial training")
            return True
        
        # Export all available data for training
        logger.info("Exporting all available historical data for training...")
        df = data_processor.force_initial_export(hours_back=8760)  # 1 year of data
        
        if df.empty:
            logger.warning("No historical data available for training")
            return False
        
        logger.info(f"Exported {len(df)} records for training")
        
        # Preprocess the data
        processed_df = data_processor.preprocess_data(df)
        
        if processed_df.empty:
            logger.error("No data after preprocessing")
            return False
        
        logger.info(f"Preprocessed data shape: {processed_df.shape}")
        
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
        # Train model on startup if needed
        await train_initial_model()
        
        # Start scheduler
        await start_scheduler()
        logger.info("Scheduler started successfully")
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
