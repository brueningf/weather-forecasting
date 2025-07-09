import uvicorn
import asyncio
import logging
from contextlib import asynccontextmanager

from api import app
from scheduler import start_scheduler, stop_scheduler
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global config
config = Config()

@asynccontextmanager
async def lifespan(app):
    """Manage application lifespan"""
    # Startup
    logger.info("Starting Weather Forecasting API...")
    try:
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
