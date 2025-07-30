import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent tkinter errors

from fastapi import FastAPI, HTTPException, Query, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Request
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from weather_data_controller import WeatherDataController
from model_predictor import ModelPredictor
from scheduler import WeatherScheduler
from config import Config
from analysis import router as analysis_router

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Weather Forecasting API",
    description="API for weather prediction with scheduled forecasting",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Mount templates
templates = Jinja2Templates(directory="templates")

# Initialize components
config = Config()
weather_data_controller = WeatherDataController()
model_predictor = ModelPredictor()
# Use the global scheduler instance from scheduler module
from scheduler import scheduler, create_scheduler

# Ensure scheduler is created if it doesn't exist
if scheduler is None:
    logger.info("Creating scheduler in API initialization...")
    create_scheduler(model_predictor)

# Pydantic models for API
class PredictionResponse(BaseModel):
    timestamp: datetime
    predicted_temperature: float
    confidence: float

class NextForecastResponse(BaseModel):
    timestamp: datetime
    predicted_temperature: float
    confidence: float

class ActualTemperatureResponse(BaseModel):
    timestamp: datetime
    temperature: float
    humidity: float
    pressure: float
    module: str

class PreprocessedDataResponse(BaseModel):
    timestamp: datetime
    temperature: float
    humidity: float
    pressure: float
    temperature_normalized: float
    humidity_normalized: float
    pressure_normalized: float
    hour: int
    minute: int
    day_of_week: int
    month: int

class SystemStatusResponse(BaseModel):
    is_initialized: bool
    is_running: bool
    preprocessed_records: int
    model_trained: bool
    last_export_time: Optional[str]

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard with all features and analysis"""
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "config": config
    })

@app.get("/api/predictions", response_model=List[PredictionResponse])
async def get_predictions(hours: int = 48):
    """Get weather predictions for the specified number of hours (default: 48 hours)"""
    try:
        # Use the correct method to get predictions from database
        predictions_df = weather_data_controller.get_latest_predictions(hours=hours)
        
        if predictions_df.empty:
            return []
        
        # Convert to response format
        data = []
        for _, row in predictions_df.iterrows():
            data.append(PredictionResponse(
                timestamp=row['timestamp'],
                predicted_temperature=float(row.get('predicted_temperature', 0)),
                confidence=float(row.get('confidence', 0.5))  # Default to 0.5 if no confidence
            ))
        
        return data
        
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving predictions")

@app.get("/api/sensor-data", response_model=List[ActualTemperatureResponse])
async def get_sensor_data(hours: int = 24, module_id: Optional[str] = None):
    """Get actual sensor data for the specified number of hours"""
    try:
        logger.info(f"Fetching sensor data with hours={hours}, module_id={module_id}")
        # Get raw sensor data using the correct method name
        raw_data = weather_data_controller.fetch_recent_sensor_data(hours=hours, module_id=module_id)
        
        if raw_data.empty:
            return []
        
        # Convert to response format
        data = []
        for _, row in raw_data.iterrows():
            data.append(ActualTemperatureResponse(
                timestamp=row['timestamp'],
                temperature=float(row.get('temperature', 0)),
                humidity=float(row.get('humidity', 0)),
                pressure=float(row.get('pressure', 0)),
                module=str(row.get('module', 'unknown'))
            ))
        
        return data
        
    except Exception as e:
        logger.error(f"Error getting sensor data: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving sensor data")

@app.get("/api/modules")
async def get_available_modules():
    """Get list of available sensor modules"""
    try:
        # Query the database directly to get available modules
        query = f"""
            SELECT DISTINCT module 
            FROM {weather_data_controller.db_manager.sensor_data_table}
            WHERE module IS NOT NULL
            ORDER BY module
        """
        
        df = pd.read_sql(query, weather_data_controller.db_manager.source_engine)
        modules = df['module'].tolist() if not df.empty else []
        
        return {"modules": modules}
        
    except Exception as e:
        logger.error(f"Error getting modules: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving modules")

@app.get("/api/preprocessed-data", response_model=List[PreprocessedDataResponse])
async def get_preprocessed_data(hours_back: Optional[int] = None, limit: Optional[int] = None):
    """Get preprocessed data for analysis"""
    try:
        logger.info(f"Fetching preprocessed data with hours_back={hours_back}, limit={limit}")
        df = weather_data_controller.fetch_preprocessed_data(hours_back=hours_back, limit=limit)
        
        if df.empty:
            return []
        
        # Convert to response format
        data = []
        for index, row in df.iterrows():
            # Extract time features from timestamp
            hour = index.hour
            minute = index.minute
            day_of_week = index.dayofweek
            month = index.month
            
            data.append(PreprocessedDataResponse(
                timestamp=index,
                temperature=float(row.get('temperature', 0)),
                humidity=float(row.get('humidity', 0)),
                pressure=float(row.get('pressure', 0)),
                temperature_normalized=float(row.get('temperature', 0)),  # Use original value as normalized
                humidity_normalized=float(row.get('humidity', 0)),       # Use original value as normalized
                pressure_normalized=float(row.get('pressure', 0)),       # Use original value as normalized
                hour=hour,
                minute=minute,
                day_of_week=day_of_week,
                month=month
            ))
        
        return data
        
    except Exception as e:
        logger.error(f"Error getting preprocessed data: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving preprocessed data")

@app.get("/api/preprocessed-stats")
async def get_preprocessed_stats():
    """Get statistics about preprocessed data"""
    try:
        stats = weather_data_controller.fetch_preprocessed_data_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting preprocessed stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving preprocessed statistics")

@app.get("/api/stats")
async def get_stats():
    """Get comprehensive statistics about the system, including system status."""
    try:
        # Get various statistics using the correct method names
        raw_stats = weather_data_controller.fetch_source_db_stats()
        preprocessed_stats = weather_data_controller.fetch_preprocessed_data_stats()
        api_stats = weather_data_controller.fetch_api_db_stats()
        # Get model info
        model_info = {
            "is_trained": model_predictor.is_trained,
            "model_path": model_predictor.config.MODEL_PATH
        }
        # Get scheduler status
        scheduler_status = {
            "is_initialized": False,
            "is_running": False
        }
        status_source = "unknown"
        if scheduler is not None:
            try:
                scheduler_status = scheduler.get_status()
                status_source = "scheduler instance"
            except Exception as e:
                logger.warning(f"Could not get scheduler status: {e}")
        # Fallback: check file
        import os
        SCHEDULER_STATUS_FILE = 'scheduler_status.flag'
        if not scheduler_status.get("is_running", False) and os.path.exists(SCHEDULER_STATUS_FILE):
            try:
                with open(SCHEDULER_STATUS_FILE, 'r') as f:
                    file_status = f.read().strip()
                if file_status == 'running':
                    scheduler_status["is_running"] = True
                    status_source = "file fallback"
            except Exception as e:
                logger.error(f"Failed to read scheduler status file: {e}")
        # System status fields
        last_export_time = weather_data_controller.preprocessor.get_last_export_time()
        # Compose response
        return {
            "raw_data": raw_stats,
            "preprocessed_data": preprocessed_stats,
            "api_data": api_stats,
            "model": model_info,
            "scheduler": scheduler_status,
            "timestamp": datetime.now().isoformat(),
            # Flattened system status fields:
            "is_initialized": scheduler_status.get("is_initialized", False),
            "is_running": scheduler_status.get("is_running", False),
            "preprocessed_records": preprocessed_stats.get("preprocessed_records", 0),
            "model_trained": model_predictor.is_trained,
            "last_export_time": last_export_time.isoformat() if last_export_time else None
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving statistics")

@app.post("/api/retrain-model")
async def retrain_model(hours_back: int = 168):
    """Manually trigger model retraining"""
    try:
        logger.info("Manual model retraining requested")
        
        # Ensure scheduler exists
        if scheduler is None:
            logger.info("Scheduler is None - creating it now...")
            create_scheduler(model_predictor)
        
        if scheduler is None:
            raise HTTPException(status_code=500, detail="Failed to create scheduler")
        
        success = scheduler.load_and_retrain_model(hours_back=hours_back)
        
        if success:
            return {"message": "Model retraining completed successfully", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="Model retraining failed")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in manual model retraining: {e}")
        raise HTTPException(status_code=500, detail="Error during model retraining")

@app.post("/api/process-data")
async def process_data(force_full_export: bool = False, hours_back: int = 168):
    """Manually trigger data processing"""
    try:
        logger.info("Manual data processing requested")
        
        raw_df, processed_df = weather_data_controller.process_and_store_new_data(
            force_full_export=force_full_export,
            hours_back=hours_back
        )
        
        if raw_df is not None and not raw_df.empty:
            if processed_df is not None and not processed_df.empty:
                return {
                    "message": "Data processing completed successfully",
                    "status": "success",
                    "raw_records": len(raw_df),
                    "processed_records": len(processed_df)
                }
            else:
                return {
                    "message": "Raw data processed but no preprocessed data generated",
                    "status": "partial_success",
                    "raw_records": len(raw_df),
                    "processed_records": 0
                }
        else:
            return {
                "message": "No new data to process",
                "status": "success",
                "raw_records": 0,
                "processed_records": 0
            }
        
    except Exception as e:
        logger.error(f"Error in manual data processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error during data processing: {str(e)}")

@app.get("/api/scheduler-status")
async def get_scheduler_status():
    """Get detailed scheduler status"""
    try:
        if scheduler is None:
            return {
                "scheduler_available": False,
                "message": "Scheduler not initialized"
            }
        
        status = scheduler.get_status()
        return {
            "scheduler_available": True,
            "status": status
        }
        
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving scheduler status")

@app.get("/api/model/benchmark")
async def get_model_benchmark():
    """Get model performance metrics (MAE, RMSE, R²)"""
    try:
        if not model_predictor.is_trained:
            raise HTTPException(status_code=404, detail="Model not trained yet")
        
        # Get benchmark metrics from the model predictor
        metrics = model_predictor.get_latest_metrics()
        
        if metrics is None:
            raise HTTPException(status_code=404, detail="No benchmark data available")
        
        return {
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "r2": metrics.get("r2")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model benchmark: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving model benchmark")

@app.get("/temperature", response_class=HTMLResponse)
async def temperature_page(request: Request):
    """Serve the temperature (actual vs predicted) graph HTML page"""
    return templates.TemplateResponse("temperature-graph.html", {"request": request}) 

app.include_router(analysis_router) 