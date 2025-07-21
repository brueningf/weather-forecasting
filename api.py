from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging

from weather_data_controller import WeatherDataController
from model_predictor import ModelPredictor
from scheduler import WeatherScheduler
from config import Config

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

# Initialize components
config = Config()
data_processor = WeatherDataController()
model_predictor = ModelPredictor()
scheduler = WeatherScheduler(model_predictor)

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

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Weather Forecasting API",
        "version": "1.0.0",
        "description": "Simple weather forecasting API with scheduled predictions every 10 minutes",
        "endpoints": {
            "predictions": "/predictions",
            "next_forecast": "/next-forecast",
            "sensor_data": "/sensor-data",
            "stats": "/stats",
            "stats_page": "/stats-page",
            "temperature_graph": "/temperature-graph",
            "preprocessed_data": "/preprocessed-data",
            "preprocessed_stats": "/preprocessed-stats",
            "system_status": "/system-status",
            "initialize_system": "/initialize-system",
            "retrain_model": "/retrain-model",
            "process_data": "/process-data",
            "clear_preprocessed_data": "/clear-preprocessed-data",
            "scheduler_status": "/scheduler-status"
        }
    }

@app.get("/stats-page")
async def stats_page():
    """Serve the stats HTML page"""
    return FileResponse("static/stats.html")

@app.get("/temperature-graph")
async def temperature_graph_page():
    """Serve the temperature comparison graph HTML page"""
    return FileResponse("static/temperature-graph.html")

@app.get("/predictions", response_model=List[PredictionResponse])
async def get_predictions(hours: int = 24):
    """Get latest predictions from database"""
    try:
        predictions_df = data_processor.get_latest_predictions(hours=hours)
        
        if predictions_df.empty:
            return []
        
        # Convert to response format
        predictions = []
        for _, row in predictions_df.iterrows():
            predictions.append(PredictionResponse(
                timestamp=row['timestamp'],
                predicted_temperature=float(row['predicted_temperature']),
                confidence=float(row['confidence'])
            ))
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving predictions")

@app.get("/next-forecast", response_model=NextForecastResponse)
async def get_next_forecast():
    """Get the next forecast (last row from predictions)"""
    try:
        predictions_df = data_processor.get_latest_predictions(hours=24)
        
        if predictions_df.empty:
            raise HTTPException(status_code=404, detail="No predictions available")
        
        # Get the last row (most recent prediction)
        last_prediction = predictions_df.iloc[-1]
        
        return NextForecastResponse(
            timestamp=last_prediction['timestamp'],
            predicted_temperature=float(last_prediction['predicted_temperature']),
            confidence=float(last_prediction['confidence'])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting next forecast: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving next forecast")

@app.get("/sensor-data", response_model=List[ActualTemperatureResponse])
async def get_sensor_data(hours: int = 24, module_id: Optional[str] = None):
    """Get actual temperature data from source database"""
    try:
        sensor_df = data_processor.fetch_recent_sensor_data(hours=hours, module_id=module_id)
        
        if sensor_df.empty:
            return []
        
        # Convert to response format
        temperatures = []
        for _, row in sensor_df.iterrows():
            temperatures.append(ActualTemperatureResponse(
                timestamp=row['timestamp'],
                temperature=float(row['temperature']),
                humidity=float(row['humidity']),
                pressure=float(row['pressure']),
                module=str(row.get('module', 'unknown'))
            ))
        
        return temperatures
        
    except Exception as e:
        logger.error(f"Error getting sensor data: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving sensor data")

@app.get("/modules")
async def get_available_modules():
    """Get list of available sensor modules"""
    try:
        query = f"""
            SELECT DISTINCT module 
            FROM {data_processor.sensor_data_table}
            WHERE module IS NOT NULL
            ORDER BY module
        """
        
        df = pd.read_sql(query, data_processor.source_engine)
        modules = df['module'].tolist() if not df.empty else []
        
        return {"modules": modules}
        
    except Exception as e:
        logger.error(f"Error getting available modules: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving modules")

@app.get("/stats")
async def get_stats():
    """Get statistics from both databases"""
    try:
        source_stats = data_processor.fetch_source_db_stats()
        api_stats = data_processor.fetch_api_db_stats()
        preprocessed_stats = data_processor.fetch_preprocessed_data_stats()
        
        return {
            "source_database": {
                "total_records": source_stats.get("sensor_records", 0),
                "latest_record": source_stats.get("latest_sensor_data"),
                "status": "online" if source_stats.get("sensor_records", 0) > 0 else "offline"
            },
            "api_database": {
                "total_predictions": api_stats.get("prediction_records", 0),
                "latest_prediction": api_stats.get("latest_prediction"),
                "status": "online" if api_stats.get("prediction_records", 0) > 0 else "offline"
            },
            "preprocessed_data": {
                "total_records": preprocessed_stats.get("preprocessed_records", 0),
                "latest_record": preprocessed_stats.get("latest_preprocessed"),
                "earliest_record": preprocessed_stats.get("earliest_preprocessed"),
                "batch_count": preprocessed_stats.get("batch_count", 0),
                "status": "available" if preprocessed_stats.get("preprocessed_records", 0) > 0 else "empty"
            },
            "model_status": {
                "is_trained": model_predictor.is_trained,
                "model_path": config.MODEL_PATH
            },
            "scheduler": {
                "interval_minutes": config.PREDICTION_INTERVAL_MINUTES,
                "next_run": "Calculated by scheduler"
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving statistics")

@app.get("/preprocessed-data", response_model=List[PreprocessedDataResponse])
async def get_preprocessed_data(hours_back: Optional[int] = None, limit: Optional[int] = None):
    """Get preprocessed data from database"""
    try:
        df = data_processor.fetch_preprocessed_data(hours_back=hours_back, limit=limit)
        
        if df.empty:
            return []
        
        # Convert to response format
        data = []
        for index, row in df.iterrows():
            data.append(PreprocessedDataResponse(
                timestamp=index,
                temperature=float(row.get('temperature', 0)),
                humidity=float(row.get('humidity', 0)),
                pressure=float(row.get('pressure', 0)),
                temperature_normalized=float(row.get('temperature_normalized', 0)),
                humidity_normalized=float(row.get('humidity_normalized', 0)),
                pressure_normalized=float(row.get('pressure_normalized', 0)),
                hour=int(row.get('hour', 0)),
                minute=int(row.get('minute', 0)),
                day_of_week=int(row.get('day_of_week', 0)),
                month=int(row.get('month', 0))
            ))
        
        return data
        
    except Exception as e:
        logger.error(f"Error getting preprocessed data: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving preprocessed data")

@app.get("/preprocessed-stats")
async def get_preprocessed_stats():
    """Get statistics about preprocessed data"""
    try:
        stats = data_processor.fetch_preprocessed_data_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting preprocessed stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving preprocessed statistics")

@app.get("/system-status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get system status including initialization and training status"""
    try:
        preprocessed_stats = data_processor.fetch_preprocessed_data_stats()
        last_export_time = data_processor.preprocessor.get_last_export_time()
        
        return SystemStatusResponse(
            is_initialized=scheduler.is_initialized,
            is_running=scheduler.is_running,
            preprocessed_records=preprocessed_stats.get("preprocessed_records", 0),
            model_trained=model_predictor.is_trained,
            last_export_time=last_export_time.isoformat() if last_export_time else None
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving system status")

@app.post("/initialize-system")
async def initialize_system(hours_back: int = 168):
    """Initialize the system with full data export and model training"""
    try:
        logger.info("Manual system initialization requested")
        
        if scheduler.is_initialized:
            return {"message": "System already initialized", "status": "success"}
        
        # Perform initial setup
        success = scheduler.perform_initial_setup(hours_back=hours_back)
        
        if success:
            scheduler.is_initialized = True
            return {"message": "System initialized successfully", "status": "success"}
        else:
            raise HTTPException(status_code=500, detail="System initialization failed")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in manual system initialization: {e}")
        raise HTTPException(status_code=500, detail="Error during system initialization")

@app.post("/retrain-model")
async def retrain_model(hours_back: int = 168):
    """Manually trigger model retraining"""
    try:
        logger.info("Manual model retraining requested")
        
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

@app.post("/process-data")
async def process_data(force_full_export: bool = False, hours_back: int = 168):
    """Manually trigger data processing"""
    try:
        logger.info("Manual data processing requested")
        
        raw_df, processed_df = data_processor.process_and_store_new_data(
            force_full_export=force_full_export,
            hours_back=hours_back
        )
        
        if processed_df is not None and not processed_df.empty:
            return {
                "message": "Data processing completed successfully",
                "raw_records": len(raw_df) if raw_df is not None else 0,
                "processed_records": len(processed_df),
                "status": "success"
            }
        else:
            return {"message": "No data to process", "status": "success"}
        
    except Exception as e:
        logger.error(f"Error in manual data processing: {e}")
        raise HTTPException(status_code=500, detail="Error during data processing")

@app.delete("/clear-preprocessed-data")
async def clear_preprocessed_data(older_than_days: Optional[int] = None):
    """Clear old preprocessed data to manage storage"""
    try:
        deleted_count = data_processor.purge_old_preprocessed_data(older_than_days=older_than_days)
        
        return {
            "message": f"Cleared {deleted_count} preprocessed records",
            "deleted_count": deleted_count,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error clearing preprocessed data: {e}")
        raise HTTPException(status_code=500, detail="Error clearing preprocessed data")

@app.get("/scheduler-status")
async def get_scheduler_status():
    """Get scheduler status and next run information"""
    try:
        return scheduler.get_status()
        
    except Exception as e:
        logger.error(f"Error getting scheduler status: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving scheduler status") 