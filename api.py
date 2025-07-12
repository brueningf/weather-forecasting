from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging

from data_processor import DataProcessor
from model_predictor import ModelPredictor
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
data_processor = DataProcessor()
model_predictor = ModelPredictor()

# Pydantic models for API
class PredictionResponse(BaseModel):
    timestamp: datetime
    predicted_temperature: float
    confidence: float

class NextForecastResponse(BaseModel):
    timestamp: datetime
    predicted_temperature: float
    confidence: float

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
            "stats": "/stats",
            "stats_page": "/stats-page"
        }
    }

@app.get("/stats-page")
async def stats_page():
    """Serve the stats HTML page"""
    return FileResponse("static/stats.html")

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

@app.get("/stats")
async def get_stats():
    """Get statistics from both databases"""
    try:
        source_stats = data_processor.get_source_database_stats()
        api_stats = data_processor.get_api_database_stats()
        
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