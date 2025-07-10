from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
    description="API for weather prediction and data management",
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

# Initialize components
config = Config()
data_processor = DataProcessor()
model_predictor = ModelPredictor()

# Pydantic models for API
class PredictionResponse(BaseModel):
    timestamp: datetime
    predicted_temperature: float
    confidence: float

class ForecastRequest(BaseModel):
    hours_ahead: int = 24

class DataExportResponse(BaseModel):
    message: str
    records_exported: int
    timestamp: datetime

class InitialExportRequest(BaseModel):
    hours_back: int = 168  # Default to 1 week

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Weather Forecasting API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predictions": "/predictions",
            "forecast": "/forecast",
            "export_data": "/export-data",
            "force_initial_export": "/force-initial-export",
            "reset_export_time": "/reset-export-time",
            "latest_data": "/latest-data",
            "stats": "/stats"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "model_loaded": model_predictor.model is not None
    }

@app.get("/predictions", response_model=List[PredictionResponse])
async def get_predictions(hours: int = 24):
    """Get latest predictions from database"""
    try:
        predictions_df = data_processor.get_latest_predictions(hours=hours)
        
        if predictions_df.empty:
            return []
        
        # Convert to response format
        predictions = []
        for index, row in predictions_df.iterrows():
            predictions.append(PredictionResponse(
                timestamp=index,
                predicted_temperature=float(row['predicted_temperature']),
                confidence=float(row['confidence'])
            ))
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving predictions")

@app.post("/forecast", response_model=List[PredictionResponse])
async def generate_forecast(request: ForecastRequest):
    """Generate new weather forecast"""
    try:
        # Export latest data
        df = data_processor.export_data()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data available for forecasting")
        
        # Preprocess data
        processed_df = data_processor.preprocess_data(df)
        
        if processed_df.empty:
            raise HTTPException(status_code=500, detail="Error preprocessing data")
        
        # Generate predictions
        predictions_df = model_predictor.predict_future(processed_df, hours_ahead=request.hours_ahead)
        
        if predictions_df.empty:
            raise HTTPException(status_code=500, detail="Error generating predictions")
        
        # Save predictions to database
        data_processor.save_predictions(predictions_df)
        
        # Convert to response format
        predictions = []
        for index, row in predictions_df.iterrows():
            predictions.append(PredictionResponse(
                timestamp=index,
                predicted_temperature=float(row['predicted_temperature']),
                confidence=float(row['confidence'])
            ))
        
        return predictions
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        raise HTTPException(status_code=500, detail="Error generating forecast")

@app.post("/export-data", response_model=DataExportResponse)
async def export_data():
    """Export new data from database"""
    try:
        df = data_processor.export_data()
        
        return DataExportResponse(
            message="Data exported successfully",
            records_exported=len(df),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail="Error exporting data")

@app.post("/force-initial-export", response_model=DataExportResponse)
async def force_initial_export(request: InitialExportRequest):
    """Force an initial data export for model training"""
    try:
        df = data_processor.force_initial_export(hours_back=request.hours_back)
        
        return DataExportResponse(
            message=f"Forced initial export from last {request.hours_back} hours",
            records_exported=len(df),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error in forced initial export: {e}")
        raise HTTPException(status_code=500, detail="Error in forced initial export")

@app.post("/reset-export-time", response_model=dict)
async def reset_export_time():
    """Reset the last export time to force a full data export on next call"""
    try:
        data_processor.reset_export_time()
        
        return {
            "message": "Export time reset successfully",
            "timestamp": datetime.now(),
            "next_export": "will include all available data"
        }
        
    except Exception as e:
        logger.error(f"Error resetting export time: {e}")
        raise HTTPException(status_code=500, detail="Error resetting export time")

@app.get("/latest-data")
async def get_latest_data(hours: int = 24):
    """Get latest sensor data from source database"""
    try:
        df = data_processor.get_latest_sensor_data(hours=hours)
        
        # Convert to list of dictionaries
        data = []
        for _, row in df.iterrows():
            data.append({
                "timestamp": row['timestamp'].isoformat() if pd.notna(row['timestamp']) else None,
                "temperature": float(row['temperature']) if pd.notna(row['temperature']) else None
            })
        
        return {
            "data": data,
            "count": len(data),
            "hours_requested": hours,
            "source": "source_database"
        }
        
    except Exception as e:
        logger.error(f"Error getting latest data from source database: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving latest data")

@app.get("/stats")
async def get_stats():
    """Get system statistics from both databases"""
    try:
        # Get stats from both databases
        source_stats = data_processor.get_source_database_stats()
        api_stats = data_processor.get_api_database_stats()
        
        return {
            "source_database": source_stats,
            "api_database": api_stats,
            "model_loaded": model_predictor.model is not None,
            "last_export_time": data_processor.last_export_time.isoformat() if data_processor.last_export_time else None
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving statistics") 