from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging

from data_processor import DataProcessor
from model_predictor import ModelPredictor
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
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
            "latest_data": "/latest-data"
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

# Background task for continuous processing
async def continuous_processing():
    """Background task for continuous data processing and prediction"""
    while True:
        try:
            logger.info("Starting continuous processing cycle")
            
            # Export new data
            df = data_processor.export_data()
            
            if not df.empty:
                # Preprocess data
                processed_df = data_processor.preprocess_data(df)
                
                if not processed_df.empty:
                    # Generate predictions
                    predictions_df = model_predictor.predict_future(processed_df, hours_ahead=24)
                    
                    if not predictions_df.empty:
                        # Save predictions
                        data_processor.save_predictions(predictions_df)
                        logger.info("Continuous processing cycle completed successfully")
                    else:
                        logger.warning("No predictions generated in continuous processing")
                else:
                    logger.warning("No processed data available for prediction")
            else:
                logger.info("No new data to process")
            
            # Wait for next cycle
            await asyncio.sleep(config.PREDICTION_INTERVAL_MINUTES * 60)
            
        except Exception as e:
            logger.error(f"Error in continuous processing: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying

# Import asyncio for background tasks
import asyncio 