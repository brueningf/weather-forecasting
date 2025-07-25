import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent tkinter errors

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, Response, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
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
from analysis import WeatherDataAnalyzer

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
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/stats-page", response_class=HTMLResponse)
async def stats_page(request: Request):
    """Serve the stats HTML page"""
    return templates.TemplateResponse("stats.html", {"request": request})

@app.get("/temperature-graph", response_class=HTMLResponse)
async def temperature_graph_page(request: Request):
    """Serve the temperature comparison graph HTML page"""
    return templates.TemplateResponse("temperature-graph.html", {"request": request})

@app.get("/sensor-data")
async def redirect_sensor_data(hours: int = 24, module_id: Optional[str] = None):
    """Redirect old sensor-data endpoint to new API endpoint"""
    params = f"?hours={hours}"
    if module_id:
        params += f"&module_id={module_id}"
    return RedirectResponse(url=f"/api/sensor-data{params}")

@app.get("/predictions")
async def redirect_predictions(hours: int = 24):
    """Redirect old predictions endpoint to new API endpoint"""
    return RedirectResponse(url=f"/api/predictions?hours={hours}")

@app.get("/next-forecast")
async def redirect_next_forecast():
    """Redirect old next-forecast endpoint to new API endpoint"""
    return RedirectResponse(url="/api/next-forecast")

@app.get("/stats")
async def redirect_stats():
    """Redirect old stats endpoint to new API endpoint"""
    return RedirectResponse(url="/api/stats")

@app.get("/api/predictions", response_model=List[PredictionResponse])
async def get_predictions(hours: int = 24):
    """Get weather predictions for the specified number of hours"""
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
                confidence=float(row.get('confidence', 0))
            ))
        
        return data
        
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving predictions")

@app.get("/api/next-forecast", response_model=NextForecastResponse)
async def get_next_forecast():
    """Get the next weather forecast"""
    try:
        # Get the latest prediction from database
        predictions_df = weather_data_controller.get_latest_predictions(hours=24)
        
        if predictions_df.empty:
            raise HTTPException(status_code=404, detail="No forecast available")
        
        # Get the most recent prediction
        latest_prediction = predictions_df.iloc[-1]
        
        return NextForecastResponse(
            timestamp=latest_prediction['timestamp'],
            predicted_temperature=float(latest_prediction.get('predicted_temperature', 0)),
            confidence=float(latest_prediction.get('confidence', 0))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting next forecast: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving next forecast")

@app.get("/api/sensor-data", response_model=List[ActualTemperatureResponse])
async def get_sensor_data(hours: int = 24, module_id: Optional[str] = None):
    """Get actual sensor data for the specified number of hours"""
    try:
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

@app.get("/api/stats")
async def get_stats():
    """Get comprehensive statistics about the system"""
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
        scheduler_info = {}
        if scheduler is not None:
            try:
                scheduler_status = scheduler.get_status()
                # Use APScheduler's running status as fallback if our flag is not set
                if not scheduler_status.get("is_running", False) and scheduler_status.get("apscheduler_running", False):
                    logger.info("APScheduler is running but our flag is not set - using APScheduler status")
                    scheduler_status["is_running"] = True
            except Exception as e:
                logger.warning(f"Could not get scheduler status: {e}")
        
        return {
            "raw_data": raw_stats,
            "preprocessed_data": preprocessed_stats,
            "api_data": api_stats,
            "model": model_info,
            "scheduler": scheduler_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving statistics")

@app.get("/api/preprocessed-data", response_model=List[PreprocessedDataResponse])
async def get_preprocessed_data(hours_back: Optional[int] = None, limit: Optional[int] = None):
    """Get preprocessed data for analysis"""
    try:
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

@app.get("/api/system-status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get system status including initialization and training status"""
    try:
        preprocessed_stats = weather_data_controller.fetch_preprocessed_data_stats()
        last_export_time = weather_data_controller.preprocessor.get_last_export_time()
        # Check if scheduler is available and initialized
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
        logger.info(f"System status: is_running={scheduler_status.get('is_running', False)} (source: {status_source})")
        return SystemStatusResponse(
            is_initialized=scheduler_status.get("is_initialized", False),
            is_running=scheduler_status.get("is_running", False),
            preprocessed_records=preprocessed_stats.get("preprocessed_records", 0),
            model_trained=model_predictor.is_trained,
            last_export_time=last_export_time.isoformat() if last_export_time else None
        )
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving system status")

@app.post("/api/initialize-system")
async def initialize_system(hours_back: int = 168):
    """Initialize the system with full data export and model training"""
    try:
        logger.info("Manual system initialization requested")
        
        # Ensure scheduler exists
        if scheduler is None:
            logger.info("Scheduler is None - creating it now...")
            try:
                created_scheduler = create_scheduler(model_predictor)
                if created_scheduler is None:
                    logger.error("create_scheduler returned None")
                    raise HTTPException(status_code=500, detail="Failed to create scheduler - returned None")
                logger.info("Scheduler created successfully")
            except Exception as scheduler_error:
                logger.error(f"Error creating scheduler: {scheduler_error}")
                import traceback
                logger.error(f"Scheduler creation traceback: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=f"Failed to create scheduler: {str(scheduler_error)}")
        
        # Double-check scheduler exists
        if scheduler is None:
            logger.error("Scheduler is still None after creation attempt")
            raise HTTPException(status_code=500, detail="Failed to create scheduler - scheduler is still None")
        
        if scheduler.is_initialized:
            logger.info("System already initialized")
            return {"message": "System already initialized", "status": "success"}
        
        # Perform initial setup
        logger.info(f"Starting system initialization with {hours_back} hours of data")
        
        # Check data availability first
        logger.info("Checking data availability...")
        try:
            raw_data = weather_data_controller.fetch_recent_sensor_data(hours=1)
            logger.info(f"Raw data check: {len(raw_data)} records found")
            if raw_data.empty:
                logger.warning("No sensor data available")
                raise HTTPException(
                    status_code=500, 
                    detail="System initialization failed: No sensor data available. Please ensure your sensor data source is connected and has data."
                )
        except Exception as data_check_error:
            logger.error(f"Data availability check failed: {data_check_error}")
            raise HTTPException(
                status_code=500, 
                detail="System initialization failed: Unable to access data source. Please check your database configuration."
            )
        
        # Try to perform initial setup
        logger.info("Attempting to perform initial setup...")
        try:
            success = scheduler.perform_initial_setup(hours_back=hours_back)
        except Exception as setup_error:
            logger.error(f"Error in perform_initial_setup: {setup_error}")
            import traceback
            logger.error(f"Initial setup traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Initial setup failed: {str(setup_error)}")
        
        if success:
            scheduler.is_initialized = True
            logger.info("System initialization completed successfully")
            return {"message": "System initialized successfully", "status": "success"}
        else:
            logger.error("System initialization failed - perform_initial_setup returned False")
            raise HTTPException(
                status_code=500, 
                detail="System initialization failed: Unable to process data or train model. Check logs for details."
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in manual system initialization: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error during system initialization: {str(e)}")

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

@app.delete("/api/clear-preprocessed-data")
async def clear_preprocessed_data(older_than_days: Optional[int] = None):
    """Clear preprocessed data older than specified days"""
    try:
        logger.info("Manual preprocessed data clearing requested")
        
        deleted_count = weather_data_controller.purge_old_preprocessed_data(older_than_days=older_than_days)
        
        return {
            "message": f"Cleared {deleted_count} preprocessed records",
            "status": "success",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Error clearing preprocessed data: {e}")
        raise HTTPException(status_code=500, detail="Error clearing preprocessed data")

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

@app.get("/api/test-scheduler")
async def test_scheduler():
    """Test endpoint to check if scheduler is working"""
    try:
        if scheduler is None:
            return {"message": "Scheduler not available", "working": False}
        
        # Try to get status
        status = scheduler.get_status()
        
        # Try to run a simple operation
        try:
            scheduler.run_once_prediction()
            prediction_working = True
        except Exception as e:
            prediction_working = False
            logger.warning(f"Prediction test failed: {e}")
        
        return {
            "message": "Scheduler test completed",
            "working": True,
            "status": status,
            "prediction_working": prediction_working
        }
        
    except Exception as e:
        logger.error(f"Error testing scheduler: {e}")
        return {"message": f"Scheduler test failed: {str(e)}", "working": False}

@app.get("/api/test-components")
async def test_components():
    """Test if basic components can be created without issues"""
    try:
        logger.info("Testing component creation...")
        
        results = {}
        
        # Test config
        try:
            from config import Config
            config = Config()
            results["config"] = {"status": "success", "message": "Config loaded successfully"}
        except Exception as e:
            results["config"] = {"status": "error", "message": str(e)}
        
        # Test database manager
        try:
            from database import DatabaseManager
            db_manager = DatabaseManager()
            results["database_manager"] = {"status": "success", "message": "Database manager created successfully"}
        except Exception as e:
            results["database_manager"] = {"status": "error", "message": str(e)}
        
        # Test weather data controller
        try:
            from weather_data_controller import WeatherDataController
            data_controller = WeatherDataController()
            results["weather_data_controller"] = {"status": "success", "message": "Weather data controller created successfully"}
        except Exception as e:
            results["weather_data_controller"] = {"status": "error", "message": str(e)}
        
        # Test model predictor
        try:
            from model_predictor import ModelPredictor
            model_predictor = ModelPredictor()
            results["model_predictor"] = {"status": "success", "message": "Model predictor created successfully"}
        except Exception as e:
            results["model_predictor"] = {"status": "error", "message": str(e)}
        
        # Test scheduler creation
        try:
            from scheduler import create_scheduler
            test_scheduler = create_scheduler(model_predictor)
            if test_scheduler is None:
                results["scheduler"] = {"status": "error", "message": "Scheduler creation returned None"}
            else:
                results["scheduler"] = {"status": "success", "message": "Scheduler created successfully"}
        except Exception as e:
            results["scheduler"] = {"status": "error", "message": str(e)}
        
        return {
            "component_tests": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error testing components: {e}")
        return {"error": str(e)}

@app.get("/api/test-database")
async def test_database():
    """Test database connectivity"""
    try:
        logger.info("Testing database connectivity...")
        
        # Test source database connection
        source_status = {"connected": False, "error": None}
        try:
            # Try to get a simple count from source database
            query = f"SELECT COUNT(*) as count FROM {weather_data_controller.db_manager.sensor_data_table}"
            df = pd.read_sql(query, weather_data_controller.db_manager.source_engine)
            source_status = {
                "connected": True, 
                "record_count": int(df['count'][0]) if not df.empty else 0
            }
        except Exception as e:
            source_status = {"connected": False, "error": str(e)}
        
        # Test API database connection
        api_status = {"connected": False, "error": None, "tables_created": False}
        try:
            # First, test if we can connect to the API database
            test_engine = weather_data_controller.db_manager.api_engine
            with test_engine.connect() as conn:
                # Test basic connection - use SQLAlchemy 2.0 compatible syntax
                from sqlalchemy import text
                conn.execute(text("SELECT 1"))
            
            # Try to create tables if they don't exist
            try:
                from database import Base
                Base.metadata.create_all(test_engine)
                api_status["tables_created"] = True
            except Exception as table_error:
                api_status["table_error"] = str(table_error)
            
            # Now try to query the Prediction table
            from database import Prediction
            session = weather_data_controller.db_manager.Session()
            prediction_count = session.query(Prediction).count()
            session.close()
            api_status = {
                "connected": True, 
                "prediction_count": prediction_count,
                "tables_created": api_status.get("tables_created", False)
            }
        except Exception as e:
            api_status = {"connected": False, "error": str(e)}
        
        return {
            "source_database": source_status,
            "api_database": api_status,
            "config": {
                "source_host": weather_data_controller.db_manager.config.SOURCE_DB_HOST,
                "source_db": weather_data_controller.db_manager.config.SOURCE_DB_NAME,
                "api_host": weather_data_controller.db_manager.config.API_DB_HOST,
                "api_db": weather_data_controller.db_manager.config.API_DB_NAME
            }
        }
        
    except Exception as e:
        logger.error(f"Error testing database: {e}")
        return {"error": str(e)}

@app.get("/api/analysis/time-series")
async def get_time_series_plot():
    """Generate time series analysis plot"""
    try:
        analyzer = WeatherDataAnalyzer()
        df = analyzer.load_data(hours_back=168)  # Last week
        
        if df is None or df.empty:
            # Create a placeholder plot when no data is available
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'No data available for analysis\nPlease initialize the system first', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Time Series Analysis - No Data Available')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            # Convert plot to image
            canvas = FigureCanvas(fig)
            canvas.draw()
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            img_data.seek(0)
            plt.close(fig)
            
            return Response(content=img_data.getvalue(), media_type="image/png")
        
        # Create time series plot
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Weather Data Time Series Analysis', fontsize=16)
        
        weather_cols = ['temperature', 'humidity', 'pressure']
        colors = ['red', 'blue', 'green']
        
        for i, (col, color) in enumerate(zip(weather_cols, colors)):
            if col in df.columns:
                axes[i].plot(df.index, df[col], color=color, alpha=0.7, linewidth=0.8)
                axes[i].set_title(f'{col.capitalize()} Over Time')
                axes[i].set_ylabel(col.capitalize())
                axes[i].grid(True, alpha=0.3)
                
                # Add rolling average
                if len(df) > 24:
                    rolling_avg = df[col].rolling(window=24, center=True).mean()
                    axes[i].plot(df.index, rolling_avg, color='black', linewidth=2, 
                               label='24-period Moving Average')
                    axes[i].legend()
        
        plt.tight_layout()
        
        # Convert plot to image
        canvas = FigureCanvas(fig)
        canvas.draw()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plt.close(fig)
        
        return Response(content=img_data.getvalue(), media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error generating time series plot: {e}")
        # Create error plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, f'Error generating plot: {str(e)}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14, color='red')
        ax.set_title('Time Series Analysis - Error')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        canvas = FigureCanvas(fig)
        canvas.draw()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plt.close(fig)
        
        return Response(content=img_data.getvalue(), media_type="image/png")

@app.get("/api/analysis/distributions")
async def get_distribution_plot():
    """Generate distribution analysis plot"""
    try:
        analyzer = WeatherDataAnalyzer()
        df = analyzer.load_data(hours_back=168)
        
        if df is None or df.empty:
            # Create a placeholder plot when no data is available
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No data available for analysis\nPlease initialize the system first', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Data Distributions - No Data Available')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            canvas = FigureCanvas(fig)
            canvas.draw()
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            img_data.seek(0)
            plt.close(fig)
            
            return Response(content=img_data.getvalue(), media_type="image/png")
        
        # Create distribution plots
        weather_cols = ['temperature', 'humidity', 'pressure']
        weather_cols_available = [col for col in weather_cols if col in df.columns]
        
        if not weather_cols_available:
            # Create a placeholder plot when no weather data is available
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No weather data available for distribution analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Data Distributions - No Weather Data')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            canvas = FigureCanvas(fig)
            canvas.draw()
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            img_data.seek(0)
            plt.close(fig)
            
            return Response(content=img_data.getvalue(), media_type="image/png")
        
        fig, axes = plt.subplots(1, len(weather_cols_available), figsize=(5*len(weather_cols_available), 4))
        if len(weather_cols_available) == 1:
            axes = [axes]
        
        fig.suptitle('Weather Data Distributions', fontsize=16)
        
        for i, col in enumerate(weather_cols_available):
            data = df[col].dropna()
            axes[i].hist(data, bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{col.capitalize()} Distribution')
            axes[i].set_xlabel(col.capitalize())
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # Add vertical lines for mean and median
            axes[i].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.1f}')
            axes[i].axvline(data.median(), color='green', linestyle='--', label=f'Median: {data.median():.1f}')
            axes[i].legend()
        
        plt.tight_layout()
        
        # Convert plot to image
        canvas = FigureCanvas(fig)
        canvas.draw()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plt.close(fig)
        
        return Response(content=img_data.getvalue(), media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error generating distribution plot: {e}")
        # Create error plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error generating plot: {str(e)}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14, color='red')
        ax.set_title('Data Distributions - Error')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        canvas = FigureCanvas(fig)
        canvas.draw()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plt.close(fig)
        
        return Response(content=img_data.getvalue(), media_type="image/png")

@app.get("/api/analysis/correlation")
async def get_correlation_plot():
    """Generate correlation matrix plot"""
    try:
        analyzer = WeatherDataAnalyzer()
        df = analyzer.load_data(hours_back=168)
        
        if df is None or df.empty:
            # Create a placeholder plot when no data is available
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'No data available for analysis\nPlease initialize the system first', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Correlation Matrix - No Data Available')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            canvas = FigureCanvas(fig)
            canvas.draw()
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            img_data.seek(0)
            plt.close(fig)
            
            return Response(content=img_data.getvalue(), media_type="image/png")
        
        # Create correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            # Create a placeholder plot when insufficient data
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'Insufficient numeric data for correlation analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Correlation Matrix - Insufficient Data')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            canvas = FigureCanvas(fig)
            canvas.draw()
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            img_data.seek(0)
            plt.close(fig)
            
            return Response(content=img_data.getvalue(), media_type="image/png")
        
        # Select key columns for cleaner visualization
        key_cols = [col for col in ['temperature', 'humidity', 'pressure', 'temperature_normalized', 
                   'humidity_normalized', 'pressure_normalized', 'hour', 'day_of_week', 'month'] 
                   if col in df.columns]
        
        if len(key_cols) < 2:
            # Create a placeholder plot when insufficient key columns
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'Insufficient key columns for correlation analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Correlation Matrix - Insufficient Columns')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            canvas = FigureCanvas(fig)
            canvas.draw()
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            img_data.seek(0)
            plt.close(fig)
            
            return Response(content=img_data.getvalue(), media_type="image/png")
        
        plt.figure(figsize=(10, 8))
        corr_matrix = df[key_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .8})
        plt.title('Weather Data Correlation Matrix')
        plt.tight_layout()
        
        # Convert plot to image
        canvas = FigureCanvas(plt.gcf())
        canvas.draw()
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plt.close()
        
        return Response(content=img_data.getvalue(), media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error generating correlation plot: {e}")
        # Create error plot
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f'Error generating plot: {str(e)}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14, color='red')
        ax.set_title('Correlation Matrix - Error')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        canvas = FigureCanvas(fig)
        canvas.draw()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plt.close(fig)
        
        return Response(content=img_data.getvalue(), media_type="image/png")

@app.get("/api/analysis/temporal")
async def get_temporal_plot():
    """Generate temporal patterns plot"""
    try:
        analyzer = WeatherDataAnalyzer()
        df = analyzer.load_data(hours_back=168)
        
        if df is None or df.empty:
            # Create a placeholder plot when no data is available
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'No data available for analysis\nPlease initialize the system first', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Temporal Patterns - No Data Available')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            canvas = FigureCanvas(fig)
            canvas.draw()
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            img_data.seek(0)
            plt.close(fig)
            
            return Response(content=img_data.getvalue(), media_type="image/png")
        
        if 'temperature' not in df.columns or len(df) < 24:
            # Create a placeholder plot when insufficient temperature data
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'Insufficient temperature data for temporal analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Temporal Patterns - Insufficient Data')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            canvas = FigureCanvas(fig)
            canvas.draw()
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            img_data.seek(0)
            plt.close(fig)
            
            return Response(content=img_data.getvalue(), media_type="image/png")
        
        # Create temporal patterns plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Temporal Patterns Analysis', fontsize=16)
        
        # Hourly temperature pattern
        hourly_temp = df.groupby(df.index.hour)['temperature'].mean()
        axes[0,0].plot(hourly_temp.index, hourly_temp.values, marker='o')
        axes[0,0].set_title('Average Temperature by Hour')
        axes[0,0].set_xlabel('Hour of Day')
        axes[0,0].set_ylabel('Temperature (°C)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Daily pattern (if enough data)
        if len(df) > 7:
            daily_temp = df.groupby(df.index.dayofweek)['temperature'].mean()
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            axes[0,1].bar(range(7), daily_temp.values)
            axes[0,1].set_title('Average Temperature by Day of Week')
            axes[0,1].set_xlabel('Day of Week')
            axes[0,1].set_ylabel('Temperature (°C)')
            axes[0,1].set_xticks(range(7))
            axes[0,1].set_xticklabels(day_names)
            axes[0,1].grid(True, alpha=0.3)
        else:
            axes[0,1].text(0.5, 0.5, 'Insufficient data for daily pattern', 
                          ha='center', va='center', transform=axes[0,1].transAxes)
            axes[0,1].set_title('Daily Pattern - Insufficient Data')
            axes[0,1].axis('off')
        
        # Monthly pattern (if enough data)
        if len(df) > 30:
            monthly_temp = df.groupby(df.index.month)['temperature'].mean()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            axes[1,0].plot(monthly_temp.index, monthly_temp.values, marker='o')
            axes[1,0].set_title('Average Temperature by Month')
            axes[1,0].set_xlabel('Month')
            axes[1,0].set_ylabel('Temperature (°C)')
            axes[1,0].set_xticks(range(1, 13))
            axes[1,0].set_xticklabels(month_names)
            axes[1,0].grid(True, alpha=0.3)
        else:
            axes[1,0].text(0.5, 0.5, 'Insufficient data for monthly pattern', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Monthly Pattern - Insufficient Data')
            axes[1,0].axis('off')
        
        # Temperature range by hour
        if len(df) > 24:
            hourly_stats = df.groupby(df.index.hour)['temperature'].agg(['mean', 'std'])
            axes[1,1].fill_between(hourly_stats.index, 
                                  hourly_stats['mean'] - hourly_stats['std'],
                                  hourly_stats['mean'] + hourly_stats['std'], 
                                  alpha=0.3, label='±1 Std Dev')
            axes[1,1].plot(hourly_stats.index, hourly_stats['mean'], marker='o', label='Mean')
            axes[1,1].set_title('Temperature Variation by Hour')
            axes[1,1].set_xlabel('Hour of Day')
            axes[1,1].set_ylabel('Temperature (°C)')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        else:
            axes[1,1].text(0.5, 0.5, 'Insufficient data for variation analysis', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Temperature Variation - Insufficient Data')
            axes[1,1].axis('off')
        
        plt.tight_layout()
        
        # Convert plot to image
        canvas = FigureCanvas(fig)
        canvas.draw()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plt.close(fig)
        
        return Response(content=img_data.getvalue(), media_type="image/png")
        
    except Exception as e:
        logger.error(f"Error generating temporal plot: {e}")
        # Create error plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, f'Error generating plot: {str(e)}', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14, color='red')
        ax.set_title('Temporal Patterns - Error')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        canvas = FigureCanvas(fig)
        canvas.draw()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plt.close(fig)
        
        return Response(content=img_data.getvalue(), media_type="image/png") 

@app.get("/api/analysis/temperature-comparison")
async def get_temperature_comparison_plot(
    hours: int = Query(24, ge=1, le=168),
    module_id: str = None,
    data_type: str = Query('preprocessed', regex='^(preprocessed|source)$')
):
    """Generate temperature comparison plot (actual vs predicted, past & future)"""
    try:
        # Fetch actual data (preprocessed or source)
        if data_type == 'preprocessed':
            actual_data = weather_data_controller.fetch_preprocessed_data(hours_back=hours)
            if actual_data is not None and not actual_data.empty:
                actual_data = actual_data.reset_index()
                # Normalize columns
                actual_data = actual_data[['timestamp', 'temperature', 'humidity', 'pressure']]
                if module_id and 'module' in actual_data.columns:
                    actual_data = actual_data[actual_data['module'] == module_id]
        else:
            actual_data = weather_data_controller.fetch_recent_sensor_data(hours=hours, module_id=module_id)
            if actual_data is not None and not actual_data.empty:
                # Normalize columns
                actual_data = actual_data[['timestamp', 'temperature', 'humidity', 'pressure']]
        # Sort by timestamp
        if actual_data is not None and not actual_data.empty:
            actual_data = actual_data.sort_values('timestamp')
        # Fetch predictions
        predictions_df = weather_data_controller.get_latest_predictions(hours=hours)
        if predictions_df is not None and not predictions_df.empty:
            predictions_df = predictions_df.copy()
            predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
            predictions_df = predictions_df.sort_values('timestamp')
        # If both are empty, show placeholder
        if (actual_data is None or actual_data.empty) and (predictions_df is None or predictions_df.empty):
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Temperature Comparison - No Data Available')
            ax.axis('off')
            canvas = FigureCanvas(fig)
            canvas.draw()
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            img_data.seek(0)
            plt.close(fig)
            return Response(content=img_data.getvalue(), media_type="image/png")
        fig, ax = plt.subplots(figsize=(12, 6))
        now = pd.Timestamp.now()
        # Plot actual data
        if actual_data is not None and not actual_data.empty:
            ax.plot(actual_data['timestamp'], actual_data['temperature'], label=f'{data_type.capitalize()} Data', color='royalblue', marker='o', alpha=0.7)
        # Plot predictions (split past/future)
        if predictions_df is not None and not predictions_df.empty:
            past_pred = predictions_df[predictions_df['timestamp'] <= now]
            future_pred = predictions_df[predictions_df['timestamp'] > now]
            if not past_pred.empty:
                ax.plot(past_pred['timestamp'], past_pred['predicted_temperature'], label='Past Predictions', color='crimson', linestyle='-', marker='x', alpha=0.7)
            if not future_pred.empty:
                ax.plot(future_pred['timestamp'], future_pred['predicted_temperature'], label='Future Predictions', color='purple', linestyle='--', marker='x', alpha=0.7)
        # Mark 'now' with a vertical line
        ax.axvline(now, color='black', linestyle=':', linewidth=2, label='Now')
        ax.set_title(f'Temperature Comparison (Last {hours} hours)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature (°C)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        plt.tight_layout()
        # Convert plot to image
        canvas = FigureCanvas(fig)
        canvas.draw()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plt.close(fig)
        return Response(content=img_data.getvalue(), media_type="image/png")
    except Exception as e:
        logger.error(f"Error generating temperature comparison plot: {e}")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f'Error generating plot: {str(e)}', ha='center', va='center', transform=ax.transAxes, fontsize=14, color='red')
        ax.set_title('Temperature Comparison - Error')
        ax.axis('off')
        canvas = FigureCanvas(fig)
        canvas.draw()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plt.close(fig)
        return Response(content=img_data.getvalue(), media_type="image/png") 

@app.get("/api/model/benchmark")
def get_model_benchmark():
    """Return latest model benchmark metrics (MAE, RMSE, R2)"""
    metrics = model_predictor.get_latest_metrics()
    if metrics is None:
        return JSONResponse(content={"error": "No benchmark metrics available. Retrain the model first."}, status_code=404)
    return metrics 