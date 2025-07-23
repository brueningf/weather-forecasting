import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent tkinter errors

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, Response, RedirectResponse
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

# Initialize components
config = Config()
data_processor = WeatherDataController()
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
async def dashboard():
    """Main dashboard with all features and analysis"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Weather Forecasting Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            .gradient-bg {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .card-hover {
                transition: transform 0.2s ease-in-out;
            }
            .card-hover:hover {
                transform: translateY(-2px);
            }
        </style>
    </head>
    <body class="bg-gray-50">
        <!-- Header -->
        <header class="gradient-bg text-white shadow-lg">
            <div class="container mx-auto px-6 py-4">
                <h1 class="text-3xl font-bold">🌤️ Weather Forecasting Dashboard</h1>
                <p class="text-blue-100 mt-2">Real-time weather predictions and analysis</p>
            </div>
        </header>

        <!-- Main Content -->
        <div class="container mx-auto px-6 py-8">
            <!-- System Status -->
            <div class="bg-white rounded-lg shadow-md p-6 mb-8">
                <h2 class="text-2xl font-semibold text-gray-800 mb-4">System Status</h2>
                <div id="system-status" class="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div class="bg-blue-50 p-4 rounded-lg">
                        <div class="text-blue-600 font-semibold">Model Status</div>
                        <div id="model-status" class="text-lg">Loading...</div>
                    </div>
                    <div class="bg-green-50 p-4 rounded-lg">
                        <div class="text-green-600 font-semibold">Scheduler</div>
                        <div id="scheduler-status" class="text-lg">Loading...</div>
                    </div>
                    <div class="bg-purple-50 p-4 rounded-lg">
                        <div class="text-purple-600 font-semibold">Data Records</div>
                        <div id="data-records" class="text-lg">Loading...</div>
                    </div>
                    <div class="bg-orange-50 p-4 rounded-lg">
                        <div class="text-orange-600 font-semibold">Last Update</div>
                        <div id="last-update" class="text-lg">Loading...</div>
                    </div>
                </div>
            </div>

            <!-- Quick Actions -->
            <div class="bg-white rounded-lg shadow-md p-6 mb-8">
                <h2 class="text-2xl font-semibold text-gray-800 mb-4">Quick Actions</h2>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <button onclick="initializeSystem()" class="bg-blue-500 hover:bg-blue-600 text-white font-semibold py-3 px-6 rounded-lg card-hover">
                        🔄 Initialize System
                    </button>
                    <button onclick="retrainModel()" class="bg-green-500 hover:bg-green-600 text-white font-semibold py-3 px-6 rounded-lg card-hover">
                        🧠 Retrain Model
                    </button>
                    <button onclick="processData()" class="bg-purple-500 hover:bg-purple-600 text-white font-semibold py-3 px-6 rounded-lg card-hover">
                        📊 Process Data
                    </button>
                </div>
            </div>

            <!-- Analysis Visualizations -->
            <div class="bg-white rounded-lg shadow-md p-6 mb-8">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-2xl font-semibold text-gray-800">Data Analysis</h2>
                    <button onclick="refreshAnalysis()" class="bg-indigo-500 hover:bg-indigo-600 text-white font-semibold py-2 px-4 rounded-lg card-hover">
                        🔄 Refresh Analysis
                    </button>
                </div>
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div>
                        <h3 class="text-lg font-semibold text-gray-700 mb-3">Time Series Analysis</h3>
                        <img id="time-series-plot" src="/api/analysis/time-series" alt="Time Series Analysis" class="w-full rounded-lg shadow-sm">
                    </div>
                    <div>
                        <h3 class="text-lg font-semibold text-gray-700 mb-3">Data Distributions</h3>
                        <img id="distribution-plot" src="/api/analysis/distributions" alt="Data Distributions" class="w-full rounded-lg shadow-sm">
                    </div>
                    <div>
                        <h3 class="text-lg font-semibold text-gray-700 mb-3">Correlation Matrix</h3>
                        <img id="correlation-plot" src="/api/analysis/correlation" alt="Correlation Matrix" class="w-full rounded-lg shadow-sm">
                    </div>
                    <div>
                        <h3 class="text-lg font-semibold text-gray-700 mb-3">Temporal Patterns</h3>
                        <img id="temporal-plot" src="/api/analysis/temporal" alt="Temporal Patterns" class="w-full rounded-lg shadow-sm">
                    </div>
                </div>
            </div>

            <!-- API Endpoints -->
            <div class="bg-white rounded-lg shadow-md p-6 mb-8">
                <h2 class="text-2xl font-semibold text-gray-800 mb-4">Available API Endpoints</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <h3 class="text-lg font-semibold text-gray-700 mb-3">Data & Predictions</h3>
                        <ul class="space-y-2 text-sm">
                            <li><a href="/api/predictions" class="text-blue-600 hover:underline">📈 /api/predictions</a> - Get weather predictions</li>
                            <li><a href="/api/next-forecast" class="text-blue-600 hover:underline">🔮 /api/next-forecast</a> - Next forecast</li>
                            <li><a href="/api/sensor-data" class="text-blue-600 hover:underline">🌡️ /api/sensor-data</a> - Raw sensor data</li>
                            <li><a href="/api/preprocessed-data" class="text-blue-600 hover:underline">⚙️ /api/preprocessed-data</a> - Processed data</li>
                        </ul>
                    </div>
                    <div>
                        <h3 class="text-lg font-semibold text-gray-700 mb-3">System &amp; Analysis</h3>
                        <ul class="space-y-2 text-sm">
                            <li><a href="/api/system-status" class="text-blue-600 hover:underline">📊 /api/system-status</a> - System status</li>
                            <li><a href="/api/stats" class="text-blue-600 hover:underline">📋 /api/stats</a> - Statistics</li>
                            <li><a href="/api/scheduler-status" class="text-blue-600 hover:underline">⏰ /api/scheduler-status</a> - Scheduler status</li>
                            <li><a href="/api/test-scheduler" class="text-blue-600 hover:underline">🧪 /api/test-scheduler</a> - Test scheduler</li>
                            <li><a href="/api/test-database" class="text-blue-600 hover:underline">🗄️ /api/test-database</a> - Test database</li>
                            <li><a href="/api/modules" class="text-blue-600 hover:underline">🔧 /api/modules</a> - Available modules</li>
                        </ul>
                    </div>
                </div>
            </div>

            <!-- Live Data -->
            <div class="bg-white rounded-lg shadow-md p-6 mb-8">
                <h2 class="text-2xl font-semibold text-gray-800 mb-4">Live Weather Data</h2>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div class="bg-red-50 p-4 rounded-lg">
                        <div class="text-red-600 font-semibold">Temperature</div>
                        <div id="current-temp" class="text-2xl font-bold">--°C</div>
                    </div>
                    <div class="bg-blue-50 p-4 rounded-lg">
                        <div class="text-blue-600 font-semibold">Humidity</div>
                        <div id="current-humidity" class="text-2xl font-bold">--%</div>
                    </div>
                    <div class="bg-green-50 p-4 rounded-lg">
                        <div class="text-green-600 font-semibold">Pressure</div>
                        <div id="current-pressure" class="text-2xl font-bold">-- hPa</div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Load system status
            async function loadSystemStatus() {
                try {
                    const response = await fetch('/api/system-status');
                    const status = await response.json();
                    
                    document.getElementById('model-status').textContent = status.model_trained ? '✅ Trained' : '❌ Needs Training';
                    document.getElementById('scheduler-status').textContent = status.is_running ? '✅ Running' : '❌ Stopped';
                    document.getElementById('data-records').textContent = status.preprocessed_records + ' records';
                    document.getElementById('last-update').textContent = status.last_export_time || 'Never';
                } catch (error) {
                    console.error('Error loading system status:', error);
                }
            }

            // Load live data
            async function loadLiveData() {
                try {
                    const response = await fetch('/api/sensor-data?hours=1&limit=1');
                    const data = await response.json();
                    
                    if (data.length > 0) {
                        const latest = data[0];
                        document.getElementById('current-temp').textContent = latest.temperature.toFixed(1) + '°C';
                        document.getElementById('current-humidity').textContent = latest.humidity.toFixed(1) + '%';
                        document.getElementById('current-pressure').textContent = latest.pressure.toFixed(1) + ' hPa';
                    }
                } catch (error) {
                    console.error('Error loading live data:', error);
                }
            }

            // Quick action functions
            async function initializeSystem() {
                try {
                    const response = await fetch('/api/initialize-system', { method: 'POST' });
                    const result = await response.json();
                    
                    if (response.ok) {
                        alert('System initialization: ' + (result.status === 'success' ? 'Success' : result.message));
                    } else {
                        alert('System initialization failed: ' + (result.detail || 'Unknown error'));
                    }
                    loadSystemStatus();
                } catch (error) {
                    alert('Error initializing system: ' + error.message);
                }
            }

            async function retrainModel() {
                try {
                    const response = await fetch('/api/retrain-model', { method: 'POST' });
                    const result = await response.json();
                    
                    if (response.ok) {
                        alert('Model retraining: ' + (result.status === 'success' ? 'Success' : result.message));
                    } else {
                        alert('Model retraining failed: ' + (result.detail || 'Unknown error'));
                    }
                    loadSystemStatus();
                } catch (error) {
                    alert('Error retraining model: ' + error.message);
                }
            }

            async function processData() {
                try {
                    const response = await fetch('/api/process-data', { method: 'POST' });
                    const result = await response.json();
                    
                    if (response.ok) {
                        alert('Data processing: ' + (result.status === 'success' ? 'Success' : result.message));
                    } else {
                        alert('Data processing failed: ' + (result.detail || 'Unknown error'));
                    }
                    loadSystemStatus();
                } catch (error) {
                    alert('Error processing data: ' + error.message);
                }
            }

            // Refresh analysis plots
            async function refreshAnalysis() {
                try {
                    // Add timestamp to prevent caching
                    const timestamp = new Date().getTime();
                    const plots = ['time-series-plot', 'distribution-plot', 'correlation-plot', 'temporal-plot'];
                    
                    plots.forEach(plotId => {
                        const img = document.getElementById(plotId);
                        if (img) {
                            img.src = img.src.split('?')[0] + '?t=' + timestamp;
                        }
                    });
                    
                    // Show a brief success message
                    const button = event.target;
                    const originalText = button.innerHTML;
                    button.innerHTML = '✅ Refreshed!';
                    button.disabled = true;
                    
                    setTimeout(() => {
                        button.innerHTML = originalText;
                        button.disabled = false;
                    }, 2000);
                    
                } catch (error) {
                    console.error('Error refreshing analysis:', error);
                    alert('Error refreshing analysis plots');
                }
            }

            // Initialize dashboard
            document.addEventListener('DOMContentLoaded', function() {
                loadSystemStatus();
                loadLiveData();
                
                // Refresh data every 30 seconds
                setInterval(() => {
                    loadSystemStatus();
                    loadLiveData();
                }, 30000);
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/stats-page")
async def stats_page():
    """Serve the stats HTML page"""
    return FileResponse("static/stats.html")

@app.get("/temperature-graph")
async def temperature_graph_page():
    """Serve the temperature comparison graph HTML page"""
    return FileResponse("static/temperature-graph.html")

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
        predictions_df = data_processor.get_latest_predictions(hours=hours)
        
        if predictions_df.empty:
            return []
        
        # Convert to response format
        data = []
        for index, row in predictions_df.iterrows():
            data.append(PredictionResponse(
                timestamp=index,
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
        predictions_df = data_processor.get_latest_predictions(hours=24)
        
        if predictions_df.empty:
            raise HTTPException(status_code=404, detail="No forecast available")
        
        # Get the most recent prediction
        latest_prediction = predictions_df.iloc[-1]
        
        return NextForecastResponse(
            timestamp=latest_prediction.name,  # Use index as timestamp
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
        raw_data = data_processor.fetch_recent_sensor_data(hours=hours, module_id=module_id)
        
        if raw_data.empty:
            return []
        
        # Convert to response format
        data = []
        for index, row in raw_data.iterrows():
            data.append(ActualTemperatureResponse(
                timestamp=index,
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
            FROM {data_processor.db_manager.sensor_data_table}
            WHERE module IS NOT NULL
            ORDER BY module
        """
        
        df = pd.read_sql(query, data_processor.db_manager.source_engine)
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
        raw_stats = data_processor.fetch_source_db_stats()
        preprocessed_stats = data_processor.fetch_preprocessed_data_stats()
        api_stats = data_processor.fetch_api_db_stats()
        
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

@app.get("/api/preprocessed-stats")
async def get_preprocessed_stats():
    """Get statistics about preprocessed data"""
    try:
        stats = data_processor.fetch_preprocessed_data_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting preprocessed stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving preprocessed statistics")

@app.get("/api/system-status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get system status including initialization and training status"""
    try:
        preprocessed_stats = data_processor.fetch_preprocessed_data_stats()
        last_export_time = data_processor.preprocessor.get_last_export_time()
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
            raw_data = data_processor.fetch_recent_sensor_data(hours=1)
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
        
        raw_df, processed_df = data_processor.process_and_store_new_data(
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
        
        deleted_count = data_processor.purge_old_preprocessed_data(older_than_days=older_than_days)
        
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
            query = f"SELECT COUNT(*) as count FROM {data_processor.db_manager.sensor_data_table}"
            df = pd.read_sql(query, data_processor.db_manager.source_engine)
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
            test_engine = data_processor.db_manager.api_engine
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
            session = data_processor.db_manager.Session()
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
                "source_host": data_processor.db_manager.config.SOURCE_DB_HOST,
                "source_db": data_processor.db_manager.config.SOURCE_DB_NAME,
                "api_host": data_processor.db_manager.config.API_DB_HOST,
                "api_db": data_processor.db_manager.config.API_DB_NAME
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