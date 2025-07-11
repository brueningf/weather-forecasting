# Weather Forecasting API

A continuous weather forecasting system that processes sensor data and provides predictions through a FastAPI REST API.

## Features

- **Automatic Model Training**: Trains the ML model on all available historical data on first startup
- **Continuous Data Processing**: Automatically exports and processes new sensor data at configurable intervals
- **Machine Learning Predictions**: Uses PyTorch neural networks for weather forecasting
- **REST API**: FastAPI-based API for accessing predictions and data
- **Database Integration**: MySQL database for storing sensor data and predictions
- **Scheduled Processing**: APScheduler for background task management
- **Real-time Statistics**: API endpoints for system monitoring
- **Manual Model Retraining**: API endpoint to retrain the model with latest data

## Project Structure

```
weather-forecasting/
├── main.py              # Main application entry point
├── api.py               # FastAPI application and endpoints
├── config.py            # Configuration management
├── data_processor.py    # Database operations and data preprocessing
├── model_predictor.py   # ML model loading and prediction
├── scheduler.py         # Background task scheduling
├── requirements.txt     # Python dependencies
├── config.env.example   # Environment configuration template
└── README.md           # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Database Setup

Create two MySQL databases and the required tables:

```sql
-- Create source database (for sensor data)
CREATE DATABASE sensor_db;

-- Create API database (for predictions)
CREATE DATABASE api_db;

-- In the source database, create the sensor data table
USE sensor_db;
CREATE TABLE sensor_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    temperature FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_timestamp (timestamp)
);

-- In the API database, create the predictions table (will be created automatically by the application)
USE api_db;
CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME,
    predicted_temperature FLOAT,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_timestamp (timestamp)
);
```

### 3. Environment Configuration

Copy the example configuration file and update it with your settings:

```bash
cp config.env.example .env
```

Edit `.env` with your database and API settings:

```env
# Source Database Configuration (where sensor data comes from)
SOURCE_DB_HOST=localhost
SOURCE_DB_USER=your_source_db_user
SOURCE_DB_PASSWORD=your_source_db_password
SOURCE_DB_NAME=sensor_db
SOURCE_DB_PORT=3306

# API Database Configuration (where predictions are stored)
API_DB_HOST=localhost
API_DB_USER=your_api_db_user
API_DB_PASSWORD=your_api_db_password
API_DB_NAME=api_db
API_DB_PORT=3306

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

# Model Configuration
MODEL_PATH=model.pth
PREDICTION_INTERVAL_MINUTES=60

# Data Export Configuration
EXPORT_BATCH_SIZE=1000
```

### 4. Model Setup

The application will automatically train a model on first startup using all available historical data. No pre-trained model is required.

## Running the Application

### Start the API Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

**First Run Behavior:**
- The application will automatically export all available historical data (up to 1 year)
- Train a neural network model on this data
- Save the trained model to `model.pth`
- Start the prediction scheduler

**Subsequent Runs:**
- The application will load the previously trained model
- Continue with normal prediction operations

### API Documentation

Once running, you can access:
- **Interactive API docs**: `http://localhost:8000/docs`
- **ReDoc documentation**: `http://localhost:8000/redoc`

## API Endpoints

### Core Endpoints

- `GET /` - API information and available endpoints
- `GET /health` - Health check with model training status
- `GET /stats` - System statistics

### Data Endpoints

- `GET /latest-data?hours=24` - Get latest sensor data
- `POST /export-data` - Manually trigger data export
- `POST /force-initial-export` - Force export of historical data
- `POST /reset-export-time` - Reset export time to force full data export

### Prediction Endpoints

- `GET /predictions?hours=24` - Get latest predictions from database
- `POST /forecast` - Generate new weather forecast

### Model Training Endpoints

- `POST /train-model` - Manually retrain the model on all available data

### Example API Usage

```bash
# Get system health and model status
curl http://localhost:8000/health

# Get latest predictions
curl http://localhost:8000/predictions?hours=48

# Generate new forecast
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"hours_ahead": 24}'

# Retrain the model
curl -X POST http://localhost:8000/train-model

# Get system statistics
curl http://localhost:8000/stats
```

## Model Training

### Automatic Training

The model is automatically trained on first startup using:
- All available historical data (up to 1 year)
- Features: temperature, hour, day_of_week, month
- Target: next hour's temperature
- Neural network architecture: 3-layer feedforward network
- Training: Adam optimizer with early stopping

### Manual Retraining

You can manually retrain the model using the `/train-model` endpoint:

```bash
curl -X POST http://localhost:8000/train-model
```

This will:
1. Export all available historical data
2. Preprocess the data
3. Train a new model
4. Save the trained model
5. Return training statistics

### Model Architecture

The model uses a simple neural network:
- Input layer: 4 features (temperature, hour, day_of_week, month)
- Hidden layers: 2 layers with 64 neurons each
- Output layer: 1 neuron (predicted temperature)
- Activation: ReLU with dropout for regularization

## Continuous Processing

The application automatically:

1. **Exports new data** from the database every configured interval
2. **Preprocesses the data** (resampling, interpolation, feature engineering)
3. **Generates predictions** using the ML model
4. **Saves predictions** back to the database

The processing interval is configurable via `PREDICTION_INTERVAL_MINUTES` in the environment file.

## Data Flow

```
Sensor Data (Source MySQL DB) → Data Export → Preprocessing → ML Model → Predictions → API MySQL DB
```

## Configuration Options

| Environment Variable      | Default    | Description                                      |
|--------------------------|------------|--------------------------------------------------|
| `SOURCE_DB_HOST`         | localhost  | Source database host (sensor data)               |
| `SOURCE_DB_USER`         | user       | Source database username                         |
| `SOURCE_DB_PASSWORD`     | pass       | Source database password                         |
| `SOURCE_DB_NAME`         | sensor_db  | Source database name                             |
| `SOURCE_DB_PORT`         | 3306       | Source database port                             |
| `API_DB_HOST`            | localhost  | API database host (predictions)                  |
| `API_DB_USER`            | user       | API database username                            |
| `API_DB_PASSWORD`        | pass       | API database password                            |
| `API_DB_NAME`            | api_db     | API database name                                |
| `API_DB_PORT`            | 3306       | API database port                                |
| `API_HOST`               | 0.0.0.0    | API server host                                  |
| `API_PORT`               | 8000       | API server port                                  |
| `API_RELOAD`             | true       | Enable auto-reload for development               |
| `MODEL_PATH`             | model.pth  | Path to trained model                            |
| `PREDICTION_INTERVAL_MINUTES` | 60    | Processing interval in minutes                   |
| `EXPORT_BATCH_SIZE`      | 1000       | Batch size for data export                       |

## Development

### Adding New Features

1. **New API endpoints**: Add to `api.py`
2. **Data processing**: Extend `data_processor.py`
3. **Model improvements**: Modify `model_predictor.py`
4. **Scheduling**: Update `scheduler.py`

### Testing

```bash
# Test the API
curl http://localhost:8000/health

# Test data export
curl -X POST http://localhost:8000/export-data

# Test prediction generation
curl -X POST http://localhost:8000/forecast
```

## Monitoring

The application provides several monitoring endpoints:

- `/health` - Basic health check
- `/stats` - Detailed system statistics
- Logs are output to console with timestamps

## Troubleshooting

### Common Issues

1. **Database Connection Error**: Check database credentials and connectivity
2. **Model Loading Error**: Ensure `model.pth` exists or the application will create a default model
3. **No Data Processing**: Verify sensor data exists in the database
4. **API Not Responding**: Check if the server is running on the correct port

### Logs

The application logs all operations with different levels:
- `INFO`: Normal operations
- `WARNING`: Non-critical issues
- `ERROR`: Critical errors

## License

This project is open source and available under the MIT License. 