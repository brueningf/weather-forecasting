# Weather Forecasting API

A continuous weather forecasting system that processes sensor data and provides predictions through a FastAPI REST API.

## Features

- **Continuous Data Processing**: Automatically exports and processes new sensor data at configurable intervals
- **Machine Learning Predictions**: Uses PyTorch models for weather forecasting
- **REST API**: FastAPI-based API for accessing predictions and data
- **Database Integration**: MySQL database for storing sensor data and predictions
- **Scheduled Processing**: APScheduler for background task management
- **Real-time Statistics**: API endpoints for system monitoring

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

Place your trained PyTorch model as `model.pth` in the project root. If no model is provided, a default model will be created.

## Running the Application

### Start the API Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### API Documentation

Once running, you can access:
- **Interactive API docs**: `http://localhost:8000/docs`
- **ReDoc documentation**: `http://localhost:8000/redoc`

## API Endpoints

### Core Endpoints

- `GET /` - API information and available endpoints
- `GET /health` - Health check
- `GET /stats` - System statistics

### Data Endpoints

- `GET /latest-data?hours=24` - Get latest sensor data
- `POST /export-data` - Manually trigger data export

### Prediction Endpoints

- `GET /predictions?hours=24` - Get latest predictions from database
- `POST /forecast` - Generate new weather forecast

### Example API Usage

```bash
# Get system health
curl http://localhost:8000/health

# Get latest predictions
curl http://localhost:8000/predictions?hours=48

# Generate new forecast
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"hours_ahead": 24}'

# Get system statistics
curl http://localhost:8000/stats
```

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