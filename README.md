# Weather Forecasting API

A simple weather forecasting API that automatically generates predictions every 10 minutes using machine learning.

## Features

- **Automated Forecasting**: Predictions generated every 10 minutes via scheduled background task
- **Simple API**: Clean, minimal endpoints for predictions and statistics
- **Real-time Stats**: Web-based dashboard for monitoring system status
- **ML-powered**: Neural network model for temperature predictions
- **Database Integration**: MySQL databases for sensor data and predictions

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   cp config.env.example config.env
   # Edit config.env with your database settings
   ```

3. **Run the application**:
   ```bash
   python main.py
   ```

## API Endpoints

### Core Endpoints

- `GET /` - API information and available endpoints
- `GET /predictions?hours=24` - Get latest predictions from database
- `GET /next-forecast` - Get the most recent prediction (next forecast)
- `GET /stats` - System statistics (JSON)
- `GET /stats-page` - Web-based statistics dashboard

### Example Usage

```bash
# Get latest predictions for the last 24 hours
curl http://localhost:8000/predictions?hours=24

# Get the next forecast (most recent prediction)
curl http://localhost:8000/next-forecast

# Get system statistics
curl http://localhost:8000/stats

# View web dashboard
# Open http://localhost:8000/stats-page in your browser
```

## System Architecture

### Data Flow

```
Sensor Data (Source MySQL DB) → Scheduled Export (every 10 min) → ML Model → Predictions → API MySQL DB
```

### Components

- **Scheduler**: Runs every 10 minutes to generate new predictions
- **Data Processor**: Handles data export, preprocessing, and database operations
- **Model Predictor**: ML model for temperature forecasting
- **API**: FastAPI server with minimal endpoints
- **Stats Dashboard**: Real-time web interface for monitoring

## Configuration

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
| `PREDICTION_INTERVAL_MINUTES` | 10     | Minutes between prediction runs                  |

## Database Schema

### Source Database (Sensor Data)
```sql
CREATE TABLE timeseries (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME,
    temperature FLOAT
);
```

### API Database (Predictions)
```sql
CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME,
    predicted_temperature FLOAT,
    confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Model Training

The model is automatically trained on startup using available historical data. The training process:

1. Exports all available sensor data
2. Preprocesses data (resampling, feature engineering)
3. Trains a neural network model
4. Saves the trained model for future predictions

## Monitoring

### Web Dashboard
Visit `http://localhost:8000/stats-page` for a real-time dashboard showing:
- Database statistics
- Model status
- Scheduler information
- System health

### API Statistics
Use `GET /stats` for programmatic access to system statistics.

## Development

### Running in Development Mode
```bash
# Enable auto-reload
export API_RELOAD=true
python main.py
```

### Testing
```bash
# Test model loading
python test_model_loading.py

# Test training
python test_training.py
```

## Troubleshooting

### Common Issues

1. **Database Connection Errors**: Check your database configuration in `config.env`
2. **No Predictions**: Ensure the model is trained and sensor data is available
3. **Scheduler Not Running**: Check logs for scheduler startup messages

### Logs
The application logs important events including:
- Scheduler start/stop
- Data export operations
- Model training progress
- Prediction generation
- Database operations

## License

This project is licensed under the MIT License. 