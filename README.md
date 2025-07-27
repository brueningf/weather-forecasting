# Weather Forecasting API

A simple weather forecasting API that automatically generates predictions every 10 minutes using machine learning.

## Features

- **48-Hour Forecasting**: Predictions generated for up to 48 hours ahead with 10-minute intervals
- **Dynamic Confidence**: Real-time confidence calculation based on model uncertainty, data quality, and forecast horizon
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
- `GET /predictions?hours=48` - Get latest predictions from database (default: 48 hours)
- `GET /next-forecast` - Get the most recent prediction (next forecast)
- `GET /stats` - System statistics (JSON)
- `GET /stats-page` - Web-based statistics dashboard

### Example Usage

```bash
# Get latest predictions for the last 48 hours (default)
curl http://localhost:8000/predictions?hours=48

# Get predictions for the last 24 hours
curl http://localhost:8000/predictions?hours=24

# Get the next forecast (most recent prediction)
curl http://localhost:8000/next-forecast

# Get system statistics
curl http://localhost:8000/stats

# View web dashboard
# Open http://localhost:8000/stats-page in your browser
```

## Prediction Features

### 48-Hour Forecast Horizon
The system now generates predictions for up to 48 hours ahead with 10-minute intervals (288 predictions total). This provides much more comprehensive forecasting capability compared to the previous 1-hour limit.

### Dynamic Confidence Calculation
Predictions now include meaningful confidence scores based on:

1. **Model Uncertainty**: Uses dropout during inference to estimate prediction variance
2. **Data Quality**: Considers data recency and consistency
3. **Forecast Horizon**: Confidence naturally decreases for predictions further in the future

Confidence values range from 0.1 to 0.95, where:
- **High (≥0.7)**: Very confident predictions
- **Medium (0.5-0.7)**: Moderately confident predictions  
- **Low (<0.5)**: Less confident predictions

## System Architecture

### Data Flow

```
Sensor Data (Source MySQL DB) → Scheduled Export (every 10 min) → ML Model → 48-Hour Predictions → API MySQL DB
```

### Components

- **Scheduler**: Runs every 10 minutes to generate new 48-hour predictions
- **Data Processor**: Handles data export, preprocessing, and database operations
- **Model Predictor**: ML model with confidence calculation for temperature forecasting
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

The system uses an LSTM neural network for temperature prediction. The model is automatically retrained when:

- New data accumulates (every 24 hours of new data)
- Model performance degrades
- System detects significant data distribution changes

### Confidence Calculation

The confidence calculation uses multiple factors:

1. **Model Uncertainty**: Multiple forward passes with dropout to estimate prediction variance
2. **Data Recency**: Confidence decreases if input data is old (>24 hours)
3. **Data Consistency**: Confidence decreases with high variance in recent temperature values
4. **Forecast Horizon**: Confidence naturally decreases for predictions further in the future

## Testing

Run the test script to verify 48-hour predictions and confidence calculation:

```bash
python test_predictions.py
```

This will test:
- 48-hour prediction generation
- Confidence calculation accuracy
- Data quality assessment
- Forecast horizon effects 