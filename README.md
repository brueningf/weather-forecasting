# End-to-End Weather Forecasting System

A complete weather forecasting system that collects local weather data using an ESP32 sensor, processes it through MQTT, and generates temperature predictions using machine learning.

## Project Overview

This project demonstrates a full-stack IoT weather forecasting system:

- **ESP32 Sensor**: Collects real-time temperature data
- **MQTT Communication**: Transmits sensor data to the server
- **Machine Learning**: Predicts future temperatures using local weather data
- **Web Dashboard**: Displays current conditions and forecasts

## System Components

### Hardware
- **ESP32 Microcontroller**: Temperature sensor with WiFi connectivity
- **MQTT Broker**: Handles communication between ESP32 and server

### Software
- **Data Collection**: ESP32 sends temperature readings every 10 minutes
- **Prediction Engine**: Machine learning model generates 48-hour forecasts
- **Web Interface**: Real-time dashboard showing current conditions and predictions

## Features

- **Real-time Data**: Live temperature readings from ESP32 sensor
- **48-Hour Forecasts**: Predictions generated every 10 minutes
- **Confidence Scores**: Each prediction includes reliability rating
- **Web Dashboard**: Monitor system status and view forecasts
- **Simple API**: Easy access to predictions and statistics

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure settings**:
   ```bash
   cp config.env.example config.env
   # Edit config.env with your database settings
   ```

3. **Run the system**:
   ```bash
   python main.py
   ```

## API Endpoints

- `GET /` - Main dashboard with all features
- `GET /api/predictions` - Get latest temperature forecasts
- `GET /api/sensor-data` - Get actual sensor readings
- `GET /api/stats` - System statistics and status
- `GET /temperature` - Temperature comparison graph

## Example Usage

```bash
# Get latest predictions
curl http://localhost:8000/api/predictions

# View web dashboard
# Open http://localhost:8000/ in your browser
```

## Technology Stack

- **Hardware**: ESP32 microcontroller with temperature sensor
- **Communication**: MQTT protocol for data transmission
- **Backend**: Python with FastAPI
- **Database**: MySQL for storing sensor data and predictions
- **Machine Learning**: LSTM neural network for temperature forecasting
- **Frontend**: HTML/CSS dashboard with real-time updates

## Data Flow

```
ESP32 Sensor → MQTT → Server → ML Model → Predictions → Web Dashboard
```

## Configuration

Key environment variables:
- Database connection settings
- MQTT broker configuration
- API server settings

## Learning Outcomes

This project demonstrates:
- IoT sensor integration with ESP32
- MQTT communication protocols
- Machine learning for time series prediction
- Full-stack web development
- Real-time data processing
- Database design and management 