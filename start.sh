#!/bin/bash

# Weather Forecasting API Startup Script

echo "Starting Weather Forecasting API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp config.env.example .env
    echo "Please edit .env file with your database credentials before running again."
    exit 1
fi

# Start the application
echo "Starting the API server..."
python main.py 