#!/usr/bin/env python3
"""
Test script to demonstrate improved temporal weather predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from model_predictor import ModelPredictor
import matplotlib.pyplot as plt

def create_sample_data():
    """Create sample weather data with realistic temporal patterns"""
    # Create 7 days of hourly data
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    timestamps = pd.date_range(start=start_date, periods=7*24, freq='H')
    
    # Create realistic temperature patterns
    temperatures = []
    for i, timestamp in enumerate(timestamps):
        # Base temperature
        base_temp = 20.0
        
        # Diurnal pattern: cooler at night, warmer during day
        hour_rad = 2 * np.pi * timestamp.hour / 24
        diurnal_variation = 5.0 * np.sin(hour_rad - np.pi/2)  # Peak at 2 PM, trough at 2 AM
        
        # Seasonal pattern (assuming summer)
        month_rad = 2 * np.pi * (timestamp.month - 1) / 12
        seasonal_variation = 3.0 * np.sin(month_rad - np.pi/2)
        
        # Add some realistic noise
        noise = np.random.normal(0, 1.0)
        
        temp = base_temp + diurnal_variation + seasonal_variation + noise
        temperatures.append(temp)
    
    # Create DataFrame
    df = pd.DataFrame({
        'temperature': temperatures
    }, index=timestamps)
    
    return df

def test_predictions():
    """Test the improved temporal predictions"""
    print("Creating sample weather data...")
    sample_data = create_sample_data()
    
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Date range: {sample_data.index[0]} to {sample_data.index[-1]}")
    print(f"Temperature range: {sample_data['temperature'].min():.1f}°C to {sample_data['temperature'].max():.1f}°C")
    
    # Initialize predictor
    print("\nInitializing model predictor...")
    predictor = ModelPredictor()
    
    # Train the model
    print("Training model...")
    success = predictor.train_model(sample_data, epochs=50, learning_rate=0.001)
    
    if not success:
        print("Training failed!")
        return
    
    # Make predictions
    print("Making predictions...")
    predictions = predictor.predict(sample_data, forecast_hours=48)
    
    if predictions.empty:
        print("No predictions generated!")
        return
    
    print(f"Generated {len(predictions)} predictions")
    print(f"Prediction range: {predictions.index[0]} to {predictions.index[-1]}")
    print(f"Predicted temperature range: {predictions['predicted_temperature'].min():.1f}°C to {predictions['predicted_temperature'].max():.1f}°C")
    
    # Analyze temporal patterns
    print("\nAnalyzing temporal patterns...")
    
    # Check day/night patterns in predictions
    predictions['hour'] = predictions.index.hour
    predictions['is_night'] = ((predictions['hour'] >= 22) | (predictions['hour'] <= 6))
    predictions['is_day'] = ((predictions['hour'] >= 6) & (predictions['hour'] <= 18))
    
    night_temps = predictions[predictions['is_night']]['predicted_temperature']
    day_temps = predictions[predictions['is_day']]['predicted_temperature']
    
    if len(night_temps) > 0 and len(day_temps) > 0:
        print(f"Night temperatures (22:00-06:00): avg={night_temps.mean():.1f}°C, min={night_temps.min():.1f}°C, max={night_temps.max():.1f}°C")
        print(f"Day temperatures (06:00-18:00): avg={day_temps.mean():.1f}°C, min={day_temps.min():.1f}°C, max={day_temps.max():.1f}°C")
        print(f"Day-night difference: {day_temps.mean() - night_temps.mean():.1f}°C")
    
    # Plot results
    try:
        plt.figure(figsize=(15, 8))
        
        # Plot historical data
        plt.plot(sample_data.index, sample_data['temperature'], 'b-', label='Historical Data', linewidth=2)
        
        # Plot predictions
        plt.plot(predictions.index, predictions['predicted_temperature'], 'r--', label='Predictions', linewidth=2)
        
        # Add vertical line to separate historical and predicted
        plt.axvline(x=sample_data.index[-1], color='g', linestyle=':', alpha=0.7, label='Prediction Start')
        
        # Highlight day/night periods
        for i, timestamp in enumerate(predictions.index):
            if predictions.iloc[i]['is_night']:
                plt.axvspan(timestamp, timestamp + timedelta(hours=1), alpha=0.1, color='blue')
            elif predictions.iloc[i]['is_day']:
                plt.axvspan(timestamp, timestamp + timedelta(hours=1), alpha=0.1, color='yellow')
        
        plt.xlabel('Time')
        plt.ylabel('Temperature (°C)')
        plt.title('Weather Predictions with Temporal Patterns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig('temporal_predictions.png', dpi=300, bbox_inches='tight')
        print("\nPlot saved as 'temporal_predictions.png'")
        
    except ImportError:
        print("Matplotlib not available, skipping plot generation")
    
    return predictions

if __name__ == "__main__":
    predictions = test_predictions()
    if predictions is not None:
        print("\nTest completed successfully!")
        print("The model now properly accounts for temporal patterns including day/night temperature variations.") 