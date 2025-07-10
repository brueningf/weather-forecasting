import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from config import Config
import logging

logger = logging.getLogger(__name__)

class SimpleWeatherModel(nn.Module):
    """Simple neural network for weather prediction"""
    def __init__(self, input_size=4, hidden_size=64, output_size=1):
        super(SimpleWeatherModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ModelPredictor:
    def __init__(self):
        self.config = Config()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if self.config.MODEL_PATH and os.path.exists(self.config.MODEL_PATH):
                self.model = torch.load(self.config.MODEL_PATH, map_location=self.device)
                logger.info(f"Model loaded from {self.config.MODEL_PATH}")
            else:
                # Create a default model if no trained model exists
                self.model = SimpleWeatherModel()
                logger.info("Created default model (no trained model found)")
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Create a default model as fallback
            self.model = SimpleWeatherModel()
            self.model.eval()
    
    def prepare_input_tensor(self, df):
        """Prepare input tensor from dataframe"""
        if df.empty:
            return None
        
        try:
            # Select features for prediction
            features = ['temperature', 'hour', 'day_of_week', 'month']
            available_features = [f for f in features if f in df.columns]
            
            if not available_features:
                logger.warning("No suitable features found for prediction")
                return None
            
            # Create feature matrix
            X = df[available_features].values
            
            # Handle missing values
            X = np.nan_to_num(X, nan=0.0)
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            return X_tensor
            
        except Exception as e:
            logger.error(f"Error preparing input tensor: {e}")
            return None
    
    def predict(self, df, forecast_hours=24):
        """Make predictions on the data"""
        if df.empty:
            logger.warning("Empty dataframe provided for prediction")
            return pd.DataFrame()
        
        try:
            # Prepare input tensor
            X_tensor = self.prepare_input_tensor(df)
            if X_tensor is None:
                return pd.DataFrame()
            
            # Make predictions
            with torch.no_grad():
                predictions = self.model(X_tensor)
                predictions = predictions.cpu().numpy().flatten()
            
            # Create predictions dataframe
            if len(predictions) > 0:
                # Use existing timestamps if available, otherwise generate future timestamps
                if len(predictions) <= len(df):
                    timestamps = df.index[:len(predictions)]
                else:
                    # Generate future timestamps
                    last_timestamp = df.index[-1] if not df.empty else datetime.now()
                    timestamps = pd.date_range(
                        start=last_timestamp + timedelta(hours=1),
                        periods=len(predictions),
                        freq='H'
                    )
                
                predictions_df = pd.DataFrame({
                    'timestamp': timestamps,
                    'predicted_temperature': predictions,
                    'confidence': 0.8  # Default confidence
                })
                
                predictions_df.set_index('timestamp', inplace=True)
                
                logger.info(f"Generated {len(predictions_df)} predictions")
                return predictions_df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return pd.DataFrame()
    
    def predict_future(self, current_data, hours_ahead=24):
        """Predict future weather based on current data"""
        try:
            if current_data.empty:
                logger.warning("No current data provided for future prediction")
                return pd.DataFrame()
            
            # Generate future timestamps
            last_timestamp = current_data.index[-1]
            future_timestamps = pd.date_range(
                start=last_timestamp + timedelta(hours=1),
                periods=hours_ahead,
                freq='H'
            )
            
            # Create future data with extrapolated features
            future_data = pd.DataFrame(index=future_timestamps)
            future_data['hour'] = future_data.index.hour
            future_data['day_of_week'] = future_data.index.dayofweek
            future_data['month'] = future_data.index.month
            
            # Use the last known temperature as a starting point
            last_temp = current_data['temperature'].iloc[-1] if 'temperature' in current_data.columns else 20.0
            future_data['temperature'] = last_temp
            
            # Make predictions on future data
            predictions_df = self.predict(future_data, forecast_hours=hours_ahead)
            
            return predictions_df
            
        except Exception as e:
            logger.error(f"Error predicting future: {e}")
            return pd.DataFrame() 