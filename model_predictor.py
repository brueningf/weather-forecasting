import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from config import Config
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class SimpleWeatherModel(nn.Module):
    """Simple neural network for weather prediction"""
    def __init__(self, input_size=10, hidden_size=64, output_size=1):
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
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if self.config.MODEL_PATH and os.path.exists(self.config.MODEL_PATH):
                # Add all necessary classes to safe globals for PyTorch 2.6+ compatibility
                torch.serialization.add_safe_globals([
                    SimpleWeatherModel,
                    torch.nn.modules.linear.Linear,
                    torch.nn.modules.activation.ReLU,
                    torch.nn.modules.dropout.Dropout
                ])
                
                # Try loading with weights_only=False for backward compatibility
                try:
                    checkpoint = torch.load(self.config.MODEL_PATH, map_location=self.device, weights_only=False)
                except Exception as e:
                    logger.warning(f"Failed to load with weights_only=False: {e}")
                    # Fallback to weights_only=True with proper safe globals
                    checkpoint = torch.load(self.config.MODEL_PATH, map_location=self.device, weights_only=True)
                
                self.model = checkpoint['model']
                self.scaler = checkpoint['scaler']
                self.is_trained = checkpoint.get('is_trained', True)
                logger.info(f"Model loaded from {self.config.MODEL_PATH}")
            else:
                # Create a default model if no trained model exists
                self.model = SimpleWeatherModel()
                self.is_trained = False
                logger.info("Created default model (no trained model found)")
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Create a default model as fallback
            self.model = SimpleWeatherModel()
            self.is_trained = False
            self.model.eval()
    
    def save_model(self):
        """Save the trained model"""
        try:
            if self.model is not None:
                checkpoint = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'is_trained': self.is_trained,
                    'timestamp': datetime.now()
                }
                torch.save(checkpoint, self.config.MODEL_PATH)
                logger.info(f"Model saved to {self.config.MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def prepare_training_data(self, df):
        """Prepare training data from dataframe"""
        if df.empty or len(df) < 10:
            logger.warning("Insufficient data for training")
            return None, None
        
        try:
            # Create enhanced temporal features
            df_enhanced = df.copy()
            
            # Basic temporal features
            df_enhanced['hour'] = df_enhanced.index.hour
            df_enhanced['day_of_week'] = df_enhanced.index.dayofweek
            df_enhanced['month'] = df_enhanced.index.month
            
            # Enhanced temporal features for better pattern recognition
            # Sinusoidal encoding of hour (captures cyclical nature of day)
            df_enhanced['hour_sin'] = np.sin(2 * np.pi * df_enhanced['hour'] / 24)
            df_enhanced['hour_cos'] = np.cos(2 * np.pi * df_enhanced['hour'] / 24)
            
            # Sinusoidal encoding of month (captures cyclical nature of year)
            df_enhanced['month_sin'] = np.sin(2 * np.pi * df_enhanced['month'] / 12)
            df_enhanced['month_cos'] = np.cos(2 * np.pi * df_enhanced['month'] / 12)
            
            # Day of week encoding
            df_enhanced['day_sin'] = np.sin(2 * np.pi * df_enhanced['day_of_week'] / 7)
            df_enhanced['day_cos'] = np.cos(2 * np.pi * df_enhanced['day_of_week'] / 7)
            
            # Time-based features
            df_enhanced['is_night'] = ((df_enhanced['hour'] >= 22) | (df_enhanced['hour'] <= 6)).astype(int)
            df_enhanced['is_day'] = ((df_enhanced['hour'] >= 6) & (df_enhanced['hour'] <= 18)).astype(int)
            df_enhanced['is_peak_hours'] = ((df_enhanced['hour'] >= 12) & (df_enhanced['hour'] <= 16)).astype(int)
            
            # Select features for training (prioritize enhanced temporal features)
            features = [
                'temperature', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'day_sin', 'day_cos', 'is_night', 'is_day', 'is_peak_hours'
            ]
            
            available_features = [f for f in features if f in df_enhanced.columns]
            
            if len(available_features) < 3:
                logger.warning("Insufficient features for training")
                return None, None
            
            # Create feature matrix
            X = df_enhanced[available_features].values
            
            # Handle missing values
            X = np.nan_to_num(X, nan=0.0)
            
            # Create target (next hour's temperature)
            y = df_enhanced['temperature'].values[1:]  # Shift by 1 hour
            X = X[:-1]  # Remove last row since we don't have next temperature
            
            if len(X) < 10:
                logger.warning("Insufficient data after creating targets")
                return None, None
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            return X_scaled, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None
    
    def train_model(self, df, epochs=100, batch_size=32, learning_rate=0.001):
        """Train the model on the provided data"""
        try:
            # Prepare training data
            X, y = self.prepare_training_data(df)
            if X is None or y is None:
                logger.error("Could not prepare training data")
                return False
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            
            # Initialize model
            input_size = X_train.shape[1]
            self.model = SimpleWeatherModel(input_size=input_size).to(self.device)
            
            # Setup training
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            
            # Training loop
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            logger.info(f"Starting training with {len(X_train)} samples, {epochs} epochs")
            
            for epoch in range(epochs):
                # Training
                self.model.train()
                optimizer.zero_grad()
                outputs = self.model(X_train_tensor)
                loss = criterion(outputs.squeeze(), y_train_tensor)
                loss.backward()
                optimizer.step()
                
                # Validation
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs.squeeze(), y_val_tensor)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 20 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
            
            self.is_trained = True
            self.save_model()
            
            logger.info(f"Training completed. Final validation loss: {best_val_loss:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def prepare_input_tensor(self, df):
        """Prepare input tensor from dataframe"""
        if df.empty:
            return None
        
        try:
            # Create enhanced temporal features (same as in prepare_training_data)
            df_enhanced = df.copy()
            
            # Basic temporal features
            df_enhanced['hour'] = df_enhanced.index.hour
            df_enhanced['day_of_week'] = df_enhanced.index.dayofweek
            df_enhanced['month'] = df_enhanced.index.month
            
            # Enhanced temporal features for better pattern recognition
            # Sinusoidal encoding of hour (captures cyclical nature of day)
            df_enhanced['hour_sin'] = np.sin(2 * np.pi * df_enhanced['hour'] / 24)
            df_enhanced['hour_cos'] = np.cos(2 * np.pi * df_enhanced['hour'] / 24)
            
            # Sinusoidal encoding of month (captures cyclical nature of year)
            df_enhanced['month_sin'] = np.sin(2 * np.pi * df_enhanced['month'] / 12)
            df_enhanced['month_cos'] = np.cos(2 * np.pi * df_enhanced['month'] / 12)
            
            # Day of week encoding
            df_enhanced['day_sin'] = np.sin(2 * np.pi * df_enhanced['day_of_week'] / 7)
            df_enhanced['day_cos'] = np.cos(2 * np.pi * df_enhanced['day_of_week'] / 7)
            
            # Time-based features
            df_enhanced['is_night'] = ((df_enhanced['hour'] >= 22) | (df_enhanced['hour'] <= 6)).astype(int)
            df_enhanced['is_day'] = ((df_enhanced['hour'] >= 6) & (df_enhanced['hour'] <= 18)).astype(int)
            df_enhanced['is_peak_hours'] = ((df_enhanced['hour'] >= 12) & (df_enhanced['hour'] <= 16)).astype(int)
            
            # Select features for prediction (same as training)
            features = [
                'temperature', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                'day_sin', 'day_cos', 'is_night', 'is_day', 'is_peak_hours'
            ]
            
            available_features = [f for f in features if f in df_enhanced.columns]
            
            if not available_features:
                logger.warning("No suitable features found for prediction")
                return None
            
            # Create feature matrix
            X = df_enhanced[available_features].values
            
            # Handle missing values
            X = np.nan_to_num(X, nan=0.0)
            
            # Scale features using the fitted scaler
            if self.is_trained and hasattr(self.scaler, 'mean_'):
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            return X_tensor
            
        except Exception as e:
            logger.error(f"Error preparing input tensor: {e}")
            return None
    
    def needs_training(self):
        """Check if the model needs to be trained"""
        # Check if model exists and has been properly loaded
        if self.model is None:
            return True
        
        # Check if model is trained
        if not self.is_trained:
            return True
        
        # Additional check: verify the model has been properly initialized with weights
        try:
            # Check if model has parameters and they're not all zeros
            has_weights = False
            for param in self.model.parameters():
                if param.data.abs().sum().item() > 0:
                    has_weights = True
                    break
            
            if not has_weights:
                logger.warning("Model exists but has no trained weights")
                return True
                
        except Exception as e:
            logger.warning(f"Error checking model weights: {e}")
            return True
        
        return False
    
    def predict(self, df, forecast_hours=24):
        """Make predictions on the data"""
        if df.empty:
            logger.warning("Empty dataframe provided for prediction")
            return pd.DataFrame()
        
        if self.needs_training():
            logger.warning("Model is not trained. Cannot make predictions.")
            return pd.DataFrame()
        
        try:
            # For forecasting, we need to create future timestamps with proper features
            last_timestamp = df.index[-1] if not df.empty else datetime.now()
            future_timestamps = pd.date_range(
                start=last_timestamp + timedelta(hours=1),
                periods=forecast_hours,
                freq='H'
            )
            
            # Create future data with proper temporal features
            future_data = pd.DataFrame(index=future_timestamps)
            future_data['hour'] = future_data.index.hour
            future_data['day_of_week'] = future_data.index.dayofweek
            future_data['month'] = future_data.index.month
            
            # Get the last known temperature as starting point
            last_temp = df['temperature'].iloc[-1] if 'temperature' in df.columns else 20.0
            
            # Create a more sophisticated temperature initialization based on time patterns
            temp_variations = []
            for i, timestamp in enumerate(future_timestamps):
                # Base diurnal pattern: cooler at night (2-6 AM), warmer during day (12-4 PM)
                hour_rad = 2 * np.pi * timestamp.hour / 24
                diurnal_variation = 3.0 * np.sin(hour_rad - np.pi/2)  # Peak at 2 PM, trough at 2 AM
                
                # Seasonal pattern: cooler in winter, warmer in summer
                month_rad = 2 * np.pi * (timestamp.month - 1) / 12
                seasonal_variation = 5.0 * np.sin(month_rad - np.pi/2)  # Peak in July, trough in January
                
                # Gradual trend: slight cooling over the forecast period
                trend = -0.05 * i
                
                # Add small random component for realism
                noise = np.random.normal(0, 0.3)
                
                temp_variations.append(diurnal_variation + seasonal_variation + trend + noise)
            
            # Initialize future temperatures with the pattern
            future_data['temperature'] = last_temp + np.array(temp_variations)
            
            # Prepare input tensor for the future data
            X_tensor = self.prepare_input_tensor(future_data)
            if X_tensor is None:
                return pd.DataFrame()
            
            # Make predictions
            with torch.no_grad():
                predictions = self.model(X_tensor)
                predictions = predictions.cpu().numpy().flatten()
            
            # Create predictions dataframe
            if len(predictions) > 0:
                predictions_df = pd.DataFrame({
                    'timestamp': future_timestamps,
                    'predicted_temperature': predictions,
                    'confidence': 0.8  # Default confidence
                })
                
                predictions_df.set_index('timestamp', inplace=True)
                
                logger.info(f"Generated {len(predictions_df)} predictions for future timestamps")
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
            
            if self.needs_training():
                logger.warning("Model is not trained. Cannot make predictions.")
                return pd.DataFrame()
            
            # Use the main predict method which now handles temporal predictions properly
            predictions_df = self.predict(current_data, forecast_hours=hours_ahead)
            
            return predictions_df
            
        except Exception as e:
            logger.error(f"Error predicting future: {e}")
            return pd.DataFrame() 