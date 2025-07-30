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
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

logger = logging.getLogger(__name__)

class LSTMWeatherModel(nn.Module):
    """LSTM-based neural network for weather prediction"""
    def __init__(self, input_size=7, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMWeatherModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])  # Take last time step
        return out

class ModelPredictor:
    def __init__(self, sequence_length=12):
        self.config = Config()
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_trained = False
        self.sequence_length = sequence_length  # e.g., use last 12 steps (2 hours if 10min freq)
        self.load_model()

    def _add_time_features(self, df):
        """Add time-based features to the dataframe."""
        df_enhanced = df.copy()
        df_enhanced['hour'] = df_enhanced.index.hour
        df_enhanced['month'] = df_enhanced.index.month
        df_enhanced['hour_sin'] = np.sin(2 * np.pi * df_enhanced['hour'] / 24)
        df_enhanced['hour_cos'] = np.cos(2 * np.pi * df_enhanced['hour'] / 24)
        df_enhanced['month_sin'] = np.sin(2 * np.pi * df_enhanced['month'] / 12)
        df_enhanced['month_cos'] = np.cos(2 * np.pi * df_enhanced['month'] / 12)
        return df_enhanced

    def calculate_confidence(self, input_tensor, prediction, history_df):
        """
        Calculate prediction confidence based on multiple factors:
        1. Model uncertainty (using dropout during inference)
        2. Data quality (recentness and consistency)
        3. Forecast horizon (confidence decreases with time)
        """
        try:
            confidence = 0.8  # Base confidence
            
            # 1. Model uncertainty using dropout
            self.model.train()  # Enable dropout for uncertainty estimation
            predictions = []
            for _ in range(10):  # Multiple forward passes with dropout
                with torch.no_grad():
                    pred = self.model(input_tensor)
                    predictions.append(pred.cpu().numpy().flatten()[0])
            
            # Calculate standard deviation of predictions
            pred_std = np.std(predictions)
            uncertainty_factor = max(0.1, min(1.0, 1.0 - pred_std / 5.0))  # Normalize to 0-1
            
            # 2. Data quality assessment
            if not history_df.empty:
                # Check data recency
                latest_time = history_df.index[-1]
                now = datetime.now()
                time_diff = (now - latest_time).total_seconds() / 3600  # hours
                recency_factor = max(0.1, 1.0 - time_diff / 24.0)  # Decrease confidence if data is old
                
                # Check data consistency (variance in recent values)
                recent_values = history_df['temperature'].tail(6).values  # Last hour
                if len(recent_values) > 1:
                    temp_variance = np.var(recent_values)
                    consistency_factor = max(0.1, 1.0 - temp_variance / 100.0)  # Normalize variance
                else:
                    consistency_factor = 0.5
            else:
                recency_factor = 0.5
                consistency_factor = 0.5
            
            # 3. Forecast horizon factor (confidence decreases with time)
            # This will be passed as a parameter from the calling function
            
            # Combine factors
            confidence = (uncertainty_factor * 0.4 + 
                        recency_factor * 0.3 + 
                        consistency_factor * 0.3)
            
            # Ensure confidence is between 0.1 and 0.95
            confidence = max(0.1, min(0.95, confidence))
            
            return confidence
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.5  # Fallback confidence

    def load_model(self):
        """Load the trained model"""
        try:
            if self.config.MODEL_PATH and os.path.exists(self.config.MODEL_PATH):
                torch.serialization.add_safe_globals([
                    LSTMWeatherModel,
                    torch.nn.modules.linear.Linear,
                    torch.nn.modules.rnn.LSTM,
                    torch.nn.modules.dropout.Dropout
                ])
                try:
                    checkpoint = torch.load(self.config.MODEL_PATH, map_location=self.device, weights_only=False)
                except Exception as e:
                    logger.warning(f"Failed to load with weights_only=False: {e}")
                    checkpoint = torch.load(self.config.MODEL_PATH, map_location=self.device, weights_only=True)
                self.model = checkpoint['model']
                self.scaler = checkpoint['scaler']
                self.is_trained = checkpoint.get('is_trained', True)
                self.sequence_length = checkpoint.get('sequence_length', 12)
                logger.info(f"Model loaded from {self.config.MODEL_PATH}")
            else:
                self.model = LSTMWeatherModel()
                self.is_trained = False
                logger.info("Created default LSTM model (no trained model found)")
            self.model.eval()
            # Load metrics from disk if available
            self._latest_metrics = None
            if os.path.exists(self.config.METRICS_PATH):
                try:
                    with open(self.config.METRICS_PATH, 'r') as f:
                        self._latest_metrics = json.load(f)
                    logger.info(f"Loaded metrics from {self.config.METRICS_PATH}")
                except Exception as e:
                    logger.warning(f"Could not load metrics from {self.config.METRICS_PATH}: {e}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = LSTMWeatherModel()
            self.is_trained = False
            self.model.eval()
            self._latest_metrics = None

    def save_model(self):
        """Save the trained model"""
        try:
            if self.model is not None:
                checkpoint = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'is_trained': self.is_trained,
                    'timestamp': datetime.now(),
                    'sequence_length': self.sequence_length
                }
                torch.save(checkpoint, self.config.MODEL_PATH)
                logger.info(f"Model saved to {self.config.MODEL_PATH}")
                # Save metrics to disk if available
                if hasattr(self, '_latest_metrics') and self._latest_metrics is not None:
                    try:
                        with open(self.config.METRICS_PATH, 'w') as f:
                            json.dump(self._latest_metrics, f)
                        logger.info(f"Metrics saved to {self.config.METRICS_PATH}")
                    except Exception as e:
                        logger.warning(f"Could not save metrics to {self.config.METRICS_PATH}: {e}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def prepare_training_data(self, df, fit_scaler=True, min_sequences=10):
        """Prepare training data for LSTM from dataframe.
        
        This method transforms raw weather data into sequences suitable for LSTM training.
        It performs the following steps:
        1. Validates data sufficiency (requires at least sequence_length + 1 samples)
        2. Validates required columns exist in the dataframe
        3. Enhances data with time-based features (hour, month cyclical encodings)
        4. Extracts available features (temperature, humidity, pressure, time features)
        5. Handles missing values using forward fill then backward fill
        6. Scales features using StandardScaler (fits only if fit_scaler=True)
        7. Creates sliding window sequences of length sequence_length for input (X)
        8. Creates corresponding target values (y) for temperature prediction
        9. Validates final sequence count (requires at least min_sequences)
        
        Args:
            df: DataFrame containing weather data with columns like temperature, humidity, pressure
            fit_scaler: Whether to fit the scaler (True for training, False for inference)
            min_sequences: Minimum number of sequences required for training
            
        Returns:
            tuple: (X_seq, y_seq) where X_seq is 3D array (samples, sequence_length, features)
                   and y_seq is 1D array of target temperatures, or (None, None) if insufficient data
        """
        # Validate input data
        if df.empty:
            logger.warning("Empty dataframe provided")
            return None, None
            
        if len(df) < self.sequence_length + 1:
            logger.warning(f"Insufficient data: {len(df)} samples, need at least {self.sequence_length + 1}")
            return None, None
            
        # Validate required columns
        required_columns = ['temperature']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return None, None
            
        try:
            # Add time features
            df_enhanced = self._add_time_features(df)
            
            # Define feature columns in order of preference
            feature_columns = [
                'temperature', 'humidity', 'pressure',
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
            ]
            
            # Select available features, maintaining order
            available_features = [f for f in feature_columns if f in df_enhanced.columns]
            
            if not available_features:
                logger.error("No features available for training")
                return None, None
                
            logger.info(f"Using features: {available_features}")
            
            # Extract features and target
            X = df_enhanced[available_features].values
            y = df_enhanced['temperature'].values
            
            # Handle missing values more intelligently
            if np.isnan(X).any():
                logger.info("Handling missing values in features")
                # Forward fill then backward fill for time series data
                X = pd.DataFrame(X, columns=available_features).fillna(method='ffill').fillna(method='bfill').values
                
            if np.isnan(y).any():
                logger.info("Handling missing values in target")
                y = pd.Series(y).fillna(method='ffill').fillna(method='bfill').values
            
            # Scale features
            if fit_scaler:
                X_scaled = self.scaler.fit_transform(X)
                logger.info("Fitted scaler on training data")
            else:
                X_scaled = self.scaler.transform(X)
                logger.info("Applied existing scaler to data")
            
            # Create sequences more efficiently
            n_samples = len(X_scaled) - self.sequence_length
            if n_samples <= 0:
                logger.warning("No samples available after sequence creation")
                return None, None
                
            # Pre-allocate arrays for better memory efficiency
            X_seq = np.zeros((n_samples, self.sequence_length, len(available_features)))
            y_seq = np.zeros(n_samples)
            
            for i in range(n_samples):
                X_seq[i] = X_scaled[i:i+self.sequence_length]
                y_seq[i] = y[i+self.sequence_length]
            
            if len(X_seq) < min_sequences:
                logger.warning(f"Insufficient sequences: {len(X_seq)}, need at least {min_sequences}")
                return None, None
                
            logger.info(f"Created {len(X_seq)} sequences with {len(available_features)} features")
            return X_seq, y_seq
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None

    def compute_metrics(self, X, y_true):
        """Compute MAE, RMSE, and R2 for the model on given data."""
        import torch
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            preds = self.model(X_tensor).cpu().numpy().flatten()
        mae = mean_absolute_error(y_true, preds)
        rmse = root_mean_squared_error(y_true, preds)
        r2 = r2_score(y_true, preds)
        return {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}

    def get_latest_metrics(self):
        """Return the latest benchmark metrics if available."""
        return getattr(self, '_latest_metrics', None)

    def train_model(self, df, epochs=100, batch_size=32, learning_rate=0.001):
        """Train the LSTM model on the provided data"""
        try:
            if df.isnull().any().any():
                logger.error("Training data contains NaN values! Please check preprocessing.")
                return False

            X, y = self.prepare_training_data(df, fit_scaler=True)
            if X is None or y is None:
                logger.error("Could not prepare training data")
                return False
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)
            input_size = X_train.shape[2]
            self.model = LSTMWeatherModel(input_size=input_size).to(self.device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0

            logger.info(f"Starting LSTM training with {len(X_train)} samples, {epochs} epochs")

            for epoch in range(epochs):
                self.model.train()
                permutation = torch.randperm(X_train_tensor.size(0))

                for i in range(0, X_train_tensor.size(0), batch_size):
                    indices = permutation[i:i+batch_size]
                    batch_X, batch_y = X_train_tensor[indices], y_train_tensor[indices]
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()

                self.model.eval()

                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs.squeeze(), y_val_tensor)

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

            # Compute metrics on validation set
            metrics = self.compute_metrics(X_val, y_val)
            self._latest_metrics = metrics
            self.save_model()  # This will now also save metrics
            logger.info(f"Training completed. Final validation loss: {best_val_loss:.4f}")
            logger.info(f"Validation metrics: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}, R2={metrics['r2']:.3f}")
            return True
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

    def prepare_input_tensor(self, df):
        """Prepare input tensor for LSTM from dataframe (last sequence_length rows).
        
        This method prepares a single input sequence for prediction by:
        1. Validating data sufficiency (requires at least sequence_length rows)
        2. Adding time-based features (hour, month cyclical encodings)
        3. Extracting available features in the same order as training
        4. Handling missing values using forward fill then backward fill
        5. Scaling features using the fitted scaler (if available)
        6. Creating a single sequence tensor for prediction
        
        Args:
            df: DataFrame containing weather data with columns like temperature, humidity, pressure
            
        Returns:
            torch.Tensor: Input tensor of shape (1, sequence_length, features) or None if insufficient data
        """
        # Validate input data
        if df.empty:
            logger.warning("Empty dataframe provided for input tensor")
            return None
            
        if len(df) < self.sequence_length:
            logger.warning(f"Insufficient data for input tensor: {len(df)} samples, need at least {self.sequence_length}")
            return None
            
        # Validate required columns
        required_columns = ['temperature']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns for input tensor: {missing_columns}")
            return None
            
        try:
            # Add time features
            df_enhanced = self._add_time_features(df)
            
            # Define feature columns in same order as training
            feature_columns = [
                'temperature', 'humidity', 'pressure',
                'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
            ]
            
            # Select available features, maintaining order
            available_features = [f for f in feature_columns if f in df_enhanced.columns]
            
            if not available_features:
                logger.error("No features available for input tensor")
                return None
                
            # Extract features
            X = df_enhanced[available_features].values
            
            # Handle missing values more intelligently
            if np.isnan(X).any():
                logger.info("Handling missing values in input tensor features")
                # Forward fill then backward fill for time series data
                X = pd.DataFrame(X, columns=available_features).fillna(method='ffill').fillna(method='bfill').values
            
            # Scale features if scaler is fitted
            if self.is_trained and hasattr(self.scaler, 'mean_'):
                X_scaled = self.scaler.transform(X)
                logger.debug("Applied fitted scaler to input tensor")
            else:
                X_scaled = X
                logger.warning("Using unscaled features for input tensor (scaler not fitted)")
            
            # Take only the last sequence_length rows
            if len(X_scaled) < self.sequence_length:
                logger.warning(f"Not enough data after scaling: {len(X_scaled)} rows, need {self.sequence_length}")
                return None
                
            X_seq = X_scaled[-self.sequence_length:]
            X_seq = np.expand_dims(X_seq, axis=0)  # (1, seq_len, input_size)
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            
            logger.debug(f"Prepared input tensor with shape {X_tensor.shape}")
            return X_tensor
            
        except Exception as e:
            logger.error(f"Error preparing input tensor: {e}")
            return None

    def needs_training(self):
        if self.model is None:
            return True
        if not self.is_trained:
            return True
        try:
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

    def predict(self, df, forecast_periods=60):
        """Make predictions on the data using LSTM"""
        if df.empty or len(df) < self.sequence_length:
            logger.warning("Insufficient data for prediction")
            return pd.DataFrame()
        if self.needs_training():
            logger.warning("Model is not trained. Cannot make predictions.")
            return pd.DataFrame()
        try:
            freq = pd.infer_freq(df.index)
            if freq is None:
                # fallback: use median diff
                diffs = df.index.to_series().diff().dropna()
                if not diffs.empty:
                    freq = diffs.median()
                else:
                    freq = pd.Timedelta(minutes=10)
            last_timestamp = df.index[-1]
            if isinstance(freq, str):
                future_timestamps = pd.date_range(
                    start=last_timestamp + pd.tseries.frequencies.to_offset(freq),
                    periods=forecast_periods,
                    freq=freq
                )
            else:
                # freq is a Timedelta
                future_timestamps = [last_timestamp + (i+1)*freq for i in range(forecast_periods)]
            
            # Prepare initial sequence
            history = df.copy()
            preds = []
            confidences = []
            
            for i in range(forecast_periods):
                X_tensor = self.prepare_input_tensor(history)
                if X_tensor is None:
                    break
                
                # Calculate confidence for this prediction
                confidence = self.calculate_confidence(X_tensor, None, history)
                
                # Apply forecast horizon factor (confidence decreases with time)
                horizon_factor = max(0.1, 1.0 - (i / forecast_periods) * 0.5)  # Decrease by up to 50%
                confidence *= horizon_factor
                
                with torch.no_grad():
                    pred = self.model(X_tensor)
                    pred = pred.cpu().numpy().flatten()[0]
                
                # Post-processing: clip output
                pred = float(np.clip(pred, -50, 60))
                preds.append(pred)
                confidences.append(confidence)
                
                # Append prediction to history for next step
                next_row = history.iloc[-1].copy()
                next_row['temperature'] = pred
                # Optionally, you could also update humidity/pressure with a model or keep last value
                next_index = future_timestamps[i]
                next_row.name = next_index
                history = pd.concat([history, pd.DataFrame([next_row])])
            
            if len(preds) > 0:
                predictions_df = pd.DataFrame({
                    'timestamp': future_timestamps[:len(preds)],
                    'predicted_temperature': preds,
                    'confidence': confidences
                })
                predictions_df.set_index('timestamp', inplace=True)
                return predictions_df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return pd.DataFrame() 