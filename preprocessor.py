import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.last_export_time = None
    
    def preprocess_data(self, df):
        """Preprocess the data for model input"""
        if df.empty:
            return df
        
        try:
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Set timestamp as index
            df = df.set_index('timestamp')
            
            # Separate numeric and non-numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
            
            # Resample numeric columns with mean (10-minute periods)
            if len(numeric_columns) > 0:
                df_numeric = df[numeric_columns].resample('10min').mean()  # 10min = 10 minutes
            else:
                df_numeric = pd.DataFrame(index=df.resample('10min').mean().index)
            
            logger.info(f"Resampled numeric columns: {list(numeric_columns)}")
            logger.info(f"Non-numeric columns found: {list(non_numeric_columns)}")
            # log first row
            if not df_numeric.empty:
                logger.info(f"First row after resampling: {df_numeric.iloc[0].to_dict()}")
            # For non-numeric columns, we can either drop them or handle them differently
            # For now, let's drop them since they're not needed for weather prediction
            if len(non_numeric_columns) > 0:
                logger.info(f"Dropping non-numeric columns for resampling: {list(non_numeric_columns)}")
            
            # Use only numeric columns for the final dataframe
            df = df_numeric
            
            # Clean outliers before interpolation
            # Define realistic ranges for weather data at 3000m altitude
            outlier_ranges = {
                'temperature': (-40, 40),  # Celsius - extreme but possible range
                'humidity': (0, 100),      # Percentage - physical limits
                'pressure': (650, 750)     # hPa - realistic range at 3000m altitude
            }
            
            outliers_removed = {}
            for column in outlier_ranges:
                if column in df.columns:
                    min_val, max_val = outlier_ranges[column]
                    original_count = len(df[column].dropna())
                    
                    # Mark outliers as NaN
                    outlier_mask = (df[column] < min_val) | (df[column] > max_val)
                    outlier_count = outlier_mask.sum()
                    
                    if outlier_count > 0:
                        logger.warning(f"{column}: Found {outlier_count} outliers outside range [{min_val}, {max_val}]")
                        # Log some example outliers
                        outlier_values = df[column][outlier_mask].head(10).tolist()
                        logger.warning(f"  Example outlier values: {outlier_values}")
                        
                        # Remove outliers
                        df[column] = df[column].where(~outlier_mask)
                        outliers_removed[column] = outlier_count
                    
                    # Additional check for sudden jumps (rate of change)
                    if len(df[column].dropna()) > 1:
                        diff = df[column].diff().abs()
                        
                        # Define maximum reasonable change per 10-minute period
                        max_change = {
                            'temperature': 5.0,   # 5°C in 10 minutes is very extreme
                            'humidity': 20.0,     # 20% in 10 minutes is unusual but possible
                            'pressure': 2.0       # 2 hPa in 10 minutes is very extreme
                        }
                        
                        if column in max_change:
                            sudden_change_mask = diff > max_change[column]
                            sudden_change_count = sudden_change_mask.sum()
                            
                            if sudden_change_count > 0:
                                logger.warning(f"{column}: Found {sudden_change_count} sudden changes >{max_change[column]} units/10min")
                                # Remove values with sudden changes
                                df[column] = df[column].where(~sudden_change_mask)
                                if column not in outliers_removed:
                                    outliers_removed[column] = 0
                                outliers_removed[column] += sudden_change_count
            
            # Log summary of outlier removal
            total_outliers = sum(outliers_removed.values())
            if total_outliers > 0:
                logger.info(f"Total outliers removed: {total_outliers}")
                for column, count in outliers_removed.items():
                    logger.info(f"  {column}: {count} outliers removed")
            else:
                logger.info("No outliers detected in the data")
            
            # Analyze data availability by day before interpolation
            df_copy = df.copy()
            df_copy['date'] = df_copy.index.date
            daily_counts = df_copy.groupby('date').size()
            total_days = len(daily_counts)
            
            # Expected measurements per day (24 hours * 6 measurements per hour = 144)
            expected_per_day = 144
            days_with_good_data = (daily_counts >= expected_per_day * 0.5).sum()  # At least 50% of expected data
            
            logger.info(f"Data availability analysis:")
            logger.info(f"Total days: {total_days}")
            logger.info(f"Days with good data (>50% coverage): {days_with_good_data}")
            logger.info(f"Data coverage: {days_with_good_data/total_days*100:.1f}%")
            
            # Log daily data counts for days with poor coverage
            poor_days = daily_counts[daily_counts < expected_per_day * 0.5]
            if len(poor_days) > 0:
                logger.warning(f"Days with poor data coverage (<50%): {len(poor_days)}")
                for date, count in poor_days.items():
                    logger.warning(f"  {date}: {count} measurements ({count/expected_per_day*100:.1f}% coverage)")
            
            # Interpolate missing values, but limit interpolation to reasonable gaps
            # Only interpolate gaps up to 2 hours (12 measurements at 10-min intervals)
            max_gap_minutes = 120  # 2 hours
            max_gap_periods = max_gap_minutes // 10  # Convert to 10-minute periods
            
            for column in df.select_dtypes(include=[np.number]).columns:
                if column in df.columns:
                    # Find gaps larger than max_gap_periods
                    null_series = df[column].isnull()
                    gap_starts = null_series & ~null_series.shift(1, fill_value=False)
                    gap_ends = null_series & ~null_series.shift(-1, fill_value=False)
                    
                    # Count consecutive nulls for each gap
                    large_gaps = 0
                    current_gap_size = 0
                    
                    for i, is_null in enumerate(null_series):
                        if is_null:
                            current_gap_size += 1
                        else:
                            if current_gap_size > max_gap_periods:
                                large_gaps += 1
                            current_gap_size = 0
                    
                    # Check final gap
                    if current_gap_size > max_gap_periods:
                        large_gaps += 1
                    
                    if large_gaps > 0:
                        logger.warning(f"{column}: Found {large_gaps} gaps larger than {max_gap_minutes} minutes")
                    
                    # Interpolate only small gaps
                    df[column] = df[column].interpolate(method='linear', limit=max_gap_periods)
                    
                    remaining_nulls = df[column].isnull().sum()
                    if remaining_nulls > 0:
                        logger.warning(f"{column}: {remaining_nulls} values remain null after limited interpolation")
            
            logger.info(f"Interpolation completed with gap limit of {max_gap_minutes} minutes")
            
            logger.info(f"Preprocessed data shape: {df.shape}")
            # Remove rows with any NaN or infinite values
            before_drop = len(df)
            df = df[~df.isnull().any(axis=1)]
            df = df[~np.isinf(df).any(axis=1)]
            after_drop = len(df)
            dropped = before_drop - after_drop
            if dropped > 0:
                logger.warning(f"Dropped {dropped} rows with NaN or infinite values after preprocessing")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return pd.DataFrame()
    
    def update_export_time(self, df):
        """Update the last export time based on the latest timestamp in the dataframe"""
        if not df.empty and 'timestamp' in df.columns:
            self.last_export_time = df['timestamp'].max()
            logger.info(f"Updated last export time to: {self.last_export_time}")
    
    def get_last_export_time(self):
        """Get the last export time"""
        return self.last_export_time
    
    def set_last_export_time(self, time):
        """Set the last export time"""
        self.last_export_time = time
        logger.info(f"Set last export time to: {self.last_export_time}")
    
    def reset_export_time(self):
        """Reset the last export time to force a full data export"""
        self.last_export_time = None
        logger.info("Reset last export time - next export will include all available data")
    
    def get_default_export_time(self, hours_back=24):
        """Get a default export time for first run"""
        return datetime.now() - timedelta(hours=hours_back) 