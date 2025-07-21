import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from weather_data_controller import WeatherDataController

class WeatherDataAnalyzer:
    def __init__(self):
        self.controller = WeatherDataController()
        
    def load_data(self, hours_back=None, limit=None):
        """Load preprocessed data for analysis"""
        print("Loading preprocessed data...")
        df = self.controller.fetch_preprocessed_data(hours_back=hours_back, limit=limit)
        
        if df is None or df.empty:
            print("No preprocessed data found.")
            return None
            
        print(f"Loaded {len(df)} records from {df.index.min()} to {df.index.max()}")
        return df
    
    def basic_statistics(self, df):
        """Generate comprehensive basic statistics"""
        print("\n" + "="*60)
        print("BASIC STATISTICS SUMMARY")
        print("="*60)
        
        # Data overview
        print(f"Dataset shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Duration: {df.index.max() - df.index.min()}")
        print(f"Frequency: ~{pd.infer_freq(df.index) or 'Variable'}")
        
        # Core weather variables statistics
        weather_cols = ['temperature', 'humidity', 'pressure']
        available_weather_cols = [col for col in weather_cols if col in df.columns]
        
        if available_weather_cols:
            print(f"\nWeather Variables Summary:")
            print("-" * 40)
            stats_df = df[available_weather_cols].describe()
            print(stats_df.round(2))
            
            # Additional statistics
            print(f"\nAdditional Statistics:")
            print("-" * 40)
            for col in available_weather_cols:
                data = df[col].dropna()
                print(f"{col.capitalize()}:")
                print(f"  Variance: {data.var():.2f}")
                print(f"  Skewness: {stats.skew(data):.2f}")
                print(f"  Kurtosis: {stats.kurtosis(data):.2f}")
                print(f"  Missing values: {df[col].isnull().sum()}")
    
    def data_quality_assessment(self, df):
        """Assess data quality issues"""
        print("\n" + "="*60)
        print("DATA QUALITY ASSESSMENT")
        print("="*60)
        
        # Missing values analysis
        missing_counts = df.isnull().sum()
        missing_percent = (missing_counts / len(df)) * 100
        
        print("Missing Values Analysis:")
        print("-" * 30)
        for col in df.columns:
            if missing_counts[col] > 0:
                print(f"{col}: {missing_counts[col]} ({missing_percent[col]:.1f}%)")
        
        if missing_counts.sum() == 0:
            print("No missing values found!")
        
        # Duplicate rows
        duplicates = df.duplicated().sum()
        print(f"\nDuplicate rows: {duplicates}")
        
        # Data range validation for weather variables
        print(f"\nData Range Validation:")
        print("-" * 30)
        weather_ranges = {
            'temperature': (-50, 60),  # Celsius
            'humidity': (0, 100),      # Percentage
            'pressure': (800, 1200)    # hPa
        }
        
        for col, (min_val, max_val) in weather_ranges.items():
            if col in df.columns:
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                print(f"{col}: {out_of_range} values outside [{min_val}, {max_val}] range")
    
    def outlier_detection(self, df):
        """Detect outliers using IQR method"""
        print("\n" + "="*60)
        print("OUTLIER DETECTION (IQR Method)")
        print("="*60)
        
        weather_cols = ['temperature', 'humidity', 'pressure']
        available_cols = [col for col in weather_cols if col in df.columns]
        
        for col in available_cols:
            data = df[col].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            outlier_percent = (outliers / len(data)) * 100
            
            print(f"{col.capitalize()}:")
            print(f"  Lower bound: {lower_bound:.2f}")
            print(f"  Upper bound: {upper_bound:.2f}")
            print(f"  Outliers: {outliers} ({outlier_percent:.1f}%)")
    
    def temporal_analysis(self, df):
        """Analyze temporal patterns"""
        print("\n" + "="*60)
        print("TEMPORAL PATTERNS ANALYSIS")
        print("="*60)
        
        # Hourly patterns
        if 'temperature' in df.columns:
            hourly_stats = df.groupby(df.index.hour)['temperature'].agg(['mean', 'std', 'min', 'max'])
            print("Temperature by Hour of Day:")
            print("-" * 40)
            print(hourly_stats.round(2))
        
        # Daily patterns
        if len(df) > 7:
            daily_stats = df.groupby(df.index.dayofweek)[['temperature', 'humidity', 'pressure']].mean()
            daily_stats.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            print(f"\nAverage Values by Day of Week:")
            print("-" * 40)
            print(daily_stats.round(2))
    
    def correlation_analysis(self, df):
        """Analyze correlations between variables"""
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            print("Correlation Matrix (top correlations):")
            print("-" * 40)
            
            # Get top correlations (excluding self-correlations)
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    if not np.isnan(corr_val):
                        corr_pairs.append((abs(corr_val), corr_val, col1, col2))
            
            # Sort by absolute correlation and show top 10
            corr_pairs.sort(reverse=True)
            for _, corr, col1, col2 in corr_pairs[:10]:
                print(f"{col1} <-> {col2}: {corr:.3f}")
    
    def create_visualizations(self, df):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Figure 1: Time series plots
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Weather Data Time Series Analysis', fontsize=16)
        
        weather_cols = ['temperature', 'humidity', 'pressure']
        colors = ['red', 'blue', 'green']
        
        for i, (col, color) in enumerate(zip(weather_cols, colors)):
            if col in df.columns:
                axes[i].plot(df.index, df[col], color=color, alpha=0.7, linewidth=0.8)
                axes[i].set_title(f'{col.capitalize()} Over Time')
                axes[i].set_ylabel(col.capitalize())
                axes[i].grid(True, alpha=0.3)
                
                # Add rolling average
                if len(df) > 24:  # If we have enough data
                    rolling_avg = df[col].rolling(window=24, center=True).mean()
                    axes[i].plot(df.index, rolling_avg, color='black', linewidth=2, 
                               label='24-period Moving Average')
                    axes[i].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Figure 2: Distribution plots
        weather_cols_available = [col for col in weather_cols if col in df.columns]
        if weather_cols_available:
            fig, axes = plt.subplots(1, len(weather_cols_available), figsize=(5*len(weather_cols_available), 4))
            if len(weather_cols_available) == 1:
                axes = [axes]
            
            fig.suptitle('Weather Data Distributions', fontsize=16)
            
            for i, col in enumerate(weather_cols_available):
                data = df[col].dropna()
                axes[i].hist(data, bins=50, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{col.capitalize()} Distribution')
                axes[i].set_xlabel(col.capitalize())
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
                
                # Add vertical lines for mean and median
                axes[i].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.1f}')
                axes[i].axvline(data.median(), color='green', linestyle='--', label=f'Median: {data.median():.1f}')
                axes[i].legend()
            
            plt.tight_layout()
            plt.show()
        
        # Figure 3: Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 2:
            plt.figure(figsize=(12, 10))
            
            # Select key columns for cleaner visualization
            key_cols = [col for col in ['temperature', 'humidity', 'pressure', 'temperature_normalized', 
                       'humidity_normalized', 'pressure_normalized', 'hour', 'day_of_week', 'month'] 
                       if col in df.columns]
            
            if len(key_cols) > 1:
                corr_matrix = df[key_cols].corr()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                
                sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                           square=True, linewidths=.5, cbar_kws={"shrink": .8})
                plt.title('Weather Data Correlation Matrix')
                plt.tight_layout()
                plt.show()
        
        # Figure 4: Hourly patterns
        if 'temperature' in df.columns and len(df) > 24:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Temporal Patterns Analysis', fontsize=16)
            
            # Hourly temperature pattern
            hourly_temp = df.groupby(df.index.hour)['temperature'].mean()
            axes[0,0].plot(hourly_temp.index, hourly_temp.values, marker='o')
            axes[0,0].set_title('Average Temperature by Hour')
            axes[0,0].set_xlabel('Hour of Day')
            axes[0,0].set_ylabel('Temperature')
            axes[0,0].grid(True, alpha=0.3)
            
            # Daily pattern (if enough data)
            if len(df) > 7:
                daily_temp = df.groupby(df.index.dayofweek)['temperature'].mean()
                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                axes[0,1].bar(range(7), daily_temp.values)
                axes[0,1].set_title('Average Temperature by Day of Week')
                axes[0,1].set_xlabel('Day of Week')
                axes[0,1].set_ylabel('Temperature')
                axes[0,1].set_xticks(range(7))
                axes[0,1].set_xticklabels(day_names)
                axes[0,1].grid(True, alpha=0.3)
            
            # Temperature vs Humidity scatter
            if 'humidity' in df.columns:
                axes[1,0].scatter(df['temperature'], df['humidity'], alpha=0.5)
                axes[1,0].set_title('Temperature vs Humidity')
                axes[1,0].set_xlabel('Temperature')
                axes[1,0].set_ylabel('Humidity')
                axes[1,0].grid(True, alpha=0.3)
            
            # Box plot of temperature by hour
            df_hourly = df.copy()
            df_hourly['hour'] = df_hourly.index.hour
            if len(df_hourly['hour'].unique()) > 5:  # Only if we have enough hours
                hourly_groups = [df_hourly[df_hourly['hour'] == h]['temperature'].dropna() 
                               for h in sorted(df_hourly['hour'].unique())]
                axes[1,1].boxplot(hourly_groups, labels=sorted(df_hourly['hour'].unique()))
                axes[1,1].set_title('Temperature Distribution by Hour')
                axes[1,1].set_xlabel('Hour of Day')
                axes[1,1].set_ylabel('Temperature')
                axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def model_features_analysis(self, df):
        """Analyze features used in the ML model"""
        print("\n" + "="*60)
        print("MODEL FEATURES ANALYSIS")
        print("="*60)
        
        # Check which model features are available
        model_features = [
            'temperature', 'humidity', 'pressure',
            'temperature_normalized', 'humidity_normalized', 'pressure_normalized',
            'hour', 'minute', 'day_of_week', 'month'
        ]
        
        available_features = [f for f in model_features if f in df.columns]
        missing_features = [f for f in model_features if f not in df.columns]
        
        print(f"Available model features ({len(available_features)}):")
        for feature in available_features:
            print(f"  ✓ {feature}")
        
        if missing_features:
            print(f"\nMissing model features ({len(missing_features)}):")
            for feature in missing_features:
                print(f"  ✗ {feature}")
        
        # Feature importance analysis (correlation with temperature)
        if 'temperature' in df.columns and len(available_features) > 1:
            print(f"\nFeature Correlations with Temperature:")
            print("-" * 40)
            correlations = []
            for feature in available_features:
                if feature != 'temperature':
                    corr = df[['temperature', feature]].corr().iloc[0, 1]
                    if not np.isnan(corr):
                        correlations.append((abs(corr), feature, corr))
            
            correlations.sort(reverse=True)
            for abs_corr, feature, corr in correlations:
                print(f"{feature}: {corr:.3f}")
    
    def run_complete_analysis(self, hours_back=None, limit=None):
        """Run complete analysis pipeline"""
        print("COMPREHENSIVE WEATHER DATA ANALYSIS")
        print("=" * 60)
        print(f"Analysis started at: {datetime.now()}")
        
        # Load data
        df = self.load_data(hours_back=hours_back, limit=limit)
        if df is None:
            return
        
        # Run all analysis components
        self.basic_statistics(df)
        self.data_quality_assessment(df)
        self.outlier_detection(df)
        self.temporal_analysis(df)
        self.correlation_analysis(df)
        self.model_features_analysis(df)
        self.create_visualizations(df)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Analysis completed at: {datetime.now()}")
        print("All visualizations have been displayed.")

if __name__ == "__main__":
    analyzer = WeatherDataAnalyzer()
    
    # Run complete analysis with all available data
    # You can modify these parameters:
    # - hours_back: None for all data, or specify hours (e.g., 48 for last 48 hours)
    # - limit: None for all records, or specify max number of records
    analyzer.run_complete_analysis(hours_back=None, limit=None) 