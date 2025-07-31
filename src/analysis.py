# Moved analysis endpoints from api.py
import matplotlib
matplotlib.use('Agg')
from fastapi import APIRouter, Query, Response
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import logging
from datetime import datetime, timedelta
from fastapi.responses import Response
from database import DatabaseManager
from scipy import stats

logger = logging.getLogger(__name__)
router = APIRouter()

# Create a single database manager instance
db_manager = DatabaseManager()

class WeatherDataAnalyzer:
    def __init__(self):
        self.controller = db_manager
        
    def load_data(self, hours_back=None, limit=None):
        """Load preprocessed data for analysis"""
        print("Loading preprocessed data...")
        print(f"Parameters: hours_back={hours_back}, limit={limit}")
        
        df = self.controller.get_preprocessed_data(hours_back=hours_back, limit=limit)
        
        print(f"DataFrame returned: {df is not None}, empty: {df.empty if df is not None else 'N/A'}")
        if df is not None and not df.empty:
            print(f"DataFrame shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"Index range: {df.index.min()} to {df.index.max()}")
        
        if df is None or df.empty:
            print("No preprocessed data found.")
            return None
            
        print(f"Loaded {len(df)} records from {df.index.min()} to {df.index.max()}")
        return df

@router.get("/api/analysis/time-series")
async def get_time_series_plot():
    """Generate time series analysis plot"""
    try:
        analyzer = WeatherDataAnalyzer()
        df = analyzer.load_data(hours_back=168)  # Last week
        if df is None or df.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'No data available for analysis\nPlease initialize the system first', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Time Series Analysis - No Data Available')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            canvas = FigureCanvas(fig)
            canvas.draw()
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            img_data.seek(0)
            plt.close(fig)
            return Response(content=img_data.getvalue(), media_type="image/png")
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Weather Data Time Series Analysis', fontsize=16)
        weather_cols = ['temperature', 'humidity', 'pressure']
        colors = ['red', 'blue', 'green']
        for i, (col, color) in enumerate(zip(weather_cols, colors)):
            if col in df.columns:
                axes[i].plot(df.index, df[col], color=color, alpha=0.7, linewidth=0.8)
                axes[i].set_title(f'{col.capitalize()} Over Time')
                axes[i].set_ylabel(col.capitalize())
                axes[i].grid(True, alpha=0.3)
                if len(df) > 24:
                    rolling_avg = df[col].rolling(window=24, center=True).mean()
                    axes[i].plot(df.index, rolling_avg, color='black', linewidth=2, label='24-period Moving Average')
                    axes[i].legend()
        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.draw()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plt.close(fig)
        return Response(content=img_data.getvalue(), media_type="image/png")
    except Exception as e:
        import logging
        logger = logging.getLogger("api")
        logger.error(f"Error generating time series plot: {e}")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, f'Error generating plot: {str(e)}', ha='center', va='center', transform=ax.transAxes, fontsize=14, color='red')
        ax.set_title('Time Series Analysis - Error')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        canvas = FigureCanvas(fig)
        canvas.draw()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plt.close(fig)
        return Response(content=img_data.getvalue(), media_type="image/png")

@router.get("/api/analysis/distributions")
async def get_distribution_plot():
    """Generate distribution analysis plot"""
    try:
        analyzer = WeatherDataAnalyzer()
        df = analyzer.load_data(hours_back=168)
        if df is None or df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No data available for analysis\nPlease initialize the system first', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Data Distributions - No Data Available')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            canvas = FigureCanvas(fig)
            canvas.draw()
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            img_data.seek(0)
            plt.close(fig)
            return Response(content=img_data.getvalue(), media_type="image/png")
        weather_cols = ['temperature', 'humidity', 'pressure']
        weather_cols_available = [col for col in weather_cols if col in df.columns]
        if not weather_cols_available:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No weather data available for distribution analysis', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Data Distributions - No Weather Data')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            canvas = FigureCanvas(fig)
            canvas.draw()
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            img_data.seek(0)
            plt.close(fig)
            return Response(content=img_data.getvalue(), media_type="image/png")
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
            axes[i].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.1f}')
            axes[i].axvline(data.median(), color='green', linestyle='--', label=f'Median: {data.median():.1f}')
            axes[i].legend()
        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.draw()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plt.close(fig)
        return Response(content=img_data.getvalue(), media_type="image/png")
    except Exception as e:
        import logging
        logger = logging.getLogger("api")
        logger.error(f"Error generating distribution plot: {e}")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error generating plot: {str(e)}', ha='center', va='center', transform=ax.transAxes, fontsize=14, color='red')
        ax.set_title('Data Distributions - Error')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        canvas = FigureCanvas(fig)
        canvas.draw()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plt.close(fig)
        return Response(content=img_data.getvalue(), media_type="image/png")

@router.get("/api/analysis/correlation")
async def get_correlation_plot():
    """Generate correlation matrix plot"""
    try:
        analyzer = WeatherDataAnalyzer()
        df = analyzer.load_data(hours_back=168)
        if df is None or df.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'No data available for analysis\nPlease initialize the system first', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Correlation Matrix - No Data Available')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            canvas = FigureCanvas(fig)
            canvas.draw()
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            img_data.seek(0)
            plt.close(fig)
            return Response(content=img_data.getvalue(), media_type="image/png")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'Insufficient numeric data for correlation analysis', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Correlation Matrix - Insufficient Data')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            canvas = FigureCanvas(fig)
            canvas.draw()
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            img_data.seek(0)
            plt.close(fig)
            return Response(content=img_data.getvalue(), media_type="image/png")
        key_cols = [col for col in ['temperature', 'humidity', 'pressure', 'temperature_normalized', 'humidity_normalized', 'pressure_normalized', 'hour', 'day_of_week', 'month'] if col in df.columns]
        if len(key_cols) < 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, 'Insufficient key columns for correlation analysis', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Correlation Matrix - Insufficient Columns')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            canvas = FigureCanvas(fig)
            canvas.draw()
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            img_data.seek(0)
            plt.close(fig)
            return Response(content=img_data.getvalue(), media_type="image/png")
        plt.figure(figsize=(10, 8))
        corr_matrix = df[key_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, square=True, linewidths=.5, cbar_kws={"shrink": .8})
        plt.title('Weather Data Correlation Matrix')
        plt.tight_layout()
        canvas = FigureCanvas(plt.gcf())
        canvas.draw()
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plt.close()
        return Response(content=img_data.getvalue(), media_type="image/png")
    except Exception as e:
        import logging
        logger = logging.getLogger("api")
        logger.error(f"Error generating correlation plot: {e}")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, f'Error generating plot: {str(e)}', ha='center', va='center', transform=ax.transAxes, fontsize=14, color='red')
        ax.set_title('Correlation Matrix - Error')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        canvas = FigureCanvas(fig)
        canvas.draw()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plt.close(fig)
        return Response(content=img_data.getvalue(), media_type="image/png")

@router.get("/api/analysis/temporal")
async def get_temporal_plot():
    """Generate temporal patterns plot"""
    try:
        analyzer = WeatherDataAnalyzer()
        df = analyzer.load_data(hours_back=168)
        if df is None or df.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'No data available for analysis\nPlease initialize the system first', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Temporal Patterns - No Data Available')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            canvas = FigureCanvas(fig)
            canvas.draw()
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            img_data.seek(0)
            plt.close(fig)
            return Response(content=img_data.getvalue(), media_type="image/png")
        if 'temperature' not in df.columns or len(df) < 24:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'Insufficient temperature data for temporal analysis', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Temporal Patterns - Insufficient Data')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            canvas = FigureCanvas(fig)
            canvas.draw()
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            img_data.seek(0)
            plt.close(fig)
            return Response(content=img_data.getvalue(), media_type="image/png")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Temporal Patterns Analysis', fontsize=16)
        hourly_temp = df.groupby(df.index.hour)['temperature'].mean()
        axes[0,0].plot(hourly_temp.index, hourly_temp.values, marker='o')
        axes[0,0].set_title('Average Temperature by Hour')
        axes[0,0].set_xlabel('Hour of Day')
        axes[0,0].set_ylabel('Temperature (°C)')
        axes[0,0].grid(True, alpha=0.3)
        if len(df) > 7:
            daily_temp = df.groupby(df.index.dayofweek)['temperature'].mean()
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            axes[0,1].bar(daily_temp.index, daily_temp.values)
            axes[0,1].set_title('Average Temperature by Day of Week')
            axes[0,1].set_xlabel('Day of Week')
            axes[0,1].set_ylabel('Temperature (°C)')
            axes[0,1].set_xticks(daily_temp.index)
            axes[0,1].set_xticklabels([day_names[i] for i in daily_temp.index])
            axes[0,1].grid(True, alpha=0.3)
        else:
            axes[0,1].text(0.5, 0.5, 'Insufficient data for daily pattern', ha='center', va='center', transform=axes[0,1].transAxes)
            axes[0,1].set_title('Daily Pattern - Insufficient Data')
            axes[0,1].axis('off')
        if len(df) > 30:
            monthly_temp = df.groupby(df.index.month)['temperature'].mean()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            axes[1,0].plot(monthly_temp.index, monthly_temp.values, marker='o')
            axes[1,0].set_title('Average Temperature by Month')
            axes[1,0].set_xlabel('Month')
            axes[1,0].set_ylabel('Temperature (°C)')
            axes[1,0].set_xticks(range(1, 13))
            axes[1,0].set_xticklabels(month_names)
            axes[1,0].grid(True, alpha=0.3)
        else:
            axes[1,0].text(0.5, 0.5, 'Insufficient data for monthly pattern', ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,0].set_title('Monthly Pattern - Insufficient Data')
            axes[1,0].axis('off')
        if len(df) > 24:
            hourly_stats = df.groupby(df.index.hour)['temperature'].agg(['mean', 'std'])
            axes[1,1].fill_between(hourly_stats.index, hourly_stats['mean'] - hourly_stats['std'], hourly_stats['mean'] + hourly_stats['std'], alpha=0.3, label='±1 Std Dev')
            axes[1,1].plot(hourly_stats.index, hourly_stats['mean'], marker='o', label='Mean')
            axes[1,1].set_title('Temperature Variation by Hour')
            axes[1,1].set_xlabel('Hour of Day')
            axes[1,1].set_ylabel('Temperature (°C)')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        else:
            axes[1,1].text(0.5, 0.5, 'Insufficient data for variation analysis', ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Temperature Variation - Insufficient Data')
            axes[1,1].axis('off')
        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.draw()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plt.close(fig)
        return Response(content=img_data.getvalue(), media_type="image/png")
    except Exception as e:
        import logging
        logger = logging.getLogger("api")
        logger.error(f"Error generating temporal plot: {e}")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, f'Error generating plot: {str(e)}', ha='center', va='center', transform=ax.transAxes, fontsize=14, color='red')
        ax.set_title('Temporal Patterns - Error')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        canvas = FigureCanvas(fig)
        canvas.draw()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plt.close(fig)
        return Response(content=img_data.getvalue(), media_type="image/png")

@router.get("/api/analysis/temperature-comparison")
async def get_temperature_comparison_plot(
    hours: int = Query(24, ge=1, le=168),
    module_id: str = None,
    data_type: str = Query('preprocessed', regex='^(preprocessed|source)$')
):
    """Generate temperature comparison plot (actual vs predicted, past & future)"""
    try:
        weather_data_controller = db_manager # Use db_manager directly
        if data_type == 'preprocessed':
            actual_data = weather_data_controller.get_preprocessed_data(hours_back=hours)
            if actual_data is not None and not actual_data.empty:
                actual_data = actual_data.reset_index()
                actual_data = actual_data[['timestamp', 'temperature', 'humidity', 'pressure']]
                if module_id and 'module' in actual_data.columns:
                    actual_data = actual_data[actual_data['module'] == module_id]
        else:
            actual_data = weather_data_controller.get_latest_sensor_data(hours=hours, module_id=module_id)
            if actual_data is not None and not actual_data.empty:
                actual_data = actual_data[['timestamp', 'temperature', 'humidity', 'pressure']]
        if actual_data is not None and not actual_data.empty:
            actual_data = actual_data.sort_values('timestamp')
        predictions_df = weather_data_controller.get_latest_predictions(hours=hours)
        if predictions_df is not None and not predictions_df.empty:
            predictions_df = predictions_df.copy()
            predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
            predictions_df = predictions_df.sort_values('timestamp')
        if (actual_data is None or actual_data.empty) and (predictions_df is None or predictions_df.empty):
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_title('Temperature Comparison - No Data Available')
            ax.axis('off')
            canvas = FigureCanvas(fig)
            canvas.draw()
            img_data = io.BytesIO()
            fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
            img_data.seek(0)
            plt.close(fig)
            return Response(content=img_data.getvalue(), media_type="image/png")
        fig, ax = plt.subplots(figsize=(12, 6))
        now = pd.Timestamp.now()
        if actual_data is not None and not actual_data.empty:
            ax.plot(actual_data['timestamp'], actual_data['temperature'], label=f'{data_type.capitalize()} Data', color='royalblue', marker='o', alpha=0.7)
        if predictions_df is not None and not predictions_df.empty:
            past_pred = predictions_df[predictions_df['timestamp'] <= now]
            future_pred = predictions_df[predictions_df['timestamp'] > now]
            if not past_pred.empty:
                ax.plot(past_pred['timestamp'], past_pred['predicted_temperature'], label='Past Predictions', color='crimson', linestyle='-', marker='x', alpha=0.7)
            if not future_pred.empty:
                ax.plot(future_pred['timestamp'], future_pred['predicted_temperature'], label='Future Predictions', color='purple', linestyle='--', marker='x', alpha=0.7)
        ax.axvline(now, color='black', linestyle=':', linewidth=2, label='Now')
        ax.set_title(f'Temperature Comparison (Last {hours} hours)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature (°C)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.draw()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plt.close(fig)
        return Response(content=img_data.getvalue(), media_type="image/png")
    except Exception as e:
        import logging
        logger = logging.getLogger("api")
        logger.error(f"Error generating temperature comparison plot: {e}")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f'Error generating plot: {str(e)}', ha='center', va='center', transform=ax.transAxes, fontsize=14, color='red')
        ax.set_title('Temperature Comparison - Error')
        ax.axis('off')
        canvas = FigureCanvas(fig)
        canvas.draw()
        img_data = io.BytesIO()
        fig.savefig(img_data, format='png', dpi=100, bbox_inches='tight')
        img_data.seek(0)
        plt.close(fig)
        return Response(content=img_data.getvalue(), media_type="image/png") 