import pandas as pd
from datetime import datetime, timedelta
from config import Config
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, DateTime, String, TIMESTAMP, Index
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()

class Prediction(Base):
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime)
    predicted_temperature = Column(Float)
    confidence = Column(Float)
    created_at = Column(TIMESTAMP)

class PreprocessedData(Base):
    __tablename__ = 'preprocessed_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime)
    temperature = Column(Float)
    humidity = Column(Float)
    pressure = Column(Float)
    batch_id = Column(String(50))
    created_at = Column(TIMESTAMP)
    __table_args__ = (
        Index('idx_timestamp', 'timestamp'),
        Index('idx_batch_id', 'batch_id'),
    )

class DatabaseManager:
    def __init__(self):
        self.config = Config()
        self.sensor_data_table = "timeseries"
        # Engines
        self.source_engine = create_engine(
            f"mysql+mysqlconnector://{self.config.SOURCE_DB_USER}:{self.config.SOURCE_DB_PASSWORD}@{self.config.SOURCE_DB_HOST}:{self.config.SOURCE_DB_PORT}/{self.config.SOURCE_DB_NAME}"
        )
        self.api_engine = create_engine(
            f"mysql+mysqlconnector://{self.config.API_DB_USER}:{self.config.API_DB_PASSWORD}@{self.config.API_DB_HOST}:{self.config.API_DB_PORT}/{self.config.API_DB_NAME}"
        )
        # Session for API DB
        self.Session = sessionmaker(bind=self.api_engine)
        # Create tables if not exist
        Base.metadata.create_all(self.api_engine)

    def export_data_since(self, last_export_time):
        """Export data from source database since the specified time"""
        try:
            # Ensure last_export_time is a naive UTC datetime
            if isinstance(last_export_time, pd.Timestamp):
                if last_export_time.tzinfo is not None:
                    last_export_time = last_export_time.tz_convert('UTC').tz_localize(None)
                else:
                    last_export_time = last_export_time.to_pydatetime()
            query = f"""
                SELECT timestamp, temperature, humidity, pressure, module 
                FROM {self.sensor_data_table} 
                WHERE timestamp > %s 
                ORDER BY timestamp
            """
            df = pd.read_sql(query, self.source_engine, params=(last_export_time,))
            logger.info(f"Exported {len(df)} new records from source database" if not df.empty else "No new data to export from source database")
            return df
        except Exception as e:
            logger.error(f"Error exporting data from source database: {e}")
            return pd.DataFrame()

    def save_predictions(self, predictions_df):
        """Save predictions to API database using ORM"""
        try:
            session = self.Session()
            preds = [Prediction(
                timestamp=index.to_pydatetime() if hasattr(index, 'to_pydatetime') else index,
                predicted_temperature=float(row.get('predicted_temperature', 0)),
                confidence=float(row.get('confidence', 0.8))
            ) for index, row in predictions_df.iterrows()]
            session.add_all(preds)
            session.commit()
            session.close()
            if len(predictions_df) > 10:
                logger.info(f"Saved {len(predictions_df)} predictions to API database")
            else:
                logger.debug(f"Saved {len(predictions_df)} predictions to API database")
        except Exception as e:
            logger.error(f"Error saving predictions to API database: {e}")

    def get_latest_predictions(self, hours=24):
        """Get latest predictions from API database using ORM"""
        try:
            session = self.Session()
            cutoff_time = datetime.now() - timedelta(hours=hours)
            preds = session.query(Prediction).filter(Prediction.timestamp >= cutoff_time).order_by(Prediction.timestamp.desc()).all()
            session.close()
            if preds:
                df = pd.DataFrame([{**{c.name: getattr(p, c.name) for c in Prediction.__table__.columns}} for p in preds])
                return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting latest predictions from API database: {e}")
            return pd.DataFrame()

    def get_latest_sensor_data(self, hours=24, module_id=None):
        """Get latest sensor data from source database (pandas)"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            if module_id:
                query = f"""
                    SELECT timestamp, temperature, humidity, pressure, module
                    FROM {self.sensor_data_table}
                    WHERE timestamp >= %s AND module = %s
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """
                df = pd.read_sql(query, self.source_engine, params=(cutoff_time, module_id))
            else:
                query = f"""
                    SELECT timestamp, temperature, humidity, pressure, module
                    FROM {self.sensor_data_table}
                    WHERE timestamp >= %s
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """
                df = pd.read_sql(query, self.source_engine, params=(cutoff_time,))
            return df
        except Exception as e:
            logger.error(f"Error getting latest sensor data from source database: {e}")
            return pd.DataFrame()

    def get_api_database_stats(self):
        """Get statistics from API database using ORM"""
        try:
            session = self.Session()
            prediction_count = session.query(Prediction).count()
            latest_prediction = session.query(Prediction).order_by(Prediction.timestamp.desc()).first()
            session.close()
            return {
                "prediction_records": prediction_count,
                "latest_prediction": latest_prediction.timestamp.isoformat() if latest_prediction else None,
                "status": "online"  # Database is online if query succeeds
            }
        except Exception as e:
            logger.error(f"Error getting API database stats: {e}")
            return {"prediction_records": 0, "latest_prediction": None, "status": "offline"}

    def get_source_database_stats(self):
        """Get statistics from source database (pandas)"""
        try:
            df = pd.read_sql(f"SELECT COUNT(*) as count, MAX(timestamp) as latest FROM {self.sensor_data_table}", self.source_engine)
            return {
                "sensor_records": int(df['count'][0]),
                "latest_sensor_data": df['latest'][0].isoformat() if pd.notnull(df['latest'][0]) else None,
                "status": "online"  # Database is online if query succeeds
            }
        except Exception as e:
            logger.error(f"Error getting source database stats: {e}")
            return {"sensor_records": 0, "latest_sensor_data": None, "status": "offline"}

    def save_preprocessed_data(self, df, batch_id=None):
        """Save preprocessed data to API database using ORM (timestamp, temperature, humidity, pressure)"""
        try:
            session = self.Session()
            if batch_id is None:
                batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            records = [PreprocessedData(
                timestamp=index.to_pydatetime() if hasattr(index, 'to_pydatetime') else index,
                temperature=float(row.get('temperature', 0)),
                humidity=float(row.get('humidity', 0)),
                pressure=float(row.get('pressure', 0)),
                batch_id=batch_id
            ) for index, row in df.iterrows()]
            session.add_all(records)
            session.commit()
            session.close()
            logger.info(f"Saved {len(records)} preprocessed records to API database (batch: {batch_id})")
            return batch_id
        except Exception as e:
            logger.error(f"Error saving preprocessed data to API database: {e}")
            return None

    def get_preprocessed_data(self, hours_back=None, batch_id=None, limit=None):
        """Get preprocessed data from API database using ORM (timestamp, temperature, humidity, pressure)"""
        try:
            session = self.Session()
            query = session.query(PreprocessedData)
            # Add debugging
            total_count = session.query(PreprocessedData).count()
            logger.debug(f"Total preprocessed records in database: {total_count}")
            if batch_id:
                query = query.filter(PreprocessedData.batch_id == batch_id)
                logger.debug(f"Filtering by batch_id: {batch_id}")
            elif hours_back:
                cutoff_time = datetime.now() - timedelta(hours=hours_back)
                query = query.filter(PreprocessedData.timestamp >= cutoff_time)
                logger.debug(f"Filtering by hours_back: {hours_back}, cutoff_time: {cutoff_time}")
            else:
                logger.debug("No filters applied - returning all records")
            query = query.order_by(PreprocessedData.timestamp)
            if limit:
                results = query.all()[-limit:]
                logger.debug(f"Limited to last {limit} records")
            else:
                results = query.all()
                logger.debug(f"Retrieved {len(results)} records")
            session.close()
            if not results and (hours_back or batch_id):
                logger.warning("No results found with filters. Trying again without filters.")
                session = self.Session()
                query = session.query(PreprocessedData).order_by(PreprocessedData.timestamp)
                results = query.all()
                session.close()
                logger.debug(f"Retrieved {len(results)} records without filters")
            if results:
                df = pd.DataFrame([{c.name: getattr(r, c.name) for c in PreprocessedData.__table__.columns} for r in results])
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                    logger.debug(f"Returning DataFrame with shape: {df.shape}")
                return df[['temperature', 'humidity', 'pressure']] if not df.empty else df
            else:
                logger.debug("No results found in query (even without filters)")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting preprocessed data from API database: {e}")
            return pd.DataFrame()

    def get_preprocessed_data_stats(self):
        """Get statistics about preprocessed data using ORM"""
        try:
            session = self.Session()
            preprocessed_count = session.query(PreprocessedData).count()
            latest = session.query(PreprocessedData).order_by(PreprocessedData.timestamp.desc()).first()
            earliest = session.query(PreprocessedData).order_by(PreprocessedData.timestamp.asc()).first()
            batch_count = session.query(PreprocessedData.batch_id).distinct().count()
            session.close()
            return {
                "preprocessed_records": preprocessed_count,
                "latest_preprocessed": latest.timestamp.isoformat() if latest else None,
                "earliest_preprocessed": earliest.timestamp.isoformat() if earliest else None,
                "batch_count": batch_count,
                "status": "online"  # Database is online if query succeeds
            }
        except Exception as e:
            logger.error(f"Error getting preprocessed data stats: {e}")
            return {
                "preprocessed_records": 0,
                "latest_preprocessed": None,
                "earliest_preprocessed": None,
                "batch_count": 0,
                "status": "offline"
            }

    def clear_preprocessed_data(self, older_than_days=None):
        """Clear old preprocessed data using ORM"""
        try:
            session = self.Session()
            if older_than_days:
                cutoff_time = datetime.now() - timedelta(days=older_than_days)
                deleted = session.query(PreprocessedData).filter(PreprocessedData.timestamp < cutoff_time).delete()
                logger.info(f"Cleared {deleted} preprocessed records older than {older_than_days} days")
            else:
                deleted = session.query(PreprocessedData).delete()
                logger.info(f"Cleared all {deleted} preprocessed records")
            session.commit()
            session.close()
            return deleted
        except Exception as e:
            logger.error(f"Error clearing preprocessed data: {e}")
            return 0 