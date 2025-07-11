#!/usr/bin/env python3
"""
Test script to verify the training functionality
"""

import asyncio
import logging
from data_processor import DataProcessor
from model_predictor import ModelPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_training():
    """Test the training functionality"""
    try:
        logger.info("Starting training test...")
        
        # Initialize components
        data_processor = DataProcessor()
        model_predictor = ModelPredictor()
        
        # Check initial state
        logger.info(f"Model needs training: {model_predictor.needs_training()}")
        logger.info(f"Model is trained: {model_predictor.is_trained}")
        
        # Export some data for testing
        logger.info("Exporting test data...")
        df = data_processor.force_initial_export(hours_back=168)  # 1 week
        
        if df.empty:
            logger.warning("No data available for testing")
            return
        
        logger.info(f"Exported {len(df)} records")
        
        # Preprocess data
        processed_df = data_processor.preprocess_data(df)
        
        if processed_df.empty:
            logger.error("No data after preprocessing")
            return
        
        logger.info(f"Preprocessed data shape: {processed_df.shape}")
        
        # Train model
        logger.info("Training model...")
        success = model_predictor.train_model(processed_df, epochs=10, learning_rate=0.001)
        
        if success:
            logger.info("Training completed successfully!")
            logger.info(f"Model is now trained: {model_predictor.is_trained}")
            
            # Test prediction
            logger.info("Testing prediction...")
            predictions = model_predictor.predict_future(processed_df, hours_ahead=24)
            
            if not predictions.empty:
                logger.info(f"Generated {len(predictions)} predictions")
                logger.info(f"Sample prediction: {predictions.head()}")
            else:
                logger.warning("No predictions generated")
        else:
            logger.error("Training failed")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_training()) 