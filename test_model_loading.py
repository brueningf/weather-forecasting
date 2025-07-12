#!/usr/bin/env python3
"""
Test script to check model loading issues
"""

import torch
import logging
from model_predictor import ModelPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test model loading and identify issues"""
    try:
        logger.info("Testing model loading...")
        
        # Test 1: Basic model predictor initialization
        logger.info("1. Initializing ModelPredictor...")
        mp = ModelPredictor()
        
        logger.info(f"   - is_trained: {mp.is_trained}")
        logger.info(f"   - model is None: {mp.model is None}")
        logger.info(f"   - device: {mp.device}")
        
        # Test 2: Check if model file exists and can be loaded
        logger.info("2. Testing direct model file loading...")
        try:
            checkpoint = torch.load('model.pth', map_location='cpu')
            logger.info(f"   - Checkpoint keys: {list(checkpoint.keys())}")
            logger.info(f"   - is_trained in checkpoint: {checkpoint.get('is_trained', 'NOT_FOUND')}")
            
            # Check if model has weights
            model = checkpoint['model']
            logger.info(f"   - Model type: {type(model)}")
            logger.info(f"   - Model state dict keys: {list(model.state_dict().keys())}")
            
        except Exception as e:
            logger.error(f"   - Error loading model file: {e}")
        
        # Test 3: Test prediction functionality
        logger.info("3. Testing prediction functionality...")
        if mp.is_trained and mp.model is not None:
            logger.info("   - Model appears to be trained and loaded")
        else:
            logger.warning("   - Model is not properly trained or loaded")
        
        logger.info("Model loading test completed")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_loading() 