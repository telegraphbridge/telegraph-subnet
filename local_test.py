#!/usr/bin/env python
import os
import json
import torch
import asyncio
import bittensor as bt
from base.types import ChainType, TokenPrediction
from neurons.miner.models.base_l2_model import LSTMTokenModel

async def test_model_directly():
    """Test the model directly without Bittensor network connection"""
    print("Telegraph Model Testing Tool")
    print("===========================\n")
    
    # Check for data directory and files
    data_dir = "data/transactions"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return
        
    print(f"✅ Found data directory: {data_dir}")
    
    # Check for transaction files
    transaction_files = [f for f in os.listdir(data_dir) if "transactions" in f and f.endswith(".json")]
    if not transaction_files:
        print("❌ No transaction files found")
        return
        
    print(f"✅ Found transaction files: {', '.join(transaction_files)}")
    
    # Check for model file
    model_path = os.path.join(data_dir, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
        
    print(f"✅ Found model file: {model_path} ({os.path.getsize(model_path):,} bytes)")
    
    # Initialize model
    print("\nInitializing model...")
    model = LSTMTokenModel(model_path=model_path, data_dir=data_dir)
    
    # Run prediction
    print("\nRunning prediction for BASE chain...")
    try:
        prediction = await model.predict(ChainType.BASE)
        
        if prediction and prediction.addresses:
            print(f"\n✅ Successfully generated {len(prediction.addresses)} token predictions!")
            
            print("\nTop 5 predicted tokens:")
            for i in range(min(5, len(prediction.addresses))):
                addr = prediction.addresses[i]
                pair = prediction.pairAddresses[i] if i < len(prediction.pairAddresses or []) else "N/A"
                conf = prediction.confidence_scores.get(addr, "N/A") if prediction.confidence_scores else "N/A"
                
                print(f"{i+1}. Token: {addr}")
                print(f"   Pair:  {pair}")
                print(f"   Confidence: {conf}")
            
            # Check if these look like real addresses
            import re
            placeholder_pattern = r"0x0{6}[0-9a-f]{34}"  # Matches the sequential pattern
            if any(re.match(placeholder_pattern, addr) for addr in prediction.addresses[:3]):
                print("\n⚠️ WARNING: Detected placeholder addresses - model likely using fallback data")
            else:
                print("\n✅ Predictions appear to be from the trained model")
                
        else:
            print("❌ No predictions generated")
    except Exception as e:
        import traceback
        print(f"❌ Error running prediction: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(test_model_directly())