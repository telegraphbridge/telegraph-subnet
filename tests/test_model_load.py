import torch
import os
import pickle

def test_model_loading():
    print("Checking if model files exist...")
    model_path = "data/transactions/best_model.pth"
    scaler_path = "data/transactions/feature_scaler.pkl"
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return False
        
    if not os.path.exists(scaler_path):
        print(f"ERROR: Scaler file not found: {scaler_path}")
        return False
    
    print("Loading model...")
    try:
        model = torch.load(model_path)
        print(f"Model loaded successfully: {type(model)}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False
        
    print("Loading scaler...")
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"Scaler loaded successfully: {type(scaler)}")
    except Exception as e:
        print(f"ERROR loading scaler: {e}")
        return False
    
    print("All tests passed!")
    return True

if __name__ == "__main__":
    test_model_loading()
