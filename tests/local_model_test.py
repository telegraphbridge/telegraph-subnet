from neurons.miner.models.base_l2_model import LSTMTokenModel
from base.types import ChainType
import asyncio
import bittensor as bt # Import bittensor for logging setup

async def test_model():
    bt.logging(debug=True) # Enable debug logging
    print("Initializing model directly...")
    model = LSTMTokenModel(model_path="data/transactions/best_model.pth")
    print("Model initialized. Calling predict...")
    try:
        prediction = await model.predict(ChainType.BASE)
        print("\n--- Prediction Result ---")
        print(f"Chain: {prediction.chain}")
        print(f"Addresses: {prediction.addresses}")
        print(f"Pair Addresses: {prediction.pairAddresses}")
        print(f"Confidence Scores: {prediction.confidence_scores}")
        print(f"Timestamp: {prediction.timestamp}")
        print("-------------------------\n")
    except Exception as e:
        print(f"Error during model prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_model())