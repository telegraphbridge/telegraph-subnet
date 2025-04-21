from telegraph.nextplace_synapsis import RealEstateSynapse, RealEstatePredictions, RealEstatePrediction
from telegraph.registry import InferenceRegistry
import bittensor as bt
import asyncio

async def test_cross_subnet_request():
    print("Testing Telegraph cross-subnet communication")
    wallet = bt.wallet(name="miner_coldkey1", hotkey="miner")
    subtensor = bt.subtensor(network="test")
    dendrite = bt.dendrite(wallet=wallet)
    metagraph = subtensor.metagraph(netuid=349)
    validator_axon = metagraph.axons[1]
    registry = InferenceRegistry()
    target_code = "nextplace"
    if registry.is_valid_code(target_code):
        test_prediction = RealEstatePrediction(
            id="test1",
            address="123 Main St",
            city="Testville",
            price=123456.0,
            beds=3,
            baths=2.0,
            sqft=1500,
            predicted_sale_price=130000.0,
            predicted_sale_date="2025-05-01"
        )
        predictions = RealEstatePredictions(predictions=[test_prediction])
        synapse = RealEstateSynapse(real_estate_predictions=predictions)
        async with bt.dendrite(wallet=wallet) as local_dendrite:
            response = await local_dendrite(
                axons=[validator_axon],
                synapse=synapse,
                deserialize=True,
                timeout=30.0
            )
        if isinstance(response, list) and len(response) > 0:
            response = response[0]
        print("\n📡 Received cross-subnet response:")
        print(f"  Response type: {type(response)}")
        if hasattr(response, 'error') and response.error:
            print(f"  ❌ Error: {response.error}")
        elif hasattr(response, 'real_estate_predictions'):
            print(f"  ✅ Real Estate Predictions: {response.real_estate_predictions}")
        elif hasattr(response, 'response'):
            print(f"  ✅ Response: {response.response}")
        else:
            print(f"  ⚠️ Unexpected response format: {response}")

if __name__ == "__main__":
    asyncio.run(test_cross_subnet_request())