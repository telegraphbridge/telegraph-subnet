import bittensor as bt
import asyncio
from telegraph.protocol import PredictionSynapse

async def main():
    wallet = bt.wallet(name="miner_coldkey1", hotkey="miner")
    dendrite = bt.dendrite(wallet=wallet)
    
    # Get validator axon info from metagraph
    subtensor = bt.subtensor(network="test")
    metagraph = subtensor.metagraph(netuid=349)
    axon = metagraph.axons[1]  # Your validator UID
    
    # Create prediction request
    synapse = PredictionSynapse(chain_name="BASE")
    
    # Send request
    print("Querying validator for predictions...")
    response = await dendrite(
        axons=[axon],
        synapse=synapse,
        deserialize=True
    )
    
    print(f"Got response type: {type(response)}")
    
    # Handle response as a list
    if isinstance(response, list) and len(response) > 0:
        synapse_response = response[0]  # Get first response from list
        print(f"First response: {synapse_response}")
        
        if hasattr(synapse_response, 'addresses') and synapse_response.addresses:
            print(f"Predicted addresses: {synapse_response.addresses}")
            print(f"Pair addresses: {synapse_response.pairAddresses}")
        else:
            print("No address predictions in response")
    else:
        print(f"Unexpected response format: {response}")

if __name__ == "__main__":
    asyncio.run(main())