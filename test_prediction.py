import bittensor as bt
import asyncio
import random
from telegraph.protocol import PredictionSynapse

async def main():
    # Use the miner hotkey to query directly
    wallet = bt.wallet(name="miner_coldkey1", hotkey="miner")
    dendrite = bt.dendrite(wallet=wallet)
    
    # Create a direct axon reference instead of using metagraph
    # This ensures we're connecting to the right endpoint
    axon = bt.axon(
        wallet=wallet,
        ip="127.0.0.1",  # Local machine - change if miner is remote
        port=8091        # Match the port in your miner config
    )
    
    print(f"Querying miner at {axon.ip}:{axon.port} for predictions...")
    
    # Create prediction request 
    synapse = PredictionSynapse(chain_name="BASE")
    
    try:
        # Send request directly to the known axon endpoint
        response = await dendrite(
            axons=[axon],
            synapse=synapse,
            deserialize=True,
            timeout=30.0
        )
        
        print(f"Got response type: {type(response)}")
        
        # Handle response with null checks
        if isinstance(response, list) and len(response) > 0:
            first_response = response[0]
            print(f"Response addresses: {getattr(first_response, 'addresses', None)}")
            print(f"Response pair addresses: {getattr(first_response, 'pairAddresses', None)}")
            print(f"Response confidence scores: {getattr(first_response, 'confidence_scores', None)}")
        else:
            print("Received empty or invalid response")
    finally:
        # Properly close the session - fix for coroutine warning
        if hasattr(dendrite, 'aio_session') and dendrite.aio_session:
            await dendrite.aio_session.close()

if __name__ == "__main__":
    asyncio.run(main())