import bittensor as bt
import asyncio
import random
from telegraph.protocol import PredictionSynapse

async def main():
    # Use the miner hotkey to query directly
    wallet = bt.wallet(name="miner_coldkey1", hotkey="miner")
    dendrite = bt.dendrite(wallet=wallet)
    
    # Create a direct axon reference instead of using metagraph
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
            
            # Check if we got a random-looking response
            addresses = getattr(first_response, 'addresses', None)
            print(f"Response addresses: {addresses}")
            
            pair_addresses = getattr(first_response, 'pairAddresses', None)
            print(f"Response pair addresses: {pair_addresses}")
            
            confidence_scores = getattr(first_response, 'confidence_scores', None)
            print(f"Response confidence scores: {confidence_scores}")
            
            # Check for placeholder address pattern
            import re
            placeholder_pattern = r"0x[0-9]{40}"
            
            if addresses and any(re.match(placeholder_pattern, addr) for addr in addresses[:3]):
                print("\n⚠️ WARNING: Detected placeholder addresses - model likely used fallback data")
                print("Run the following to verify your data files:")
                print("  python verify_data.py")
                print("\nIf files are missing, run:")
                print("  python create_sample_data.py")
            else:
                print("\n✅ Received real token predictions from the model")
        else:
            print("Received empty or invalid response")
    finally:
        # Properly close the session - fix for coroutine warning
        if hasattr(dendrite, 'aio_session') and dendrite.aio_session:
            await dendrite.aio_session.close()

if __name__ == "__main__":
    asyncio.run(main())