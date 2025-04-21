import time
import bittensor as bt
from neurons.miner.miner import TelegraphMiner

if __name__ == "__main__":
    # Parse command line arguments
    config = TelegraphMiner.config()
    
    # Create and run the miner
    with TelegraphMiner(config) as miner:
        while True:
            try:
                # Log running status
                bt.logging.info(f"Telegraph Miner running... Block: {miner.block}")
                
                # Sleep to avoid excessive logging
                time.sleep(60)
            except KeyboardInterrupt:
                bt.logging.info("Keyboard interrupt detected. Exiting...")
                break
            except Exception as e:
                bt.logging.error(f"Error in miner loop: {e}")
                time.sleep(60)