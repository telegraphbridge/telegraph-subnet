import time
import bittensor as bt
from neurons.validator.validator import TelegraphValidator
SUBNET_UID = 349

if __name__ == "__main__":
    # Parse command line arguments
    config = TelegraphValidator.config()
    config.netuid = SUBNET_UID
    
    # Create and run the validator
    with TelegraphValidator(config) as validator:
        while True:
            try:
                # Log running status
                bt.logging.info(f"Telegraph Validator running... Block: {validator.block}")
                
                # Sleep to avoid excessive logging
                time.sleep(60)
            except KeyboardInterrupt:
                bt.logging.info("Keyboard interrupt detected. Exiting...")
                break
            except Exception as e:
                bt.logging.error(f"Error in validator loop: {e}")
                time.sleep(60)