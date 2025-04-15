import bittensor as bt
import argparse
from typing import Optional

_config = None

def get_config(config: Optional[bt.config] = None) -> bt.config:
    """Get validator configuration with subnet defaults
    
    Args:
        config: Optional existing config object to use
        
    Returns:
        bt.config: Configuration with telegraph subnet defaults
    """
    global _config
    
    # Return cached config if available
    if config is not None:
        return config
    if _config is not None:
        return _config
        
    # Create new config parser
    parser = argparse.ArgumentParser()
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.axon.add_args(parser)
    
    # Add telegraph subnet specific arguments
    parser.add_argument('--netuid', 
                      type=int, 
                      default=1,
                      help='Subnet netuid')
    
    parser.add_argument('--neuron.vpermit_tao_limit', 
                      type=int, 
                      default=1024,
                      help='Maximum stake for validators with permits')
    
    # Add validator specific arguments
    parser.add_argument('--validator.min_evaluation_time',
                      type=int,
                      default=3600,  # 1 hour
                      help='Minimum time (seconds) before evaluating token performance')
    
    # Parse config
    _config = bt.config(parser)
    
    return _config