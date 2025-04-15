import bittensor as bt
import numpy as np
from typing import List, Optional

def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    # Filter non serving axons.
    if not metagraph.axons[uid].is_serving:
        return False
    # Filter validator permit > 1024 stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
    # Available otherwise.
    return True

def get_miner_uids() -> np.ndarray:
    """Get all miner uids from the network
    
    Returns:
        np.ndarray: Array of miner UIDs that are serving on the network
    """
    # Import locally to avoid circular import
    from ..config import get_config
    
    # Get config for network parameters
    config = get_config()
    
    # Get the metagraph for our subnet
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)
    metagraph.sync(subtensor=subtensor)
    
    # Filter for miners using the check function
    miner_uids = [uid for uid in range(metagraph.n) 
                  if check_uid_availability(metagraph, uid, config.neuron.vpermit_tao_limit)]
    
    # Return as numpy array
    return np.array(miner_uids)