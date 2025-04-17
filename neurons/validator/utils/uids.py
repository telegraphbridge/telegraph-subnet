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
    # Filter non serving axons
    if not metagraph.axons[uid].is_serving:
        return False
        
    # Filter validator permit > 1024 stake
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            return False
            
    # Available otherwise
    return True

def get_miner_uids() -> List[int]:
    """Get miner UIDs for querying
    
    Returns:
        List[int]: List of miner UIDs
    """
    # Import config locally to avoid circular imports
    from neurons.validator.config import get_config
    
    # Get config with subnet defaults
    config = get_config()
    
    # Get the metagraph for this subnet
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(netuid=config.netuid)
    metagraph.sync(subtensor=subtensor)
    
    # Filter UIDs based on availability
    available_uids = []
    for uid in range(metagraph.n):
        if check_uid_availability(metagraph, uid, config.neuron.vpermit_tao_limit):
            available_uids.append(uid)
            
    return available_uids