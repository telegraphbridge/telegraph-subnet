import bittensor as bt
import numpy as np
from typing import List

def check_uid_availability(metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int) -> bool:
    # Only include nodes that are serving.
    if not metagraph.axons[uid].is_serving:
        return False
    # Optionally filter based on stake; if a validator permit is active.
    if hasattr(metagraph, 'validator_permit'):
        if metagraph.validator_permit[uid] and metagraph.S[uid] > vpermit_tao_limit:
            return False
    return True

def get_miner_uids(metagraph=None) -> List[int]:
    """Get UIDs of miners from the metagraph using availability checks."""
    if metagraph is None:
        try:
            subtensor = bt.subtensor(network="test")
            metagraph = subtensor.metagraph(netuid=349)
            metagraph.sync(subtensor=subtensor)
        except Exception as e:
            bt.logging.error(f"Error getting metagraph: {e}")
            return []

    miner_uids = []
    vpermit_limit = 1024  # Example limit
    for uid in range(metagraph.n):
        if check_uid_availability(metagraph, uid, vpermit_limit):
            miner_uids.append(uid)

    bt.logging.debug(f"Found {len(miner_uids)} miner UIDs")
    return miner_uids

# def get_miner_uids() -> List[int]:
#     """Get miner UIDs for querying
    
#     Returns:
#         List[int]: List of miner UIDs
#     """
#     # Import config locally to avoid circular imports
#     from neurons.validator.config import get_config
    
#     # Get config with subnet defaults
#     config = get_config()
    
#     # Get the metagraph for this subnet
#     subtensor = bt.subtensor(config=config)
#     metagraph = subtensor.metagraph(netuid=config.netuid)
#     metagraph.sync(subtensor=subtensor)
    
#     # Filter UIDs based on availability
#     available_uids = []
#     for uid in range(metagraph.n):
#         if check_uid_availability(metagraph, uid, config.neuron.vpermit_tao_limit):
#             available_uids.append(uid)
            
#     return available_uids