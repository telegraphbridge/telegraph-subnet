import bittensor as bt
import numpy as np
from typing import List
from telegraph.protocol import PredictionSynapse # Assuming PredictionSynapse is the response type

def check_uid_availability(metagraph: "bt.metagraph.Metagraph", uid: int, vpermit_tao_limit: int) -> bool:
    """Check if a UID is available for querying."""
    # Check if UID is valid
    if uid >= metagraph.n:
        # bt.logging.trace(f"UID {uid} is invalid, metagraph size is {metagraph.n}")
        return False
    # Check if the axon is serving.
    if not metagraph.axons[uid].is_serving:
        # bt.logging.trace(f"UID {uid} is not serving.")
        return False
    # Check if the miner has a validator permit and has too much stake.
    if metagraph.validator_permit[uid]:
        if metagraph.S[uid] > vpermit_tao_limit:
            # bt.logging.trace(f"UID {uid} has validator permit and stake {metagraph.S[uid]} > {vpermit_tao_limit}")
            return False
    # bt.logging.trace(f"UID {uid} is available.")
    return True

def get_miner_uids(metagraph: "bt.metagraph.Metagraph", vpermit_tao_limit: int = 1024) -> List[int]:
    """Get miner UIDs that are available for querying."""
    if metagraph is None:
        bt.logging.error("Metagraph not provided to get_miner_uids")
        return []

    available_uids = [
        uid for uid in range(metagraph.n)
        if check_uid_availability(metagraph, uid, vpermit_tao_limit)
    ]
    # bt.logging.debug(f"Found {len(available_uids)} available miner UIDs")
    return available_uids

def get_uids_from_responses(responses: List[PredictionSynapse], queried_uids: List[int]) -> List[int]:
    """ Maps responses back to their original UIDs based on axon hotkey. """
    uid_map = {axon.hotkey: uid for uid, axon in enumerate(queried_uids)} # Map hotkey to original queried UID index
    response_uids = []
    for resp in responses:
        hotkey = resp.axon.hotkey
        if hotkey in uid_map:
             response_uids.append(queried_uids[uid_map[hotkey]]) # Get the actual UID
        else:
             # This case should ideally not happen if responses map correctly
             bt.logging.warning(f"Could not map response from hotkey {hotkey} back to a queried UID.")
             # Handle appropriately, maybe skip or assign a default/error UID? For now, skip.
             pass
    return response_uids
