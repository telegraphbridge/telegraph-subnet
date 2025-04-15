import bittensor as bt
from typing import List, Optional
from base.types import ChainType, TokenPrediction

class PredictionSynapse(bt.Synapse):
    """
    Protocol for querying predictions from miners.
    
    Attributes:
        chain_name: Input chain name to query
        addresses: Response containing list of predicted addresses
        pairAddresses: Response containing list of predicted pair addresses
    """
    # Required request input
    chain_name: str
    
    # Optional response output
    addresses: Optional[List[str]] = None
    pairAddresses: Optional[List[str]] = None  # Added missing field

class PerformanceSynapse(bt.Synapse):
    """
    Protocol for querying miner performance metrics.
    
    Attributes:
        query: Input query string
        performance: Response containing performance metrics
    """
    # Required request input  
    query: str
    
    # Optional response output
    performance: Optional[str] = None

def validate_prediction_response(synapse: PredictionSynapse) -> bool:
    """Validates if the prediction response meets minimum requirements"""
    # Check addresses are present
    if not synapse.addresses or len(synapse.addresses) < 2:
        return False
    
    # Check pair addresses are present - essential for liquidity checking
    if not synapse.pairAddresses or len(synapse.pairAddresses) < 2:
        return False
        
    # Check we have matching pairs
    if len(synapse.addresses) != len(synapse.pairAddresses):
        return False
        
    return True

class TelegraphProtocol:
    @staticmethod
    def validate_request(chain: ChainType) -> bool:
        """Validates if the request is properly formatted"""
        return chain in ChainType

    @staticmethod
    def validate_response(prediction: TokenPrediction) -> bool:
        """Validates if the response meets minimum requirements"""
        if not prediction.addresses or len(prediction.addresses) < 2:
            return False
        if not prediction.pairAddresses or len(prediction.pairAddresses) < 2:
            return False
        return True