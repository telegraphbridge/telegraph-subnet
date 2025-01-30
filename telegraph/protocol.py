import bittensor as bt
from typing import List, Optional
from base.types import ChainType, TokenPrediction

class PredictionSynapse(bt.Synapse):
    """
    Protocol for querying predictions from miners.
    
    Attributes:
        chain_name: Input chain name to query
        addresses: Response containing list of predicted addresses
    """
    # Required request input
    chain_name: str
    
    # Optional response output
    addresses: Optional[List[str]] = None

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
    if not synapse.addresses or len(synapse.addresses) < 2:
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
        return True 