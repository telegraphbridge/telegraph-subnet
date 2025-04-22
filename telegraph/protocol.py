import bittensor as bt
import json
from typing import List, Optional, Any, Dict, Union
from base.types import ChainType, TokenPrediction
from telegraph.nextplace_synapsis import RealEstateSynapse
bt.synapse.register(RealEstateSynapse)

class PredictionSynapse(bt.Synapse):
    """Synapse for token price predictions"""
    chain_name: str = ""
    addresses: Optional[List[str]] = None
    pairAddresses: Optional[List[str]] = None
    confidence_scores: Optional[Dict[str, float]] = None
    # Add serialized as a field to avoid the error
    serialized: Optional[bytes] = None
    
    def deserialize(self, data=None) -> 'PredictionSynapse':
        """Deserialize data into this synapse"""
        # Use data parameter if provided, otherwise use self.serialized
        if data is None:
            if not hasattr(self, 'serialized') or self.serialized is None:
                return self
            data = self.serialized
            
        obj = json.loads(data.decode())
        self.chain_name = obj.get("chain_name", "")
        self.addresses = obj.get("addresses", None)
        self.pairAddresses = obj.get("pairAddresses", None)
        self.confidence_scores = obj.get("confidence_scores", None)
        return self
    
    def serialize(self) -> bytes:
        """Serialize this synapse into bytes"""
        data = {
            "chain_name": self.chain_name,
            "addresses": self.addresses,
            "pairAddresses": self.pairAddresses,
            "confidence_scores": self.confidence_scores
        }
        serialized_data = json.dumps(data).encode()
        # Store serialized data in self.serialized (now a valid field)
        self.serialized = serialized_data
        return serialized_data
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        data = {
            "chain_name": self.chain_name,
            "addresses": self.addresses,
            "pairAddresses": self.pairAddresses,
            "confidence_scores": self.confidence_scores
        }
        return json.dumps(data)
    
    def from_json(self, json_str: str) -> 'PredictionSynapse':
        """Load from JSON string"""
        obj = json.loads(json_str)
        self.chain_name = obj.get("chain_name", "")
        self.addresses = obj.get("addresses", None)
        self.pairAddresses = obj.get("pairAddresses", None)
        self.confidence_scores = obj.get("confidence_scores", None)
        return self

class PerformanceSynapse(bt.Synapse):
    """
    Protocol for querying miner performance metrics.
    
    Attributes:
        query: Input query string
        performance: Response containing performance metrics
    """
    # Required request input  
    query: str = ""
    # Optional response output
    performance: Optional[str] = None
    # Add serialized as a field
    serialized: Optional[bytes] = None
    
    def deserialize(self, data=None) -> 'PerformanceSynapse':
        """Deserialize data into this synapse"""
        # Use data parameter if provided, otherwise use self.serialized
        if data is None:
            if not hasattr(self, 'serialized') or self.serialized is None:
                return self
            data = self.serialized
            
        obj = json.loads(data.decode())
        self.query = obj.get("query", "")
        self.performance = obj.get("performance", None)
        return self
    
    def serialize(self) -> bytes:
        """Serialize this synapse into bytes"""
        data = {
            "query": self.query,
            "performance": self.performance
        }
        serialized_data = json.dumps(data).encode()
        # Store serialized data in self.serialized
        self.serialized = serialized_data
        return serialized_data

class InferenceRequestSynapse(bt.Synapse):
    """Synapse for cross‑subnet inference requests"""
    inference_code: str = ""
    data: Any = None
    response: Any = None
    error: Optional[str] = None
    serialized: Optional[bytes] = None

    def deserialize(self, data=None) -> "InferenceRequestSynapse":
        """Deserialize data into this synapse"""
        if data is None:
            if not hasattr(self, 'serialized') or self.serialized is None:
                return self
            data = self.serialized

        obj = json.loads(data.decode())
        self.inference_code = obj.get("inference_code", "")
        self.data = obj.get("data", None)
        self.response = obj.get("response", None)
        self.error = obj.get("error", None)
        return self
    
    def serialize(self) -> bytes:
        """Serialize this synapse into bytes"""
        data = {
            "inference_code": self.inference_code,
            "data": self.data,
            "response": self.response,
            "error": self.error
        }
        serialized_data = json.dumps(data).encode()
        self.serialized = serialized_data
        return serialized_data
        
    def to_json(self) -> str:
        """Convert to JSON string"""
        data = {
            "inference_code": self.inference_code,
            "data": self.data,
            "response": self.response,
            "error": self.error
        }
        return json.dumps(data)

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