import bittensor as bt
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional

from telegraph.protocol import InferenceRequestSynapse
from telegraph.registry import InferenceRegistry

class TelegraphClient:
    """Client for Telegraph cross-subnet communication"""
    
    def __init__(self, wallet: "bt.wallet", telegraph_netuid: int = 1):
        """Initialize Telegraph client
        
        Args:
            wallet: Bittensor wallet for signing requests
            telegraph_netuid: Netuid of the Telegraph subnet
        """
        self.wallet = wallet
        self.telegraph_netuid = telegraph_netuid
        self.subtensor = bt.subtensor()
        self._metagraph = None
        self.registry = InferenceRegistry()
        
    def _get_metagraph(self):
        """Get and cache Telegraph subnet metagraph"""
        if self._metagraph is None:
            self._metagraph = self.subtensor.metagraph(self.telegraph_netuid)
            self._metagraph.sync(subtensor=self.subtensor)
        return self._metagraph
        
    def list_validators(self) -> List[Dict[str, Any]]:
        """Get available Telegraph validators"""
        metagraph = self._get_metagraph()
        validators = []
        
        for uid in metagraph.validators:
            if metagraph.axons[uid].is_serving:
                validators.append({
                    "uid": int(uid),
                    "hotkey": metagraph.hotkeys[uid],
                    "stake": float(metagraph.S[uid]),
                })
        
        return validators
        
    def get_available_codes(self) -> List[str]:
        """Get available inference codes"""
        return self.registry.get_all_codes()
        
    async def query_model(
        self, 
        inference_code: str, 
        data: Any,
        validator_uid: Optional[int] = None,
        timeout: float = 15.0
    ) -> Dict[str, Any]:
        """Send inference request to target subnet via Telegraph
        
        Args:
            inference_code: Code identifying target model (e.g., "gpt3")
            data: Input data for the model
            validator_uid: Specific validator UID to use (optional)
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary with either "response" or "error" key
        """
        metagraph = self._get_metagraph()
        
        # Verify the inference code is valid
        if not self.registry.is_valid_code(inference_code):
            return {"error": f"Unknown inference code: {inference_code}"}
        
        # Select validator
        if validator_uid is not None:
            if validator_uid not in range(len(metagraph.axons)):
                return {"error": f"Invalid validator UID: {validator_uid}"}
            target_axon = metagraph.axons[validator_uid]
        else:
            # Find validator with highest stake
            if not metagraph.validators:
                return {"error": "No Telegraph validators available"}
                
            validator_stakes = [metagraph.S[uid] for uid in metagraph.validators]
            highest_stake_idx = np.argmax(validator_stakes)
            target_uid = metagraph.validators[highest_stake_idx]
            target_axon = metagraph.axons[target_uid]
            
            bt.logging.info(f"Selected validator {target_uid} with highest stake")
        
        # Create synapse for request
        synapse = InferenceRequestSynapse(
            inference_code=inference_code,
            data=data
        )
        
        # Send request to validator
        dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Sending inference request for code {inference_code}")
        
        try:
            response = await dendrite(
                axons=[target_axon],
                synapse=synapse,
                deserialize=True,
                timeout=timeout
            )
            
            # Check for errors
            if hasattr(response, "error") and response.error:
                return {"error": response.error}
            
            return {"response": response.response}
            
        except Exception as e:
            bt.logging.error(f"Failed to query model: {str(e)}")
            return {"error": f"Request failed: {str(e)}"}