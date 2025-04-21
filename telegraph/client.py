from typing import Dict, Any, List, Optional
import bittensor as bt
import asyncio
import json

class TelegraphClient:
    """Client for Telegraph Cross-Subnet Communication"""
    
    def __init__(
        self, 
        wallet: 'bt.wallet', 
        telegraph_netuid: int,
        network: str = "test"  # Add network parameter, default to test
    ):
        """Initialize the Telegraph client"""
        self.wallet = wallet
        self.subtensor = bt.subtensor(network=network)  # Connect to the specific network
        self.telegraph_netuid = telegraph_netuid
        self._metagraph = None
        self._dendrite = None
        
    @property
    def metagraph(self):
        """Get or create metagraph"""
        if self._metagraph is None:
            try:
                self._metagraph = self.subtensor.metagraph(netuid=self.telegraph_netuid)
                self._metagraph.sync(subtensor=self.subtensor)
            except Exception as e:
                print(f"Error connecting to subnet {self.telegraph_netuid}: {e}")
                self._metagraph = None
        return self._metagraph
        
    @property
    def dendrite(self):
        """Get or create dendrite"""
        if self._dendrite is None:
            self._dendrite = bt.dendrite(wallet=self.wallet)
        return self._dendrite
    
    def list_validators(self) -> List[Dict[str, Any]]:
        """List available validators on the Telegraph subnet"""
        if not self.metagraph:
            return []
            
        validators = []
        # Updated to work with current metagraph structure
        for uid in range(self.metagraph.n):
            # Check if S[uid] meets validator threshold (typically >= 1024 stake is validator)
            if self.metagraph.S[uid] > 1024:
                validators.append({
                    "uid": uid,
                    "hotkey": self.metagraph.hotkeys[uid],
                    "stake": self.metagraph.S[uid],
                    "axon_info": self.metagraph.axons[uid]
                })
                
        for uid in range(self.metagraph.n):
            # For subnet 349, validator is UID 1 (per your setup)
            # Or use lower threshold: if self.metagraph.S[uid] >= 0:
            if uid == 1:  # UID 1 is your validator
                validators.append({
                    "uid": uid,
                    "hotkey": self.metagraph.hotkeys[uid],
                    "stake": float(self.metagraph.S[uid]),
                    "axon_info": self.metagraph.axons[uid]
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
        """Send inference request to target subnet via Telegraph"""
        metagraph = self._get_metagraph()
        if not self.registry.is_valid_code(inference_code):
            return {"error": f"Unknown inference code: {inference_code}"}
        if validator_uid is not None:
            if validator_uid not in range(len(metagraph.axons)):
                return {"error": f"Invalid validator UID: {validator_uid}"}
            target_axon = metagraph.axons[validator_uid]
        else:
            if not metagraph.validators:
                return {"error": "No Telegraph validators available"}
            validator_stakes = [metagraph.S[uid] for uid in metagraph.validators]
            highest_stake_idx = np.argmax(validator_stakes)
            target_uid = metagraph.validators[highest_stake_idx]
            target_axon = metagraph.axons[target_uid]
            bt.logging.info(f"Selected validator {target_uid} with highest stake")
        # Use standard Synapse for cross-subnet
        synapse = bt.Synapse()
        synapse.data = {
            "inference_code": inference_code,
            "payload": data
        }
        dendrite = bt.dendrite(wallet=self.wallet)
        bt.logging.info(f"Sending inference request for code {inference_code}")
        try:
            response = await dendrite(
                axons=[target_axon],
                synapse=synapse,
                deserialize=True,
                timeout=timeout
            )
            if hasattr(response, "error") and response.error:
                return {"error": response.error}
            return {"response": getattr(response, "response", None)}
        except Exception as e:
            bt.logging.error(f"Failed to query model: {str(e)}")
            return {"error": f"Request failed: {str(e)}"}