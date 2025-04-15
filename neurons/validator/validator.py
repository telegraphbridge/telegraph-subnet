import bittensor as bt
import time
import numpy as np
from typing import Dict, List, Any, Optional
from base.validator import BaseValidatorNeuron
from telegraph.protocol import PredictionSynapse, InferenceRequestSynapse
from telegraph.registry import InferenceRegistry
from ...base.types import ChainType, TokenPrediction
from .storage.prediction_store import PredictionStore
from .utils.performance_calculator import PerformanceCalculator
from .utils.uids import get_miner_uids
import random

class TelegraphValidator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(TelegraphValidator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        self.prediction_store = PredictionStore()
        self.performance_calculator = PerformanceCalculator()
        
        # Initialize inference registry
        self.inference_registry = InferenceRegistry()
        
        # Setup axon handlers for cross-subnet communication
        if not self.config.neuron.axon_off:
            self.setup_inference_handlers()
    
    def setup_inference_handlers(self):
        """Configure axon handlers for cross-subnet inference requests"""
        if hasattr(self, 'axon') and self.axon:
            # Attach handler for inference requests
            self.axon.attach(
                forward_fn=self.process_inference_request,
                blacklist_fn=self.blacklist_inference_request,
                priority_fn=self.priority_inference_request
            )
            bt.logging.info("Attached inference request handler to validator axon")
    
    async def process_inference_request(self, synapse: InferenceRequestSynapse) -> InferenceRequestSynapse:
        """Process incoming inference requests from other networks/users
        
        This method handles requests as described in section 3.3 of the whitepaper
        """
        bt.logging.debug(f"Received inference request with code: {synapse.inference_code}")
        
        try:
            # Validate inference code
            if not synapse.inference_code or not self.inference_registry.is_valid_code(synapse.inference_code):
                synapse.error = f"Invalid or unknown inference code: {synapse.inference_code}"
                return synapse
                
            # Route request to target subnet
            result = await self.route_inference_request(synapse.inference_code, synapse.data)
            
            # Handle routing errors
            if isinstance(result, dict) and "error" in result:
                synapse.error = result["error"]
                return synapse
                
            # Set response
            synapse.response = result
            return synapse
            
        except Exception as e:
            bt.logging.error(f"Error processing inference request: {str(e)}")
            synapse.error = f"Internal error: {str(e)}"
            return synapse
    
    async def route_inference_request(
        self, 
        inference_code: str, 
        data: Any
    ) -> Any:
        """Route an inference request to the appropriate subnet
        
        Args:
            inference_code: Code identifying the target model
            data: Input data for the model
            
        Returns:
            Model output or error information
        """
        # Get target subnet ID
        target_netuid = self.inference_registry.get_netuid(inference_code)
        if target_netuid is None:
            return {"error": f"Unknown inference code: {inference_code}"}
            
        try:
            # Create request synapse
            synapse = InferenceRequestSynapse(
                inference_code=inference_code,
                data=data
            )
            
            # Get target subnet metagraph 
            target_metagraph = self.subtensor.metagraph(target_netuid)
            target_metagraph.sync(subtensor=self.subtensor)
            
            # Select a validator from target subnet (using stake-weighted selection)
            if len(target_metagraph.validators) == 0:
                return {"error": f"No validators available on subnet {target_netuid}"}
                    
            # Use stake as weighting for selection
            validator_stake = np.array([target_metagraph.S[uid] for uid in target_metagraph.validators])
            total_stake = np.sum(validator_stake)
            if total_stake == 0:
                # Random selection if no stake
                selected_idx = random.randint(0, len(target_metagraph.validators)-1)
            else:
                # Stake-weighted selection
                probabilities = validator_stake / total_stake
                selected_idx = np.random.choice(len(target_metagraph.validators), p=probabilities)
                    
            selected_uid = target_metagraph.validators[selected_idx]
            target_axon = target_metagraph.axons[selected_uid]
            
            bt.logging.info(f"Routing request to subnet {target_netuid}, validator {selected_uid}")
            
            # Send request to target validator
            response = await self.dendrite(
                axons=[target_axon],
                synapse=synapse,
                deserialize=True,
                timeout=15.0
            )
            
            # Check for response errors
            if hasattr(response, "error") and response.error:
                return {"error": f"Target subnet error: {response.error}"}
                
            # Return the model output
            return response.response
            
        except Exception as e:
            bt.logging.error(f"Failed to route inference request: {str(e)}")
            return {"error": f"Routing failed: {str(e)}"}
    
    async def blacklist_inference_request(self, synapse: InferenceRequestSynapse) -> tuple[bool, str]:
        """Determine if an inference request should be blacklisted"""
        # For MVP, we don't blacklist inference requests
        # In production, implement token-gating or other access controls
        return False, "Allowed"
        
    async def priority_inference_request(self, synapse: InferenceRequestSynapse) -> float:
        """Determine priority of inference request"""
        # Give higher priority to validators on this subnet
        if synapse.dendrite.hotkey in self.metagraph.validators:
            return 1.0
            
        # Medium priority to registered users
        if synapse.dendrite.hotkey in self.metagraph.hotkeys:
            return 0.5
            
        # Low priority to everyone else
        return 0.1
    async def forward(self):
        """Main validator loop"""
        try:
            miner_uids = get_miner_uids()
            
            # Check if we have miners to query
            if len(miner_uids) == 0:
                bt.logging.warning("No miners available to query")
                return
                
            bt.logging.info(f"Querying {len(miner_uids)} miners")
            
            # Query miners
            responses = await self.dendrite(
                # Send the query to all miners
                axons=[self.metagraph.axons[uid] for uid in miner_uids],
                synapse=PredictionSynapse(chain_name=ChainType.BASE.value),  # Fixed the chain_name
                deserialize=True,
            )

            bt.logging.info(f"Received {len(responses)} responses")
            
            # Store predictions
            await self._store_predictions(responses)
            
            # Calculate rewards based on historical performance
            rewards_dict = await self._calculate_rewards(miner_uids)

            # Convert dictionary to arrays for update_scores
            uids = np.array(list(rewards_dict.keys()))
            rewards = np.array(list(rewards_dict.values()))

            bt.logging.info(f"Calculated {len(rewards)} rewards")

            # Update scores with numpy arrays
            self.update_scores(rewards, uids)
            time.sleep(5)
            
        except Exception as e:
            bt.logging.error(f"Error in validator forward: {e}")


    async def _store_predictions(self, responses):
        """Store predictions for each miner
        
        Args:
            responses (Dict[int, any]): Responses from miners, keyed by UID
        """
        try:
            from datetime import datetime
            for uid, response in responses.items():
                # Check if response has the required fields
                if not hasattr(response, 'addresses') or not response.addresses:
                    bt.logging.warning(f"Invalid response from UID {uid}: missing addresses")
                    continue
                    
                # Convert PredictionSynapse to TokenPrediction
                prediction = TokenPrediction(
                    chain=ChainType(response.chain_name),
                    addresses=response.addresses,
                    pairAddresses=getattr(response, 'pairAddresses', []),
                    timestamp=datetime.now(),
                    confidence_scores={}  # Not available in synapse
                )
                
                # Store valid prediction
                await self.prediction_store.store_prediction(uid, prediction)
                bt.logging.debug(f"Stored prediction from UID {uid}")
                
        except Exception as e:
            bt.logging.error(f"Error storing predictions: {e}")


    async def _calculate_rewards(self, miner_uids: List[int]) -> Dict[int, float]:
        """Calculate rewards based on historical performance"""
        # Get historical predictions for all miners
        histories = []
        for uid in miner_uids:
            history = await self.prediction_store.get_miner_history(uid)
            if history:
                histories.extend(history.predictions)
        
        # Calculate performance metrics for all predictions at once
        if not histories:
            return {uid: 0 for uid in miner_uids}
        
        # Get normalized performances and their corresponding miner IDs
        performances, sorted_uids = await self.performance_calculator.calculate_performance(histories, miner_uids)
        
        # Map performances back to miner UIDs using the sorted IDs
        rewards = {}
        for uid, perf in zip(sorted_uids, performances):
            rewards[uid] = perf
        
        return rewards 