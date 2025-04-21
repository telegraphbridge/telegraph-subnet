import asyncio
import bittensor as bt
import time
import numpy as np
from typing import Dict, List, Any, Optional
from base.validator import BaseValidatorNeuron
from telegraph.protocol import PredictionSynapse, InferenceRequestSynapse
from telegraph.registry import InferenceRegistry
from base.types import ChainType, TokenPrediction
from .storage.prediction_store import PredictionStore
from .utils.performance_calculator import PerformanceCalculator
from .utils.uids import get_miner_uids
import random

class TelegraphValidator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(TelegraphValidator, self).__init__(config=config)
        self.inference_registry = InferenceRegistry()

        import os
        os.makedirs("data/transactions", exist_ok=True)
        os.makedirs("data/predictions", exist_ok=True)

        bt.logging.info("load_state()")
        self.load_state()

        self.prediction_store = PredictionStore()
        self.performance_calculator = PerformanceCalculator()
        
        # Initialize inference registry
        self.inference_registry = InferenceRegistry()
        self._dendrite = bt.dendrite(wallet=self.wallet)
        if not self.config.neuron.axon_off:
            self.setup_inference_handlers()
    
    def setup_inference_handlers(self):
        def handle(syn: InferenceRequestSynapse) -> InferenceRequestSynapse:
            return self.process_inference_request_sync(syn)
        self.axon.attach(
            forward_fn=handle,
            # blacklist_fn=self.blacklist_inference_request,
            priority_fn=self.priority_inference_request
        )
        bt.logging.info("Attached inference handler")

    def process_inference_request_sync(self, syn: InferenceRequestSynapse) -> InferenceRequestSynapse:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.process_inference_request(syn))

    async def process_inference_request(self, syn: InferenceRequestSynapse) -> InferenceRequestSynapse:
        bt.logging.info(f"Incoming cross‑subnet request: {syn.inference_code}")
        if not self.inference_registry.is_valid_code(syn.inference_code):
            syn.error = f"Unknown code {syn.inference_code}"
            return syn

        syn.response = await self.route_inference_request(syn.inference_code, syn.data)
        return syn


    async def route_inference_request(self, inference_code: str, data: Any) -> Any:
        target_netuid = self.inference_registry.get_netuid(inference_code)
        if target_netuid is None:
            return {"error": f"Invalid code {inference_code}"}

        bt.logging.info(f"Routing to subnet {target_netuid} code={inference_code}")

        # fetch & sync that subnet's metagraph
        subtensor = bt.subtensor(network=self.config.subtensor.network)
        m = subtensor.metagraph(netuid=target_netuid)
        m.sync(subtensor=subtensor)

        # pick a validator uid
        validators = m.validators if len(m.validators)>0 else list(range(m.n))
        stakes = np.array([m.S[uid] for uid in validators], dtype=float)
        if stakes.sum() > 0:
            idx = np.random.choice(len(validators), p=stakes/stakes.sum())
        else:
            idx = random.randrange(len(validators))
        uid = validators[idx]
        ax = m.axons[uid]
        bt.logging.info(f"Selected peer uid={uid}, hotkey={ax.hotkey}")

        # build a fresh synapse to send
        req = InferenceRequestSynapse(inference_code=inference_code, data=data)
        # send and await
        resp = await bt.dendrite(wallet=self.wallet)(
            axons=[ax],
            synapse=req,
            deserialize=True,
            timeout=30.0
        )
        # resp is a list
        syn = resp[0] if isinstance(resp, list) else resp
        if syn.error:
            return {"error": syn.error}
        # consumer subnets should have populated syn.response
        return getattr(syn, "response", None)
    def blacklist_inference_request(self, synapse: InferenceRequestSynapse) -> tuple[bool, str]:
        """Determine if an inference request should be blacklisted"""
        # For MVP, we don't blacklist inference requests
        # In production, implement token-gating or other access controls
        return False, "ok"
        
    def priority_inference_request(self, synapse: InferenceRequestSynapse) -> float:
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
            # Get available miners
            miner_uids = get_miner_uids()
            
            # Filter miner UIDs to ensure they exist in metagraph
            valid_miner_uids = []
            for uid in miner_uids:
                if uid < len(self.metagraph.axons):
                    valid_miner_uids.append(uid)
                else:
                    bt.logging.warning(f"Miner UID {uid} out of range (metagraph size: {len(self.metagraph.axons)})")
            
            # Check if we have miners to query
            if len(valid_miner_uids) == 0:
                bt.logging.warning("No valid miners available to query")
                return
                
            bt.logging.info(f"Querying {len(valid_miner_uids)} miners")
            
            # Query only BASE chain for now
            chain_type = ChainType.BASE
            
            # Create PredictionSynapse for the query
            synapse = PredictionSynapse(chain_name=chain_type.value)
            
            # Query miners
            responses = await self.dendrite(
                axons=[self.metagraph.axons[uid] for uid in valid_miner_uids],
                synapse=synapse,
                deserialize=True,
                timeout=20.0  # Increased timeout for production
            )

            bt.logging.info(f"Received {len(responses)} responses")
            
            # Store predictions
            await self._store_predictions(responses)
            
            # Calculate rewards based on historical performance
            rewards_dict = await self._calculate_rewards(valid_miner_uids)

            # Convert dictionary to arrays for update_scores
            uids = np.array(list(rewards_dict.keys()))
            rewards = np.array(list(rewards_dict.values()))

            # Handle empty rewards case
            if len(rewards) == 0:
                bt.logging.warning("No rewards to update")
                return

            bt.logging.info(f"Calculated {len(rewards)} rewards")

            # Update scores with numpy arrays
            self.update_scores(rewards, uids)
            
        except Exception as e:
            bt.logging.error(f"Error in validator forward: {str(e)}")
            import traceback
            bt.logging.debug(traceback.format_exc())

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
                    
                # Extract confidence scores if available
                confidence_scores = {}
                if hasattr(response, 'confidence_scores') and isinstance(response.confidence_scores, dict):
                    confidence_scores = response.confidence_scores
                
                # Ensure we have a valid chain_name
                chain_name = ChainType.BASE.value
                if hasattr(response, 'chain_name') and response.chain_name:
                    chain_name = response.chain_name
                    
                # Convert PredictionSynapse to TokenPrediction
                prediction = TokenPrediction(
                    chain=ChainType(chain_name),
                    addresses=response.addresses,
                    pairAddresses=getattr(response, 'pairAddresses', []),
                    timestamp=datetime.now(),
                    confidence_scores=confidence_scores
                )
                
                # Store valid prediction
                await self.prediction_store.store_prediction(uid, prediction)
                bt.logging.debug(f"Stored prediction from UID {uid}")
                    
        except Exception as e:
            bt.logging.error(f"Error storing predictions: {str(e)}")
            import traceback
            bt.logging.debug(traceback.format_exc())

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