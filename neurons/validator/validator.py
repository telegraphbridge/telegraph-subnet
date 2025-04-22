import asyncio
import bittensor as bt
import time
import numpy as np
import random
from typing import Dict, List, Any, Optional
from base.validator import BaseValidatorNeuron
from telegraph.protocol import PredictionSynapse, InferenceRequestSynapse
from telegraph.registry import InferenceRegistry
from base.types import ChainType, TokenPrediction
from .storage.prediction_store import PredictionStore
from .utils.performance_calculator import PerformanceCalculator
from .utils.uids import get_miner_uids
from datetime import datetime

class TelegraphValidator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(TelegraphValidator, self).__init__(config=config)
        self.inference_registry = InferenceRegistry()
        import os
        os.makedirs("data/transactions", exist_ok=True)
        os.makedirs("data/predictions", exist_ok=True)
        bt.logging.info("Loading validator state")
        self.load_state()
        self.prediction_store = PredictionStore()
        self.performance_calculator = PerformanceCalculator()
        # Reinitialize inference registry
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

    def blacklist_inference_request(self, synapse: InferenceRequestSynapse) -> tuple[bool, str]:
        # For MVP, allow all requests.
        return False, "ok"

    async def priority_inference_request(self, synapse: InferenceRequestSynapse) -> float:
        # Metagraph doesn't have 'validators' attribute, use hotkeys instead
        if synapse.dendrite.hotkey in self.metagraph.hotkeys:
            return 1.0
        return 0.5

    def process_inference_request_sync(self, syn: InferenceRequestSynapse) -> InferenceRequestSynapse:
        return asyncio.get_event_loop().run_until_complete(self.process_inference_request(syn))

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
        bt.logging.info(f"Routing to subnet {target_netuid} with code={inference_code}")
        subtensor = bt.subtensor(network=self.config.subtensor.network)
        m = subtensor.metagraph(netuid=target_netuid)
        m.sync(subtensor=subtensor)
        # Use all nodes (or filter using get_miner_uids) since m.validators is unavailable.
        peers = list(range(m.n))
        stakes = np.array([m.S[uid] for uid in peers], dtype=float)
        if stakes.sum() > 0:
            idx = np.random.choice(len(peers), p=stakes/stakes.sum())
        else:
            idx = random.randrange(len(peers))
        uid = peers[idx]
        ax = m.axons[uid]
        bt.logging.info(f"Selected peer uid={uid}, hotkey={m.hotkeys[uid]}")
        req = InferenceRequestSynapse(inference_code=inference_code, data=data)
        resp = await bt.dendrite(wallet=self.wallet)(
            axons=[ax],
            synapse=req,
            deserialize=True,
            timeout=30.0
        )
        syn = resp[0] if isinstance(resp, list) else resp
        if syn.error:
            return {"error": syn.error}
        return getattr(syn, "response", None)

    async def forward(self):
        """Main validator loop: Queries miners, processes responses, calculates rewards, and updates scores."""
        try:
            # 1. Get available miner UIDs from the metagraph
            miner_uids = get_miner_uids(self.metagraph) # Pass metagraph
            if not miner_uids:
                bt.logging.warning("No available miners found in the metagraph.")
                await asyncio.sleep(60) # Wait before retrying
                return

            # Filter UIDs to ensure they are within the current metagraph size (n)
            valid_miner_uids = [uid for uid in miner_uids if uid < self.metagraph.n]
            if not valid_miner_uids:
                bt.logging.warning("No valid miners available in the current metagraph slice (UIDs might be out of range).")
                await asyncio.sleep(60) # Wait before retrying
                return

            bt.logging.info(f"Querying {len(valid_miner_uids)} miners: {valid_miner_uids}")

            # 2. Prepare and send synapse query to miners
            chain_type = ChainType.BASE
            synapse = PredictionSynapse(chain_name=chain_type.value)

            responses = await self._dendrite( # Use the validator's dendrite instance
                axons=[self.metagraph.axons[uid] for uid in valid_miner_uids],
                synapse=synapse,
                deserialize=True, # Deserialize responses into PredictionSynapse objects
                timeout=20.0      # Set a timeout for responses
            )
            bt.logging.info(f"Received {len(responses)} raw responses.")

            # 3. Process responses and map to UIDs for the CURRENT round
            valid_responses: List[PredictionSynapse] = []
            current_uids: List[int] = [] # UIDs corresponding to valid_responses

            for i, resp in enumerate(responses):
                uid = valid_miner_uids[i] # Get UID based on original query order

                # Check if the dendrite call itself was successful AND the response synapse has addresses
                # resp.is_success checks the TerminalInfo status code (e.g., 200 OK)
                if resp.is_success and getattr(resp, 'addresses', None):
                    valid_responses.append(resp)
                    current_uids.append(uid)
                    # bt.logging.trace(f"Received valid response from UID {uid}")
                else:
                    # Log detailed failure information from TerminalInfo
                    status_code = resp.dendrite.status_code if hasattr(resp, 'dendrite') else 'N/A'
                    status_message = resp.dendrite.status_message if hasattr(resp, 'dendrite') else 'N/A'
                    axon_ip = resp.axon.ip if hasattr(resp, 'axon') else 'N/A'
                    reason = f"Axon: {axon_ip}, Code: {status_code}, Message: {status_message}"
                    if not getattr(resp, 'addresses', None) and resp.is_success:
                         reason += " (Missing addresses attribute)" # Add specific reason if success but no addresses
                    bt.logging.warning(f"Invalid or failed response from UID {uid}. Reason: {reason}")

            # 4. Check if any valid responses were received
            if not valid_responses:
                bt.logging.warning("No valid responses received from miners in this round.")
                await asyncio.sleep(60) # Wait before next cycle
                return

            bt.logging.info(f"Processing {len(valid_responses)} valid responses from UIDs: {current_uids}")

            # 5. Store valid predictions from the current round
            # Pass the UIDs corresponding to the valid responses
            await self._store_predictions(valid_responses, current_uids)

            # 6. Calculate rewards based ONLY on the current round's valid responses
            # Pass the valid responses and their corresponding UIDs
            rewards_dict = await self._calculate_rewards(valid_responses, current_uids)

            # 7. Prepare rewards and UIDs for score update
            uids_to_update = np.array(list(rewards_dict.keys()), dtype=int)
            rewards_to_update = np.array(list(rewards_dict.values()), dtype=float)

            if len(rewards_to_update) == 0:
                bt.logging.warning("No rewards calculated for the current round (performance might be zero or calculation failed).")
                # Still proceed to potentially save state, but don't update scores if empty
            else:
                bt.logging.info(f"Calculated rewards for {len(rewards_to_update)} miners: {rewards_dict}")
                # Update scores in the base class
                self.update_scores(rewards_to_update, uids_to_update)

            # 8. Save validator state periodically (consider moving if needed)
            self.save_state()

        except Exception as e:
            # Catch any unexpected errors in the main loop
            bt.logging.error(f"Error in validator forward loop: {str(e)}")
            import traceback
            bt.logging.error(traceback.format_exc()) # Log the full traceback for debugging
            # Optional: Implement specific error handling or recovery logic here
            await asyncio.sleep(60) # Wait before retrying after a major error


    async def _store_predictions(self, responses: List[PredictionSynapse], uids: List[int]):
        """Store predictions for each miner from the current round."""
        from datetime import datetime
        if len(responses) != len(uids):
             bt.logging.error(f"Mismatch storing predictions: {len(responses)} responses, {len(uids)} UIDs")
             return # Avoid processing if lists don't match

        for i, response in enumerate(responses):
            uid = uids[i]
            try:
                # Ensure addresses exist before creating TokenPrediction
                if not getattr(response, "addresses", None):
                    bt.logging.warning(f"Skipping storing prediction for UID {uid}: Missing addresses.")
                    continue

                confidence_scores = response.confidence_scores if hasattr(response, "confidence_scores") else {}
                chain_name = response.chain_name if hasattr(response, "chain_name") and response.chain_name else ChainType.BASE.value

                prediction = TokenPrediction(
                    chain=ChainType(chain_name),
                    addresses=response.addresses,
                    pairAddresses=getattr(response, 'pairAddresses', []),
                    timestamp=datetime.now(), # Use current time for storage timestamp
                    confidence_scores=confidence_scores
                )
                await self.prediction_store.store_prediction(uid, prediction)
                bt.logging.debug(f"Stored prediction from UID {uid} for current round.")
            except Exception as e:
                bt.logging.error(f"Error storing prediction for UID {uid}: {str(e)}")
                import traceback
                bt.logging.debug(traceback.format_exc()) # Keep debug for detailed trace

    async def _calculate_rewards(self, responses: List[PredictionSynapse], uids: List[int]) -> Dict[int, float]:
        """Calculate rewards based ONLY on the current round's responses."""
        if len(responses) != len(uids):
             bt.logging.error(f"Mismatch calculating rewards: {len(responses)} responses, {len(uids)} UIDs")
             return {} # Return empty dict if lists don't match

        rewards = {}
        # Use the responses directly, assuming they are TokenPrediction compatible or extract data
        current_predictions = []
        valid_uids_for_perf = []

        for i, response in enumerate(responses):
             uid = uids[i]
             # Create TokenPrediction objects from responses for performance calculation
             if getattr(response, "addresses", None):
                 chain_name = response.chain_name if hasattr(response, "chain_name") and response.chain_name else ChainType.BASE.value
                 current_predictions.append(TokenPrediction(
                     chain=ChainType(chain_name),
                     addresses=response.addresses,
                     pairAddresses=getattr(response, 'pairAddresses', []),
                     timestamp=datetime.now(), # Timestamp for evaluation context
                     confidence_scores=response.confidence_scores if hasattr(response, "confidence_scores") else {}
                 ))
                 valid_uids_for_perf.append(uid)
             else:
                 bt.logging.warning(f"Skipping reward calculation for UID {uid}: Invalid response.")


        if not current_predictions:
            bt.logging.warning("No valid predictions from current round to calculate performance.")
            return {}

        # Calculate performance scores for the current predictions
        # calculate_performance now expects predictions and corresponding UIDs
        performance_uids, performance_scores = await self.performance_calculator.calculate_performance(
            current_predictions,
            valid_uids_for_perf # Pass the UIDs corresponding to current_predictions
        )

        if len(performance_uids) == 0:
            bt.logging.warning("Performance calculation returned no results.")
            return {}

        # Normalize scores
        normalized_scores = self.performance_calculator.normalize_performance(performance_scores)

        # Map normalized scores back to UIDs
        for i, uid in enumerate(performance_uids):
            rewards[uid] = normalized_scores[i]

        return rewards
