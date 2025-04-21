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
        """Main validator loop."""
        try:
            miner_uids = get_miner_uids()
            valid_miner_uids = [uid for uid in miner_uids if uid < len(self.metagraph.axons)]
            if len(valid_miner_uids) == 0:
                bt.logging.warning("No valid miners available to query")
                return
            bt.logging.info(f"Querying {len(valid_miner_uids)} miners")
            chain_type = ChainType.BASE
            synapse = PredictionSynapse(chain_name=chain_type.value)
            responses = await self.dendrite(
                axons=[self.metagraph.axons[uid] for uid in valid_miner_uids],
                synapse=synapse,
                deserialize=True,
                timeout=20.0
            )
            bt.logging.info(f"Received {len(responses)} responses")
            await self._store_predictions(responses)
            rewards_dict = await self._calculate_rewards(valid_miner_uids)
            uids = np.array(list(rewards_dict.keys()))
            rewards = np.array(list(rewards_dict.values()))
            if len(rewards) == 0:
                bt.logging.warning("No rewards to update")
                return
            bt.logging.info(f"Calculated rewards for {len(rewards)} miners")
            self.update_scores(rewards, uids)
        except Exception as e:
            bt.logging.error(f"Error in validator forward: {str(e)}")
            import traceback
            bt.logging.debug(traceback.format_exc())

    async def _store_predictions(self, responses):
        """Store predictions for each miner from a list of responses."""
        from datetime import datetime
        try:
            for idx, response in enumerate(responses):
                uid = getattr(response, "uid", idx)
                if not getattr(response, "addresses", None):
                    bt.logging.warning(f"Invalid response (index {idx}) missing addresses")
                    continue
                confidence_scores = response.confidence_scores if hasattr(response, "confidence_scores") else {}
                chain_name = response.chain_name if hasattr(response, "chain_name") and response.chain_name else ChainType.BASE.value
                # Default pairAddresses to empty list if missing.
                prediction = TokenPrediction(
                    chain=ChainType(chain_name),
                    addresses=response.addresses,
                    pairAddresses=getattr(response, 'pairAddresses', []),
                    timestamp=datetime.now(),
                    confidence_scores=confidence_scores
                )
                await self.prediction_store.store_prediction(uid, prediction)
                bt.logging.debug(f"Stored prediction from UID {uid}")
        except Exception as e:
            bt.logging.error(f"Error storing predictions: {str(e)}")
            import traceback
            bt.logging.debug(traceback.format_exc())
    async def _calculate_rewards(self, miner_uids: List[int]) -> Dict[int, float]:
        """Calculate rewards based on historical performance."""
        histories = []
        for uid in miner_uids:
            history = await self.prediction_store.get_miner_history(uid)
            if history:
                histories.extend(history.predictions)
        if not histories:
            return {uid: 0 for uid in miner_uids}
        performances, sorted_uids = await self.performance_calculator.calculate_performance(histories, miner_uids)
        rewards = {uid: perf for uid, perf in zip(sorted_uids, performances)}
        return rewards