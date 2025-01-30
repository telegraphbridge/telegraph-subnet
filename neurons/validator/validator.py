import bittensor as bt
import time
import numpy as np
from typing import Dict, List
from base.validator import BaseValidatorNeuron
from telegraph.protocol import PredictionSynapse
from ...base.types import ChainType, TokenPrediction
from .storage.prediction_store import PredictionStore
from .utils.performance_calculator import PerformanceCalculator
from .utils.uids import get_miner_uids


class TelegraphValidator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(TelegraphValidator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()

        self.prediction_store = PredictionStore()
        self.performance_calculator = PerformanceCalculator()


    async def forward(self):
        """Main validator loop"""
        try:
            miner_uids = get_miner_uids()
            # Query miners
            responses = await self.dendrite(
                # Send the query to all miners
                # TODO: send to all miners
                axons =[self.metagraph.axons[uid] for uid in miner_uids],
                synapse=PredictionSynapse(chain_name=PredictionSynapse.chain_name),
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


    async def _store_predictions(self, responses: Dict[TokenPrediction], miner_uids: List[int]):
        """Store predictions for each miner"""
        for uid in miner_uids:
            await self.prediction_store.store_prediction(uid, responses[uid])


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