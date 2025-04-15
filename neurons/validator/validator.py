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