import time
import typing
import bittensor as bt
from typing import Dict
from ...base.types import ChainType, TokenPrediction
from .models.base_l2_model import BaseTokenModel

import telegraph.protocol as protocol
from base.miner import BaseMinerNeuron

class TelegraphMiner(BaseMinerNeuron):

    def __init__(self, config=None):
        super(TelegraphMiner, self).__init__(config=config)
        self.models: Dict[ChainType, BaseTokenModel] = {}


    async def forward(self, synapse: protocol.TelegraphProtocol) -> TokenPrediction:
        """Process the chain request and return token predictions"""
        try:
            chain = synapse.chain_name
            if chain not in self.models:
                raise ValueError(f"Unsupported chain: {chain}")

            model = self.models[chain]
            prediction = await model.predict(chain)

            return prediction

        except Exception as e:
            bt.logging.error(f"Error in miner forward: {e}")
            raise 

    async def blacklist(self, synapse: protocol.TelegraphProtocol) -> typing.Tuple[bool, str]:
        # TODO: define and implement blacklist
        return True, "Blacklisted"
    
    async def priority(self, synapse: protocol.TelegraphProtocol) -> float:
        # TODO: define and implement priority
        return 0.0
    
if __name__ == "__main__":
    with TelegraphMiner() as miner:
        while True:
            bt.logging.info("Miner running...", time.time())
            time.sleep(5)
