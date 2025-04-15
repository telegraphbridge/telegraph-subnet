import time
import typing
import bittensor as bt
from typing import Dict
from ...base.types import ChainType, TokenPrediction
from .models.base_l2_model import BaseTokenModel, LSTMTokenModel

import telegraph.protocol as protocol
from base.miner import BaseMinerNeuron
import os
import datetime

class TelegraphMiner(BaseMinerNeuron):

    def __init__(self, config=None):
        super(TelegraphMiner, self).__init__(config=config)
        self.models: Dict[str, BaseTokenModel] = {}
        self._initialize_models()

    def _initialize_models(self):
        """Initialize prediction models for supported chains"""
        # For Base L2 chain using the LSTM model
        try:
            # Use default mock LSTM model for now (no path provided)
            self.models[ChainType.BASE.value] = LSTMTokenModel()
            bt.logging.info(f"Initialized model for {ChainType.BASE.value}")
        except Exception as e:
            bt.logging.error(f"Failed to initialize model for {ChainType.BASE.value}: {e}")


    async def forward(self, synapse: protocol.PredictionSynapse) -> protocol.PredictionSynapse:
        """Process the chain request and return token predictions"""
        try:
            chain_name = synapse.chain_name
            
            # Check if we support this chain
            if chain_name not in self.models:
                synapse.addresses = []
                # Set empty pair addresses if the field exists
                if hasattr(synapse, 'pairAddresses'):
                    synapse.pairAddresses = []
                return synapse
            
            # Get corresponding chain enum
            try:
                chain_enum = ChainType(chain_name)
            except ValueError:
                bt.logging.warning(f"Invalid chain name: {chain_name}")
                synapse.addresses = []
                # Set empty pair addresses if the field exists
                if hasattr(synapse, 'pairAddresses'):
                    synapse.pairAddresses = []
                return synapse

            # Generate prediction using the model
            model = self.models[chain_name]
            prediction = await model.predict(chain_enum)
            
            # Set the responses in the synapse
            synapse.addresses = prediction.addresses
            
            # Safely set pair addresses if the field exists in both objects
            if hasattr(synapse, 'pairAddresses') and hasattr(prediction, 'pairAddresses'):
                synapse.pairAddresses = prediction.pairAddresses
            elif hasattr(synapse, 'pairAddresses'):
                synapse.pairAddresses = []
            
            return synapse

        except Exception as e:
            bt.logging.error(f"Error in miner forward: {e}")
            synapse.addresses = []
            # Set empty pair addresses if the field exists
            if hasattr(synapse, 'pairAddresses'):
                synapse.pairAddresses = []
            return synapse

    async def blacklist(self, synapse: protocol.PredictionSynapse) -> typing.Tuple[bool, str]:
        """Determine if the request should be blacklisted
        
        Args:
            synapse: The request synapse
        Returns:
            Tuple[bool, str]: (blacklisted, reason)
        """
        # Only accept requests from validators if force_validator_permit is True
        if self.config.blacklist.force_validator_permit:
            if synapse.dendrite.hotkey not in self.metagraph.validators:
                return True, "Not a validator hotkey"
        
        # Blacklist non-registered users if allow_non_registered is False
        if not self.config.blacklist.allow_non_registered:
            if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
                return True, "Hotkey not registered in metagraph"
        
        # Accept the request
        return False, "Allowed"

    async def priority(self, synapse: protocol.PredictionSynapse) -> float:
        """Return priority score for request
        
        Args:
            synapse: The request synapse
        Returns:
            float: Priority score (higher is more priority)
        """
        # Give higher priority to validators
        if synapse.dendrite.hotkey in self.metagraph.validators:
            return 1.0
        
        # Give medium priority to registered users
        if synapse.dendrite.hotkey in self.metagraph.hotkeys:
            return 0.5
            
        # Give lowest priority to unknown users
        return 0.1
    
if __name__ == "__main__":
    with TelegraphMiner() as miner:
        while True:
            bt.logging.info(f"Miner running... {time.time()}")
            time.sleep(5)