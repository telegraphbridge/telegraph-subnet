import os
import time
import bittensor as bt
import asyncio
from typing import Dict, List, Tuple, Any
from base.types import ChainType, TokenPrediction
from .models.base_l2_model import BaseTokenModel, LSTMTokenModel
from base.miner import BaseMinerNeuron
from telegraph.protocol import PredictionSynapse, InferenceRequestSynapse

class TelegraphMiner(BaseMinerNeuron):
    """Telegraph miner implementation for token price prediction"""

    def __init__(self, config=None):
        super(TelegraphMiner, self).__init__(config=config)
        
        # Create necessary directories
        os.makedirs("data/transactions", exist_ok=True)
        
        # Initialize token prediction models
        self.models = {}
        self._initialize_models()
        
        # Set up axon handlers
        if self.axon:
            self.axon.attach(
                forward_fn=self.forward,
                blacklist_fn=self.blacklist,
                priority_fn=self.priority
            )
    
    def _initialize_models(self):
        """Initialize prediction models for supported chains"""
        try:
            # For now, only initialize BASE L2 model
            model_path = os.path.join("data/transactions", "best_model.pth")
            
            # Check if we have a trained model
            if os.path.exists(model_path):
                bt.logging.info(f"Using trained model from {model_path}")
                self.models[ChainType.BASE.value] = LSTMTokenModel(model_path=model_path)
            else:
                bt.logging.warning(f"No trained model found at {model_path}, using default initialization")
                self.models[ChainType.BASE.value] = LSTMTokenModel()
                
            bt.logging.info(f"Initialized model for {ChainType.BASE.value}")
        except Exception as e:
            bt.logging.error(f"Failed to initialize model: {str(e)}")
            # Use default model as fallback
            self.models[ChainType.BASE.value] = LSTMTokenModel()
    
    async def forward(self, synapse: PredictionSynapse) -> PredictionSynapse:
        """Process a request for token predictions
        
        Args:
            synapse: The request synapse containing the chain name
            
        Returns:
            PredictionSynapse with token predictions
        """
        try:
            # Validate chain name
            if not synapse.chain_name:
                bt.logging.warning("No chain specified in request")
                synapse.chain_name = ChainType.BASE.value
            
            # Standardize the chain name to match our model keys
            chain_key = synapse.chain_name
            
            # Check if we support this chain
            if chain_key not in self.models:
                bt.logging.warning(f"Unsupported chain: {chain_key}")
                synapse.addresses = [f"0x{i:040x}" for i in range(10)]
                return synapse
            
            # Get chain type from name
            try:
                chain = ChainType(chain_key)
            except ValueError:
                bt.logging.warning(f"Invalid chain name: {chain_key}, using BASE")
                chain = ChainType.BASE
            
            # Get predictions from model
            prediction = await self.models[chain_key].predict(chain)
            
            # Copy predictions to synapse
            synapse.addresses = prediction.addresses
            synapse.pairAddresses = prediction.pairAddresses
            synapse.confidence_scores = prediction.confidence_scores
            
            return synapse
            
        except Exception as e:
            bt.logging.error(f"Error in forward: {str(e)}")
            # Return empty prediction in case of error
            synapse.addresses = [f"0x{i:040x}" for i in range(10)]
            return synapse
    
    async def blacklist(self, synapse: PredictionSynapse) -> Tuple[bool, str]:
        """Determine if a request should be blacklisted
        
        Args:
            synapse: The request synapse
            
        Returns:
            Tuple[bool, str]: (blacklisted, reason)
        """
        # For BASE chain queries, everyone is allowed
        if synapse.chain_name == ChainType.BASE.value:
            return False, "Allowed"
            
        # Otherwise, only allow registered validators
        if synapse.dendrite.hotkey in self.metagraph.hotkeys:
            return False, "Registered user"
            
        # Blacklist all others
        return True, "Not a registered user"
    
    async def priority(self, synapse: PredictionSynapse) -> float:
        """Determine priority for a request
        
        Args:
            synapse: The request synapse
            
        Returns:
            float: Priority value (higher is more important)
        """
        # Give validators highest priority
        if synapse.dendrite.hotkey in self.metagraph.validators:
            return 1.0
            
        # Registered users get medium priority
        if synapse.dendrite.hotkey in self.metagraph.hotkeys:
            return 0.5
            
        # Everyone else gets lowest priority
        return 0.1

if __name__ == "__main__":
    # Run the miner
    with TelegraphMiner() as miner:
        while True:
            print(f"Miner running, block: {miner.block}")
            time.sleep(60)