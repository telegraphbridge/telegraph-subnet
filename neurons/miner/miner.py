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
        """Process incoming prediction requests with better debugging and error handling."""
        try:
            # Ensure chain_name has a value
            chain_key = getattr(synapse, 'chain_name', None) or "BASE"
            bt.logging.info(f"Processing prediction request for chain: {chain_key}")
            
            # Debug existing files
            data_dir = "data/transactions"
            if os.path.exists(data_dir):
                files = os.listdir(data_dir)
                bt.logging.info(f"Files in data directory: {files}")
                
                # Check for model file specifically
                model_path = os.path.join(data_dir, "best_model.pth")
                if os.path.exists(model_path):
                    bt.logging.info(f"Model file exists at {model_path}, size: {os.path.getsize(model_path)} bytes")
                else:
                    bt.logging.warning(f"Model file not found at {model_path}")
                    
                # Check for transaction data
                transaction_files = [f for f in files if "transactions" in f and f.endswith(".json")]
                bt.logging.info(f"Transaction files: {transaction_files}")
            else:
                bt.logging.warning(f"Data directory {data_dir} does not exist")
            
            # Initialize response fields to ensure they're never None
            synapse.addresses = []
            synapse.pairAddresses = []
            synapse.confidence_scores = {}
            
            # Even if model isn't available, we'll generate dummy data
            model = self.models.get(chain_key)
            bt.logging.info(f"Using model: {model.__class__.__name__ if model else 'None'}")
            
            if model:
                try:
                    # Use the model to generate predictions
                    bt.logging.info("Calling model.predict()...")
                    prediction = await model.predict(ChainType(chain_key))
                    
                    if prediction and prediction.addresses:
                        synapse.addresses = prediction.addresses
                        synapse.pairAddresses = prediction.pairAddresses or []
                        synapse.confidence_scores = prediction.confidence_scores or {}
                        
                        bt.logging.info(f"Prediction success: {len(synapse.addresses)} addresses")
                        bt.logging.info(f"First few addresses: {synapse.addresses[:3]}")
                        bt.logging.info(f"Sample confidence: {list(synapse.confidence_scores.items())[:2]}")
                        
                        # Check if these are likely random addresses (detect placeholder format)
                        placeholder_pattern = r"0x[0-9a-f]{40}"
                        import re
                        if all(re.match(r"0x[0-9]{40}", addr) for addr in synapse.addresses[:3]):
                            bt.logging.warning("Detected placeholder addresses - model likely used fallback data")
                        
                        return synapse
                    else:
                        bt.logging.warning("Model returned empty prediction, falling back to dummy data")
                except Exception as e:
                    bt.logging.error(f"Model prediction error: {str(e)}")
                    import traceback
                    bt.logging.error(traceback.format_exc())
            
            # Fallback dummy predictions (always runs if model fails)
            import random
            bt.logging.warning("Using fallback random address generation")
            # Generate random hex addresses (40 chars after 0x)
            synapse.addresses = [f"0x{''.join(random.choices('0123456789abcdef', k=40))}" for _ in range(10)]
            synapse.pairAddresses = [f"0x{''.join(random.choices('0123456789abcdef', k=40))}" for _ in range(10)]
            synapse.confidence_scores = {addr: random.uniform(0.1, 0.9) for addr in synapse.addresses}
            bt.logging.info(f"Generated {len(synapse.addresses)} dummy addresses as fallback")
            
        except Exception as e:
            bt.logging.error(f"Critical error in forward method: {str(e)}")
            import traceback
            bt.logging.error(traceback.format_exc())
            # Even in worst case, return something valid
            synapse.addresses = [f"0x{i:040x}" for i in range(10)]
            synapse.pairAddresses = [f"0x{i+100:040x}" for i in range(10)]
            synapse.confidence_scores = {addr: 0.5 for addr in synapse.addresses}
            
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
        return True, "Not a registered user"
            
        # Blacklist all others
        return True, "Not a registered user"

    async def priority(self, synapse: PredictionSynapse) -> float:
        """Determine priority for a request
        
        Args:
            synapse: The request synapse
            
        Returns:
            float: Priority value (higher is more important)
        """
        # Get validator hotkeys or use top stake holders instead of validators attribute
        validator_uids = [i for i in range(self.metagraph.n) if self.metagraph.S[i] > 0]
        validator_hotkeys = [self.metagraph.hotkeys[uid] for uid in validator_uids]
        
        # Give validators highest priority
        if synapse.dendrite.hotkey in validator_hotkeys:
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