import os
import json
import time
import bittensor as bt
import numpy as np
from typing import List, Dict, Any, Tuple  # Added Tuple here
from datetime import datetime, timedelta
from .liquidity_checker import LiquidityChecker
from base.types import ChainType, TokenPrediction

class PerformanceCalculator:
    def __init__(self, storage_path: str = "storage/liquidity_data.json"):
        """Initialize performance calculator with persistent storage
        
        Args:
            storage_path: Path to store liquidity data
        """
        self.liquidity_checker = LiquidityChecker()
        self.storage_path = storage_path
        self.min_evaluation_time = 3600  # 1 hour minimum between initial and current check
        
        # Load cached values from storage
        self.initial_liquidity_cache: Dict[str, Dict[str, Any]] = self._load_from_storage()
        
        # Ensure storage directory exists
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

    def _load_from_storage(self) -> Dict[str, Dict[str, Any]]:
        """Load initial liquidity values from storage"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            bt.logging.warning(f"Failed to load liquidity data: {e}")
        return {}

    def _save_to_storage(self):
        """Save initial liquidity values to storage"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.initial_liquidity_cache, f)
        except Exception as e:
            bt.logging.warning(f"Failed to save liquidity data: {e}")

    async def get_or_cache_initial_liquidity(
        self, 
        chain: ChainType, 
        token_addr: str, 
        pair_addr: str
    ) -> Dict[str, Any]:
        """Get or cache initial liquidity for a token pair.
        
        Returns:
            Dict with 'value' and 'timestamp' fields.
        """
        cache_key = f"{chain.value}:{token_addr}:{pair_addr}"
        
        # Check if we have this in cache
        if cache_key in self.initial_liquidity_cache:
            bt.logging.debug(f"Using cached liquidity data for {cache_key}")
            return self.initial_liquidity_cache[cache_key]
            
        # Not in cache; fetch initial data.
        try:
            metrics = await self.liquidity_checker.check_token_liquidity(
                chain, token_addr, pair_addr
            )
        except Exception as e:
            bt.logging.error(f"Error getting initial liquidity: {e}")
            # Fallback: use default liquidity (e.g. 1000)
            metrics = type("DummyMetrics", (), {"current_liquidity": 1000})
            
        # Store with timestamp.
        self.initial_liquidity_cache[cache_key] = {
            "value": metrics.current_liquidity,
            "timestamp": time.time()
        }
        # Save to persistent storage.
        self._save_to_storage()
        return self.initial_liquidity_cache[cache_key]
    
    async def calculate_token_performance(self, prediction: TokenPrediction) -> float:
        """Calculate average percentage liquidity increase for a single prediction"""
        try:
            total_percentage_change = 0.0
            valid_tokens = 0
            
            bt.logging.info(f"Evaluating {len(prediction.addresses or [])} tokens for {prediction.chain.value}")
            
            for i, token_addr in enumerate(prediction.addresses or []):
                try:
                    # Skip if index is out of range or pair is None
                    if i >= len(prediction.pairAddresses or []) or not prediction.pairAddresses[i]:
                        continue
                    
                    pair_addr = prediction.pairAddresses[i]
                    
                    # Get initial liquidity with timestamp (cached after first check)
                    initial_data = await self.get_or_cache_initial_liquidity(
                        prediction.chain,
                        token_addr,
                        pair_addr
                    )
                    
                    initial_liquidity = initial_data["value"]
                    initial_timestamp = initial_data["timestamp"]
                    
                    # Skip tokens with zero initial liquidity
                    if initial_liquidity == 0:
                        bt.logging.debug(f"Skipping token {token_addr}, zero initial liquidity")
                        continue
                        
                    # Skip if not enough time has passed since initial check
                    current_time = time.time()
                    if current_time - initial_timestamp < self.min_evaluation_time:
                        bt.logging.debug(f"Skipping token {token_addr}, not enough time elapsed since initial check")
                        continue
                    
                    # Get current liquidity
                    try:
                        current_metrics = await self.liquidity_checker.check_token_liquidity(
                            prediction.chain,
                            token_addr,
                            pair_addr
                        )
                        
                        # Calculate percentage change
                        percentage_change = ((current_metrics.current_liquidity - initial_liquidity) 
                                         / initial_liquidity) * 100
                        
                        total_percentage_change += percentage_change
                        valid_tokens += 1
                        
                        bt.logging.debug(f"Token {token_addr} liquidity change: {percentage_change:.2f}%")
                        
                    except Exception as e:
                        bt.logging.warning(f"Error checking current liquidity for {token_addr}: {e}")
                        continue
                
                except Exception as e:
                    bt.logging.warning(f"Error processing token {token_addr}: {e}")
                    continue
                
            # Return average percentage change across all valid tokens
            return total_percentage_change / valid_tokens if valid_tokens > 0 else 0.0
            
        except Exception as e:
            bt.logging.error(f"Failed to calculate token performance: {e}")
            return 0.0

    async def calculate_performance(
        self, 
        predictions: List[TokenPrediction], 
        miner_ids: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate normalized performance scores for all predictions
        
        Args:
            predictions: List of token predictions to evaluate
            miner_ids: List of miner UIDs corresponding to each prediction
            
        Returns:
            Tuple of (miner_ids, performance_scores) as numpy arrays
        """
        # Validate input
        if not predictions or len(predictions) == 0:
            bt.logging.warning("No predictions provided for performance calculation")
            return np.array([]), np.array([])
            
        # Check if number of predictions matches number of miner IDs
        if len(predictions) != len(miner_ids):
            bt.logging.error(f"Mismatch between predictions ({len(predictions)}) and miner IDs ({len(miner_ids)})")
            
        # Create a mapping between prediction and miner ID to handle potential mismatch
        prediction_map = {}
        for i, pred in enumerate(predictions):
            if not pred.addresses:
                continue
            # Create a unique key for each prediction based on its attributes
            key = f"{pred.chain.value}-{','.join(pred.addresses or [])}-{pred.timestamp.isoformat()}"
            # If we have this miner ID in the list, associate it with this prediction
            if i < len(miner_ids):
                prediction_map[key] = miner_ids[i]
        
        # Calculate average liquidity change for each prediction
        performances = []
        valid_miner_ids = []
        
        bt.logging.info(f"Calculating performance for {len(prediction_map)} predictions")
        
        for i, pred in enumerate(predictions):
            try:
                # Skip predictions without addresses
                if not pred.addresses or len(pred.addresses) == 0:
                    bt.logging.warning(f"Skipping prediction #{i} - no addresses")
                    continue
                    
                # Create key to lookup miner ID
                key = f"{pred.chain.value}-{','.join(pred.addresses or [])}-{pred.timestamp.isoformat()}"
                
                # Skip if no matching miner ID found
                if key not in prediction_map:
                    bt.logging.warning(f"Skipping prediction #{i} - no matching miner ID")
                    continue
                    
                miner_id = prediction_map[key]
                avg_performance = await self.calculate_token_performance(pred)
                performances.append(avg_performance)
                valid_miner_ids.append(miner_id)
                
                bt.logging.info(f"Miner {miner_id} performance: {avg_performance:.2f}%")
                
            except Exception as e:
                bt.logging.error(f"Error calculating performance for prediction #{i}: {e}")
                continue
        
        # Convert to numpy arrays
        performances_array = np.array(performances) if performances else np.array([])
        valid_miner_ids_array = np.array(valid_miner_ids) if valid_miner_ids else np.array([])
        
        # Log performance summary
        avg_performance = np.mean(performances_array) if len(performances_array) > 0 else 0.0
        bt.logging.info(f"Average performance across {len(performances_array)} tokens: {avg_performance:.2f}%")
        
        # Return both arrays for reward allocation
        return valid_miner_ids_array, performances_array
    
    def normalize_performance(self, performance_scores: np.ndarray) -> np.ndarray:
        """
        Normalize performance scores to a 0.1-1.0 range according to the whitepaper.
        Ensures that scores are always positive and proportionate.
        
        Args:
            performance_scores: Raw performance percentage scores
            
        Returns:
            Normalized scores between 0.1 and 1.0
        """
        # If we have no valid scores, return an empty array
        if len(performance_scores) == 0:
            return np.array([])
            
        # Find min and max, ensuring we have at least some range
        min_score = min(np.min(performance_scores), 0)
        max_score = max(np.max(performance_scores), min_score + 1)
        
        # Normalize to 0-1 range first
        normalized = (performance_scores - min_score) / (max_score - min_score) 
        
        # Scale to 0.1-1.0 range as per whitepaper
        scaled = 0.1 + (normalized * 0.9)
        
        return scaled