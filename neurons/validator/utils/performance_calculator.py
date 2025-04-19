import os
import json
import time
import bittensor as bt
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta
from .liquidity_checker import LiquidityChecker
from ....base.types import ChainType, TokenPrediction

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
        """Get or cache initial liquidity for a token pair
        
        Returns:
            Dict with 'value' and 'timestamp' fields
        """
        cache_key = f"{chain.value}:{token_addr}:{pair_addr}"
        
        # Check if we have this in cache
        if cache_key in self.initial_liquidity_cache:
            return self.initial_liquidity_cache[cache_key]
            
        # Not in cache, fetch initial data
        try:
            metrics = await self.liquidity_checker.check_token_liquidity(
                chain, token_addr, pair_addr
            )
            
            # Store with timestamp
            self.initial_liquidity_cache[cache_key] = {
                "value": metrics.current_liquidity,
                "timestamp": time.time()
            }
            
            # Save to persistent storage
            self._save_to_storage()
            
            return self.initial_liquidity_cache[cache_key]
            
        except Exception as e:
            bt.logging.error(f"Error getting initial liquidity: {e}")
            return {"value": 0, "timestamp": time.time()}

    async def calculate_token_performance(self, prediction: TokenPrediction) -> float:
        """Calculate average percentage liquidity increase for a single prediction"""
        try:
            total_percentage_change = 0.0
            valid_tokens = 0
            
            for i, token_addr in enumerate(prediction.addresses):
                try:
                    # Skip if index is out of range or pair is None
                    if i >= len(prediction.pairAddresses) or not prediction.pairAddresses[i]:
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
                        continue
                        
                    # Skip if not enough time has passed since initial check
                    current_time = time.time()
                    if current_time - initial_timestamp < self.min_evaluation_time:
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

    # Rest of the code remains unchanged
    async def calculate_performance(
        self, 
        predictions: List[TokenPrediction], 
        miner_ids: List[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate normalized performance scores for all predictions
        
        Args:
            predictions: List of TokenPrediction objects
            miner_ids: List of miner IDs corresponding to each prediction
            
        Returns:
            tuple containing:
                - numpy array of normalized performance scores
                - numpy array of corresponding miner IDs
        """
        # Calculate average liquidity change for each prediction
        performances = []
        for pred in predictions:
            avg_performance = await self.calculate_token_performance(pred)
            performances.append(avg_performance)
            
        performances = np.array(performances)
        miner_ids = np.array(miner_ids)
        
        # If we have no valid performances, return equal scores
        if len(performances) == 0 or np.all(performances == 0):
            return np.full_like(performances, 0.1), miner_ids
            
        # Normalize scores between 0.1 and 1.0
        min_score = 0.1
        max_score = 1.0
        
        # Min-max normalization
        min_perf = np.min(performances)
        max_perf = np.max(performances)
        
        if min_perf == max_perf:
            return np.full_like(performances, max_score), miner_ids
            
        normalized_scores = min_score + (max_score - min_score) * (
            (performances - min_perf) / (max_perf - min_perf)
        )
        
        return normalized_scores, miner_ids