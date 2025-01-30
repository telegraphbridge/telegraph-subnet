from typing import List, Dict
import numpy as np
from .liquidity_checker import LiquidityChecker
from ....base.types import ChainType, TokenPrediction

class PerformanceCalculator:
    def __init__(self):
        self.liquidity_checker = LiquidityChecker()
        self.initial_liquidity_cache: Dict[str, float] = {}  # Cache initial liquidity values


    async def get_or_cache_initial_liquidity(
        self, 
        chain: ChainType, 
        token_addr: str, 
        pair_addr: str
    ) -> float:
        """Get or cache initial liquidity for a token pair"""
        cache_key = f"{chain.value}:{token_addr}:{pair_addr}"
        if cache_key not in self.initial_liquidity_cache:
            metrics = await self.liquidity_checker.check_token_liquidity(
                chain,
                token_addr,
                pair_addr
            )
            self.initial_liquidity_cache[cache_key] = metrics.current_liquidity
        return self.initial_liquidity_cache[cache_key]


    async def calculate_token_performance(self, prediction: TokenPrediction) -> float:
        """Calculate average percentage liquidity increase for a single prediction"""
        total_percentage_change = 0.0
        valid_tokens = 0
        
        for i, token_addr in enumerate(prediction.addresses):
            pair_addr = prediction.pairAddresses[i] if i < len(prediction.pairAddresses) else None
            if not pair_addr:
                continue
                
            # Get initial liquidity (cached after first check)
            initial_liquidity = await self.get_or_cache_initial_liquidity(
                prediction.chain,
                token_addr,
                pair_addr
            )
            
            # Get current liquidity
            current_metrics = await self.liquidity_checker.check_token_liquidity(
                prediction.chain,
                token_addr,
                pair_addr
            )
            
            # Skip if initial liquidity was 0 to avoid division by zero
            if initial_liquidity == 0:
                continue
                
            # Calculate percentage change
            percentage_change = ((current_metrics.current_liquidity - initial_liquidity) 
                               / initial_liquidity) * 100
            
            total_percentage_change += percentage_change
            valid_tokens += 1
            
        # Return average percentage change across all valid tokens
        return total_percentage_change / valid_tokens if valid_tokens > 0 else 0.0


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
