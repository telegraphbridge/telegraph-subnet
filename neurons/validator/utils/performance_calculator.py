import os
import json
import time
import bittensor as bt
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import traceback # Import traceback for detailed error logging

# Assuming LiquidityChecker is correctly defined in its own file
from .liquidity_checker import LiquidityChecker
# Assuming base types are correctly defined
from base.types import ChainType, TokenPrediction, LiquidityMetrics

class PerformanceCalculator:
    """
    Calculates the performance of miner predictions based on token liquidity changes.
    It caches initial liquidity values to avoid redundant RPC calls and ensures
    a minimum evaluation time window.
    """
    def __init__(self, storage_path: str = "storage/liquidity_data.json", min_evaluation_time_seconds: int = 3600):
        """
        Initialize the PerformanceCalculator.

        Args:
            storage_path: Path to the JSON file for caching initial liquidity data.
            min_evaluation_time_seconds: Minimum time (in seconds) that must pass
                                         before evaluating a token's performance. Defaults to 1 hour.
        """
        self.liquidity_checker = LiquidityChecker()
        self.storage_path = storage_path
        self.min_evaluation_time = min_evaluation_time_seconds # Use the argument

        # Load cached initial liquidity values from storage
        self.initial_liquidity_cache: Dict[str, Dict[str, Any]] = self._load_from_storage()

        # Ensure the storage directory exists
        storage_dir = os.path.dirname(self.storage_path)
        if storage_dir and not os.path.exists(storage_dir):
             os.makedirs(storage_dir, exist_ok=True)
             bt.logging.info(f"Created storage directory: {storage_dir}")

    def _load_from_storage(self) -> Dict[str, Dict[str, Any]]:
        """Load initial liquidity values from the JSON storage file."""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    bt.logging.info(f"Loaded {len(data)} initial liquidity records from {self.storage_path}")
                    return data
            else:
                bt.logging.info(f"Liquidity cache file not found at {self.storage_path}. Starting fresh cache.")
        except json.JSONDecodeError:
            bt.logging.error(f"Error decoding JSON from {self.storage_path}. Starting fresh cache.")
        except Exception as e:
            bt.logging.warning(f"Failed to load liquidity data from {self.storage_path}: {e}. Starting fresh cache.")
        return {}

    def _save_to_storage(self):
        """Save the current initial liquidity cache to the JSON storage file."""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.initial_liquidity_cache, f, indent=4) # Add indent for readability
            # bt.logging.debug(f"Saved {len(self.initial_liquidity_cache)} liquidity records to {self.storage_path}")
        except Exception as e:
            bt.logging.error(f"Failed to save liquidity data to {self.storage_path}: {e}")

    async def get_or_cache_initial_liquidity(
        self,
        chain: ChainType,
        token_addr: str,
        pair_addr: str
    ) -> Dict[str, Any]:
        """
        Get initial liquidity for a token pair, using cache if available, otherwise fetch and cache.

        Args:
            chain: The blockchain chain type.
            token_addr: The token address.
            pair_addr: The liquidity pool pair address.

        Returns:
            A dictionary containing 'value' (initial liquidity) and 'timestamp' (when it was fetched).
            Returns a default value if fetching fails.
        """
        # Standardize addresses for cache key consistency
        token_addr_std = token_addr.lower()
        pair_addr_std = pair_addr.lower()
        cache_key = f"{chain.value}:{token_addr_std}:{pair_addr_std}"

        # Check cache first
        if cache_key in self.initial_liquidity_cache:
            # bt.logging.trace(f"Using cached liquidity data for {cache_key}")
            return self.initial_liquidity_cache[cache_key]

        # Not in cache, fetch initial liquidity data
        bt.logging.debug(f"Fetching initial liquidity for {cache_key}...")
        try:
            metrics = await self.liquidity_checker.check_token_liquidity(
                chain, token_addr_std, pair_addr_std # Use standardized addresses
            )
            initial_liquidity_value = metrics.current_liquidity
            bt.logging.info(f"Fetched initial liquidity for {cache_key}: {initial_liquidity_value}")

        except Exception as e:
            bt.logging.error(f"Error getting initial liquidity for {cache_key}: {e}")
            # Fallback: Use a default value (e.g., 0) to indicate failure or unknown state
            initial_liquidity_value = 0.0
            bt.logging.warning(f"Using fallback initial liquidity (0.0) for {cache_key} due to error.")

        # Store the fetched (or fallback) value with the current timestamp
        current_timestamp = time.time()
        self.initial_liquidity_cache[cache_key] = {
            "value": initial_liquidity_value,
            "timestamp": current_timestamp
        }

        # Save updated cache to persistent storage
        self._save_to_storage()

        return self.initial_liquidity_cache[cache_key]

    async def calculate_token_performance(self, prediction: TokenPrediction) -> float:
        """
        Calculate the average percentage liquidity change for tokens in a single prediction.

        Args:
            prediction: A TokenPrediction object containing addresses and pair addresses.

        Returns:
            The average percentage change in liquidity across all valid, evaluatable tokens
            in the prediction. Returns 0.0 if no tokens can be evaluated.
        """
        total_percentage_change = 0.0
        valid_tokens_evaluated = 0

        if not prediction.addresses:
            bt.logging.warning("Cannot calculate performance: Prediction has no addresses.")
            return 0.0

        bt.logging.debug(f"Evaluating {len(prediction.addresses)} tokens for {prediction.chain.value} prediction made at {prediction.timestamp}")

        for i, token_addr in enumerate(prediction.addresses):
            try:
                # Ensure we have a corresponding pair address
                if i >= len(prediction.pairAddresses or []) or not prediction.pairAddresses[i] or prediction.pairAddresses[i] == "0x0000000000000000000000000000000000000000":
                    # bt.logging.trace(f"Skipping token {token_addr}: Missing or invalid pair address.")
                    continue

                pair_addr = prediction.pairAddresses[i]

                # Get initial liquidity (fetches/caches on first call)
                initial_data = await self.get_or_cache_initial_liquidity(
                    prediction.chain, token_addr, pair_addr
                )
                initial_liquidity = initial_data["value"]
                initial_timestamp = initial_data["timestamp"]

                # CRITICAL CHECK: Skip tokens if initial liquidity was zero or couldn't be fetched.
                if initial_liquidity <= 0:
                    bt.logging.warning(f"Token {token_addr} (Pair: {pair_addr}) has zero or negative initial liquidity ({initial_liquidity}). Skipping performance calculation.")
                    continue

                # Check if enough time has passed since the initial liquidity check
                current_time = time.time()
                elapsed_time = current_time - initial_timestamp
                if elapsed_time < self.min_evaluation_time:
                    # bt.logging.trace(f"Skipping token {token_addr}: Not enough time elapsed ({elapsed_time:.0f}s < {self.min_evaluation_time}s).")
                    continue

                # Fetch current liquidity
                try:
                    current_metrics = await self.liquidity_checker.check_token_liquidity(
                        prediction.chain, token_addr, pair_addr
                    )
                    current_liquidity = current_metrics.current_liquidity

                    # Calculate percentage change (handle division by zero just in case, though checked above)
                    if initial_liquidity > 0:
                        percentage_change = ((current_liquidity - initial_liquidity) / initial_liquidity) * 100.0
                    else:
                        percentage_change = 0.0 # Should not happen due to check above

                    total_percentage_change += percentage_change
                    valid_tokens_evaluated += 1

                    bt.logging.debug(f"Token {token_addr}: Initial Liq={initial_liquidity:.4f} (at {datetime.fromtimestamp(initial_timestamp)}), Current Liq={current_liquidity:.4f}, Change={percentage_change:.2f}%")

                except Exception as e:
                    bt.logging.warning(f"Error checking current liquidity for {token_addr} (Pair: {pair_addr}): {e}")
                    # Continue to the next token if current check fails
                    continue

            except Exception as e:
                # Catch errors during processing of a single token within the prediction
                bt.logging.error(f"Error processing token {token_addr} in prediction: {e}")
                bt.logging.debug(traceback.format_exc()) # Log traceback for debugging
                continue # Continue to the next token

        # Calculate average percentage change across all successfully evaluated tokens
        if valid_tokens_evaluated > 0:
            average_performance = total_percentage_change / valid_tokens_evaluated
            bt.logging.info(f"Prediction evaluated {valid_tokens_evaluated} tokens. Average performance: {average_performance:.2f}%")
            return average_performance
        else:
            bt.logging.warning("No valid tokens could be evaluated for this prediction.")
            return 0.0

    async def calculate_performance(
        self,
        predictions: List[TokenPrediction], # Predictions from the CURRENT round
        miner_ids: List[int] # UIDs corresponding *exactly* to the predictions list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate performance scores for a list of predictions from the current round.

        Args:
            predictions: List of TokenPrediction objects to evaluate.
            miner_ids: List of miner UIDs, where miner_ids[i] corresponds to predictions[i].

        Returns:
            A tuple containing two numpy arrays:
            1. An array of miner UIDs for which performance was successfully calculated.
            2. An array of corresponding raw performance scores (average percentage change).
            Returns empty arrays if no performance could be calculated.
        """
        if not predictions:
            bt.logging.warning("No predictions provided for performance calculation.")
            return np.array([], dtype=int), np.array([], dtype=float)

        # CRITICAL CHECK: Ensure the number of predictions matches the number of miner IDs
        if len(predictions) != len(miner_ids):
            bt.logging.error(f"CRITICAL MISMATCH: Cannot calculate performance. Received {len(predictions)} predictions but {len(miner_ids)} miner IDs.")
            # Returning empty arrays prevents incorrect reward assignment
            return np.array([], dtype=int), np.array([], dtype=float)

        performances = []
        valid_miner_ids_for_scores = [] # Store UIDs for which a score was calculated

        bt.logging.info(f"Calculating performance for {len(predictions)} predictions from UIDs: {miner_ids}")

        # Iterate through predictions and their corresponding miner IDs
        for i, pred in enumerate(predictions):
            miner_id = miner_ids[i] # Direct 1-to-1 mapping thanks to the check above
            try:
                # Calculate the performance for this specific prediction
                avg_performance = await self.calculate_token_performance(pred)

                # Store the calculated performance and the corresponding miner ID
                performances.append(avg_performance)
                valid_miner_ids_for_scores.append(miner_id)

                bt.logging.info(f"Performance calculated for UID {miner_id}: {avg_performance:.2f}%")

            except Exception as e:
                # Catch errors during the performance calculation for a specific miner's prediction
                bt.logging.error(f"Error calculating performance for UID {miner_id}'s prediction: {e}")
                bt.logging.debug(traceback.format_exc()) # Log traceback for debugging
                # Skip this miner's score if calculation failed
                continue

        # Convert the results to numpy arrays
        performances_array = np.array(performances, dtype=float)
        final_miner_ids_array = np.array(valid_miner_ids_for_scores, dtype=int)

        # Log summary of the round's performance calculation
        if len(final_miner_ids_array) > 0:
             avg_overall_performance = np.mean(performances_array)
             bt.logging.info(f"Finished performance calculation for {len(final_miner_ids_array)} predictions. Average raw score: {avg_overall_performance:.2f}%")
        else:
             bt.logging.warning("Performance calculation finished, but no valid scores were generated.")

        # Return the arrays of UIDs and their corresponding scores
        return final_miner_ids_array, performances_array

    def normalize_performance(self, performance_scores: np.ndarray) -> np.ndarray:
        """
        Normalize raw performance scores (percentage changes) to a 0.1 - 1.0 range.
        This scaling ensures rewards are positive and distributed relative to performance.

        Args:
            performance_scores: A numpy array of raw performance scores.

        Returns:
            A numpy array of normalized scores, ranging from 0.1 to 1.0.
            Returns an empty array if the input is empty.
        """
        if len(performance_scores) == 0:
            return np.array([], dtype=float) # Return float array

        # Handle case where all scores are the same to avoid division by zero
        if np.all(performance_scores == performance_scores[0]):
             # If all scores are identical, assign a mid-range normalized score (e.g., 0.55)
             # or handle as per specific requirements (e.g., all get 0.1 or 1.0).
             # Assigning 0.55 provides some differentiation if needed later.
             bt.logging.info("All performance scores are identical. Assigning uniform normalized score.")
             return np.full(performance_scores.shape, 0.55, dtype=float)


        # Find the minimum and maximum scores in the array
        min_score = np.min(performance_scores)
        max_score = np.max(performance_scores)

        # Normalize scores to the 0-1 range using min-max scaling
        # Add a small epsilon to the denominator to prevent division by zero if max_score == min_score (handled above, but safe)
        range_score = max_score - min_score
        if range_score == 0: range_score = 1e-9 # Prevent division by zero

        normalized_scores = (performance_scores - min_score) / range_score

        # Scale the normalized scores to the desired 0.1 - 1.0 range
        # Formula: scaled = new_min + normalized * (new_max - new_min)
        scaled_scores = 0.1 + (normalized_scores * 0.9)

        # Ensure scores are clipped within the bounds just in case of floating point issues
        scaled_scores = np.clip(scaled_scores, 0.1, 1.0)

        bt.logging.debug(f"Normalized scores: Min={np.min(scaled_scores):.4f}, Max={np.max(scaled_scores):.4f}, Avg={np.mean(scaled_scores):.4f}")

        return scaled_scores