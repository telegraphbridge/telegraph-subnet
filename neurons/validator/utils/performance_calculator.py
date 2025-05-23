import asyncio
import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import bittensor as bt

from base.types import ChainType, TokenPrediction, LiquidityMetrics
from .liquidity_checker import LiquidityChecker

class PerformanceCalculator:
    """
    Enhanced performance calculator that integrates with model evaluation framework
    and implements whitepaper-compliant Netheril performance metrics.
    """
    
    def __init__(self):
        self.liquidity_checker = LiquidityChecker()
        
        # Performance tracking
        self.historical_scores = defaultdict(list)  # uid -> list of scores
        self.performance_windows = {}  # uid -> sliding window data
        self.wallet_performance_cache = {}  # Cache wallet performance data
        
        # Scoring configuration based on whitepaper
        self.scoring_weights = {
            'accuracy': 0.35,           # Primary metric: prediction accuracy
            'liquidity_alpha': 0.25,    # Liquidity correlation performance
            'wallet_profiling': 0.20,   # Quality of wallet-based predictions
            'response_efficiency': 0.10, # Speed and resource efficiency
            'consistency': 0.10         # Historical consistency
        }
        
        # Performance thresholds
        self.min_predictions_threshold = 3
        self.historical_window_hours = 168  # 7 days
        self.performance_decay_factor = 0.95  # Slight decay for older performance
        
        bt.logging.info("Enhanced PerformanceCalculator initialized with model evaluation integration")

    async def calculate_performance_with_evaluation(self, predictions: List[TokenPrediction], 
                                                uids: List[int], round_id: str = None) -> Tuple[List[int], List[float]]:
        """
        Calculate performance using the comprehensive model evaluation framework.
        This replaces the basic performance calculation with comprehensive evaluation.
        """
        try:
            if not hasattr(self, 'model_evaluator'):
                from .model_evaluator import ModelEvaluationFramework
                from .benchmark_dataset import BenchmarkDatasetManager
                from .liquidity_checker import LiquidityChecker
                
                # Initialize evaluation framework
                benchmark_manager = BenchmarkDatasetManager()
                liquidity_checker = LiquidityChecker()
                self.model_evaluator = ModelEvaluationFramework(benchmark_manager, liquidity_checker)
                
                bt.logging.info("Initialized ModelEvaluationFramework in PerformanceCalculator")
            
            # Prepare submissions for evaluation
            submissions = [(uid, pred) for uid, pred in zip(uids, predictions)]
            
            # Run comprehensive evaluation
            evaluation_results = await self.model_evaluator.evaluate_models(
                submissions, round_id or f"perf_calc_{int(time.time())}"
            )
            
            if not evaluation_results:
                bt.logging.warning("Model evaluation returned no results, falling back to basic calculation")
                return await self.calculate_performance(predictions, uids)
            
            # Enhance evaluation results with historical performance and Netheril-specific metrics
            enhanced_scores = await self._enhance_with_historical_performance(evaluation_results)
            
            # Extract UIDs and scores from enhanced results
            evaluated_uids = [result.miner_uid for result in evaluation_results]
            enhanced_final_scores = [enhanced_scores.get(uid, 0.0) for uid in evaluated_uids]
            
            # Update historical tracking
            self._update_performance_tracking(evaluation_results, enhanced_scores)
            
            bt.logging.info(f"Enhanced performance calculation completed: {len(evaluation_results)} miners evaluated")
            bt.logging.info(f"Enhanced score range: {min(enhanced_final_scores):.4f} - {max(enhanced_final_scores):.4f}")
            
            return evaluated_uids, enhanced_final_scores
            
        except Exception as e:
            bt.logging.error(f"Error in comprehensive performance calculation: {str(e)}")
            import traceback
            bt.logging.error(traceback.format_exc())
            # Fallback to basic performance calculation
            return await self.calculate_performance(predictions, uids)


    async def _enhance_with_historical_performance(self, evaluation_results) -> Dict[int, float]:
        """
        Enhance evaluation results with historical performance and Netheril-specific metrics.
        Implements the whitepaper's emphasis on wallet profiling and consistency.
        """
        enhanced_scores = {}
        
        for result in evaluation_results:
            uid = result.miner_uid
            base_score = result.overall_score
            
            try:
                # Get historical performance for this miner
                historical_data = self._get_historical_performance(uid)
                
                # Calculate Netheril-specific enhancements
                wallet_profiling_score = await self._calculate_wallet_profiling_score(uid, result)
                consistency_score = self._calculate_consistency_score(uid, historical_data)
                liquidity_alpha_score = await self._calculate_liquidity_alpha_score(uid, result)
                efficiency_score = self._calculate_efficiency_score(result)
                
                # Apply whitepaper-compliant weighted scoring
                enhanced_score = (
                    base_score * self.scoring_weights['accuracy'] +
                    liquidity_alpha_score * self.scoring_weights['liquidity_alpha'] +
                    wallet_profiling_score * self.scoring_weights['wallet_profiling'] +
                    efficiency_score * self.scoring_weights['response_efficiency'] +
                    consistency_score * self.scoring_weights['consistency']
                )
                
                # Apply historical performance decay
                if historical_data:
                    historical_average = np.mean([h['score'] for h in historical_data[-10:]])  # Last 10 scores
                    enhanced_score = (enhanced_score * 0.7) + (historical_average * 0.3)  # Blend with history
                
                enhanced_scores[uid] = max(0.0, min(1.0, enhanced_score))  # Clamp to [0, 1]
                
                bt.logging.debug(f"Enhanced scoring for miner {uid}: "
                                f"base={base_score:.4f}, "
                                f"wallet_profiling={wallet_profiling_score:.4f}, "
                                f"consistency={consistency_score:.4f}, "
                                f"liquidity_alpha={liquidity_alpha_score:.4f}, "
                                f"final={enhanced_scores[uid]:.4f}")
                
            except Exception as e:
                bt.logging.error(f"Error enhancing score for miner {uid}: {str(e)}")
                enhanced_scores[uid] = base_score  # Fallback to base score
        
        return enhanced_scores

    async def _calculate_wallet_profiling_score(self, uid: int, evaluation_result) -> float:
        """
        Calculate wallet profiling score based on Netheril approach (Section 3.5).
        Measures how well the miner implements wallet-based token selection.
        """
        try:
            # Check if miner's predictions show wallet-based analysis patterns
            metadata = evaluation_result.metadata
            
            # Look for indicators of wallet profiling in prediction metadata
            wallet_analysis_indicators = 0
            max_indicators = 5
            
            # 1. Diversity in predictions (indicates wallet diversity analysis)
            diversity_score = evaluation_result.scores.get('diversity', 0)
            if diversity_score > 0.7:
                wallet_analysis_indicators += 1
            
            # 2. Confidence calibration (indicates proper wallet ROI weighting)
            confidence_calibration = evaluation_result.scores.get('confidence_calibration', 0)
            if confidence_calibration > 0.6:
                wallet_analysis_indicators += 1
            
            # 3. Response time efficiency (indicates optimized wallet lookups)
            response_time_score = evaluation_result.scores.get('response_time', 0)
            if response_time_score > 0.8:
                wallet_analysis_indicators += 1
            
            # 4. Liquidity correlation (indicates understanding of wallet impact on liquidity)
            liquidity_correlation = evaluation_result.scores.get('liquidity_correlation', 0)
            if liquidity_correlation > 0.5:
                wallet_analysis_indicators += 1
            
            # 5. Check if predictions have appropriate confidence distribution
            if 'num_predictions' in metadata and metadata['num_predictions'] >= 5:
                wallet_analysis_indicators += 1
            
            wallet_profiling_score = wallet_analysis_indicators / max_indicators
            
            bt.logging.debug(f"Wallet profiling score for miner {uid}: "
                           f"{wallet_profiling_score:.4f} "
                           f"({wallet_analysis_indicators}/{max_indicators} indicators)")
            
            return wallet_profiling_score
            
        except Exception as e:
            bt.logging.error(f"Error calculating wallet profiling score for miner {uid}: {str(e)}")
            return 0.0

    def _calculate_consistency_score(self, uid: int, historical_data: List[Dict]) -> float:
        """
        Calculate consistency score based on historical performance stability.
        Rewards miners who consistently perform well over time.
        """
        try:
            if not historical_data or len(historical_data) < 3:
                return 0.5  # Neutral score for new miners
            
            # Get recent scores (last 20 evaluations)
            recent_scores = [h['score'] for h in historical_data[-20:]]
            
            if len(recent_scores) < 3:
                return 0.5
            
            # Calculate consistency metrics
            mean_score = np.mean(recent_scores)
            std_score = np.std(recent_scores)
            
            # Consistency score: high mean, low variance is best
            if std_score == 0:
                consistency_score = 1.0 if mean_score > 0.5 else 0.5
            else:
                # Normalize by coefficient of variation
                cv = std_score / (mean_score + 1e-8)  # Add small epsilon to avoid division by zero
                consistency_score = max(0.0, 1.0 - cv)  # Lower CV = higher consistency
            
            # Apply trend analysis - reward improving performance
            if len(recent_scores) >= 10:
                first_half = np.mean(recent_scores[:len(recent_scores)//2])
                second_half = np.mean(recent_scores[len(recent_scores)//2:])
                improvement_factor = min(1.2, max(0.8, second_half / (first_half + 1e-8)))
                consistency_score *= improvement_factor
            
            consistency_score = max(0.0, min(1.0, consistency_score))
            
            bt.logging.debug(f"Consistency score for miner {uid}: {consistency_score:.4f} "
                           f"(mean={mean_score:.4f}, std={std_score:.4f})")
            
            return consistency_score
            
        except Exception as e:
            bt.logging.error(f"Error calculating consistency score for miner {uid}: {str(e)}")
            return 0.5

    async def _calculate_liquidity_alpha_score(self, uid: int, evaluation_result) -> float:
        """
        Calculate liquidity alpha score - measures ability to predict liquidity-impacting tokens.
        This implements the core Netheril approach of energy dissipation tracking.
        """
        try:
            # Base liquidity correlation from evaluation
            base_liquidity_score = evaluation_result.scores.get('liquidity_correlation', 0)
            
            # Check for additional liquidity intelligence indicators
            metadata = evaluation_result.metadata
            
            # Factor in prediction timing efficiency (faster predictions for high-liquidity tokens)
            timing_factor = 1.0
            inference_time = metadata.get('inference_time_ms', 1000)
            if inference_time < 500:  # Very fast predictions
                timing_factor = 1.1
            elif inference_time > 2000:  # Slow predictions
                timing_factor = 0.9
            
            # Factor in prediction precision (appropriate number of predictions)
            precision_factor = 1.0
            num_predictions = metadata.get('num_predictions', 0)
            if 5 <= num_predictions <= 15:  # Optimal range
                precision_factor = 1.1
            elif num_predictions > 20:  # Too many predictions
                precision_factor = 0.9
            
            # Combine factors
            liquidity_alpha_score = base_liquidity_score * timing_factor * precision_factor
            liquidity_alpha_score = max(0.0, min(1.0, liquidity_alpha_score))
            
            bt.logging.debug(f"Liquidity alpha score for miner {uid}: {liquidity_alpha_score:.4f} "
                           f"(base={base_liquidity_score:.4f}, timing={timing_factor:.2f}, "
                           f"precision={precision_factor:.2f})")
            
            return liquidity_alpha_score
            
        except Exception as e:
            bt.logging.error(f"Error calculating liquidity alpha score for miner {uid}: {str(e)}")
            return 0.0

    def _calculate_efficiency_score(self, evaluation_result) -> float:
        """
        Calculate efficiency score based on computational resources and response time.
        """
        try:
            # Get response time score from evaluation
            response_time_score = evaluation_result.scores.get('response_time', 0)
            
            # Get resource usage from metadata
            metadata = evaluation_result.metadata
            memory_usage = metadata.get('memory_usage_mb', 0)
            inference_time = metadata.get('inference_time_ms', 1000)
            
            # Score memory efficiency (lower is better, up to reasonable limits)
            memory_efficiency = 1.0
            if memory_usage > 0:
                if memory_usage <= 100:  # Very efficient
                    memory_efficiency = 1.0
                elif memory_usage <= 500:  # Reasonable
                    memory_efficiency = 0.8
                elif memory_usage <= 1000:  # Acceptable
                    memory_efficiency = 0.6
                else:  # Inefficient
                    memory_efficiency = 0.4
            
            # Combine response time and memory efficiency
            efficiency_score = (response_time_score * 0.7) + (memory_efficiency * 0.3)
            
            bt.logging.debug(f"Efficiency score: {efficiency_score:.4f} "
                           f"(response_time={response_time_score:.4f}, "
                           f"memory_eff={memory_efficiency:.4f})")
            
            return efficiency_score
            
        except Exception as e:
            bt.logging.error(f"Error calculating efficiency score: {str(e)}")
            return 0.5

    def _get_historical_performance(self, uid: int) -> List[Dict]:
        """Get historical performance data for a miner"""
        if uid not in self.historical_scores:
            return []
        
        # Filter to recent history (last 7 days)
        cutoff_time = datetime.utcnow() - timedelta(hours=self.historical_window_hours)
        recent_history = [
            h for h in self.historical_scores[uid]
            if h['timestamp'] > cutoff_time
        ]
        
        return recent_history

    def _update_performance_tracking(self, evaluation_results, enhanced_scores: Dict[int, float]):
        """Update historical performance tracking with new results"""
        current_time = datetime.utcnow()
        
        for result in evaluation_results:
            uid = result.miner_uid
            enhanced_score = enhanced_scores.get(uid, result.overall_score)
            
            # Add to historical tracking
            if uid not in self.historical_scores:
                self.historical_scores[uid] = []
            
            score_entry = {
                'timestamp': current_time,
                'score': enhanced_score,
                'base_score': result.overall_score,
                'round_id': result.metadata.get('round_id', ''),
                'evaluation_time': result.metadata.get('evaluation_time', ''),
                'scores_breakdown': result.scores.copy()
            }
            
            self.historical_scores[uid].append(score_entry)
            
            # Keep only recent history (last 100 entries per miner)
            self.historical_scores[uid] = self.historical_scores[uid][-100:]
            
            bt.logging.debug(f"Updated performance tracking for miner {uid}: "
                           f"score={enhanced_score:.4f}")

    async def calculate_performance(self, predictions: List[TokenPrediction], uids: List[int]) -> Tuple[List[int], List[float]]:
        """
        Fallback performance calculation method for when model evaluation is unavailable.
        Implements basic Netheril-inspired scoring.
        """
        try:
            bt.logging.info("Using fallback performance calculation method")
            
            if len(predictions) != len(uids):
                bt.logging.error(f"Mismatch: {len(predictions)} predictions vs {len(uids)} UIDs")
                return [], []
            
            scores = []
            
            for i, prediction in enumerate(predictions):
                uid = uids[i]
                
                try:
                    # Basic scoring factors
                    prediction_quality = self._score_prediction_quality(prediction)
                    diversity_score = self._score_diversity(prediction)
                    confidence_quality = self._score_confidence_quality(prediction)
                    timing_score = self._score_timing(prediction)
                    
                    # Combine scores with weights
                    overall_score = (
                        prediction_quality * 0.4 +
                        diversity_score * 0.25 +
                        confidence_quality * 0.2 +
                        timing_score * 0.15
                    )
                    
                    # Apply historical consistency if available
                    historical_data = self._get_historical_performance(uid)
                    if historical_data:
                        consistency_score = self._calculate_consistency_score(uid, historical_data)
                        overall_score = (overall_score * 0.8) + (consistency_score * 0.2)
                    
                    scores.append(max(0.0, min(1.0, overall_score)))
                    
                    bt.logging.debug(f"Fallback scoring for miner {uid}: "
                                   f"quality={prediction_quality:.3f}, "
                                   f"diversity={diversity_score:.3f}, "
                                   f"confidence={confidence_quality:.3f}, "
                                   f"timing={timing_score:.3f}, "
                                   f"overall={scores[-1]:.3f}")
                    
                except Exception as e:
                    bt.logging.error(f"Error scoring prediction for UID {uid}: {str(e)}")
                    scores.append(0.0)
            
            bt.logging.info(f"Fallback performance calculation completed for {len(uids)} miners")
            return uids, scores
            
        except Exception as e:
            bt.logging.error(f"Error in fallback performance calculation: {str(e)}")
            return [], []

    def _score_prediction_quality(self, prediction: TokenPrediction) -> float:
        """Score based on prediction quality indicators"""
        if not prediction.addresses:
            return 0.0
        
        # Quality factors
        num_predictions = len(prediction.addresses)
        has_pairs = bool(prediction.pairAddresses)
        has_confidence = bool(prediction.confidence_scores)
        
        # Optimal prediction count (5-15 as per whitepaper guidance)
        count_score = 1.0 if 5 <= num_predictions <= 15 else max(0.3, 1.0 - abs(num_predictions - 10) * 0.05)
        pair_score = 1.0 if has_pairs else 0.7
        confidence_score = 1.0 if has_confidence else 0.5
        
        return (count_score + pair_score + confidence_score) / 3.0

    def _score_diversity(self, prediction: TokenPrediction) -> float:
        """Score prediction diversity (important for Netheril wallet profiling)"""
        if not prediction.addresses:
            return 0.0
        
        # Check for address diversity (no duplicates, reasonable spread)
        unique_addresses = len(set(prediction.addresses))
        diversity_ratio = unique_addresses / len(prediction.addresses)
        
        return diversity_ratio

    def _score_confidence_quality(self, prediction: TokenPrediction) -> float:
        """Score confidence score quality and calibration"""
        if not prediction.confidence_scores:
            return 0.5
        
        confidence_values = list(prediction.confidence_scores.values())
        
        # Check for reasonable confidence distribution
        mean_confidence = np.mean(confidence_values)
        std_confidence = np.std(confidence_values)
        
        # Prefer diverse but reasonable confidence scores
        if 0.3 <= mean_confidence <= 0.8 and std_confidence > 0.1:
            return 1.0
        elif 0.1 <= mean_confidence <= 0.9:
            return 0.7
        else:
            return 0.3

    def _score_timing(self, prediction: TokenPrediction) -> float:
        """Score based on inference timing efficiency"""
        inference_time = getattr(prediction, 'inference_time_ms', 1000)
        
        if inference_time <= 0:
            return 1.0  # No timing data
        
        # Score based on speed (faster is better, up to reasonable limits)
        if inference_time <= 500:
            return 1.0
        elif inference_time <= 1000:
            return 0.8
        elif inference_time <= 2000:
            return 0.6
        elif inference_time <= 5000:
            return 0.4
        else:
            return 0.2

    def normalize_performance(self, scores: List[float]) -> List[float]:
        """
        Normalize performance scores to ensure fair distribution.
        Uses softmax-like normalization to maintain relative rankings.
        """
        try:
            if not scores:
                return []
            
            scores_array = np.array(scores, dtype=float)
            
            # Handle edge cases
            if len(scores) == 1:
                return [1.0]
            
            if np.all(scores_array == 0):
                return [1.0 / len(scores)] * len(scores)  # Equal distribution
            
            # Apply temperature scaling for better distribution
            temperature = 0.5
            scaled_scores = scores_array / temperature
            
            # Prevent overflow in softmax
            scaled_scores = scaled_scores - np.max(scaled_scores)
            
            # Softmax normalization
            exp_scores = np.exp(scaled_scores)
            normalized_scores = exp_scores / np.sum(exp_scores)
            
            # Ensure minimum score threshold
            min_score = 0.01
            normalized_scores = np.maximum(normalized_scores, min_score)
            
            # Renormalize after applying minimum
            normalized_scores = normalized_scores / np.sum(normalized_scores)
            
            bt.logging.debug(f"Normalized {len(scores)} scores: "
                           f"original_range=[{min(scores):.4f}, {max(scores):.4f}], "
                           f"normalized_range=[{min(normalized_scores):.4f}, {max(normalized_scores):.4f}]")
            
            return normalized_scores.tolist()
            
        except Exception as e:
            bt.logging.error(f"Error normalizing performance scores: {str(e)}")
            # Fallback to equal distribution
            return [1.0 / len(scores)] * len(scores)

    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get overall performance statistics across all miners"""
        try:
            total_evaluations = sum(len(history) for history in self.historical_scores.values())
            active_miners = len(self.historical_scores)
            
            # Calculate recent performance metrics
            recent_scores = []
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            for history in self.historical_scores.values():
                recent_evals = [
                    eval_data for eval_data in history
                    if eval_data['timestamp'] > cutoff_time
                ]
                recent_scores.extend([eval_data['score'] for eval_data in recent_evals])
            
            avg_score = np.mean(recent_scores) if recent_scores else 0.0
            max_score = np.max(recent_scores) if recent_scores else 0.0
            
            return {
                'total_evaluations': total_evaluations,
                'active_miners': active_miners,
                'recent_evaluations_24h': len(recent_scores),
                'average_score_24h': float(avg_score),
                'max_score_24h': float(max_score),
                'scoring_weights': self.scoring_weights.copy(),
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            bt.logging.error(f"Error getting performance statistics: {str(e)}")
            return {}