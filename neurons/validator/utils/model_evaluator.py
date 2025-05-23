import asyncio
import time
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import bittensor as bt

from base.types import ChainType, TokenPrediction
from .benchmark_dataset import BenchmarkDatasetManager
from .liquidity_checker import LiquidityChecker

class ModelEvaluationResult:
    """Container for evaluation results"""
    def __init__(self, miner_uid: int, scores: Dict[str, float], 
                 metadata: Dict[str, Any] = None):
        self.miner_uid = miner_uid
        self.scores = scores
        self.metadata = metadata or {}
        self.overall_score = self._calculate_overall_score()
        
    def _calculate_overall_score(self) -> float:
        """Calculate weighted overall score from individual metrics"""
        weights = {
            'accuracy': 0.3,
            'precision': 0.2,
            'recall': 0.15,
            'liquidity_correlation': 0.2,
            'confidence_calibration': 0.1,
            'response_time': 0.05
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in self.scores:
                weighted_sum += self.scores[metric] * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

class ModelEvaluationFramework:
    """
    Production-ready model evaluation framework for competition assessment.
    Evaluates predictions against benchmark datasets with multiple metrics.
    """
    
    def __init__(self, benchmark_manager: BenchmarkDatasetManager, 
                 liquidity_checker: LiquidityChecker):
        self.benchmark_manager = benchmark_manager
        self.liquidity_checker = liquidity_checker
        
        # Evaluation configuration
        self.evaluation_window_hours = 24  # How long to track actual performance
        self.min_predictions_for_eval = 3   # Minimum predictions needed for evaluation
        self.confidence_threshold = 0.1     # Minimum confidence for predictions
        
        # Performance tracking
        self.historical_performance = {}    # Track real performance over time
        self.evaluation_cache = {}          # Cache evaluation results
        
        bt.logging.info("ModelEvaluationFramework initialized")

    async def evaluate_models(self, submissions: List[Tuple[int, TokenPrediction]], 
                            round_id: str) -> List[ModelEvaluationResult]:
        """
        Evaluate multiple model submissions against benchmark dataset.
        
        Args:
            submissions: List of (miner_uid, prediction) tuples
            round_id: Competition round identifier
            
        Returns:
            List of evaluation results sorted by performance
        """
        try:
            bt.logging.info(f"Starting model evaluation for round {round_id} with {len(submissions)} submissions")
            
            if not submissions:
                bt.logging.warning("No submissions to evaluate")
                return []
            
            # Get benchmark dataset for evaluation
            chain = submissions[0][1].chain  # Assume all submissions are for same chain
            benchmark_data = self.benchmark_manager.get_latest_dataset(chain.value.lower())
            
            if not benchmark_data:
                bt.logging.error(f"No benchmark dataset available for chain {chain.value}")
                return self._fallback_evaluation(submissions)
            
            bt.logging.info(f"Using benchmark dataset version: {benchmark_data['version']}")
            bt.logging.info(f"Benchmark contains {benchmark_data['record_count']} records")
            
            # Evaluate each submission
            evaluation_results = []
            
            for miner_uid, prediction in submissions:
                try:
                    result = await self._evaluate_single_model(
                        miner_uid, prediction, benchmark_data, round_id
                    )
                    evaluation_results.append(result)
                    
                    bt.logging.info(f"Evaluated model from miner {miner_uid}: "
                                  f"overall_score={result.overall_score:.4f}")
                    
                except Exception as e:
                    bt.logging.error(f"Error evaluating model from miner {miner_uid}: {str(e)}")
                    # Create fallback result
                    evaluation_results.append(ModelEvaluationResult(
                        miner_uid=miner_uid,
                        scores={'accuracy': 0.0, 'overall': 0.0},
                        metadata={'error': str(e)}
                    ))
            
            # Sort by overall score (descending)
            evaluation_results.sort(key=lambda x: x.overall_score, reverse=True)
            
            bt.logging.info(f"Model evaluation completed. Top 3 scores: {[r.overall_score for r in evaluation_results[:3]]}")
            
            return evaluation_results
            
        except Exception as e:
            bt.logging.error(f"Critical error in model evaluation: {str(e)}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return self._fallback_evaluation(submissions)

    async def _evaluate_single_model(self, miner_uid: int, prediction: TokenPrediction,
                                   benchmark_data: Dict[str, Any], round_id: str) -> ModelEvaluationResult:
        """
        Evaluate a single model prediction against benchmark dataset.
        
        Args:
            miner_uid: Miner unique identifier
            prediction: Model prediction to evaluate
            benchmark_data: Benchmark dataset for comparison
            round_id: Competition round identifier
            
        Returns:
            ModelEvaluationResult with detailed scoring
        """
        start_time = time.perf_counter()
        
        try:
            # Extract ground truth from benchmark dataset
            ground_truth_tokens = self._extract_ground_truth(benchmark_data['data'])
            predicted_tokens = set(prediction.addresses)
            
            bt.logging.debug(f"Evaluating miner {miner_uid}: {len(predicted_tokens)} predictions "
                           f"vs {len(ground_truth_tokens)} ground truth tokens")
            
            # Calculate accuracy metrics
            accuracy_scores = self._calculate_accuracy_metrics(
                predicted_tokens, ground_truth_tokens
            )
            
            # Calculate liquidity correlation
            liquidity_score = await self._calculate_liquidity_correlation(
                prediction, benchmark_data['data']
            )
            
            # Calculate confidence calibration
            confidence_score = self._calculate_confidence_calibration(
                prediction, ground_truth_tokens
            )
            
            # Calculate response time score
            response_time_score = self._calculate_response_time_score(prediction)
            
            # Calculate diversity score
            diversity_score = self._calculate_prediction_diversity(prediction)
            
            # Combine all scores
            scores = {
                'accuracy': accuracy_scores['accuracy'],
                'precision': accuracy_scores['precision'],
                'recall': accuracy_scores['recall'],
                'f1_score': accuracy_scores['f1_score'],
                'liquidity_correlation': liquidity_score,
                'confidence_calibration': confidence_score,
                'response_time': response_time_score,
                'diversity': diversity_score
            }
            
            # Additional metadata
            metadata = {
                'round_id': round_id,
                'evaluation_time': datetime.utcnow().isoformat(),
                'benchmark_version': benchmark_data['version'],
                'num_predictions': len(predicted_tokens),
                'num_ground_truth': len(ground_truth_tokens),
                'evaluation_duration_ms': int((time.perf_counter() - start_time) * 1000),
                'prediction_timestamp': prediction.timestamp.isoformat(),
                'inference_time_ms': getattr(prediction, 'inference_time_ms', 0),
                'memory_usage_mb': getattr(prediction, 'memory_usage_mb', 0)
            }
            
            result = ModelEvaluationResult(miner_uid, scores, metadata)
            
            # Update historical performance tracking
            self._update_historical_performance(miner_uid, result)
            
            bt.logging.debug(f"Evaluation completed for miner {miner_uid} in "
                           f"{metadata['evaluation_duration_ms']}ms")
            
            return result
            
        except Exception as e:
            bt.logging.error(f"Error in single model evaluation for miner {miner_uid}: {str(e)}")
            return ModelEvaluationResult(
                miner_uid=miner_uid,
                scores={'accuracy': 0.0, 'error': 1.0},
                metadata={'error': str(e), 'round_id': round_id}
            )

    def _extract_ground_truth(self, benchmark_data: List[Dict[str, Any]]) -> set:
        """
        Extract ground truth tokens from benchmark dataset.
        In production, this would be based on actual performance criteria.
        """
        try:
            # Sort by liquidity and volume to identify top-performing tokens
            sorted_tokens = sorted(
                benchmark_data,
                key=lambda x: (
                    float(x.get('liquidityPoolSize', 0)) * 
                    float(x.get('volume24hUsd', 0))
                ),
                reverse=True
            )
            
            # Take top 20% as ground truth high-performing tokens
            num_ground_truth = max(1, len(sorted_tokens) // 5)
            ground_truth = {token['tokenID'] for token in sorted_tokens[:num_ground_truth]}
            
            bt.logging.debug(f"Extracted {len(ground_truth)} ground truth tokens from "
                           f"{len(benchmark_data)} benchmark records")
            
            return ground_truth
            
        except Exception as e:
            bt.logging.error(f"Error extracting ground truth: {str(e)}")
            return set()

    def _calculate_accuracy_metrics(self, predicted: set, ground_truth: set) -> Dict[str, float]:
        """Calculate standard accuracy metrics (precision, recall, F1)"""
        try:
            if not ground_truth:
                bt.logging.warning("No ground truth tokens available for accuracy calculation")
                return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
            
            if not predicted:
                bt.logging.warning("No predicted tokens available for accuracy calculation")
                return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
            
            # Calculate intersection and metrics
            true_positives = len(predicted.intersection(ground_truth))
            false_positives = len(predicted - ground_truth)
            false_negatives = len(ground_truth - predicted)
            
            # Precision: TP / (TP + FP)
            precision = true_positives / len(predicted) if len(predicted) > 0 else 0.0
            
            # Recall: TP / (TP + FN)
            recall = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0.0
            
            # F1 Score: 2 * (precision * recall) / (precision + recall)
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Overall accuracy: TP / (TP + FP + FN)
            total_relevant = len(predicted.union(ground_truth))
            accuracy = true_positives / total_relevant if total_relevant > 0 else 0.0
            
            bt.logging.debug(f"Accuracy metrics: precision={precision:.4f}, recall={recall:.4f}, "
                           f"f1={f1_score:.4f}, accuracy={accuracy:.4f}")
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score
            }
            
        except Exception as e:
            bt.logging.error(f"Error calculating accuracy metrics: {str(e)}")
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

    async def _calculate_liquidity_correlation(self, prediction: TokenPrediction, 
                                             benchmark_data: List[Dict[str, Any]]) -> float:
        """
        Calculate how well predictions correlate with actual liquidity changes.
        This would use real-time liquidity data in production.
        """
        try:
            if not prediction.addresses:
                return 0.0
            
            # Create liquidity lookup from benchmark data
            liquidity_lookup = {}
            for token_data in benchmark_data:
                token_id = token_data.get('tokenID')
                if token_id:
                    liquidity_lookup[token_id] = float(token_data.get('liquidityPoolSize', 0))
            
            # Calculate average liquidity of predicted tokens
            predicted_liquidities = []
            for addr in prediction.addresses:
                if addr in liquidity_lookup:
                    predicted_liquidities.append(liquidity_lookup[addr])
            
            if not predicted_liquidities:
                bt.logging.warning("No liquidity data found for predicted tokens")
                return 0.0
            
            # Calculate score based on liquidity distribution
            avg_predicted_liquidity = np.mean(predicted_liquidities)
            all_liquidities = list(liquidity_lookup.values())
            
            if not all_liquidities:
                return 0.0
            
            # Score based on percentile of average predicted liquidity
            percentile = np.percentile(all_liquidities, 
                                     [avg_predicted_liquidity > liq for liq in all_liquidities].count(True) / len(all_liquidities) * 100)
            
            # Normalize to 0-1 scale (higher liquidity = better score)
            score = min(1.0, avg_predicted_liquidity / np.percentile(all_liquidities, 90))
            
            bt.logging.debug(f"Liquidity correlation score: {score:.4f} "
                           f"(avg predicted: {avg_predicted_liquidity:.2f})")
            
            return score
            
        except Exception as e:
            bt.logging.error(f"Error calculating liquidity correlation: {str(e)}")
            return 0.0

    def _calculate_confidence_calibration(self, prediction: TokenPrediction, 
                                        ground_truth: set) -> float:
        """
        Calculate how well-calibrated the confidence scores are.
        Well-calibrated means high confidence predictions are more accurate.
        """
        try:
            if not prediction.confidence_scores or not ground_truth:
                return 0.0
            
            # Group predictions by confidence level
            confidence_buckets = defaultdict(list)
            
            for addr, confidence in prediction.confidence_scores.items():
                bucket = int(confidence * 10) / 10  # Round to nearest 0.1
                is_correct = addr in ground_truth
                confidence_buckets[bucket].append(is_correct)
            
            if not confidence_buckets:
                return 0.0
            
            # Calculate calibration error
            calibration_error = 0.0
            total_predictions = sum(len(bucket) for bucket in confidence_buckets.values())
            
            for confidence_level, correctness_list in confidence_buckets.items():
                if not correctness_list:
                    continue
                
                actual_accuracy = sum(correctness_list) / len(correctness_list)
                expected_accuracy = confidence_level
                
                # Weight by number of predictions in this bucket
                weight = len(correctness_list) / total_predictions
                calibration_error += weight * abs(actual_accuracy - expected_accuracy)
            
            # Convert to score (lower error = higher score)
            calibration_score = max(0.0, 1.0 - calibration_error)
            
            bt.logging.debug(f"Confidence calibration score: {calibration_score:.4f} "
                           f"(error: {calibration_error:.4f})")
            
            return calibration_score
            
        except Exception as e:
            bt.logging.error(f"Error calculating confidence calibration: {str(e)}")
            return 0.0

    def _calculate_response_time_score(self, prediction: TokenPrediction) -> float:
        """Calculate score based on inference time (faster is better)"""
        try:
            inference_time = getattr(prediction, 'inference_time_ms', 0)
            
            if inference_time <= 0:
                return 1.0  # No timing data, assume perfect
            
            # Score based on response time (exponential decay)
            # Target: under 1000ms = 1.0, over 5000ms = 0.0
            max_time_ms = 5000
            target_time_ms = 1000
            
            if inference_time <= target_time_ms:
                score = 1.0
            elif inference_time >= max_time_ms:
                score = 0.0
            else:
                # Exponential decay between target and max
                ratio = (inference_time - target_time_ms) / (max_time_ms - target_time_ms)
                score = np.exp(-3 * ratio)  # Decay factor of 3
            
            bt.logging.debug(f"Response time score: {score:.4f} "
                           f"(inference time: {inference_time}ms)")
            
            return score
            
        except Exception as e:
            bt.logging.error(f"Error calculating response time score: {str(e)}")
            return 1.0  # Default to perfect score

    def _calculate_prediction_diversity(self, prediction: TokenPrediction) -> float:
        """Calculate diversity score based on prediction spread"""
        try:
            if not prediction.addresses:
                return 0.0
            
            num_predictions = len(prediction.addresses)
            
            # Score based on number of predictions (more diversity = better, up to a point)
            optimal_predictions = 10
            max_predictions = 20
            
            if num_predictions <= optimal_predictions:
                score = num_predictions / optimal_predictions
            else:
                # Diminishing returns after optimal number
                excess = num_predictions - optimal_predictions
                max_excess = max_predictions - optimal_predictions
                penalty = min(1.0, excess / max_excess) * 0.5  # Max 50% penalty
                score = 1.0 - penalty
            
            bt.logging.debug(f"Diversity score: {score:.4f} "
                           f"({num_predictions} predictions)")
            
            return score
            
        except Exception as e:
            bt.logging.error(f"Error calculating diversity score: {str(e)}")
            return 0.0

    def _update_historical_performance(self, miner_uid: int, result: ModelEvaluationResult):
        """Update historical performance tracking for miner"""
        try:
            if miner_uid not in self.historical_performance:
                self.historical_performance[miner_uid] = []
            
            # Add current result
            self.historical_performance[miner_uid].append({
                'timestamp': datetime.utcnow(),
                'overall_score': result.overall_score,
                'scores': result.scores.copy(),
                'round_id': result.metadata.get('round_id', '')
            })
            
            # Keep only last 50 evaluations per miner
            self.historical_performance[miner_uid] = \
                self.historical_performance[miner_uid][-50:]
            
            bt.logging.debug(f"Updated historical performance for miner {miner_uid}")
            
        except Exception as e:
            bt.logging.error(f"Error updating historical performance: {str(e)}")

    def _fallback_evaluation(self, submissions: List[Tuple[int, TokenPrediction]]) -> List[ModelEvaluationResult]:
        """Fallback evaluation when benchmark data is unavailable"""
        bt.logging.warning("Using fallback evaluation method")
        
        results = []
        for miner_uid, prediction in submissions:
            # Simple scoring based on prediction quality metrics
            num_predictions = len(prediction.addresses) if prediction.addresses else 0
            has_confidence = bool(prediction.confidence_scores)
            inference_time = getattr(prediction, 'inference_time_ms', 1000)
            
            # Basic scoring
            prediction_score = min(1.0, num_predictions / 10.0)  # 10 predictions = 1.0
            confidence_score = 1.0 if has_confidence else 0.5
            time_score = max(0.0, 1.0 - (inference_time / 5000.0))  # 5000ms = 0.0
            
            overall_score = (prediction_score + confidence_score + time_score) / 3.0
            
            result = ModelEvaluationResult(
                miner_uid=miner_uid,
                scores={
                    'accuracy': overall_score,
                    'prediction_quality': prediction_score,
                    'confidence_quality': confidence_score,
                    'response_time': time_score
                },
                metadata={'evaluation_method': 'fallback'}
            )
            
            results.append(result)
        
        results.sort(key=lambda x: x.overall_score, reverse=True)
        return results

    def get_miner_performance_history(self, miner_uid: int, days: int = 7) -> List[Dict[str, Any]]:
        """Get historical performance for a specific miner"""
        if miner_uid not in self.historical_performance:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        recent_performance = [
            perf for perf in self.historical_performance[miner_uid]
            if perf['timestamp'] > cutoff_time
        ]
        
        return recent_performance

    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get overall evaluation statistics"""
        try:
            total_evaluations = sum(
                len(history) for history in self.historical_performance.values()
            )
            
            active_miners = len(self.historical_performance)
            
            # Calculate average scores across all recent evaluations
            recent_scores = []
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            for history in self.historical_performance.values():
                recent_evals = [
                    eval_data for eval_data in history
                    if eval_data['timestamp'] > cutoff_time
                ]
                recent_scores.extend([eval_data['overall_score'] for eval_data in recent_evals])
            
            avg_score = np.mean(recent_scores) if recent_scores else 0.0
            max_score = np.max(recent_scores) if recent_scores else 0.0
            
            return {
                'total_evaluations': total_evaluations,
                'active_miners': active_miners,
                'recent_evaluations_24h': len(recent_scores),
                'average_score_24h': float(avg_score),
                'max_score_24h': float(max_score),
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            bt.logging.error(f"Error getting evaluation statistics: {str(e)}")
            return {}