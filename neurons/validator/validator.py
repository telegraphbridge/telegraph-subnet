import asyncio
import bittensor as bt
import time
import numpy as np
import random
import traceback
from typing import Dict, List, Any, Optional
from base.validator import BaseValidatorNeuron
from telegraph.protocol import PredictionSynapse, InferenceRequestSynapse
from telegraph.registry import InferenceRegistry
from base.types import ChainType, TokenPrediction
from .storage.prediction_store import PredictionStore
from .utils.performance_calculator import PerformanceCalculator
from .utils.uids import get_miner_uids
from datetime import datetime
from .utils.consensus_manager import ConsensusManager
from .utils.leaderboard import Leaderboard
from .utils.submission_validator import SubmissionValidator
from .utils.competition_fee_manager import CompetitionFeeManager
from .utils.competition_round_manager import CompetitionRoundManager, RoundState
from .utils.benchmark_dataset import BenchmarkDatasetManager
from .utils.model_evaluator import ModelEvaluationFramework, ModelEvaluationResult
from .utils.submission_validator import SubmissionValidator
from .utils.consensus_manager import ConsensusManager
from .utils.leaderboard import Leaderboard
from .analytics.analytics_collector import AnalyticsCollector
from .analytics.dashboard_server import DashboardServer

class TelegraphValidator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super(TelegraphValidator, self).__init__(config=config)
        self.inference_registry = InferenceRegistry()
        import os
        os.makedirs("data/transactions", exist_ok=True)
        os.makedirs("data/predictions", exist_ok=True)
        bt.logging.info("Loading validator state")
        self.load_state()
        self.prediction_store = PredictionStore()
        self.performance_calculator = PerformanceCalculator()
        # Reinitialize inference registry
        self.inference_registry = InferenceRegistry()
        self._dendrite = bt.dendrite(wallet=self.wallet)
        self.consensus_manager = ConsensusManager()
        self.leaderboard = Leaderboard()
        self.submission_validator = SubmissionValidator()
        self.competition_fee_manager = CompetitionFeeManager(
            storage_path="data/fees",
            base_fee_tao=0.001  # 0.001 TAO base fee
        )
        bt.logging.info("Competition fee manager initialized")
        self.competition_manager = CompetitionRoundManager()
        bt.logging.info("Competition round manager initialized")
        self.benchmark_manager = BenchmarkDatasetManager(
                dataset_dir="data/benchmark_datasets",
                auto_update_hours=6  # Update datasets every 6 hours
        )
        # Initialize datasets for supported chains if they don't exist
        asyncio.create_task(self._initialize_benchmark_datasets())
        bt.logging.info("Benchmark dataset manager initialized")
        self.model_evaluator = ModelEvaluationFramework(
            benchmark_manager=self.benchmark_manager,
            liquidity_checker=self.performance_calculator.liquidity_checker
        )
        
        bt.logging.info("Model evaluation framework initialized")

        self.submission_validator = SubmissionValidator(
            min_confidence=0.0,  # Allow low confidence for development
            max_submission_age_minutes=30  # 30 minute window for submissions
        )
        
        # Initialize consensus manager with proper storage
        self.consensus_manager = ConsensusManager(
            storage_path="data/consensus/preferred_miner.json",
            vote_threshold=0.51,  # 51% threshold for consensus
            max_history_entries=1000
        )
        
        # Initialize leaderboard with proper storage
        self.leaderboard = Leaderboard(
            leaderboard_path="data/leaderboard.json",
            max_history_entries=1000
        )
        
        bt.logging.info("Submission validator, consensus manager, and leaderboard initialized")
        self.analytics_collector = AnalyticsCollector(
            storage_path="data/analytics",
            buffer_size=1000
        )
        
        # Initialize dashboard server
        self.dashboard_server = DashboardServer(
            analytics_collector=self.analytics_collector,
            validator_instance=self,
            host="0.0.0.0",
            port=8080
        )
        
        # Start background tasks
        asyncio.create_task(self._start_analytics_background_tasks())
        bt.logging.info("Analytics system and dashboard initialized")

        if not self.config.neuron.axon_off:
            self.setup_inference_handlers()

    async def _initialize_benchmark_datasets(self):
        """Initialize benchmark datasets for all supported chains"""
        try:
            supported_chains = ['base']  # Add more chains as needed
            
            for chain in supported_chains:
                # Check if dataset exists and is recent
                if not self.benchmark_manager.get_dataset_version(chain):
                    bt.logging.info(f"Creating initial benchmark dataset for {chain}")
                    
                    # Generate initial sample dataset
                    sample_data = self.benchmark_manager.generate_sample_dataset(chain, size=200)
                    
                    version = self.benchmark_manager.add_dataset(
                        chain=chain,
                        dataset=sample_data,
                        description=f"Initial benchmark dataset for {chain}",
                        validate=True
                    )
                    
                    if version:
                        bt.logging.info(f"Created initial dataset for {chain}: {version}")
                    else:
                        bt.logging.error(f"Failed to create initial dataset for {chain}")
            
            # Set up periodic dataset updates
            asyncio.create_task(self._periodic_dataset_updates())
            
        except Exception as e:
            bt.logging.error(f"Error initializing benchmark datasets: {e}")

    # Add this new method to TelegraphValidator class:
    async def _periodic_dataset_updates(self):
        """Periodically update benchmark datasets"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                supported_chains = ['base']
                await self.benchmark_manager.auto_update_datasets(supported_chains)
                
                # Clean up old dataset files
                self.benchmark_manager.cleanup_old_datasets(keep_versions=3)
                
            except Exception as e:
                bt.logging.error(f"Error in periodic dataset updates: {e}")

    def setup_inference_handlers(self):
        def handle(syn: InferenceRequestSynapse) -> InferenceRequestSynapse:
            return self.process_inference_request_sync(syn)
        self.axon.attach(
            forward_fn=handle,
            # blacklist_fn=self.blacklist_inference_request,
            priority_fn=self.priority_inference_request
        )
        bt.logging.info("Attached inference handler")

    def blacklist_inference_request(self, synapse: InferenceRequestSynapse) -> tuple[bool, str]:
        # For MVP, allow all requests.
        return False, "ok"

    async def priority_inference_request(self, synapse: InferenceRequestSynapse) -> float:
        # Metagraph doesn't have 'validators' attribute, use hotkeys instead
        if synapse.dendrite.hotkey in self.metagraph.hotkeys:
            return 1.0
        return 0.5

    def process_inference_request_sync(self, syn: InferenceRequestSynapse) -> InferenceRequestSynapse:
        return asyncio.get_event_loop().run_until_complete(self.process_inference_request(syn))

    async def process_inference_request(self, syn: InferenceRequestSynapse) -> InferenceRequestSynapse:
        bt.logging.info(f"Incoming cross‑subnet request: {syn.inference_code}")
        if not self.inference_registry.is_valid_code(syn.inference_code):
            syn.error = f"Unknown code {syn.inference_code}"
            return syn
        syn.response = await self.route_inference_request(syn.inference_code, syn.data)
        return syn

    async def route_inference_request(self, inference_code: str, data: Any) -> Any:
        target_netuid = self.inference_registry.get_netuid(inference_code)
        if target_netuid is None:
            return {"error": f"Invalid code {inference_code}"}
        bt.logging.info(f"Routing to subnet {target_netuid} with code={inference_code}")
        subtensor = bt.subtensor(network=self.config.subtensor.network)
        m = subtensor.metagraph(netuid=target_netuid)
        m.sync(subtensor=subtensor)
        # Use all nodes (or filter using get_miner_uids) since m.validators is unavailable.
        peers = list(range(m.n))
        stakes = np.array([m.S[uid] for uid in peers], dtype=float)
        if stakes.sum() > 0:
            idx = np.random.choice(len(peers), p=stakes/stakes.sum())
        else:
            idx = random.randrange(len(peers))
        uid = peers[idx]
        ax = m.axons[uid]
        bt.logging.info(f"Selected peer uid={uid}, hotkey={m.hotkeys[uid]}")
        req = InferenceRequestSynapse(inference_code=inference_code, data=data)
        resp = await bt.dendrite(wallet=self.wallet)(
            axons=[ax],
            synapse=req,
            deserialize=True,
            timeout=30.0
        )
        syn = resp[0] if isinstance(resp, list) else resp
        if syn.error:
            return {"error": syn.error}
        return getattr(syn, "response", None)


    async def forward(self):
        """Main validator loop with competition round management."""
        try:
            # 1. Update round states and start new rounds if needed
            self.competition_manager.update_round_states()
            
            # Check if we should start new rounds for any chains
            for chain in [ChainType.BASE]:  # Add other chains as needed
                    if self.competition_manager.should_start_new_round(chain):
                        # Get current dataset version from benchmark manager
                        dataset_version = self.benchmark_manager.get_dataset_version(chain.value.lower())
                        if not dataset_version:
                            # Create dataset if it doesn't exist
                            sample_data = self.benchmark_manager.generate_sample_dataset(chain.value.lower(), size=100)
                            dataset_version = self.benchmark_manager.add_dataset(
                                chain=chain.value.lower(),
                                dataset=sample_data,
                                description=f"Competition dataset for {chain.value}"
                            )
                        
                        if dataset_version:
                            round_id = self.competition_manager.start_new_round(chain, dataset_version)
                            bt.logging.info(f"Started new competition round {round_id} for {chain.value} with dataset {dataset_version}")
                        else:
                            bt.logging.error(f"Failed to get dataset version for {chain.value}")
                

            # 2. Get available miner UIDs from the metagraph
            miner_uids = get_miner_uids(self.metagraph)
            if not miner_uids:
                bt.logging.warning("No available miners found in the metagraph.")
                await asyncio.sleep(60)
                return

            valid_miner_uids = [uid for uid in miner_uids if uid < self.metagraph.n]
            if not valid_miner_uids:
                bt.logging.warning("No valid miners available in the current metagraph slice.")
                await asyncio.sleep(60)
                return

            bt.logging.info(f"Querying {len(valid_miner_uids)} miners: {valid_miner_uids}")

            # 3. Auto-register miners for active competition rounds
            active_rounds = self.competition_manager.get_active_rounds()
            for round_id, round_info in active_rounds.items():
                if round_info['state'] == RoundState.REGISTRATION.value:
                    for uid in valid_miner_uids:
                        self.competition_manager.register_participant(round_id, uid)

            # 4. Query miners for predictions
            chain_type = ChainType.BASE
            synapse = PredictionSynapse(chain_name=chain_type.value)

            responses = await self._dendrite(
                axons=[self.metagraph.axons[uid] for uid in valid_miner_uids],
                synapse=synapse,
                deserialize=True,
                timeout=20.0
            )
            bt.logging.info(f"Received {len(responses)} raw responses.")

            # 5. Process responses and handle competition submissions
            valid_responses: List[PredictionSynapse] = []
            current_uids: List[int] = []

            # Get the current submission round for BASE chain
            submission_round_id = self.competition_manager.get_round_for_submission(ChainType.BASE)
            
            for i, resp in enumerate(responses):
                uid = valid_miner_uids[i]
                
                if resp and hasattr(resp, 'addresses') and resp.addresses:
                    # Validate the response
                    error = self.submission_validator.validate(self._response_to_prediction(resp))
                    if error:
                        bt.logging.warning(f"Invalid submission from UID {uid}: {error}")
                        continue
                        
                    valid_responses.append(resp)
                    current_uids.append(uid)
                    
                    # Submit to competition if there's an active round
                    if submission_round_id:
                        model_data = {
                            "addresses": resp.addresses,
                            "confidence_scores": getattr(resp, 'confidence_scores', {}),
                            "timestamp": datetime.utcnow().isoformat(),
                            "pairAddresses": getattr(resp, 'pairAddresses', [])
                        }
                        self.competition_manager.submit_model(submission_round_id, uid, model_data)
                else:
                    bt.logging.debug(f"Invalid or empty response from UID {uid}")

            if not valid_responses:
                bt.logging.warning("No valid responses received from miners.")
                await asyncio.sleep(30)
                return

            bt.logging.info(f"Processing {len(valid_responses)} valid responses from UIDs: {current_uids}")

            # 6. Store predictions and calculate rewards
            await self._store_predictions(valid_responses, current_uids)
            
            # 7. Process competition evaluations
            await self._process_competition_evaluations()
            
            # 8. Calculate rewards (now includes competition performance)
            rewards_dict = await self._calculate_rewards(valid_responses, current_uids)
            await self.run_consensus_and_select_preferred(rewards_dict)
            
            bt.logging.info(f"Rewards calculated: {rewards_dict}")
            await self.update_leaderboard(rewards_dict)
            await self._process_revenue_distribution()
            
            # Log fee statistics periodically
            if hasattr(self, '_last_fee_stats_log'):
                time_since_last_log = time.time() - self._last_fee_stats_log
                if time_since_last_log > 3600:  # Log every hour
                    await self._log_fee_statistics()
                    self._last_fee_stats_log = time.time()
            else:
                self._last_fee_stats_log = time.time()
        
            if hasattr(self, 'analytics_collector'):
                # Collect performance metrics
                response_times = {}
                memory_usage = {}
                
                # Extract response times and memory usage from responses
                for i, response in enumerate(valid_responses):
                    uid = current_uids[i]
                    response_times[uid] = getattr(response, 'inference_time_ms', 0)
                    memory_usage[uid] = getattr(response, 'memory_usage_mb', 0)
                
                # Collect all analytics
                self.analytics_collector.collect_performance_metrics(
                    rewards_dict, response_times, memory_usage
                )
                
                self.analytics_collector.collect_network_metrics(
                    self.metagraph, current_uids
                )
                
                self.analytics_collector.collect_competition_metrics(
                    self.competition_manager
                )
                
                self.analytics_collector.collect_fee_metrics(
                    self.competition_fee_manager
                )
                
                self.analytics_collector.collect_consensus_metrics(
                    self.consensus_manager
                )
                
                bt.logging.debug("Analytics data collected for current round")
        
            # 9. Update scores
            uids_to_update = np.array(list(rewards_dict.keys()), dtype=int)
            rewards_to_update = np.array(list(rewards_dict.values()), dtype=float)

            if len(rewards_to_update) > 0:
                await self.update_scores(rewards_to_update, uids_to_update)
                bt.logging.info(f"Updated scores for {len(uids_to_update)} miners")
            else:
                bt.logging.warning("No rewards to update")

            self.save_state()

        except Exception as e:
            bt.logging.error(f"Error in validator forward loop: {str(e)}")
            bt.logging.error(traceback.format_exc())
            await asyncio.sleep(60)

    async def _process_competition_evaluations(self):
        """Process rounds that are ready for evaluation using comprehensive model evaluation"""
        evaluation_rounds = self.competition_manager.get_rounds_for_evaluation()
        
        for round_id in evaluation_rounds:
            round_info = self.competition_manager.get_round_info(round_id)
            if not round_info:
                continue
                
            bt.logging.info(f"Processing comprehensive evaluation for round {round_id}")
            
            # Get submissions for this round
            submissions = round_info.get('submissions', {})
            if not submissions:
                bt.logging.warning(f"No submissions found for round {round_id}")
                self.competition_manager.complete_round(round_id)
                continue
            
            # Convert submissions to evaluation format
            evaluation_submissions = []
            
            for uid_str, submission_data in submissions.items():
                uid = int(uid_str)
                model_data = submission_data['model_data']
                
                prediction = TokenPrediction(
                    chain=ChainType(round_info['chain']),
                    addresses=model_data['addresses'],
                    pairAddresses=model_data.get('pairAddresses', []),
                    timestamp=datetime.fromisoformat(model_data['timestamp']),
                    confidence_scores=model_data['confidence_scores'],
                    inference_time_ms=model_data.get('inference_time_ms', 0),
                    memory_usage_mb=model_data.get('memory_usage_mb', 0)
                )
                
                evaluation_submissions.append((uid, prediction))
            
            # Run comprehensive model evaluation
            try:
                evaluation_results = await self.model_evaluator.evaluate_models(
                    evaluation_submissions, round_id
                )
                
                if evaluation_results:
                    # Convert evaluation results to competition results
                    results = {}
                    for eval_result in evaluation_results:
                        results[eval_result.miner_uid] = eval_result.overall_score
                        
                        # Log detailed results
                        bt.logging.info(f"Miner {eval_result.miner_uid} evaluation: "
                                    f"overall={eval_result.overall_score:.4f}, "
                                    f"accuracy={eval_result.scores.get('accuracy', 0):.4f}, "
                                    f"precision={eval_result.scores.get('precision', 0):.4f}")
                    
                    # Store results in competition manager
                    self.competition_manager.store_evaluation_results(round_id, results)
                    bt.logging.info(f"Stored comprehensive evaluation results for round {round_id}")
                    
                else:
                    bt.logging.warning(f"No evaluation results for round {round_id}")
                    self.competition_manager.complete_round(round_id)
                    
            except Exception as e:
                bt.logging.error(f"Error in comprehensive evaluation for round {round_id}: {str(e)}")
                self.competition_manager.complete_round(round_id)

    async def _store_predictions(self, responses: List[PredictionSynapse], uids: List[int]):
        from datetime import datetime
        if len(responses) != len(uids):
            bt.logging.error(f"Mismatch storing predictions: {len(responses)} responses, {len(uids)} UIDs")
            return

        recent_hashes = set()
        num_submissions = len(responses)
        fee = self.competition_fee_manager.calculate_fee(num_submissions)
        stored_count = 0  
        total_fees_collected = 0.0  

        for i, response in enumerate(responses):
            uid = uids[i]
            try:
                if not getattr(response, "addresses", None):
                    bt.logging.warning(f"Skipping storing prediction for UID {uid}: Missing addresses.")
                    continue

                confidence_scores = response.confidence_scores if hasattr(response, "confidence_scores") else {}
                chain_name = response.chain_name if hasattr(response, "chain_name") and response.chain_name else ChainType.BASE.value
                inference_time = getattr(response, "inference_time_ms", 0)
                memory_usage = getattr(response, "memory_usage_mb", 0)
                bt.logging.info(f"Miner UID {uid} inference time: {inference_time} ms, memory usage: {memory_usage} MB")

                prediction = TokenPrediction(
                    chain=ChainType(chain_name),
                    addresses=response.addresses,
                    pairAddresses=getattr(response, 'pairAddresses', []),
                    timestamp=datetime.now(),
                    confidence_scores=confidence_scores,
                    inference_time_ms=inference_time,
                    memory_usage_mb=memory_usage
                )

                # Validate submission using the submission validator
                validation_error = self.submission_validator.validate(prediction)
                if validation_error:
                    bt.logging.warning(f"Rejected prediction from UID {uid}: {validation_error}")
                    continue

                # Check for duplicates
                if self.submission_validator.is_duplicate(prediction, recent_hashes):
                    bt.logging.warning(f"Duplicate prediction from UID {uid} detected, skipping.")
                    continue

                # Calculate and process fee
                miner_stats = self.competition_fee_manager.get_miner_fee_stats(uid)
                
                # Get miner's performance history for quality scoring
                performance_history = []
                if hasattr(self, 'performance_calculator') and hasattr(self.performance_calculator, 'historical_scores'):
                    miner_history = self.performance_calculator.historical_scores.get(uid, [])
                    performance_history = [entry['score'] for entry in miner_history[-10:]]  # Last 10 scores
                
                quality_score = self.competition_fee_manager.calculate_miner_quality_score(uid, performance_history)
                
                fee_amount = self.competition_fee_manager.calculate_fee(
                    miner_uid=uid,
                    submission_count=1,
                    quality_score=quality_score,
                    is_repeat_miner=miner_stats['is_repeat_miner']
                )
                
                # Record the fee payment (in production, this would involve actual TAO transfer)
                submission_data = {
                    'addresses_count': len(prediction.addresses),
                    'has_confidence_scores': bool(prediction.confidence_scores),
                    'chain': prediction.chain.value,
                    'quality_score': quality_score,
                    'inference_time_ms': prediction.inference_time_ms,
                    'memory_usage_mb': prediction.memory_usage_mb
                }
                
                fee_recorded = self.competition_fee_manager.record_submission(
                    miner_uid=uid,
                    fee_paid=fee_amount,
                    submission_data=submission_data
                )
                
                if fee_recorded:
                    total_fees_collected += fee_amount
                    
                # Add to recent hashes for this batch
                prediction_hash = f"{uid}_{hash(tuple(prediction.addresses))}"
                recent_hashes.add(prediction_hash)

                # Store the prediction
                await self.prediction_store.store_prediction(uid, prediction)
                stored_count += 1
                bt.logging.debug(f"Stored prediction from UID {uid} for current round.")
                
            except Exception as e:
                bt.logging.error(f"Error storing prediction for UID {uid}: {str(e)}")
                import traceback
                bt.logging.debug(traceback.format_exc())
        bt.logging.info(f"Successfully stored {stored_count}/{len(responses)} predictions")
        bt.logging.info(f"Total fees collected this round: {total_fees_collected:.6f} TAO")

    async def _calculate_rewards(self, responses: List[PredictionSynapse], uids: List[int]) -> Dict[int, float]:
        """Calculate rewards using enhanced performance evaluation with Netheril-compliant scoring"""
        if len(responses) != len(uids):
            bt.logging.error(f"Mismatch calculating rewards: {len(responses)} responses, {len(uids)} UIDs")
            return {}

        rewards = {}
        current_predictions = []
        valid_uids_for_perf = []

        for i, response in enumerate(responses):
            uid = uids[i]
            if getattr(response, "addresses", None):
                chain_name = response.chain_name if hasattr(response, "chain_name") and response.chain_name else ChainType.BASE.value
                
                prediction = TokenPrediction(
                    chain=ChainType(chain_name),
                    addresses=response.addresses,
                    pairAddresses=getattr(response, 'pairAddresses', []),
                    timestamp=datetime.now(),
                    confidence_scores=response.confidence_scores if hasattr(response, "confidence_scores") else {},
                    inference_time_ms=getattr(response, 'inference_time_ms', 0),
                    memory_usage_mb=getattr(response, 'memory_usage_mb', 0)
                )
                
                current_predictions.append(prediction)
                valid_uids_for_perf.append(uid)
            else:
                bt.logging.warning(f"Skipping reward calculation for UID {uid}: Invalid response.")

        if not current_predictions:
            bt.logging.warning("No valid predictions from current round to calculate performance.")
            return {}

        # Use enhanced performance calculation with comprehensive evaluation
        performance_uids, performance_scores = await self.performance_calculator.calculate_performance_with_evaluation(
            current_predictions, valid_uids_for_perf, f"reward_calc_{int(time.time())}"
        )

        if len(performance_uids) == 0:
            bt.logging.warning("Enhanced performance calculation returned no results.")
            return {}

        # Normalize scores using the enhanced normalization
        normalized_scores = self.performance_calculator.normalize_performance(performance_scores)

        # Map normalized scores back to UIDs
        for i, uid in enumerate(performance_uids):
            rewards[uid] = normalized_scores[i]

        # Log enhanced scoring details
        bt.logging.info(f"Enhanced reward calculation completed for {len(rewards)} miners")
        bt.logging.info(f"Performance score distribution: mean={np.mean(performance_scores):.4f}, "
                    f"std={np.std(performance_scores):.4f}")
        bt.logging.info(f"Top 3 performers: {sorted(rewards.items(), key=lambda x: x[1], reverse=True)[:3]}")

        return rewards

    async def run_consensus_and_select_preferred(self, rewards_dict: Dict[int, float]):
        """
        Runs consensus voting and selects the preferred miner for this round.
        """
        try:
            round_id = f"consensus_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Record votes based on performance rewards
            votes = rewards_dict.copy()  # {miner_uid: weight}
            self.consensus_manager.record_votes(round_id, votes)
            
            # Select preferred miner based on consensus
            preferred_uid = self.consensus_manager.select_preferred_miner(round_id, votes)
            
            if preferred_uid is not None:
                bt.logging.info(f"[Consensus] Preferred miner selected: UID {preferred_uid}")
                
                # --- MSG REWARD HOOK ---
                # TODO: Mint and distribute MSG reward to preferred miner here.
                # Example:
                # self.msg_reward_manager.reward_preferred_miner(preferred_uid, amount)
                bt.logging.info(f"[MSG REWARD] TODO: Mint/distribute MSG to preferred miner UID {preferred_uid}")
            else:
                bt.logging.warning("[Consensus] No preferred miner selected this round.")
            
            return preferred_uid
            
        except Exception as e:
            bt.logging.error(f"Error in consensus and preferred miner selection: {str(e)}")
            return None

    async def update_leaderboard(self, rewards_dict: Dict[int, float]):
        """
        Updates the leaderboard with current miner stats, preferred miner, and consensus history.
        """
        try:
            # Gather miner stats from rewards
            miner_stats = {}
            for uid, score in rewards_dict.items():
                miner_stats[uid] = {
                    "normalized_score": score,
                    "timestamp": datetime.utcnow().isoformat(),
                    # Add more stats here if available from performance calculator
                }
            
            # Get preferred miner
            preferred_uid = self.consensus_manager.get_preferred_miner()
            
            # Get recent consensus history
            consensus_history = self.consensus_manager.get_consensus_history(limit=10)
            
            # Update leaderboard
            self.leaderboard.update(miner_stats, preferred_uid, consensus_history)
            
            bt.logging.info(f"[Leaderboard] Updated with {len(miner_stats)} miners, preferred_uid={preferred_uid}")
            
        except Exception as e:
            bt.logging.error(f"Error updating leaderboard: {str(e)}")
            
    # Add this new method to handle revenue distribution:
    async def _process_revenue_distribution(self):
        """Process revenue distribution if enough fees have been collected"""
        try:
            # Get preferred miner for bonus distribution
            preferred_miner_uid = self.consensus_manager.get_preferred_miner()
            
            # Distribute revenue
            distribution = self.competition_fee_manager.distribute_revenue(preferred_miner_uid)
            
            if distribution:
                bt.logging.info("Revenue distribution completed:")
                for category, amount in distribution.items():
                    if category != 'preferred_miner_uid':
                        bt.logging.info(f"  {category}: {amount:.6f} TAO")
                
                # TODO: Implement actual TAO transfers here
                # In production, this would involve:
                # 1. Transfer to validator pool
                # 2. Transfer to development fund
                # 3. Transfer bonus to preferred miner
                # 4. Burn specified amount
                
            return distribution
            
        except Exception as e:
            bt.logging.error(f"Error processing revenue distribution: {str(e)}")
            return None

    async def _log_fee_statistics(self):
        """Log comprehensive fee statistics for monitoring"""
        try:
            stats = self.competition_fee_manager.get_fee_statistics()
            
            bt.logging.info("=== Fee Management Statistics ===")
            bt.logging.info(f"Total collected (all time): {stats.get('total_collected_all_time', 0):.6f} TAO")
            bt.logging.info(f"Fees collected (24h): {stats.get('fees_collected_24h', 0):.6f} TAO")
            bt.logging.info(f"Submissions (24h): {stats.get('submissions_24h', 0)}")
            bt.logging.info(f"Active miners (24h): {stats.get('active_miners_24h', 0)}")
            bt.logging.info(f"Average fee (24h): {stats.get('average_fee_24h', 0):.6f} TAO")
            bt.logging.info(f"Current base fee: {stats.get('current_base_fee', 0):.6f} TAO")
            bt.logging.info(f"Surge multiplier: {stats.get('surge_multiplier', 1.0):.2f}x")
            
            last_distribution = stats.get('last_distribution')
            if last_distribution:
                dist_time = last_distribution['timestamp']
                dist_amount = last_distribution['total_distributed']
                bt.logging.info(f"Last distribution: {dist_amount:.6f} TAO at {dist_time}")
            
            bt.logging.info("================================")
            
        except Exception as e:
            bt.logging.error(f"Error logging fee statistics: {str(e)}")
        

    async def _start_analytics_background_tasks(self):
        """Start analytics background tasks"""
        try:
            # Start dashboard server
            await self.dashboard_server.start_server()
            
            # Start periodic aggregation task
            asyncio.create_task(self.analytics_collector.run_periodic_aggregation())
            
            bt.logging.info("Analytics background tasks started successfully")
            
        except Exception as e:
            bt.logging.error(f"Error starting analytics background tasks: {str(e)}")

    async def cleanup(self):
        """Cleanup resources when validator shuts down"""
        try:
            if hasattr(self, 'dashboard_server'):
                await self.dashboard_server.stop_server()
                
            if hasattr(self, 'analytics_collector'):
                self.analytics_collector._save_historical_data()
                
            bt.logging.info("Validator cleanup completed")
            
        except Exception as e:
            bt.logging.error(f"Error during validator cleanup: {str(e)}")
            """Cleanup resources when validator shuts down"""
            try:
                if hasattr(self, 'dashboard_server'):
                    await self.dashboard_server.stop_server()
                    
                if hasattr(self, 'analytics_collector'):
                    self.analytics_collector._save_historical_data()
                    
                bt.logging.info("Validator cleanup completed")
                
            except Exception as e:
                bt.logging.error(f"Error during validator cleanup: {str(e)}")
                

    # Complete the missing methods in the existing code:
    def _auto_register_miners_for_round(self, round_id: str, valid_miner_uids: List[int]):
        """Auto-register miners for active competition rounds"""
        for uid in valid_miner_uids:
            self.competition_manager.register_participant(round_id, uid)

    def _submit_to_competition_round(self, submission_round_id: str, uid: int, prediction):
        """Submit a prediction to the current competition round"""
        model_data = {
            'addresses': prediction.addresses,
            'pairAddresses': prediction.pairAddresses,
            'timestamp': prediction.timestamp.isoformat(),
            'confidence_scores': prediction.confidence_scores,
            'inference_time_ms': prediction.inference_time_ms,
            'memory_usage_mb': prediction.memory_usage_mb
        }
        
        success = self.competition_manager.submit_model(submission_round_id, uid, model_data)
        if success:
            bt.logging.debug(f"Submitted model from miner {uid} to competition round {submission_round_id}")
        else:
            bt.logging.warning(f"Failed to submit model from miner {uid} to round {submission_round_id}")

    def _response_to_prediction(self, response: PredictionSynapse) -> TokenPrediction:
        """Convert response to TokenPrediction for validation"""
        return TokenPrediction(
            chain=ChainType.BASE,
            addresses=response.addresses or [],
            pairAddresses=getattr(response, 'pairAddresses', []),
            timestamp=datetime.utcnow(),
            confidence_scores=getattr(response, 'confidence_scores', {}),
            inference_time_ms=getattr(response, 'inference_time_ms', 0),
            memory_usage_mb=getattr(response, 'memory_usage_mb', 0)
        )

    def _extract_response_metrics(self, response: PredictionSynapse) -> tuple:
        """Extract response time and memory usage from response"""
        response_time = getattr(response, 'inference_time_ms', 0)
        memory_usage = getattr(response, 'memory_usage_mb', 0)
        return response_time, memory_usage
