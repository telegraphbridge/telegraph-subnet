import os
import json
import time
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import bittensor as bt

class AnalyticsCollector:
    """
    Production-ready analytics data collector for comprehensive subnet monitoring.
    Collects performance metrics, network health, and operational statistics.
    """
    
    def __init__(self, storage_path: str = "data/analytics", buffer_size: int = 1000):
        self.storage_path = storage_path
        self.buffer_size = buffer_size
        
        os.makedirs(storage_path, exist_ok=True)
        
        # Real-time data buffers
        self.performance_buffer = deque(maxlen=buffer_size)
        self.network_buffer = deque(maxlen=buffer_size)
        self.competition_buffer = deque(maxlen=buffer_size)
        self.fee_buffer = deque(maxlen=buffer_size)
        self.consensus_buffer = deque(maxlen=buffer_size)
        
        # Aggregated data storage
        self.hourly_aggregates = defaultdict(dict)
        self.daily_aggregates = defaultdict(dict)
        
        # Historical data
        self.historical_data = {
            'performance_trends': [],
            'network_health': [],
            'competition_stats': [],
            'fee_analytics': [],
            'consensus_metrics': []
        }
        
        # Load existing historical data
        self._load_historical_data()
        
        bt.logging.info(f"AnalyticsCollector initialized with storage at {storage_path}")

    def collect_performance_metrics(self, rewards: Dict[int, float], 
                                  response_times: Dict[int, float], 
                                  memory_usage: Dict[int, float]):
        """Collect performance metrics from current round"""
        try:
            timestamp = datetime.utcnow()
            
            # Calculate performance statistics
            reward_values = list(rewards.values())
            response_time_values = list(response_times.values())
            memory_values = list(memory_usage.values())
            
            performance_data = {
                'timestamp': timestamp.isoformat(),
                'rewards': {
                    'count': len(reward_values),
                    'mean': float(np.mean(reward_values)) if reward_values else 0.0,
                    'std': float(np.std(reward_values)) if reward_values else 0.0,
                    'max': float(np.max(reward_values)) if reward_values else 0.0,
                    'min': float(np.min(reward_values)) if reward_values else 0.0,
                    'distribution': self._calculate_distribution(reward_values)
                },
                'response_times': {
                    'count': len(response_time_values),
                    'mean': float(np.mean(response_time_values)) if response_time_values else 0.0,
                    'median': float(np.median(response_time_values)) if response_time_values else 0.0,
                    'p95': float(np.percentile(response_time_values, 95)) if response_time_values else 0.0,
                    'p99': float(np.percentile(response_time_values, 99)) if response_time_values else 0.0
                },
                'memory_usage': {
                    'count': len(memory_values),
                    'mean': float(np.mean(memory_values)) if memory_values else 0.0,
                    'max': float(np.max(memory_values)) if memory_values else 0.0,
                    'total': float(np.sum(memory_values)) if memory_values else 0.0
                },
                'individual_miners': {
                    uid: {
                        'reward': rewards.get(uid, 0.0),
                        'response_time': response_times.get(uid, 0.0),
                        'memory_usage': memory_usage.get(uid, 0.0)
                    }
                    for uid in set(list(rewards.keys()) + list(response_times.keys()) + list(memory_usage.keys()))
                }
            }
            
            self.performance_buffer.append(performance_data)
            
            bt.logging.debug(f"Collected performance metrics: {len(rewards)} miners, "
                           f"avg_reward={performance_data['rewards']['mean']:.4f}")
            
        except Exception as e:
            bt.logging.error(f"Error collecting performance metrics: {str(e)}")

    def collect_network_metrics(self, metagraph, active_uids: List[int]):
        """Collect network health and topology metrics"""
        try:
            timestamp = datetime.utcnow()
            
            # Calculate network statistics
            total_stake = float(np.sum([metagraph.S[uid] for uid in range(metagraph.n)]))
            active_stake = float(np.sum([metagraph.S[uid] for uid in active_uids if uid < metagraph.n]))
            
            network_data = {
                'timestamp': timestamp.isoformat(),
                'network_size': {
                    'total_nodes': metagraph.n,
                    'active_miners': len(active_uids),
                    'participation_rate': len(active_uids) / max(metagraph.n, 1)
                },
                'stake_distribution': {
                    'total_stake': total_stake,
                    'active_stake': active_stake,
                    'stake_concentration': self._calculate_stake_concentration(metagraph),
                    'top_10_stake_percent': self._calculate_top_stake_percentage(metagraph, 10)
                },
                'network_health': {
                    'consensus_weight': active_stake / max(total_stake, 1),
                    'decentralization_score': self._calculate_decentralization_score(metagraph),
                    'node_diversity': len(set(metagraph.hotkeys))
                },
                'active_miners': active_uids
            }
            
            self.network_buffer.append(network_data)
            
            bt.logging.debug(f"Collected network metrics: {metagraph.n} total nodes, "
                           f"{len(active_uids)} active, {network_data['network_health']['decentralization_score']:.3f} decentralization")
            
        except Exception as e:
            bt.logging.error(f"Error collecting network metrics: {str(e)}")

    def collect_competition_metrics(self, competition_manager):
        """Collect competition round and performance metrics"""
        try:
            timestamp = datetime.utcnow()
            
            # Get competition statistics
            active_rounds = competition_manager.get_active_rounds()
            completed_rounds = competition_manager.get_completed_rounds()
            round_stats = competition_manager.get_round_statistics()
            
            competition_data = {
                'timestamp': timestamp.isoformat(),
                'rounds': {
                    'active_count': len(active_rounds),
                    'completed_count': len(completed_rounds),
                    'total_rounds': round_stats.get('total_rounds', 0),
                    'active_details': {
                        round_id: {
                            'state': round_info['state'],
                            'participants': len(round_info.get('participants', [])),
                            'submissions': len(round_info.get('submissions', {})),
                            'chain': round_info['chain'],
                            'created_at': round_info['created_at']
                        }
                        for round_id, round_info in active_rounds.items()
                    }
                },
                'participation': {
                    'total_participants': round_stats.get('total_participants', 0),
                    'avg_participants_per_round': round_stats.get('avg_participants_per_round', 0),
                    'submission_success_rate': round_stats.get('submission_success_rate', 0)
                },
                'performance': {
                    'avg_evaluation_time': round_stats.get('avg_evaluation_time', 0),
                    'round_completion_rate': round_stats.get('completion_rate', 0)
                }
            }
            
            self.competition_buffer.append(competition_data)
            
            bt.logging.debug(f"Collected competition metrics: {len(active_rounds)} active rounds, "
                           f"{len(completed_rounds)} completed")
            
        except Exception as e:
            bt.logging.error(f"Error collecting competition metrics: {str(e)}")

    def collect_fee_metrics(self, fee_manager):
        """Collect fee and revenue metrics"""
        try:
            timestamp = datetime.utcnow()
            
            # Get fee statistics
            fee_stats = fee_manager.get_fee_statistics()
            
            fee_data = {
                'timestamp': timestamp.isoformat(),
                'revenue': {
                    'total_collected': fee_stats.get('total_collected_all_time', 0),
                    'collected_24h': fee_stats.get('fees_collected_24h', 0),
                    'collected_7d': fee_stats.get('fees_collected_7d', 0),
                    'collection_rate_24h': fee_stats.get('fees_collected_24h', 0) / 24  # Per hour
                },
                'submissions': {
                    'count_24h': fee_stats.get('submissions_24h', 0),
                    'count_7d': fee_stats.get('submissions_7d', 0),
                    'active_miners_24h': fee_stats.get('active_miners_24h', 0)
                },
                'pricing': {
                    'base_fee': fee_stats.get('current_base_fee', 0),
                    'average_fee_24h': fee_stats.get('average_fee_24h', 0),
                    'surge_multiplier': fee_stats.get('surge_multiplier', 1.0)
                },
                'distribution': fee_stats.get('last_distribution', {})
            }
            
            self.fee_buffer.append(fee_data)
            
            bt.logging.debug(f"Collected fee metrics: {fee_data['revenue']['total_collected']:.6f} TAO total, "
                           f"{fee_data['submissions']['count_24h']} submissions 24h")
            
        except Exception as e:
            bt.logging.error(f"Error collecting fee metrics: {str(e)}")

    def collect_consensus_metrics(self, consensus_manager):
        """Collect consensus and governance metrics"""
        try:
            timestamp = datetime.utcnow()
            
            # Get consensus statistics
            consensus_stats = consensus_manager.get_consensus_stats()
            consensus_history = consensus_manager.get_consensus_history(limit=24)  # Last 24 rounds
            
            # Calculate consensus health metrics
            recent_success_rate = 0.0
            if consensus_history:
                successful_rounds = sum(1 for h in consensus_history if 'preferred_miner' in h)
                recent_success_rate = successful_rounds / len(consensus_history)
            
            consensus_data = {
                'timestamp': timestamp.isoformat(),
                'consensus_health': {
                    'success_rate_overall': consensus_stats.get('consensus_success_rate', 0),
                    'success_rate_recent': recent_success_rate,
                    'total_rounds': consensus_stats.get('total_rounds', 0),
                    'vote_threshold': consensus_stats.get('vote_threshold', 0)
                },
                'preferred_miner': {
                    'current_uid': consensus_stats.get('current_preferred_miner'),
                    'selection_age_hours': consensus_stats.get('preferred_miner_age_hours'),
                    'stability_score': self._calculate_preferred_miner_stability(consensus_history)
                },
                'voting_patterns': {
                    'participation_diversity': self._calculate_voting_diversity(consensus_history),
                    'consensus_strength': self._calculate_consensus_strength(consensus_history)
                }
            }
            
            self.consensus_buffer.append(consensus_data)
            
            bt.logging.debug(f"Collected consensus metrics: {recent_success_rate:.2%} recent success rate")
            
        except Exception as e:
            bt.logging.error(f"Error collecting consensus metrics: {str(e)}")

    async def run_periodic_aggregation(self):
        """Run periodic data aggregation and cleanup"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Aggregate data
                self._aggregate_hourly_data()
                self._aggregate_daily_data()
                
                # Update historical trends
                self._update_historical_trends()
                
                # Save data
                self._save_historical_data()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                bt.logging.debug("Periodic analytics aggregation completed")
                
            except Exception as e:
                bt.logging.error(f"Error in periodic aggregation: {str(e)}")

    def get_realtime_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time data for dashboard display"""
        try:
            current_time = datetime.utcnow()
            
            # Get latest data from each buffer
            latest_performance = list(self.performance_buffer)[-1] if self.performance_buffer else {}
            latest_network = list(self.network_buffer)[-1] if self.network_buffer else {}
            latest_competition = list(self.competition_buffer)[-1] if self.competition_buffer else {}
            latest_fee = list(self.fee_buffer)[-1] if self.fee_buffer else {}
            latest_consensus = list(self.consensus_buffer)[-1] if self.consensus_buffer else {}
            
            # Calculate trends (last hour)
            hour_ago = current_time - timedelta(hours=1)
            
            dashboard_data = {
                'timestamp': current_time.isoformat(),
                'system_status': 'operational',  # Could be enhanced with health checks
                'overview': {
                    'active_miners': latest_network.get('network_size', {}).get('active_miners', 0),
                    'total_nodes': latest_network.get('network_size', {}).get('total_nodes', 0),
                    'current_rewards_mean': latest_performance.get('rewards', {}).get('mean', 0),
                    'fees_collected_24h': latest_fee.get('revenue', {}).get('collected_24h', 0),
                    'active_rounds': latest_competition.get('rounds', {}).get('active_count', 0),
                    'consensus_success_rate': latest_consensus.get('consensus_health', {}).get('success_rate_recent', 0)
                },
                'performance': {
                    'current_metrics': latest_performance,
                    'trends': self._calculate_performance_trends(hour_ago)
                },
                'network': {
                    'current_metrics': latest_network,
                    'health_score': self._calculate_network_health_score(latest_network)
                },
                'competition': {
                    'current_metrics': latest_competition,
                    'participation_trends': self._calculate_participation_trends(hour_ago)
                },
                'fees': {
                    'current_metrics': latest_fee,
                    'revenue_trends': self._calculate_revenue_trends(hour_ago)
                },
                'consensus': {
                    'current_metrics': latest_consensus,
                    'governance_health': self._calculate_governance_health(latest_consensus)
                }
            }
            
            return dashboard_data
            
        except Exception as e:
            bt.logging.error(f"Error getting dashboard data: {str(e)}")
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}

    def get_historical_analytics(self, timeframe: str = '24h') -> Dict[str, Any]:
        """Get historical analytics data for specified timeframe"""
        try:
            if timeframe == '1h':
                data_source = list(self.performance_buffer)[-12:]  # Last 12 entries (1 hour at 5min intervals)
            elif timeframe == '24h':
                data_source = self._get_data_for_timeframe(timedelta(hours=24))
            elif timeframe == '7d':
                data_source = self._get_data_for_timeframe(timedelta(days=7))
            else:
                data_source = list(self.performance_buffer)[-288:]  # Last 24 hours
            
            return {
                'timeframe': timeframe,
                'data_points': len(data_source),
                'performance_history': self._extract_performance_history(data_source),
                'network_history': self._extract_network_history(data_source),
                'revenue_history': self._extract_revenue_history(data_source),
                'trends': self._calculate_trend_analysis(data_source)
            }
            
        except Exception as e:
            bt.logging.error(f"Error getting historical analytics: {str(e)}")
            return {'error': str(e)}

    # Helper methods for calculations
    def _calculate_distribution(self, values: List[float]) -> Dict[str, float]:
        """Calculate distribution statistics"""
        if not values:
            return {'q25': 0, 'q50': 0, 'q75': 0, 'q90': 0, 'q95': 0}
        
        return {
            'q25': float(np.percentile(values, 25)),
            'q50': float(np.percentile(values, 50)),
            'q75': float(np.percentile(values, 75)),
            'q90': float(np.percentile(values, 90)),
            'q95': float(np.percentile(values, 95))
        }

    def _calculate_stake_concentration(self, metagraph) -> float:
        """Calculate stake concentration using Gini coefficient"""
        try:
            stakes = [metagraph.S[uid] for uid in range(metagraph.n)]
            stakes = sorted([s for s in stakes if s > 0])
            
            if not stakes:
                return 0.0
            
            # Gini coefficient calculation
            n = len(stakes)
            cumsum = np.cumsum(stakes)
            gini = (n + 1 - 2 * sum((n + 1 - i) * stake for i, stake in enumerate(stakes, 1)) / cumsum[-1]) / n
            
            return float(gini)
            
        except Exception:
            return 0.0

    def _calculate_top_stake_percentage(self, metagraph, top_n: int) -> float:
        """Calculate percentage of stake held by top N miners"""
        try:
            stakes = sorted([metagraph.S[uid] for uid in range(metagraph.n)], reverse=True)
            total_stake = sum(stakes)
            top_stake = sum(stakes[:top_n])
            
            return float(top_stake / max(total_stake, 1))
            
        except Exception:
            return 0.0

    def _calculate_decentralization_score(self, metagraph) -> float:
        """Calculate network decentralization score"""
        try:
            # Based on stake distribution and node diversity
            stake_concentration = self._calculate_stake_concentration(metagraph)
            unique_hotkeys = len(set(metagraph.hotkeys))
            total_nodes = metagraph.n
            
            # Higher score = more decentralized
            diversity_score = unique_hotkeys / max(total_nodes, 1)
            concentration_score = 1 - stake_concentration
            
            return float((diversity_score + concentration_score) / 2)
            
        except Exception:
            return 0.5

    def _calculate_preferred_miner_stability(self, consensus_history: List[Dict]) -> float:
        """Calculate stability of preferred miner selections"""
        try:
            if len(consensus_history) < 2:
                return 1.0
            
            # Count how often the preferred miner changes
            changes = 0
            for i in range(1, len(consensus_history)):
                prev_miner = consensus_history[i-1].get('preferred_miner')
                curr_miner = consensus_history[i].get('preferred_miner')
                if prev_miner != curr_miner:
                    changes += 1
            
            stability = 1 - (changes / (len(consensus_history) - 1))
            return float(stability)
            
        except Exception:
            return 0.5

    def _calculate_voting_diversity(self, consensus_history: List[Dict]) -> float:
        """Calculate diversity in voting patterns"""
        try:
            all_miners = set()
            for entry in consensus_history:
                votes = entry.get('votes', {})
                all_miners.update(votes.keys())
            
            # Diversity based on number of unique participants
            return float(len(all_miners) / max(len(consensus_history), 1))
            
        except Exception:
            return 0.0

    def _calculate_consensus_strength(self, consensus_history: List[Dict]) -> float:
        """Calculate strength of consensus decisions"""
        try:
            if not consensus_history:
                return 0.0
            
            strengths = []
            for entry in consensus_history:
                votes = entry.get('votes', {})
                if votes:
                    vote_values = list(votes.values())
                    max_vote = max(vote_values)
                    total_votes = sum(vote_values)
                    strength = max_vote / max(total_votes, 1)
                    strengths.append(strength)
            
            return float(np.mean(strengths)) if strengths else 0.0
            
        except Exception:
            return 0.0

    def _aggregate_hourly_data(self):
        """Aggregate data into hourly buckets"""
        try:
            current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
            hour_key = current_hour.isoformat()
            
            # Aggregate performance data
            hour_performance = [p for p in self.performance_buffer 
                              if datetime.fromisoformat(p['timestamp']).replace(minute=0, second=0, microsecond=0) == current_hour]
            
            if hour_performance:
                self.hourly_aggregates[hour_key] = {
                    'timestamp': hour_key,
                    'performance': self._aggregate_performance_data(hour_performance),
                    'data_points': len(hour_performance)
                }
            
        except Exception as e:
            bt.logging.error(f"Error in hourly aggregation: {str(e)}")

    def _aggregate_daily_data(self):
        """Aggregate data into daily buckets"""
        try:
            current_day = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            day_key = current_day.isoformat()
            
            # Get all hourly data for the current day
            day_hourly_data = [data for hour_key, data in self.hourly_aggregates.items()
                              if datetime.fromisoformat(hour_key).date() == current_day.date()]
            
            if day_hourly_data:
                self.daily_aggregates[day_key] = {
                    'timestamp': day_key,
                    'summary': self._aggregate_daily_summary(day_hourly_data),
                    'hourly_data_points': len(day_hourly_data)
                }
            
        except Exception as e:
            bt.logging.error(f"Error in daily aggregation: {str(e)}")

    def _save_historical_data(self):
        """Save historical data to persistent storage"""
        try:
            historical_file = os.path.join(self.storage_path, "historical_analytics.json")
            
            data_to_save = {
                'last_updated': datetime.utcnow().isoformat(),
                'hourly_aggregates': dict(list(self.hourly_aggregates.items())[-168:]),  # Last 7 days
                'daily_aggregates': dict(list(self.daily_aggregates.items())[-30:]),     # Last 30 days
                'buffer_sizes': {
                    'performance': len(self.performance_buffer),
                    'network': len(self.network_buffer),
                    'competition': len(self.competition_buffer),
                    'fee': len(self.fee_buffer),
                    'consensus': len(self.consensus_buffer)
                }
            }
            
            # Atomic write
            temp_file = historical_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            
            os.rename(temp_file, historical_file)
            
        except Exception as e:
            bt.logging.error(f"Error saving historical data: {str(e)}")

    def _load_historical_data(self):
        """Load historical data from persistent storage"""
        try:
            historical_file = os.path.join(self.storage_path, "historical_analytics.json")
            
            if os.path.exists(historical_file):
                with open(historical_file, 'r') as f:
                    loaded_data = json.load(f)
                    
                self.hourly_aggregates = defaultdict(dict, loaded_data.get('hourly_aggregates', {}))
                self.daily_aggregates = defaultdict(dict, loaded_data.get('daily_aggregates', {}))
                
                bt.logging.info(f"Loaded historical analytics data: "
                              f"{len(self.hourly_aggregates)} hourly, "
                              f"{len(self.daily_aggregates)} daily aggregates")
            
        except Exception as e:
            bt.logging.error(f"Error loading historical data: {str(e)}")

    def _cleanup_old_data(self):
        """Clean up old data to prevent unbounded growth"""
        try:
            # Keep only recent hourly aggregates (7 days)
            cutoff_time = datetime.utcnow() - timedelta(days=7)
            old_hourly_keys = [
                key for key in self.hourly_aggregates.keys()
                if datetime.fromisoformat(key) < cutoff_time
            ]
            for key in old_hourly_keys:
                del self.hourly_aggregates[key]
            
            # Keep only recent daily aggregates (30 days)
            cutoff_time = datetime.utcnow() - timedelta(days=30)
            old_daily_keys = [
                key for key in self.daily_aggregates.keys()
                if datetime.fromisoformat(key) < cutoff_time
            ]
            for key in old_daily_keys:
                del self.daily_aggregates[key]
            
            if old_hourly_keys or old_daily_keys:
                bt.logging.debug(f"Cleaned up {len(old_hourly_keys)} hourly and {len(old_daily_keys)} daily aggregates")
            
        except Exception as e:
            bt.logging.error(f"Error cleaning up old data: {str(e)}")

    # Additional helper methods for trend calculations
    def _calculate_performance_trends(self, since: datetime) -> Dict[str, float]:
        """Calculate performance trends since given time"""
        try:
            recent_data = [p for p in self.performance_buffer 
                          if datetime.fromisoformat(p['timestamp']) > since]
            
            if len(recent_data) < 2:
                return {'reward_trend': 0.0, 'response_time_trend': 0.0}
            
            # Calculate trends using linear regression or simple comparison
            rewards = [d['rewards']['mean'] for d in recent_data]
            response_times = [d['response_times']['mean'] for d in recent_data]
            
            reward_trend = (rewards[-1] - rewards[0]) / max(rewards[0], 0.001) if rewards else 0.0
            response_time_trend = (response_times[-1] - response_times[0]) / max(response_times[0], 0.001) if response_times else 0.0
            
            return {
                'reward_trend': float(reward_trend),
                'response_time_trend': float(response_time_trend)
            }
            
        except Exception:
            return {'reward_trend': 0.0, 'response_time_trend': 0.0}

    def _calculate_network_health_score(self, network_data: Dict) -> float:
        """Calculate overall network health score"""
        try:
            participation_rate = network_data.get('network_size', {}).get('participation_rate', 0)
            decentralization_score = network_data.get('network_health', {}).get('decentralization_score', 0)
            consensus_weight = network_data.get('network_health', {}).get('consensus_weight', 0)
            
            # Weighted health score
            health_score = (participation_rate * 0.4 + decentralization_score * 0.4 + consensus_weight * 0.2)
            
            return float(health_score)
            
        except Exception:
            return 0.5

    def _calculate_participation_trends(self, since: datetime) -> Dict[str, float]:
        """Calculate competition participation trends"""
        try:
            recent_data = [c for c in self.competition_buffer 
                          if datetime.fromisoformat(c['timestamp']) > since]
            
            if not recent_data:
                return {'participation_trend': 0.0, 'completion_trend': 0.0}
            
            avg_participants = np.mean([d['participation']['total_participants'] for d in recent_data])
            completion_rate = np.mean([d['performance']['round_completion_rate'] for d in recent_data])
            
            return {
                'participation_trend': float(avg_participants),
                'completion_trend': float(completion_rate)
            }
            
        except Exception:
            return {'participation_trend': 0.0, 'completion_trend': 0.0}

    def _calculate_revenue_trends(self, since: datetime) -> Dict[str, float]:
        """Calculate revenue and fee trends"""
        try:
            recent_data = [f for f in self.fee_buffer 
                          if datetime.fromisoformat(f['timestamp']) > since]
            
            if not recent_data:
                return {'revenue_trend': 0.0, 'fee_trend': 0.0}
            
            revenue_rate = np.mean([d['revenue']['collection_rate_24h'] for d in recent_data])
            avg_fee = np.mean([d['pricing']['average_fee_24h'] for d in recent_data])
            
            return {
                'revenue_trend': float(revenue_rate),
                'fee_trend': float(avg_fee)
            }
            
        except Exception:
            return {'revenue_trend': 0.0, 'fee_trend': 0.0}

    def _calculate_governance_health(self, consensus_data: Dict) -> float:
        """Calculate governance system health"""
        try:
            success_rate = consensus_data.get('consensus_health', {}).get('success_rate_recent', 0)
            stability_score = consensus_data.get('preferred_miner', {}).get('stability_score', 0)
            participation = consensus_data.get('voting_patterns', {}).get('participation_diversity', 0)
            
            # Combined governance health score
            health = (success_rate * 0.5 + stability_score * 0.3 + participation * 0.2)
            
            return float(health)
            
        except Exception:
            return 0.5
        
    def _update_historical_trends(self):
        """Update historical trend data"""
        try:
            current_time = datetime.utcnow()
            
            # Update performance trends
            if self.performance_buffer:
                latest_performance = list(self.performance_buffer)[-10:]  # Last 10 entries
                avg_rewards = [p['rewards']['mean'] for p in latest_performance if 'rewards' in p]
                
                if avg_rewards:
                    trend_entry = {
                        'timestamp': current_time.isoformat(),
                        'avg_reward': float(np.mean(avg_rewards)),
                        'trend_direction': 'up' if len(avg_rewards) > 1 and avg_rewards[-1] > avg_rewards[0] else 'down'
                    }
                    self.historical_data['performance_trends'].append(trend_entry)
                    
                    # Keep only last 100 entries
                    self.historical_data['performance_trends'] = self.historical_data['performance_trends'][-100:]
            
            # Update other trend data similarly
            bt.logging.debug("Updated historical trends")
            
        except Exception as e:
            bt.logging.error(f"Error updating historical trends: {str(e)}")

    def _aggregate_performance_data(self, hour_performance):
        """Aggregate performance data for hourly buckets"""
        try:
            if not hour_performance:
                return {}
            
            # Calculate hourly aggregates
            all_rewards = []
            all_response_times = []
            total_miners = set()
            
            for entry in hour_performance:
                if 'rewards' in entry:
                    all_rewards.append(entry['rewards']['mean'])
                if 'response_times' in entry:
                    all_response_times.append(entry['response_times']['mean'])
                if 'individual_miners' in entry:
                    total_miners.update(entry['individual_miners'].keys())
            
            return {
                'avg_reward': float(np.mean(all_rewards)) if all_rewards else 0.0,
                'avg_response_time': float(np.mean(all_response_times)) if all_response_times else 0.0,
                'unique_miners': len(total_miners),
                'data_points': len(hour_performance)
            }
            
        except Exception as e:
            bt.logging.error(f"Error aggregating performance data: {str(e)}")
            return {}

    def _aggregate_daily_summary(self, day_hourly_data):
        """Aggregate hourly data into daily summary"""
        try:
            if not day_hourly_data:
                return {}
            
            daily_rewards = []
            daily_response_times = []
            all_miners = set()
            
            for hour_data in day_hourly_data:
                perf_data = hour_data.get('performance', {})
                if 'avg_reward' in perf_data:
                    daily_rewards.append(perf_data['avg_reward'])
                if 'avg_response_time' in perf_data:
                    daily_response_times.append(perf_data['avg_response_time'])
                if 'unique_miners' in perf_data:
                    all_miners.add(perf_data['unique_miners'])
            
            return {
                'daily_avg_reward': float(np.mean(daily_rewards)) if daily_rewards else 0.0,
                'daily_avg_response_time': float(np.mean(daily_response_times)) if daily_response_times else 0.0,
                'peak_miners': max(all_miners) if all_miners else 0,
                'active_hours': len(day_hourly_data)
            }
            
        except Exception as e:
            bt.logging.error(f"Error aggregating daily summary: {str(e)}")
            return {}

    def _get_data_for_timeframe(self, timeframe_delta):
        """Get data for specified timeframe"""
        try:
            cutoff_time = datetime.utcnow() - timeframe_delta
            
            # Get data from all buffers within timeframe
            performance_data = [
                p for p in self.performance_buffer
                if datetime.fromisoformat(p['timestamp']) > cutoff_time
            ]
            
            return performance_data
            
        except Exception as e:
            bt.logging.error(f"Error getting data for timeframe: {str(e)}")
            return []

    def _extract_performance_history(self, data_source):
        """Extract performance history from data source"""
        try:
            history = []
            for entry in data_source:
                if 'rewards' in entry:
                    history.append({
                        'timestamp': entry['timestamp'],
                        'avg_reward': entry['rewards']['mean'],
                        'std_reward': entry['rewards']['std'],
                        'miner_count': entry['rewards']['count']
                    })
            
            return history
            
        except Exception as e:
            bt.logging.error(f"Error extracting performance history: {str(e)}")
            return []

    def _extract_network_history(self, data_source):
        """Extract network history from data source"""
        try:
            history = []
            for entry in data_source:
                if 'network_size' in entry:
                    history.append({
                        'timestamp': entry['timestamp'],
                        'active_miners': entry['network_size']['active_miners'],
                        'total_nodes': entry['network_size']['total_nodes'],
                        'participation_rate': entry['network_size']['participation_rate']
                    })
            
            return history
            
        except Exception as e:
            bt.logging.error(f"Error extracting network history: {str(e)}")
            return []

    def _extract_revenue_history(self, data_source):
        """Extract revenue history from data source"""
        try:
            history = []
            for entry in data_source:
                if 'revenue' in entry:
                    history.append({
                        'timestamp': entry['timestamp'],
                        'total_collected': entry['revenue']['total_collected'],
                        'collected_24h': entry['revenue']['collected_24h'],
                        'collection_rate': entry['revenue']['collection_rate_24h']
                    })
            
            return history
            
        except Exception as e:
            bt.logging.error(f"Error extracting revenue history: {str(e)}")
            return []

    def _calculate_trend_analysis(self, data_source):
        """Calculate trend analysis from data source"""
        try:
            if len(data_source) < 2:
                return {'trend': 'insufficient_data'}
            
            # Calculate trends for key metrics
            rewards = [d['rewards']['mean'] for d in data_source if 'rewards' in d]
            
            if len(rewards) >= 2:
                # Simple linear trend
                trend_slope = (rewards[-1] - rewards[0]) / len(rewards)
                trend_direction = 'increasing' if trend_slope > 0 else 'decreasing'
            else:
                trend_direction = 'stable'
                trend_slope = 0.0
            
            return {
                'trend': trend_direction,
                'slope': float(trend_slope),
                'data_points': len(data_source),
                'timespan_hours': len(data_source) * 0.083  # Assuming 5-minute intervals
            }
            
        except Exception as e:
            bt.logging.error(f"Error calculating trend analysis: {str(e)}")
            return {'trend': 'error'}