import os
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
import bittensor as bt

class CompetitionFeeManager:
    """
    Production-ready fee management system for competition submissions.
    Handles dynamic fee calculation, payment tracking, and revenue distribution
    according to whitepaper specifications.
    """
    
    def __init__(self, storage_path: str = "data/fees", base_fee_tao: float = 0.001):
        self.storage_path = storage_path
        self.base_fee_tao = base_fee_tao  # Base fee in TAO
        
        os.makedirs(storage_path, exist_ok=True)
        
        # Fee calculation parameters
        self.fee_config = {
            'base_fee': base_fee_tao,
            'volume_multiplier': 1.5,      # Fee increases with submission volume
            'quality_discount': 0.2,       # Discount for high-quality submissions
            'repeat_miner_discount': 0.1,  # Discount for consistent miners
            'min_fee': 0.0001,             # Minimum fee (0.0001 TAO)
            'max_fee': 0.01,               # Maximum fee (0.01 TAO)
            'surge_threshold': 100,        # Submissions per hour that trigger surge
            'surge_multiplier': 2.0        # Surge pricing multiplier
        }
        
        # Fee tracking state
        self.fee_state = {
            'total_collected': 0.0,
            'fees_by_miner': {},           # miner_uid -> total fees paid
            'submission_history': [],      # Recent submission history
            'revenue_distribution': {      # How collected fees are distributed
                'validator_pool': 0.6,     # 60% to validators
                'development_fund': 0.2,   # 20% for development
                'preferred_miner': 0.15,   # 15% to preferred miner
                'burn': 0.05              # 5% burned
            },
            'last_distribution': None
        }
        
        # Load existing state
        self._load_state()
        
        bt.logging.info(f"CompetitionFeeManager initialized with base fee: {base_fee_tao} TAO")

    def _load_state(self):
        """Load fee management state from persistent storage"""
        state_file = os.path.join(self.storage_path, "fee_state.json")
        
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    loaded_state = json.load(f)
                    
                # Merge loaded state with defaults
                if isinstance(loaded_state, dict):
                    self.fee_state.update(loaded_state)
                    
                bt.logging.info(f"Loaded fee state: {self.fee_state['total_collected']:.6f} TAO collected")
                
            except Exception as e:
                bt.logging.error(f"Failed to load fee state: {e}")
        else:
            bt.logging.info("No existing fee state found, starting fresh")

    def _save_state(self):
        """Save fee management state to persistent storage with atomic write"""
        try:
            state_file = os.path.join(self.storage_path, "fee_state.json")
            
            # Write to temporary file first for atomic operation
            temp_file = state_file + '.tmp'
            with open(temp_file, 'w') as f:
                json.dump(self.fee_state, f, indent=2, default=str)
            
            # Atomic move
            os.rename(temp_file, state_file)
            
            bt.logging.debug("Fee state saved successfully")
            
        except Exception as e:
            bt.logging.error(f"Failed to save fee state: {e}")

    def calculate_fee(self, miner_uid: int, submission_count: int = 1, 
                     quality_score: float = 0.5, is_repeat_miner: bool = False) -> float:
        """
        Calculate dynamic fee for competition submission based on multiple factors.
        
        Args:
            miner_uid: Unique identifier of the miner
            submission_count: Number of submissions in current batch
            quality_score: Quality score of previous submissions (0.0-1.0)
            is_repeat_miner: Whether miner has submitted before
            
        Returns:
            Fee amount in TAO
        """
        try:
            # Start with base fee
            fee = self.fee_config['base_fee']
            
            # Apply volume multiplier for batch submissions
            if submission_count > 1:
                volume_multiplier = 1 + (submission_count - 1) * 0.1  # 10% increase per additional submission
                fee *= min(volume_multiplier, self.fee_config['volume_multiplier'])
            
            # Apply quality discount for high-performing miners
            if quality_score > 0.7:
                quality_discount = self.fee_config['quality_discount'] * (quality_score - 0.7) / 0.3
                fee *= (1 - quality_discount)
            
            # Apply repeat miner discount
            if is_repeat_miner:
                fee *= (1 - self.fee_config['repeat_miner_discount'])
            
            # Check for surge pricing based on recent submission volume
            surge_multiplier = self._calculate_surge_multiplier()
            fee *= surge_multiplier
            
            # Apply min/max bounds
            fee = max(self.fee_config['min_fee'], min(fee, self.fee_config['max_fee']))
            
            # Round to 6 decimal places for TAO precision
            fee = float(Decimal(str(fee)).quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP))
            
            bt.logging.debug(f"Calculated fee for miner {miner_uid}: {fee:.6f} TAO "
                           f"(submissions={submission_count}, quality={quality_score:.3f}, "
                           f"repeat={is_repeat_miner}, surge={surge_multiplier:.2f})")
            
            return fee
            
        except Exception as e:
            bt.logging.error(f"Error calculating fee for miner {miner_uid}: {str(e)}")
            return self.fee_config['base_fee']  # Fallback to base fee

    def _calculate_surge_multiplier(self) -> float:
        """Calculate surge pricing multiplier based on recent submission volume"""
        try:
            # Count submissions in the last hour
            current_time = datetime.utcnow()
            hour_ago = current_time - timedelta(hours=1)
            
            recent_submissions = [
                entry for entry in self.fee_state['submission_history']
                if datetime.fromisoformat(entry['timestamp']) > hour_ago
            ]
            
            submission_rate = len(recent_submissions)
            
            if submission_rate > self.fee_config['surge_threshold']:
                # Linear surge pricing above threshold
                excess_rate = submission_rate - self.fee_config['surge_threshold']
                surge_factor = 1 + (excess_rate / self.fee_config['surge_threshold'])
                surge_multiplier = min(surge_factor, self.fee_config['surge_multiplier'])
                
                bt.logging.info(f"Surge pricing active: {submission_rate} submissions/hour, "
                              f"multiplier: {surge_multiplier:.2f}")
                
                return surge_multiplier
            
            return 1.0  # No surge pricing
            
        except Exception as e:
            bt.logging.error(f"Error calculating surge multiplier: {str(e)}")
            return 1.0

    def record_submission(self, miner_uid: int, fee_paid: float, submission_data: Dict[str, Any]) -> bool:
        """
        Record a fee payment for a competition submission.
        
        Args:
            miner_uid: Unique identifier of the miner
            fee_paid: Amount paid in TAO
            submission_data: Additional submission metadata
            
        Returns:
            True if recorded successfully, False otherwise
        """
        try:
            current_time = datetime.utcnow()
            
            # Record the submission
            submission_record = {
                'miner_uid': miner_uid,
                'fee_paid': fee_paid,
                'timestamp': current_time.isoformat(),
                'submission_data': submission_data
            }
            
            # Add to submission history
            self.fee_state['submission_history'].append(submission_record)
            
            # Update total collected
            self.fee_state['total_collected'] += fee_paid
            
            # Update miner's total fees
            miner_uid_str = str(miner_uid)
            if miner_uid_str not in self.fee_state['fees_by_miner']:
                self.fee_state['fees_by_miner'][miner_uid_str] = {
                    'total_paid': 0.0,
                    'submission_count': 0,
                    'first_submission': current_time.isoformat(),
                    'last_submission': current_time.isoformat()
                }
            
            miner_stats = self.fee_state['fees_by_miner'][miner_uid_str]
            miner_stats['total_paid'] += fee_paid
            miner_stats['submission_count'] += 1
            miner_stats['last_submission'] = current_time.isoformat()
            
            # Clean up old submission history (keep last 1000 entries)
            if len(self.fee_state['submission_history']) > 1000:
                self.fee_state['submission_history'] = self.fee_state['submission_history'][-1000:]
            
            # Save state
            self._save_state()
            
            bt.logging.info(f"Recorded fee payment: miner {miner_uid} paid {fee_paid:.6f} TAO")
            bt.logging.info(f"Total collected: {self.fee_state['total_collected']:.6f} TAO")
            
            return True
            
        except Exception as e:
            bt.logging.error(f"Error recording submission fee for miner {miner_uid}: {str(e)}")
            return False

    def get_miner_fee_stats(self, miner_uid: int) -> Dict[str, Any]:
        """Get fee statistics for a specific miner"""
        miner_uid_str = str(miner_uid)
        
        if miner_uid_str not in self.fee_state['fees_by_miner']:
            return {
                'total_paid': 0.0,
                'submission_count': 0,
                'average_fee': 0.0,
                'is_repeat_miner': False,
                'first_submission': None,
                'last_submission': None
            }
        
        miner_stats = self.fee_state['fees_by_miner'][miner_uid_str]
        
        return {
            'total_paid': miner_stats['total_paid'],
            'submission_count': miner_stats['submission_count'],
            'average_fee': miner_stats['total_paid'] / max(miner_stats['submission_count'], 1),
            'is_repeat_miner': miner_stats['submission_count'] > 1,
            'first_submission': miner_stats['first_submission'],
            'last_submission': miner_stats['last_submission']
        }

    def calculate_miner_quality_score(self, miner_uid: int, performance_history: List[float]) -> float:
        """
        Calculate quality score for a miner based on performance history.
        
        Args:
            miner_uid: Miner unique identifier
            performance_history: List of recent performance scores
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        try:
            if not performance_history:
                return 0.5  # Neutral score for new miners
            
            # Calculate weighted average (more recent scores have higher weight)
            weights = [0.8 ** i for i in range(len(performance_history))]
            weights.reverse()  # Most recent gets highest weight
            
            weighted_sum = sum(score * weight for score, weight in zip(performance_history, weights))
            weight_sum = sum(weights)
            
            quality_score = weighted_sum / weight_sum if weight_sum > 0 else 0.5
            
            # Apply consistency bonus (reward stable performance)
            if len(performance_history) >= 5:
                import numpy as np
                std_dev = np.std(performance_history)
                consistency_bonus = max(0, 0.1 * (1 - std_dev))  # Up to 10% bonus for low variance
                quality_score = min(1.0, quality_score + consistency_bonus)
            
            bt.logging.debug(f"Quality score for miner {miner_uid}: {quality_score:.3f} "
                           f"(based on {len(performance_history)} scores)")
            
            return quality_score
            
        except Exception as e:
            bt.logging.error(f"Error calculating quality score for miner {miner_uid}: {str(e)}")
            return 0.5

    def distribute_revenue(self, preferred_miner_uid: Optional[int] = None) -> Dict[str, float]:
        """
        Distribute collected revenue according to whitepaper specifications.
        
        Args:
            preferred_miner_uid: UID of the preferred miner for bonus distribution
            
        Returns:
            Dictionary showing distribution amounts
        """
        try:
            # Check if we have enough revenue to distribute (minimum 0.1 TAO)
            total_revenue = self.fee_state['total_collected']
            min_distribution_threshold = 0.1
            
            if total_revenue < min_distribution_threshold:
                bt.logging.info(f"Revenue {total_revenue:.6f} TAO below distribution threshold {min_distribution_threshold}")
                return {}
            
            # Calculate distribution amounts
            distribution = {}
            distribution_config = self.fee_state['revenue_distribution']
            
            distribution['validator_pool'] = total_revenue * distribution_config['validator_pool']
            distribution['development_fund'] = total_revenue * distribution_config['development_fund']
            distribution['burn_amount'] = total_revenue * distribution_config['burn']
            
            # Preferred miner bonus (if applicable)
            if preferred_miner_uid is not None:
                distribution['preferred_miner_bonus'] = total_revenue * distribution_config['preferred_miner']
                distribution['preferred_miner_uid'] = preferred_miner_uid
            else:
                # If no preferred miner, add to validator pool
                distribution['validator_pool'] += total_revenue * distribution_config['preferred_miner']
                distribution['preferred_miner_bonus'] = 0.0
            
            # Record distribution
            distribution_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'total_distributed': total_revenue,
                'distribution': distribution.copy()
            }
            
            # Reset collected amount
            self.fee_state['total_collected'] = 0.0
            self.fee_state['last_distribution'] = distribution_record
            
            # Save state
            self._save_state()
            
            bt.logging.info(f"Revenue distribution completed: {total_revenue:.6f} TAO distributed")
            bt.logging.info(f"Validator pool: {distribution['validator_pool']:.6f} TAO")
            bt.logging.info(f"Development fund: {distribution['development_fund']:.6f} TAO")
            bt.logging.info(f"Preferred miner bonus: {distribution.get('preferred_miner_bonus', 0):.6f} TAO")
            bt.logging.info(f"Burn amount: {distribution['burn_amount']:.6f} TAO")
            
            return distribution
            
        except Exception as e:
            bt.logging.error(f"Error distributing revenue: {str(e)}")
            return {}

    def get_fee_statistics(self) -> Dict[str, Any]:
        """Get comprehensive fee management statistics"""
        try:
            current_time = datetime.utcnow()
            
            # Calculate time-based statistics
            day_ago = current_time - timedelta(days=1)
            week_ago = current_time - timedelta(days=7)
            
            recent_submissions_24h = [
                entry for entry in self.fee_state['submission_history']
                if datetime.fromisoformat(entry['timestamp']) > day_ago
            ]
            
            recent_submissions_7d = [
                entry for entry in self.fee_state['submission_history']
                if datetime.fromisoformat(entry['timestamp']) > week_ago
            ]
            
            # Calculate fee totals
            fees_24h = sum(entry['fee_paid'] for entry in recent_submissions_24h)
            fees_7d = sum(entry['fee_paid'] for entry in recent_submissions_7d)
            
            # Get active miners
            active_miners_24h = len(set(entry['miner_uid'] for entry in recent_submissions_24h))
            active_miners_7d = len(set(entry['miner_uid'] for entry in recent_submissions_7d))
            
            # Calculate averages
            avg_fee_24h = fees_24h / max(len(recent_submissions_24h), 1)
            avg_fee_7d = fees_7d / max(len(recent_submissions_7d), 1)
            
            # Current fee calculation
            current_fee = self.calculate_fee(0)  # Sample fee for new miner
            surge_multiplier = self._calculate_surge_multiplier()
            
            statistics = {
                'total_collected_all_time': self.fee_state['total_collected'],
                'fees_collected_24h': fees_24h,
                'fees_collected_7d': fees_7d,
                'submissions_24h': len(recent_submissions_24h),
                'submissions_7d': len(recent_submissions_7d),
                'active_miners_24h': active_miners_24h,
                'active_miners_7d': active_miners_7d,
                'average_fee_24h': avg_fee_24h,
                'average_fee_7d': avg_fee_7d,
                'current_base_fee': self.fee_config['base_fee'],
                'current_sample_fee': current_fee,
                'surge_multiplier': surge_multiplier,
                'total_unique_miners': len(self.fee_state['fees_by_miner']),
                'fee_config': self.fee_config.copy(),
                'last_distribution': self.fee_state.get('last_distribution'),
                'last_updated': current_time.isoformat()
            }
            
            return statistics
            
        except Exception as e:
            bt.logging.error(f"Error getting fee statistics: {str(e)}")
            return {}

    def update_fee_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Update fee configuration parameters.
        
        Args:
            new_config: Dictionary with configuration updates
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            # Validate configuration values
            valid_keys = set(self.fee_config.keys())
            invalid_keys = set(new_config.keys()) - valid_keys
            
            if invalid_keys:
                bt.logging.error(f"Invalid configuration keys: {invalid_keys}")
                return False
            
            # Validate value ranges
            if 'base_fee' in new_config and not (0.0001 <= new_config['base_fee'] <= 1.0):
                bt.logging.error("Base fee must be between 0.0001 and 1.0 TAO")
                return False
            
            if 'min_fee' in new_config and 'max_fee' in new_config:
                if new_config['min_fee'] >= new_config['max_fee']:
                    bt.logging.error("Minimum fee must be less than maximum fee")
                    return False
            
            # Update configuration
            old_config = self.fee_config.copy()
            self.fee_config.update(new_config)
            
            # Save state
            self._save_state()
            
            bt.logging.info(f"Fee configuration updated: {new_config}")
            bt.logging.debug(f"Old config: {old_config}")
            bt.logging.debug(f"New config: {self.fee_config}")
            
            return True
            
        except Exception as e:
            bt.logging.error(f"Error updating fee configuration: {str(e)}")
            return False

    def get_miner_discount_eligibility(self, miner_uid: int, performance_history: List[float]) -> Dict[str, Any]:
        """
        Check what discounts a miner is eligible for.
        
        Args:
            miner_uid: Miner unique identifier
            performance_history: Recent performance scores
            
        Returns:
            Dictionary with discount eligibility information
        """
        try:
            miner_stats = self.get_miner_fee_stats(miner_uid)
            quality_score = self.calculate_miner_quality_score(miner_uid, performance_history)
            
            discounts = {
                'repeat_miner_discount': {
                    'eligible': miner_stats['is_repeat_miner'],
                    'discount_percent': self.fee_config['repeat_miner_discount'] * 100,
                    'reason': f"Has {miner_stats['submission_count']} previous submissions"
                },
                'quality_discount': {
                    'eligible': quality_score > 0.7,
                    'discount_percent': 0.0,
                    'quality_score': quality_score,
                    'reason': ""
                }
            }
            
            if discounts['quality_discount']['eligible']:
                discount_percent = self.fee_config['quality_discount'] * (quality_score - 0.7) / 0.3 * 100
                discounts['quality_discount']['discount_percent'] = discount_percent
                discounts['quality_discount']['reason'] = f"High quality score: {quality_score:.3f}"
            else:
                discounts['quality_discount']['reason'] = f"Quality score too low: {quality_score:.3f} (need > 0.7)"
            
            return discounts
            
        except Exception as e:
            bt.logging.error(f"Error checking discount eligibility for miner {miner_uid}: {str(e)}")
            return {}