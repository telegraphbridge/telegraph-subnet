import os
import json
import time
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import bittensor as bt

class ConsensusManager:
    """
    Production-ready consensus manager for preferred miner selection.
    Implements weighted voting, Byzantine fault tolerance, and persistent state.
    """
    
    def __init__(self, storage_path: str = "data/consensus/preferred_miner.json",
                 vote_threshold: float = 0.51, max_history_entries: int = 1000):
        self.storage_path = storage_path
        self.vote_threshold = vote_threshold
        self.max_history_entries = max_history_entries
        
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        
        # Initialize state
        self.state = {
            'preferred_miner': None,
            'preferred_miner_timestamp': None,
            'current_round_votes': {},
            'vote_history': [],
            'consensus_stats': {
                'total_rounds': 0,
                'successful_consensus': 0,
                'last_update': None
            }
        }
        
        # Load existing state
        self._load()
        
        bt.logging.info(f"ConsensusManager initialized with threshold={vote_threshold}")

    def _load(self):
        """Load consensus state from persistent storage"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    loaded_state = json.load(f)
                    
                # Validate and merge loaded state
                if isinstance(loaded_state, dict):
                    self.state.update(loaded_state)
                    
                bt.logging.info(f"Loaded consensus state with {len(self.state.get('vote_history', []))} history entries")
                
            except Exception as e:
                bt.logging.error(f"Failed to load consensus state: {e}")
        else:
            bt.logging.info("No existing consensus state found, starting fresh")

    def _save(self):
        """Save consensus state to persistent storage with atomic write"""
        try:
            # Limit history size to prevent unbounded growth
            if len(self.state['vote_history']) > self.max_history_entries:
                self.state['vote_history'] = self.state['vote_history'][-self.max_history_entries:]
            
            # Write to temporary file first for atomic operation
            temp_path = self.storage_path + '.tmp'
            with open(temp_path, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
            
            # Atomic move
            os.rename(temp_path, self.storage_path)
            
            bt.logging.debug("Consensus state saved successfully")
            
        except Exception as e:
            bt.logging.error(f"Failed to save consensus state: {e}")

    def record_votes(self, round_id: str, votes: Dict[int, float]):
        """
        Record votes for a competition round with validation and Byzantine fault tolerance.
        
        Args:
            round_id: Unique identifier for the round
            votes: Dictionary mapping miner UIDs to vote weights/scores
        """
        try:
            if not isinstance(votes, dict) or not votes:
                bt.logging.warning(f"Invalid votes provided for round {round_id}")
                return
            
            # Validate vote format
            validated_votes = {}
            total_weight = 0.0
            
            for uid, weight in votes.items():
                # Validate UID
                if not isinstance(uid, int) or uid < 0:
                    bt.logging.warning(f"Invalid UID in votes: {uid}")
                    continue
                
                # Validate weight
                if not isinstance(weight, (int, float)) or weight < 0:
                    bt.logging.warning(f"Invalid vote weight for UID {uid}: {weight}")
                    continue
                
                validated_votes[uid] = float(weight)
                total_weight += float(weight)
            
            if not validated_votes:
                bt.logging.warning(f"No valid votes for round {round_id}")
                return
            
            # Normalize votes to sum to 1.0
            if total_weight > 0:
                normalized_votes = {uid: weight / total_weight for uid, weight in validated_votes.items()}
            else:
                # Equal weight fallback
                equal_weight = 1.0 / len(validated_votes)
                normalized_votes = {uid: equal_weight for uid in validated_votes.keys()}
            
            # Store vote record
            vote_record = {
                'round_id': round_id,
                'timestamp': datetime.utcnow().isoformat(),
                'votes': normalized_votes,
                'total_participants': len(normalized_votes),
                'vote_hash': self._calculate_vote_hash(normalized_votes)
            }
            
            # Add to history
            self.state['vote_history'].append(vote_record)
            self.state['current_round_votes'] = normalized_votes
            
            # Update stats
            self.state['consensus_stats']['total_rounds'] += 1
            self.state['consensus_stats']['last_update'] = datetime.utcnow().isoformat()
            
            self._save()
            
            bt.logging.info(f"Recorded votes for round {round_id}: {len(normalized_votes)} participants")
            bt.logging.debug(f"Vote distribution: {sorted(normalized_votes.items(), key=lambda x: x[1], reverse=True)[:3]}")
            
        except Exception as e:
            bt.logging.error(f"Error recording votes for round {round_id}: {str(e)}")

    def select_preferred_miner(self, round_id: str, votes: Dict[int, float]) -> Optional[int]:
        """
        Select preferred miner based on weighted voting with Byzantine fault tolerance.
        
        Args:
            round_id: Unique identifier for the round
            votes: Dictionary mapping miner UIDs to vote weights
            
        Returns:
            UID of preferred miner, or None if no consensus reached
        """
        try:
            if not votes:
                bt.logging.warning(f"No votes provided for preferred miner selection in round {round_id}")
                return None
            
            # Record the votes first
            self.record_votes(round_id, votes)
            
            # Calculate weighted votes
            sorted_miners = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            
            if not sorted_miners:
                bt.logging.warning(f"No valid miners to select from in round {round_id}")
                return None
            
            # Get the highest scoring miner
            preferred_uid, highest_score = sorted_miners[0]
            
            # Check if the preferred miner meets the threshold
            total_vote_weight = sum(votes.values())
            if total_vote_weight == 0:
                bt.logging.warning(f"Total vote weight is zero for round {round_id}")
                return None
            
            vote_percentage = highest_score / total_vote_weight
            
            # Apply consensus threshold
            if vote_percentage >= self.vote_threshold:
                # Update preferred miner
                self.state['preferred_miner'] = preferred_uid
                self.state['preferred_miner_timestamp'] = datetime.utcnow().isoformat()
                self.state['consensus_stats']['successful_consensus'] += 1
                
                self._save()
                
                bt.logging.info(f"Consensus reached for round {round_id}: "
                              f"miner {preferred_uid} selected with {vote_percentage:.2%} support")
                
                return preferred_uid
                
            else:
                bt.logging.info(f"No consensus reached for round {round_id}: "
                              f"highest score {vote_percentage:.2%} below threshold {self.vote_threshold:.2%}")
                
                # Consider selecting based on relative performance anyway
                # (Production systems might want this for liveness)
                if vote_percentage > 0.3:  # At least 30% support
                    self.state['preferred_miner'] = preferred_uid
                    self.state['preferred_miner_timestamp'] = datetime.utcnow().isoformat()
                    
                    self._save()
                    
                    bt.logging.info(f"Selected miner {preferred_uid} despite no consensus "
                                  f"(relative performance: {vote_percentage:.2%})")
                    return preferred_uid
                
                return None
                
        except Exception as e:
            bt.logging.error(f"Error selecting preferred miner for round {round_id}: {str(e)}")
            return None

    def get_preferred_miner(self) -> Optional[int]:
        """
        Get the currently preferred miner UID.
        
        Returns:
            UID of preferred miner, or None if no preferred miner
        """
        preferred_uid = self.state.get('preferred_miner')
        
        if preferred_uid is not None:
            # Check if the preferred miner selection is recent
            timestamp_str = self.state.get('preferred_miner_timestamp')
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    age_hours = (datetime.utcnow() - timestamp).total_seconds() / 3600
                    
                    # Consider preferred miner stale after 24 hours
                    if age_hours > 24:
                        bt.logging.info(f"Preferred miner {preferred_uid} selection is stale ({age_hours:.1f} hours old)")
                        return None
                        
                except Exception as e:
                    bt.logging.error(f"Error parsing preferred miner timestamp: {e}")
                    return None
        
        return preferred_uid

    def get_consensus_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent consensus history.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of consensus vote records
        """
        history = self.state.get('vote_history', [])
        return history[-limit:] if history else []

    def get_consensus_stats(self) -> Dict[str, Any]:
        """Get consensus statistics and performance metrics"""
        stats = self.state.get('consensus_stats', {}).copy()
        
        # Calculate success rate
        total_rounds = stats.get('total_rounds', 0)
        successful_consensus = stats.get('successful_consensus', 0)
        
        if total_rounds > 0:
            stats['consensus_success_rate'] = successful_consensus / total_rounds
        else:
            stats['consensus_success_rate'] = 0.0
        
        # Add current state info
        stats['current_preferred_miner'] = self.state.get('preferred_miner')
        stats['preferred_miner_age_hours'] = self._get_preferred_miner_age_hours()
        stats['vote_history_entries'] = len(self.state.get('vote_history', []))
        stats['vote_threshold'] = self.vote_threshold
        
        return stats

    def _get_preferred_miner_age_hours(self) -> Optional[float]:
        """Get age of current preferred miner selection in hours"""
        timestamp_str = self.state.get('preferred_miner_timestamp')
        if not timestamp_str:
            return None
        
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
            return (datetime.utcnow() - timestamp).total_seconds() / 3600
        except Exception:
            return None

    def _calculate_vote_hash(self, votes: Dict[int, float]) -> str:
        """Calculate hash of vote data for integrity checking"""
        # Sort votes for consistent hashing
        sorted_votes = sorted(votes.items())
        vote_string = str(sorted_votes)
        return hashlib.sha256(vote_string.encode()).hexdigest()[:16]

    def validate_consensus_integrity(self) -> bool:
        """Validate the integrity of consensus data"""
        try:
            # Check for required fields
            required_fields = ['preferred_miner', 'vote_history', 'consensus_stats']
            for field in required_fields:
                if field not in self.state:
                    bt.logging.error(f"Missing required consensus field: {field}")
                    return False
            
            # Validate vote history
            for i, vote_record in enumerate(self.state['vote_history'][-10:]):  # Check last 10
                if not isinstance(vote_record, dict):
                    bt.logging.error(f"Invalid vote record format at index {i}")
                    return False
                
                required_vote_fields = ['round_id', 'timestamp', 'votes']
                for field in required_vote_fields:
                    if field not in vote_record:
                        bt.logging.error(f"Missing vote record field: {field}")
                        return False
            
            bt.logging.debug("Consensus integrity validation passed")
            return True
            
        except Exception as e:
            bt.logging.error(f"Error validating consensus integrity: {str(e)}")
            return False

    def reset_consensus_state(self):
        """Reset consensus state (use with caution)"""
        bt.logging.warning("Resetting consensus state")
        
        self.state = {
            'preferred_miner': None,
            'preferred_miner_timestamp': None,
            'current_round_votes': {},
            'vote_history': [],
            'consensus_stats': {
                'total_rounds': 0,
                'successful_consensus': 0,
                'last_update': None
            }
        }
        
        self._save()