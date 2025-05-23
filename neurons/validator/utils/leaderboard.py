import os
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import bittensor as bt

class Leaderboard:
    """
    Production-ready leaderboard system for tracking miner performance,
    submission history, and preferred miner status.
    """
    
    def __init__(self, leaderboard_path: str = "data/leaderboard.json", max_history_entries: int = 1000):
        self.leaderboard_path = leaderboard_path
        self.max_history_entries = max_history_entries
        
        os.makedirs(os.path.dirname(leaderboard_path), exist_ok=True)
        
        # Initialize leaderboard data structure
        self.data = {
            'miners': {},  # uid -> miner stats
            'preferred_miner_history': [],  # History of preferred miner selections
            'consensus_history': [],  # History of consensus rounds
            'last_updated': None,
            'statistics': {
                'total_submissions': 0,
                'active_miners': 0,
                'total_rounds': 0
            }
        }
        
        # Load existing data
        self._load()
        
        bt.logging.info(f"Leaderboard initialized with {len(self.data['miners'])} miners")

    def _load(self) -> Dict[str, Any]:
        """Load leaderboard data from persistent storage"""
        if os.path.exists(self.leaderboard_path):
            try:
                with open(self.leaderboard_path, 'r') as f:
                    loaded_data = json.load(f)
                    
                # Validate and merge loaded data
                if isinstance(loaded_data, dict):
                    # Ensure all required keys exist
                    for key in ['miners', 'preferred_miner_history', 'consensus_history', 'statistics']:
                        if key not in loaded_data:
                            loaded_data[key] = self.data[key]
                    
                    self.data = loaded_data
                    
                bt.logging.info(f"Loaded leaderboard with {len(self.data['miners'])} miners")
                
            except Exception as e:
                bt.logging.error(f"Failed to load leaderboard data: {e}")
                
        return self.data

    def _save(self):
        """Save leaderboard data to persistent storage with atomic write"""
        try:
            # Limit history sizes to prevent unbounded growth
            if len(self.data['preferred_miner_history']) > self.max_history_entries:
                self.data['preferred_miner_history'] = self.data['preferred_miner_history'][-self.max_history_entries:]
                
            if len(self.data['consensus_history']) > self.max_history_entries:
                self.data['consensus_history'] = self.data['consensus_history'][-self.max_history_entries:]
            
            # Update timestamp
            self.data['last_updated'] = datetime.utcnow().isoformat()
            
            # Write to temporary file first for atomic operation
            temp_path = self.leaderboard_path + '.tmp'
            with open(temp_path, 'w') as f:
                json.dump(self.data, f, indent=2, default=str)
            
            # Atomic move
            os.rename(temp_path, self.leaderboard_path)
            
            bt.logging.debug("Leaderboard data saved successfully")
            
        except Exception as e:
            bt.logging.error(f"Failed to save leaderboard data: {e}")

    def update(self, miner_stats: Dict[int, Dict[str, Any]], preferred_uid: Optional[int], 
               consensus_history: List[Dict[str, Any]]):
        """
        Update leaderboard with latest miner stats, preferred miner, and consensus history.
        
        Args:
            miner_stats: Dictionary mapping miner UIDs to their stats
            preferred_uid: UID of the preferred miner for this round
            consensus_history: List of recent consensus round data
        """
        try:
            current_time = datetime.utcnow()
            
            # Update individual miner statistics
            for uid, stats in miner_stats.items():
                self._update_miner_stats(uid, stats, current_time)
            
            # Update preferred miner history
            if preferred_uid is not None:
                self._update_preferred_miner_history(preferred_uid, current_time)
            
            # Update consensus history
            self._update_consensus_history(consensus_history, current_time)
            
            # Update global statistics
            self._update_global_statistics()
            
            # Save changes
            self._save()
            
            bt.logging.info(f"Leaderboard updated: {len(miner_stats)} miners, "
                          f"preferred_uid={preferred_uid}")
            
        except Exception as e:
            bt.logging.error(f"Error updating leaderboard: {str(e)}")

    def _update_miner_stats(self, uid: int, stats: Dict[str, Any], timestamp: datetime):
        """Update statistics for a specific miner"""
        uid_str = str(uid)  # JSON keys must be strings
        
        # Initialize miner entry if not exists
        if uid_str not in self.data['miners']:
            self.data['miners'][uid_str] = {
                'uid': uid,
                'total_submissions': 0,
                'total_score': 0.0,
                'average_score': 0.0,
                'best_score': 0.0,
                'worst_score': 1.0,
                'recent_scores': [],  # Last 50 scores
                'preferred_selections': 0,
                'last_submission': None,
                'first_seen': timestamp.isoformat(),
                'status': 'active'
            }
        
        miner_data = self.data['miners'][uid_str]
        
        # Update basic stats
        current_score = stats.get('normalized_score', 0.0)
        
        miner_data['total_submissions'] += 1
        miner_data['total_score'] += current_score
        miner_data['average_score'] = miner_data['total_score'] / miner_data['total_submissions']
        miner_data['best_score'] = max(miner_data['best_score'], current_score)
        miner_data['worst_score'] = min(miner_data['worst_score'], current_score)
        miner_data['last_submission'] = timestamp.isoformat()
        
        # Update recent scores (keep last 50)
        miner_data['recent_scores'].append({
            'score': current_score,
            'timestamp': timestamp.isoformat()
        })
        miner_data['recent_scores'] = miner_data['recent_scores'][-50:]
        
        # Calculate recent performance metrics
        if len(miner_data['recent_scores']) >= 5:
            recent_score_values = [s['score'] for s in miner_data['recent_scores'][-10:]]
            miner_data['recent_average'] = sum(recent_score_values) / len(recent_score_values)
            
            # Calculate trend (improving/declining)
            if len(recent_score_values) >= 10:
                first_half = sum(recent_score_values[:5]) / 5
                second_half = sum(recent_score_values[5:]) / 5
                miner_data['performance_trend'] = 'improving' if second_half > first_half else 'declining'
            else:
                miner_data['performance_trend'] = 'stable'
        
        # Add any additional stats from the input
        for key, value in stats.items():
            if key not in ['normalized_score']:  # Don't duplicate the main score
                miner_data[f'latest_{key}'] = value

    def _update_preferred_miner_history(self, preferred_uid: int, timestamp: datetime):
        """Update preferred miner selection history"""
        # Update the preferred miner's selection count
        uid_str = str(preferred_uid)
        if uid_str in self.data['miners']:
            self.data['miners'][uid_str]['preferred_selections'] += 1
        
        # Add to preferred miner history
        history_entry = {
            'uid': preferred_uid,
            'timestamp': timestamp.isoformat(),
            'round_id': f"round_{int(timestamp.timestamp())}"
        }
        
        self.data['preferred_miner_history'].append(history_entry)

    def _update_consensus_history(self, consensus_history: List[Dict[str, Any]], timestamp: datetime):
        """Update consensus round history"""
        for consensus_round in consensus_history:
            # Add timestamp if not present
            if 'timestamp' not in consensus_round:
                consensus_round['timestamp'] = timestamp.isoformat()
            
            # Add to consensus history if not already present
            round_id = consensus_round.get('round_id', '')
            existing_rounds = [h.get('round_id', '') for h in self.data['consensus_history']]
            
            if round_id not in existing_rounds:
                self.data['consensus_history'].append(consensus_round)

    def _update_global_statistics(self):
        """Update global leaderboard statistics"""
        stats = self.data['statistics']
        
        # Count active miners (submitted in last 24 hours)
        active_count = 0
        total_submissions = 0
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        for miner_data in self.data['miners'].values():
            total_submissions += miner_data.get('total_submissions', 0)
            
            last_submission = miner_data.get('last_submission')
            if last_submission:
                try:
                    last_time = datetime.fromisoformat(last_submission)
                    if last_time > cutoff_time:
                        active_count += 1
                        miner_data['status'] = 'active'
                    else:
                        miner_data['status'] = 'inactive'
                except Exception:
                    pass
        
        stats['active_miners'] = active_count
        stats['total_submissions'] = total_submissions
        stats['total_rounds'] = len(self.data['consensus_history'])
        stats['total_miners'] = len(self.data['miners'])

    def get_leaderboard(self, limit: int = 50, sort_by: str = 'average_score') -> Dict[str, Any]:
        """
        Get current leaderboard rankings.
        
        Args:
            limit: Maximum number of miners to return
            sort_by: Field to sort by ('average_score', 'recent_average', 'total_submissions', etc.)
            
        Returns:
            Dictionary containing ranked miners and metadata
        """
        try:
            # Get all miners and convert to list
            miners_list = []
            for uid_str, miner_data in self.data['miners'].items():
                miner_copy = miner_data.copy()
                miner_copy['uid'] = int(uid_str)  # Ensure UID is integer
                miners_list.append(miner_copy)
            
            # Sort miners by the specified field
            if sort_by in ['average_score', 'recent_average', 'best_score', 'total_submissions', 'preferred_selections']:
                miners_list.sort(key=lambda x: x.get(sort_by, 0), reverse=True)
            else:
                bt.logging.warning(f"Invalid sort field: {sort_by}, using average_score")
                miners_list.sort(key=lambda x: x.get('average_score', 0), reverse=True)
            
            # Add rankings
            for i, miner in enumerate(miners_list[:limit]):
                miner['rank'] = i + 1
            
            # Get top performers
            top_miners = miners_list[:limit]
            
            # Get recent activity
            recent_activity = self._get_recent_activity()
            
            # Get preferred miner stats
            preferred_stats = self._get_preferred_miner_stats()
            
            leaderboard = {
                'rankings': top_miners,
                'metadata': {
                    'total_miners': len(miners_list),
                    'sort_by': sort_by,
                    'last_updated': self.data.get('last_updated'),
                    'statistics': self.data['statistics']
                },
                'recent_activity': recent_activity,
                'preferred_miner_stats': preferred_stats
            }
            
            bt.logging.debug(f"Generated leaderboard with {len(top_miners)} miners, sorted by {sort_by}")
            
            return leaderboard
            
        except Exception as e:
            bt.logging.error(f"Error generating leaderboard: {str(e)}")
            return {'rankings': [], 'metadata': {}, 'recent_activity': [], 'preferred_miner_stats': {}}

    def _get_recent_activity(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent miner activity"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_activity = []
        
        for uid_str, miner_data in self.data['miners'].items():
            recent_scores = miner_data.get('recent_scores', [])
            
            for score_entry in recent_scores:
                try:
                    score_time = datetime.fromisoformat(score_entry['timestamp'])
                    if score_time > cutoff_time:
                        recent_activity.append({
                            'uid': int(uid_str),
                            'score': score_entry['score'],
                            'timestamp': score_entry['timestamp']
                        })
                except Exception:
                    continue
        
        # Sort by timestamp (most recent first)
        recent_activity.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return recent_activity[:100]  # Return last 100 activities

    def _get_preferred_miner_stats(self) -> Dict[str, Any]:
        """Get statistics about preferred miner selections"""
        preferred_history = self.data.get('preferred_miner_history', [])
        
        if not preferred_history:
            return {}
        
        # Get current preferred miner
        current_preferred = preferred_history[-1] if preferred_history else None
        
        # Count selections per miner
        selection_counts = {}
        for entry in preferred_history:
            uid = entry['uid']
            selection_counts[uid] = selection_counts.get(uid, 0) + 1
        
        # Get most selected miner
        most_selected_uid = max(selection_counts, key=selection_counts.get) if selection_counts else None
        
        stats = {
            'current_preferred_miner': current_preferred,
            'total_selections': len(preferred_history),
            'unique_miners_selected': len(selection_counts),
            'most_selected_miner': {
                'uid': most_selected_uid,
                'selections': selection_counts.get(most_selected_uid, 0)
            } if most_selected_uid else None,
            'selection_distribution': selection_counts
        }
        
        return stats

    def get_miner_profile(self, uid: int) -> Optional[Dict[str, Any]]:
        """Get detailed profile for a specific miner"""
        uid_str = str(uid)
        
        if uid_str not in self.data['miners']:
            return None
        
        miner_data = self.data['miners'][uid_str].copy()
        
        # Add position in leaderboard
        leaderboard = self.get_leaderboard(limit=1000)
        for rank, miner in enumerate(leaderboard['rankings'], 1):
            if miner['uid'] == uid:
                miner_data['current_rank'] = rank
                break
        
        return miner_data

    def cleanup_old_data(self, days: int = 30):
        """Clean up old data to prevent unbounded growth"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            
            # Clean up old scores from miner data
            for miner_data in self.data['miners'].values():
                recent_scores = miner_data.get('recent_scores', [])
                filtered_scores = []
                
                for score_entry in recent_scores:
                    try:
                        score_time = datetime.fromisoformat(score_entry['timestamp'])
                        if score_time > cutoff_time:
                            filtered_scores.append(score_entry)
                    except Exception:
                        continue
                
                miner_data['recent_scores'] = filtered_scores
            
            # Clean up old consensus history
            filtered_consensus = []
            for consensus_entry in self.data.get('consensus_history', []):
                try:
                    consensus_time = datetime.fromisoformat(consensus_entry.get('timestamp', ''))
                    if consensus_time > cutoff_time:
                        filtered_consensus.append(consensus_entry)
                except Exception:
                    continue
            
            self.data['consensus_history'] = filtered_consensus
            
            # Save changes
            self._save()
            
            bt.logging.info(f"Cleaned up leaderboard data older than {days} days")
            
        except Exception as e:
            bt.logging.error(f"Error cleaning up old leaderboard data: {str(e)}")