import os
import json
from typing import Dict, Any, List
from datetime import datetime
import bittensor as bt

class Leaderboard:
    """
    Maintains and exposes a leaderboard of miner performance, submission history, and preferred miner status.
    """
    def __init__(self, leaderboard_path: str = "data/leaderboard.json"):
        self.leaderboard_path = leaderboard_path
        os.makedirs(os.path.dirname(leaderboard_path), exist_ok=True)
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.leaderboard_path):
            try:
                with open(self.leaderboard_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                bt.logging.error(f"[Leaderboard] Failed to load: {e}")
        return {}

    def _save(self):
        try:
            with open(self.leaderboard_path, "w") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            bt.logging.error(f"[Leaderboard] Failed to save: {e}")

    def update(self, miner_stats: Dict[int, Dict[str, Any]], preferred_uid: int, consensus_history: List[Dict[str, Any]]):
        """
        Update leaderboard with latest miner stats, preferred miner, and consensus history.
        """
        now = datetime.utcnow().isoformat()
        self.data = {
            "last_updated": now,
            "miners": miner_stats,
            "preferred_miner": preferred_uid,
            "consensus_history": consensus_history[-20:]  # Keep last 20 rounds for audit
        }
        bt.logging.info(f"[Leaderboard] Leaderboard updated at {now}")
        self._save()

    def get_leaderboard(self) -> Dict[str, Any]:
        return self.data