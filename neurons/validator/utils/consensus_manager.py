import os
import json
import bittensor as bt
from typing import Dict, List, Optional
from datetime import datetime

class ConsensusManager:
    """
    Manages consensus voting and preferred miner selection.
    Stores votes, calculates weighted results, and persists preferred miner status.
    """
    def __init__(self, storage_path: str = "data/consensus/preferred_miner.json"):
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    self.state = json.load(f)
            except Exception as e:
                bt.logging.error(f"[ConsensusManager] Failed to load: {e}")
                self.state = {}
        else:
            self.state = {}

    def _save(self):
        try:
            with open(self.storage_path, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            bt.logging.error(f"[ConsensusManager] Failed to save: {e}")

    def record_votes(self, round_id: str, votes: Dict[int, float]):
        """
        Record votes for a round. votes: {miner_uid: weight}
        """
        now = datetime.utcnow().isoformat()
        self.state.setdefault("history", []).append({
            "round_id": round_id,
            "votes": votes,
            "timestamp": now
        })
        bt.logging.info(f"[ConsensusManager] Votes recorded for round {round_id}: {votes}")
        self._save()

    def select_preferred_miner(self, round_id: str, votes: Dict[int, float]) -> Optional[int]:
        """
        Selects the preferred miner by weighted voting.
        Returns the UID of the preferred miner.
        """
        if not votes:
            bt.logging.warning("[ConsensusManager] No votes to select preferred miner.")
            return None
        preferred_uid = max(votes, key=votes.get)
        self.state["preferred_miner"] = {
            "uid": preferred_uid,
            "round_id": round_id,
            "votes": votes,
            "timestamp": datetime.utcnow().isoformat()
        }
        bt.logging.info(f"[ConsensusManager] Preferred miner for round {round_id}: UID {preferred_uid}")
        self._save()
        return preferred_uid

    def get_preferred_miner(self) -> Optional[int]:
        return self.state.get("preferred_miner", {}).get("uid")