import os
import json
from datetime import datetime, timedelta
import bittensor as bt

class SubmissionTracker:
    """
    Tracks model submissions for retraining enforcement and auditability.
    """
    def __init__(self, storage_path: str = "data/submission_history.json"):
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(storage_path), exist_ok=True)
        self._load()

    def _load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r") as f:
                    self.history = json.load(f)
            except Exception as e:
                bt.logging.error(f"[SubmissionTracker] Failed to load: {e}")
                self.history = {}
        else:
            self.history = {}

    def _save(self):
        try:
            with open(self.storage_path, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            bt.logging.error(f"[SubmissionTracker] Failed to save: {e}")

    def record_submission(self, chain: str, model_hash: str, dataset_version: str):
        now = datetime.utcnow().isoformat()
        self.history.setdefault(chain, []).append({
            "timestamp": now,
            "model_hash": model_hash,
            "dataset_version": dataset_version
        })
        bt.logging.info(f"[SubmissionTracker] Submission recorded: {chain}, {model_hash}, {dataset_version}, {now}")
        self._save()

    def last_submission_time(self, chain: str) -> datetime:
        if chain in self.history and self.history[chain]:
            last = self.history[chain][-1]["timestamp"]
            return datetime.fromisoformat(last)
        return datetime.min

    def is_submission_allowed(self, chain: str, min_interval_minutes: int = 60) -> bool:
        last_time = self.last_submission_time(chain)
        now = datetime.utcnow()
        allowed = (now - last_time) > timedelta(minutes=min_interval_minutes)
        if not allowed:
            bt.logging.warning(f"[SubmissionTracker] Submission not allowed for {chain}: last at {last_time}")
        return allowed