import bittensor as bt
from base.types import TokenPrediction, ChainType
from typing import Optional

class SubmissionValidator:
    """
    Validates incoming miner submissions for format, recency, and uniqueness.
    """
    def __init__(self, min_confidence: float = 0.0):
        self.min_confidence = min_confidence

    def validate(self, prediction: TokenPrediction) -> Optional[str]:
        # Check addresses
        if not prediction.addresses or not isinstance(prediction.addresses, list):
            return "Missing or invalid addresses"
        if not all(isinstance(addr, str) and addr.startswith("0x") for addr in prediction.addresses):
            return "Invalid address format"
        # Check confidence scores
        if not isinstance(prediction.confidence_scores, dict):
            return "Missing or invalid confidence_scores"
        for addr, score in prediction.confidence_scores.items():
            if not isinstance(score, (float, int)) or score < self.min_confidence:
                return f"Invalid confidence score for {addr}"
        # Check timestamp
        if not hasattr(prediction, "timestamp"):
            return "Missing timestamp"
        # Check chain type
        if not isinstance(prediction.chain, ChainType):
            return "Invalid chain type"
        # TODO: Add signature verification for anti-replay and authenticity
        return None  # No error

    def is_duplicate(self, prediction: TokenPrediction, recent_hashes: set) -> bool:
        # Simple hash for uniqueness (addresses + timestamp)
        pred_hash = hash((tuple(prediction.addresses), prediction.timestamp))
        return pred_hash in recent_hashes