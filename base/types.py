from enum import Enum
from typing import List, Dict, NamedTuple
from dataclasses import dataclass
from datetime import datetime

class ChainType(Enum):
    ETHEREUM = "ethereum"
    BASE = "base"
    SOLANA = "solana"

@dataclass
class TokenPrediction:
    chain: ChainType
    addresses: List[str]
    pairAddresses: List[str]
    timestamp: datetime
    confidence_scores: Dict[str, float]

@dataclass
class PredictionHistory:
    miner_uid: int
    predictions: List[TokenPrediction]
    performance_metrics: Dict[str, float]

class LiquidityMetrics(NamedTuple):
    current_liquidity: float
