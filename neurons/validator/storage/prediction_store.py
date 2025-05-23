import os
import json
from datetime import datetime
from typing import List, Optional
from base.types import TokenPrediction, PredictionHistory, ChainType

class PredictionStore:
    def __init__(self, storage_path: str = "./predictions"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

    async def store_prediction(self, miner_uid: int, prediction: TokenPrediction):
        """Store a new prediction, maintaining max 5 predictions per day for last 7 days"""
        file_path = os.path.join(self.storage_path, f"miner_{miner_uid}.json")
        predictions = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    predictions = [
                        TokenPrediction(
                            chain=ChainType(p['chain']),
                            addresses=p['addresses'],
                            pairAddresses=p.get('pairAddresses', []),  # FIX: Provide default empty list
                            timestamp=datetime.fromisoformat(p['timestamp']),
                            confidence_scores=p['confidence_scores']
                        ) for p in data
                    ]
                except json.JSONDecodeError:
                    predictions = []

        # Remove old predictions (older than 7 days)
        current_time = datetime.now()
        predictions = [p for p in predictions if (current_time - p.timestamp).days < 7]

        # Group predictions by day and keep only five per day (omitting details)
        predictions_by_day = {}
        for p in predictions:
            day_key = p.timestamp.date().isoformat()
            predictions_by_day.setdefault(day_key, []).append(p)
        for day in predictions_by_day:
            predictions_by_day[day] = sorted(predictions_by_day[day], key=lambda x: x.timestamp, reverse=True)[:5]

        day_key = prediction.timestamp.date().isoformat()
        predictions_by_day.setdefault(day_key, []).append(prediction)
        predictions_by_day[day_key] = sorted(predictions_by_day[day_key], key=lambda x: x.timestamp, reverse=True)[:5]

        # Flatten and save
        all_predictions = [p for day_preds in predictions_by_day.values() for p in day_preds]
        with open(file_path, 'w') as f:
            json.dump([{
                'chain': p.chain.value,
                'addresses': p.addresses,
                'pairAddresses': p.pairAddresses,  # FIX: Save pairAddresses
                'timestamp': p.timestamp.isoformat(),
                'confidence_scores': p.confidence_scores
            } for p in all_predictions], f)

    async def get_miner_history(self, miner_uid: int, days: int = 7) -> Optional[PredictionHistory]:
        """Retrieve miner's prediction history for the specified number of days (default 7)"""
        file_path = os.path.join(self.storage_path, f"miner_{miner_uid}.json")
        if not os.path.exists(file_path):
            return None
        
        
        with open(file_path, 'w') as f:
            json.dump([{
                'chain': p.chain.value,
                'addresses': p.addresses,
                'pairAddresses': p.pairAddresses,
                'timestamp': p.timestamp.isoformat(),
                'confidence_scores': p.confidence_scores,
                'inference_time_ms': getattr(p, 'inference_time_ms', 0),
                'memory_usage_mb': getattr(p, 'memory_usage_mb', 0)
            } for p in all_predictions], f)
        
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                predictions = [
                    TokenPrediction(
                        chain=ChainType(p['chain']),
                        addresses=p['addresses'],
                        pairAddresses=p.get('pairAddresses', []),  # FIX here as well
                        timestamp=datetime.fromisoformat(p['timestamp']),
                        confidence_scores=p['confidence_scores']
                    ) for p in data
                ]
                current_time = datetime.now()
                filtered_predictions = [p for p in predictions if (current_time - p.timestamp).days < days]
                return PredictionHistory(
                    miner_uid=miner_uid,
                    predictions=filtered_predictions,
                    performance_metrics={}
                )
            except json.JSONDecodeError:
                return None