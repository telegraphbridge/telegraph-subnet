from typing import List, Optional
from ....base.types import TokenPrediction, PredictionHistory
import json
import os
from datetime import datetime
from ....base.types import ChainType

class PredictionStore:
    def __init__(self, storage_path: str = "./predictions"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

    async def store_prediction(self, miner_uid: int, prediction: TokenPrediction):
        """Store a new prediction, maintaining max 5 predictions per day for last 7 days"""
        file_path = os.path.join(self.storage_path, f"miner_{miner_uid}.json")
        
        # Load existing predictions or create new structure
        predictions = []
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    predictions = [
                        TokenPrediction(
                            chain=ChainType(p['chain']),
                            addresses=p['addresses'],
                            timestamp=datetime.fromisoformat(p['timestamp']),
                            confidence_scores=p['confidence_scores']
                        ) for p in data
                    ]
                except json.JSONDecodeError:
                    predictions = []

        # Remove predictions older than 7 days
        current_time = datetime.now()
        predictions = [
            p for p in predictions 
            if (current_time - p.timestamp).days < 7
        ]

        # Group predictions by day
        predictions_by_day = {}
        for p in predictions:
            day_key = p.timestamp.date().isoformat()
            if day_key not in predictions_by_day:
                predictions_by_day[day_key] = []
            predictions_by_day[day_key].append(p)

        # Keep only 5 most recent predictions per day
        for day in predictions_by_day:
            predictions_by_day[day] = sorted(
                predictions_by_day[day],
                key=lambda x: x.timestamp,
                reverse=True
            )[:5]

        # Add new prediction
        day_key = prediction.timestamp.date().isoformat()
        if day_key not in predictions_by_day:
            predictions_by_day[day_key] = []
        predictions_by_day[day_key].append(prediction)
        predictions_by_day[day_key] = sorted(
            predictions_by_day[day_key],
            key=lambda x: x.timestamp,
            reverse=True
        )[:5]

        # Flatten predictions and save
        all_predictions = [
            p for day_predictions in predictions_by_day.values()
            for p in day_predictions
        ]
        
        with open(file_path, 'w') as f:
            json.dump([{
                'chain': p.chain.value,
                'addresses': p.addresses,
                'timestamp': p.timestamp.isoformat(),
                'confidence_scores': p.confidence_scores
            } for p in all_predictions], f)

    async def get_miner_history(self, miner_uid: int, days: int = 7) -> Optional[PredictionHistory]:
        """Retrieve miner's prediction history for the specified number of days (default 7)"""
        file_path = os.path.join(self.storage_path, f"miner_{miner_uid}.json")
        
        if not os.path.exists(file_path):
            return None

        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                predictions = [
                    TokenPrediction(
                        chain=ChainType(p['chain']),
                        addresses=p['addresses'],
                        timestamp=datetime.fromisoformat(p['timestamp']),
                        confidence_scores=p['confidence_scores']
                    ) for p in data
                ]
                
                # Filter for requested number of days
                current_time = datetime.now()
                filtered_predictions = [
                    p for p in predictions 
                    if (current_time - p.timestamp).days < days
                ]

                return PredictionHistory(
                    miner_uid=miner_uid,
                    predictions=filtered_predictions,
                    performance_metrics={} # Add performance metrics as needed
                )
            except json.JSONDecodeError:
                return None 