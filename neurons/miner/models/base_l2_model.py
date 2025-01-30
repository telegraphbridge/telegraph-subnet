from abc import ABC, abstractmethod
from ....base.types import ChainType, TokenPrediction
import numpy as np
import tensorflow as tf
import joblib

class BaseTokenModel(ABC):
    @abstractmethod
    async def predict(self, chain: ChainType) -> TokenPrediction:
        """Generate token predictions for the specified chain"""
        pass

class LSTMTokenModel(BaseTokenModel):
    