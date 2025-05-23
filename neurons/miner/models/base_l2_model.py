from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
import numpy as np
import os
import json
import bittensor as bt
from datetime import datetime
import threading
import pickle
import glob
from collections import defaultdict
import traceback
import glob
from base.types import ChainType, TokenPrediction
from neurons.miner.utils.submission_tracker import SubmissionTracker
import hashlib
import time
import psutil

class BaseTokenModel(ABC):
    @abstractmethod
    async def predict(self, chain: ChainType) -> TokenPrediction:
        """Generate token predictions for the specified chain"""
        pass

class NetherilLSTM(nn.Module):
    """LSTM network for token prediction based on Netheril whitepaper approach"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):

        """Initialize LSTM network for token prediction
        
        Args:
            input_size: Number of features in input
            hidden_size: Number of features in hidden state
            num_layers: Number of recurrent layers
            dropout: Dropout rate for regularization
        """
        super(NetherilLSTM, self).__init__()
    
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers for token scoring
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, 1)  # Output is a single token score
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        unbatched = False
        if x.dim() == 2:                  # (seq_len, input_size)
            x = x.unsqueeze(0)            # (1, seq_len, input_size)
            unbatched = True

        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))  # (batch, seq_len, hidden)
        out = out[:, -1, :]              # (batch, hidden)

        # use the declared fc1/fc2 layers
        out = self.fc1(out)              # -> hidden/2
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)              # -> 1
        out = self.sigmoid(out)          # -> (batch, 1)

        if unbatched:
            out = out.squeeze(0)         # back to (1,) or scalar

        return out

class LSTMTokenModel(BaseTokenModel):
    def __init__(self, 
                model_path: str = None,
                data_dir: str = "data/transactions", 
                hidden_size: int = 128, 
                num_layers: int = 2,
                device: str = None):
        """Initialize LSTM token prediction model based on Netheril approach
        
        Args:
            model_path: Path to pre-trained model weights (optional)
            data_dir: Directory containing transaction data files
            hidden_size: Number of features in LSTM hidden state
            num_layers: Number of recurrent layers in LSTM
            device: Device to run model on (cuda:0, cpu, etc.)
        """
        # Set device for computation
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
        # Directory for transaction data
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Token caches - populated during prediction
        self.token_cache = {}  # Maps token addresses to their metadata
        self.wallet_cache = {}  # Maps wallet addresses to their stats/history
        self.token_scores = {}  # Cached token scores from recent predictions
        
        # Cache for pair addresses (token -> pair mapping)
        self.pair_lookup = {}
        
        # Features from the Netheril whitepaper
        self.features = [
            'liquidityPoolSize', 
            'volume24hUsd', 
            'marketCapUsd',
            'buyAmount',
            'buyValueEth',
            'price',
            'priceInUsd',
            'walletEthBalance',
            # Wallet stats (calculated during preprocessing)
            'walletHistoricalROI',
            'walletTradeFrequency',
            'walletDiversityScore'
        ]
        
        # Number of historical time steps to use for liquidity and volume sequences
        self.seq_length = 12  # Use 12 historical points as shown in example data
        
        # Calculate input size based on features plus historical sequences
        self.input_size = len(self.features) + (2 * self.seq_length)  # Add liquidity and volume sequences
        
        # Create model
        self.model = NetherilLSTM(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        ).to(self.device)
        
        self.scaler = None
        scaler_path = os.path.join(data_dir, "feature_scaler.pkl")
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                bt.logging.info(f"Loaded feature scaler from {scaler_path}")
            except Exception as e:
                bt.logging.error(f"Error loading feature scaler: {str(e)}")
        else:
            bt.logging.warning("No feature scaler found. Predictions may be less accurate.")
        
        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
            bt.logging.info(f"Loaded pre-trained model from {model_path}")
        else:
            bt.logging.warning("No pre-trained model found. Using initialized model.")
        
        # Load token pair data if available
        self._load_token_pairs()
        
        # Threading lock for concurrent access
        self.lock = threading.Lock()
        
    def _load_token_pairs(self):
        """Load token pair information from data files"""
        try:
            pair_file = os.path.join(self.data_dir, "token_pairs.json")
            if os.path.exists(pair_file):
                with open(pair_file, 'r') as f:
                    self.pair_lookup = json.load(f)
                bt.logging.info(f"Loaded {len(self.pair_lookup)} token pairs")
            else:
                bt.logging.info("No token pair data found")
        except Exception as e:
            bt.logging.error(f"Error loading token pairs: {str(e)}")
            
    def _save_token_pairs(self):
        """Save token pair information to data file"""
        try:
            pair_file = os.path.join(self.data_dir, "token_pairs.json")
            with open(pair_file, 'w') as f:
                json.dump(self.pair_lookup, f)
            bt.logging.debug(f"Saved {len(self.pair_lookup)} token pairs")
        except Exception as e:
            bt.logging.error(f"Error saving token pairs: {str(e)}")
            
    # Modify the _load_model method to add better error handling and debugging:
    def _load_model(self, model_path: str):
        """Load pre-trained model weights with better error handling"""
        try:
            bt.logging.info(f"Attempting to load model from: {model_path}")
            if not os.path.exists(model_path):
                bt.logging.error(f"Model file not found at: {model_path}")
                return
                
            # Print file size for debugging
            file_size = os.path.getsize(model_path)
            bt.logging.info(f"Model file size: {file_size} bytes")
            
            checkpoint = torch.load(model_path, map_location=self.device)
            bt.logging.info(f"Model loaded successfully. Keys: {checkpoint.keys()}")
            
            if 'model_state_dict' not in checkpoint:
                bt.logging.error(f"Invalid model file - missing 'model_state_dict' key")
                return
                
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Set to evaluation mode
            bt.logging.info(f"Model weights loaded successfully")
            
            # Load token scoring stats if available
            if 'token_scores' in checkpoint:
                self.token_scores = checkpoint['token_scores']
                bt.logging.info(f"Loaded {len(self.token_scores)} token scores")
                
            # Load wallet cache if available
            if 'wallet_cache' in checkpoint:
                self.wallet_cache = checkpoint['wallet_cache']
                bt.logging.info(f"Loaded {len(self.wallet_cache)} wallet entries")
                
        except Exception as e:
            bt.logging.error(f"Error loading model: {str(e)}")
            bt.logging.error(traceback.format_exc())
            bt.logging.warning("Using initialized model instead")

    # Modify the _load_transaction_data method to better handle file paths and errors:
    def _load_transaction_data(self, chain: ChainType) -> List[Dict[str, Any]]:
        """Load transaction data for the specified chain with better debugging"""
        chain_name = chain.name.lower()
        transactions = []
        
        try:
            # Look for transaction files matching the chain
            pattern = os.path.join(self.data_dir, f"{chain_name}_transactions_*.json")
            bt.logging.info(f"Looking for transaction files matching: {pattern}")
            
            files = glob.glob(pattern)
            if not files:
                # Try alternative pattern (uppercase)
                pattern = os.path.join(self.data_dir, f"{chain_name.upper()}_transactions_*.json")
                bt.logging.info(f"Trying alternative pattern: {pattern}")
                files = glob.glob(pattern)
                
            # List all files in directory for debugging
            all_files = os.listdir(self.data_dir)
            bt.logging.info(f"All files in data directory: {all_files}")
            
            if not files:
                # Try looking for any transaction file
                pattern = os.path.join(self.data_dir, "*_transactions_*.json")
                files = glob.glob(pattern)
                if files:
                    bt.logging.warning(f"Found non-matching transaction files: {files}")
                
                bt.logging.warning(f"No transaction data found for chain {chain_name}")
                return []
                
            # Load the most recent file (sorted by timestamp in filename)
            files.sort(reverse=True)
            latest_file = files[0]
            bt.logging.info(f"Loading transaction data from: {latest_file}")
            
            with open(latest_file, 'r') as f:
                transactions = json.load(f)
                
            bt.logging.info(f"Loaded {len(transactions)} transactions")
            
            # Validate transaction data format
            if transactions:
                bt.logging.info(f"Sample transaction keys: {list(transactions[0].keys())}")
            
            return transactions
        except Exception as e:
            bt.logging.error(f"Error loading transaction data: {str(e)}")
            bt.logging.error(traceback.format_exc())
            return []
                
    def _calculate_wallet_stats(self, transactions: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate wallet statistics from transaction data
        
        Args:
            transactions: List of transaction data dictionaries
            
        Returns:
            Dictionary mapping wallet addresses to their statistics
        """
        wallet_stats = {}
        wallet_tokens = defaultdict(set)
        wallet_trades = defaultdict(int)
        
        # Process transactions to gather wallet stats
        for tx in transactions:
            wallet = tx.get('walletID')
            if not wallet:
                continue
                
            # Track tokens traded by each wallet
            token = tx.get('tokenID')
            if token:
                wallet_tokens[wallet].add(token)
                
            # Count trades per wallet
            wallet_trades[wallet] += 1
            
            # Cache token data
            if token and token not in self.token_cache:
                self.token_cache[token] = {
                    'symbol': tx.get('tokenSymbol', ''),
                    'price': tx.get('price', 0),
                    'liquidity': tx.get('liquidityPoolSize', 0),
                    'volume': tx.get('volume24hUsd', 0),
                    'marketCap': tx.get('marketCapUsd', 0)
                }
                
                # If we have pool address, add to pair lookup
                if 'poolAddress' in tx:
                    self.pair_lookup[token] = tx['poolAddress']
        
        # Compute statistics for each wallet
        for wallet in wallet_trades.keys():
            # Diversity score = number of unique tokens traded
            diversity_score = len(wallet_tokens[wallet])
            
            # Trade frequency = number of trades
            trade_frequency = wallet_trades[wallet]
            
            # For ROI, we would need historical data - use placeholder for now
            # In production, this would be calculated from actual wallet performance
            roi = 0.0
            
            if wallet in self.wallet_cache:
                # Use historical ROI if available
                roi = self.wallet_cache[wallet].get('roi', 0.0)
                
                # Update with new data
                self.wallet_cache[wallet].update({
                    'diversity_score': diversity_score,
                    'trade_frequency': trade_frequency
                })
            else:
                # Create new wallet cache entry
                self.wallet_cache[wallet] = {
                    'diversity_score': diversity_score,
                    'trade_frequency': trade_frequency,
                    'roi': roi
                }
            
            # Add to stats dictionary
            wallet_stats[wallet] = {
                'historical_roi': roi,
                'trade_frequency': trade_frequency,
                'diversity_score': diversity_score
            }
            
        return wallet_stats
        
    def _preprocess_transaction(self, tx: Dict[str, Any], wallet_stats: Dict[str, Dict[str, float]]) -> torch.Tensor:
        """Preprocess a single transaction for model input
        
        Args:
            tx: Transaction data dictionary
            wallet_stats: Dictionary of wallet statistics
            
        Returns:
            Tensor representation of the transaction features
        """
        try:
            wallet = tx.get('walletID', '')
            
            # Get wallet stats
            wallet_roi = wallet_stats.get(wallet, {}).get('historical_roi', 0.0)
            wallet_trades = wallet_stats.get(wallet, {}).get('trade_frequency', 0)
            wallet_diversity = wallet_stats.get(wallet, {}).get('diversity_score', 0)
            
            # Extract base features
            features = [
                float(tx.get('liquidityPoolSize', 0)),
                float(tx.get('volume24hUsd', 0)),
                float(tx.get('marketCapUsd', 0)),
                float(tx.get('buyAmount', 0)),
                float(tx.get('buyValueEth', 0)),
                float(tx.get('price', 0)),
                float(tx.get('priceInUsd', 0)),
                float(tx.get('walletEthBalance', 0)),
                float(wallet_roi),
                float(wallet_trades),
                float(wallet_diversity)
            ]
            
            # Extract historical liquidity sequence
            liquidity_seq = []
            if 'historicalLiquidity' in tx:
                for item in tx['historicalLiquidity']:
                    liquidity_seq.append(float(item.get('value', 0)))
                    
                # Pad if needed
                while len(liquidity_seq) < self.seq_length:
                    liquidity_seq.append(0.0)
                    
                # Truncate if too long
                liquidity_seq = liquidity_seq[:self.seq_length]
            else:
                liquidity_seq = [0.0] * self.seq_length
                
            # Extract historical volume sequence
            volume_seq = []
            if 'historicalVolume' in tx:
                for item in tx['historicalVolume']:
                    volume_seq.append(float(item.get('value', 0)))
                    
                # Pad if needed
                while len(volume_seq) < self.seq_length:
                    volume_seq.append(0.0)
                    
                # Truncate if too long
                volume_seq = volume_seq[:self.seq_length]
            else:
                volume_seq = [0.0] * self.seq_length
                
            all_features = features + liquidity_seq + volume_seq
            
            # Apply scaling if scaler is available
            if self.scaler is not None:
                all_features = self.scaler.transform([all_features])[0]
            # Create tensor
            return torch.tensor([all_features], dtype=torch.float32).to(self.device)
            
        except Exception as e:
            bt.logging.error(f"Error preprocessing transaction: {str(e)}")
            # Return zero tensor in case of error
            return torch.zeros((1, self.input_size), dtype=torch.float32).to(self.device)
            
    def _preprocess_transactions(self, transactions: List[Dict[str, Any]]) -> torch.Tensor:
        """Preprocess multiple transactions for batch prediction
        
        Args:
            transactions: List of transaction data dictionaries
            
        Returns:
            Tensor batch of transaction features
        """
        if not transactions:
            # Return empty tensor if no transactions
            return torch.zeros((0, self.input_size), dtype=torch.float32).to(self.device)
            
        try:
            # Calculate wallet statistics
            wallet_stats = self._calculate_wallet_stats(transactions)
            
            # Process each transaction
            features_list = []
            for tx in transactions:
                tx_tensor = self._preprocess_transaction(tx, wallet_stats)
                features_list.append(tx_tensor)
                
            # Combine into batch
            if features_list:
                return torch.cat(features_list, dim=0)
            else:
                return torch.zeros((0, self.input_size), dtype=torch.float32).to(self.device)
                
        except Exception as e:
            bt.logging.error(f"Error preprocessing transactions: {str(e)}")
            return torch.zeros((0, self.input_size), dtype=torch.float32).to(self.device)
            
    def _get_top_tokens(self, transactions: list, scores: torch.Tensor, k: int = 10) -> Tuple[list, list, dict]:
        try:
            score_list = scores.squeeze().tolist()
            # Ensure score_list is a list
            if isinstance(score_list, float):
                score_list = [score_list]
            # Lower threshold if desired (here using 0.001)
            valid_indices = [i for i, score in enumerate(score_list) if score > 0.001]
            bt.logging.info(f"Valid token indices: {valid_indices}")
            if not valid_indices:
                bt.logging.warning("No valid token scores found")
                return ([], [], {})
            # Sort valid tokens by score (highest first)
            sorted_indices = sorted(valid_indices, key=lambda i: score_list[i], reverse=True)
            token_addresses = []
            pair_addresses = []
            confidence_scores = {}
            # Only return tokens that are not placeholders
            for idx in sorted_indices[:k]:
                tx = transactions[idx]
                token = tx.get('tokenID')
                pair = tx.get('poolAddress')
                # Only accept if model provided a nonzero token; otherwise skip it.
                if token and token != "0x0000000000000000000000000000000000000000":
                    token_addresses.append(token)
                    pair_addresses.append(pair or "0x0000000000000000000000000000000000000000")
                    confidence_scores[token] = score_list[idx]
            bt.logging.info(f"Selected tokens: {token_addresses}")
            return token_addresses, pair_addresses, confidence_scores
        except Exception as e:
            bt.logging.error(f"Error in _get_top_tokens: {str(e)}")
            return ([], [], {})
            
    async def predict(self, chain: ChainType) -> TokenPrediction:
        """Generate token predictions for the specified chain, with timing and resource metrics."""
        with self.lock:
            try:
                # --- Start timing and resource usage ---
                start_time = time.perf_counter()
                process = psutil.Process(os.getpid())
                start_mem = process.memory_info().rss

                # Load transaction data for the specified chain
                transactions = self._load_transaction_data(chain)
                if not transactions:
                    bt.logging.warning(f"No transaction data available for chain {chain}")
                    return TokenPrediction(
                        chain=chain,
                        addresses=[f"0x{i:040x}" for i in range(10, 20)],
                        pairAddresses=[f"0x{i:040x}" for i in range(100, 110)],
                        timestamp=datetime.now(),
                        confidence_scores={f"0x{i:040x}": 0.5 for i in range(10, 20)},
                        inference_time_ms=0,
                        memory_usage_mb=0
                    )

                features = self._preprocess_transactions(transactions)
                with torch.no_grad():
                    scores = self.model(features)

                addresses, pair_addresses, confidence_scores = self._get_top_tokens(
                    transactions=transactions,
                    scores=scores,
                    k=10
                )

                # --- End timing and resource usage ---
                end_time = time.perf_counter()
                end_mem = process.memory_info().rss
                inference_time_ms = int((end_time - start_time) * 1000)
                memory_usage_mb = int((end_mem - start_mem) / (1024 * 1024))

                bt.logging.info(f"[LSTMTokenModel] Inference time: {inference_time_ms} ms, Memory usage: {memory_usage_mb} MB")

                self._save_token_pairs()

                return TokenPrediction(
                    chain=chain,
                    addresses=addresses,
                    pairAddresses=pair_addresses,
                    timestamp=datetime.now(),
                    confidence_scores=confidence_scores,
                    inference_time_ms=inference_time_ms,
                    memory_usage_mb=memory_usage_mb
                )

            except Exception as e:
                bt.logging.error(f"Error making prediction: {str(e)}")
                return TokenPrediction(
                    chain=chain,
                    addresses=[f"0x{i:040x}" for i in range(10, 20)],
                    pairAddresses=[f"0x{i:040x}" for i in range(100, 110)],
                    timestamp=datetime.now(),
                    confidence_scores={f"0x{i:040x}": 0.5 for i in range(10, 20)},
                    inference_time_ms=0,
                    memory_usage_mb=0
                )
                            
    def get_model_hash(self) -> str:
        """Returns a hash of the current model weights for uniqueness tracking."""
        try:
            weights = self.model.state_dict()
            weights_bytes = b"".join([v.cpu().numpy().tobytes() for v in weights.values()])
            return hashlib.sha256(weights_bytes).hexdigest()
        except Exception as e:
            bt.logging.error(f"[LSTMTokenModel] Error hashing model: {e}")
            return "unknown"

    def submit_model(self, chain: ChainType, dataset_version: str, min_interval_minutes: int = 60):
        """
        Handles model submission: checks interval, records submission, and logs.
        """
        if not hasattr(self, "submission_tracker"):
            self.submission_tracker = SubmissionTracker()
        if not self.submission_tracker.is_submission_allowed(chain.value, min_interval_minutes):
            bt.logging.warning(f"[LSTMTokenModel] Submission blocked: too soon for {chain.value}")
            return False
        model_hash = self.get_model_hash()
        self.submission_tracker.record_submission(chain.value, model_hash, dataset_version)
        bt.logging.info(f"[LSTMTokenModel] Model submitted for {chain.value} with hash {model_hash} and dataset {dataset_version}")
        return True

    def save_model(self, path: str):
        """Save model weights and associated data to file
        
        Args:
            path: Path to save model file
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'token_scores': self.token_scores,
                'wallet_cache': self.wallet_cache
            }, path)
            bt.logging.info(f"Model saved to {path}")
        except Exception as e:
            bt.logging.error(f"Error saving model: {str(e)}")