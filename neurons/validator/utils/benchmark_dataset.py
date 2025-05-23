import os
import json
import hashlib
import asyncio
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import bittensor as bt

class BenchmarkDatasetManager:
    """
    Production-ready benchmark dataset management system.
    Handles dataset creation, versioning, validation, and automated updates.
    """
    
    def __init__(self, dataset_dir: str = "data/benchmark_datasets", 
                 auto_update_hours: int = 6):
        self.dataset_dir = dataset_dir
        self.auto_update_hours = auto_update_hours
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        self.metadata_file = os.path.join(self.dataset_dir, "metadata.json")
        self.integrity_file = os.path.join(self.dataset_dir, "integrity.json")
        
        # Initialize state
        self.metadata = {}
        self.integrity_hashes = {}
        
        # Load existing data
        self._load_metadata()
        self._load_integrity_data()
        
        bt.logging.info(f"BenchmarkDatasetManager initialized with {len(self.metadata)} datasets")

    def _load_metadata(self):
        """Load dataset metadata with error handling"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r") as f:
                    self.metadata = json.load(f)
                bt.logging.info(f"Loaded benchmark dataset metadata: {len(self.metadata)} chains")
            except Exception as e:
                bt.logging.error(f"Failed to load benchmark metadata: {e}")
                self.metadata = {}
        else:
            bt.logging.info("No existing benchmark metadata found, starting fresh")
            self.metadata = {}

    def _save_metadata(self):
        """Save dataset metadata with atomic write"""
        try:
            # Write to temporary file first for atomic operation
            temp_file = self.metadata_file + ".tmp"
            with open(temp_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
            
            # Atomic move
            os.rename(temp_file, self.metadata_file)
            bt.logging.debug("Benchmark dataset metadata saved successfully")
        except Exception as e:
            bt.logging.error(f"Failed to save benchmark metadata: {e}")

    def _load_integrity_data(self):
        """Load file integrity hashes"""
        if os.path.exists(self.integrity_file):
            try:
                with open(self.integrity_file, "r") as f:
                    self.integrity_hashes = json.load(f)
                bt.logging.debug(f"Loaded integrity data for {len(self.integrity_hashes)} files")
            except Exception as e:
                bt.logging.error(f"Failed to load integrity data: {e}")
                self.integrity_hashes = {}
        else:
            self.integrity_hashes = {}

    def _save_integrity_data(self):
        """Save file integrity hashes"""
        try:
            with open(self.integrity_file, "w") as f:
                json.dump(self.integrity_hashes, f, indent=2)
            bt.logging.debug("Integrity data saved successfully")
        except Exception as e:
            bt.logging.error(f"Failed to save integrity data: {e}")

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file for integrity checking"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            bt.logging.error(f"Error calculating hash for {file_path}: {e}")
            return ""

    def _validate_dataset_format(self, dataset: List[Dict[str, Any]]) -> bool:
        """Validate dataset format and required fields"""
        if not isinstance(dataset, list):
            bt.logging.error("Dataset must be a list")
            return False
        
        if len(dataset) == 0:
            bt.logging.warning("Dataset is empty")
            return False
        
        # Check required fields in each record
        required_fields = [
            'tokenID', 'liquidityPoolSize', 'volume24hUsd', 
            'marketCapUsd', 'price', 'timestamp'
        ]
        
        for i, record in enumerate(dataset[:10]):  # Check first 10 records
            if not isinstance(record, dict):
                bt.logging.error(f"Record {i} is not a dictionary")
                return False
            
            for field in required_fields:
                if field not in record:
                    bt.logging.error(f"Missing required field '{field}' in record {i}")
                    return False
            
            # Validate data types
            try:
                float(record['liquidityPoolSize'])
                float(record['volume24hUsd'])
                float(record['marketCapUsd'])
                float(record['price'])
            except (ValueError, TypeError):
                bt.logging.error(f"Invalid numeric data in record {i}")
                return False
        
        bt.logging.info(f"Dataset validation passed: {len(dataset)} records")
        return True

    def add_dataset(self, chain: str, dataset: List[Dict[str, Any]], 
                   description: str = "", validate: bool = True) -> str:
        """
        Add a new benchmark dataset for a chain with validation and versioning.
        
        Args:
            chain: Chain identifier (e.g., 'base', 'ethereum')
            dataset: List of transaction/token data records
            description: Human-readable description of the dataset
            validate: Whether to validate dataset format
            
        Returns:
            Version identifier of the created dataset
        """
        try:
            bt.logging.info(f"Adding new benchmark dataset for chain {chain}")
            
            # Validate dataset format if requested
            if validate and not self._validate_dataset_format(dataset):
                bt.logging.error(f"Dataset validation failed for chain {chain}")
                return ""
            
            # Create version identifier
            timestamp = datetime.utcnow()
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            version = f"{chain}_{timestamp_str}"
            
            # Save dataset file
            file_path = os.path.join(self.dataset_dir, f"{version}.json")
            
            with open(file_path, "w") as f:
                json.dump(dataset, f, indent=2)
            
            # Calculate file hash for integrity
            file_hash = self._calculate_file_hash(file_path)
            
            # Update metadata
            self.metadata[chain] = {
                "version": version,
                "file": file_path,
                "timestamp": timestamp.isoformat(),
                "description": description,
                "record_count": len(dataset),
                "file_size": os.path.getsize(file_path),
                "hash": file_hash
            }
            
            # Update integrity tracking
            self.integrity_hashes[version] = file_hash
            
            # Save metadata and integrity data
            self._save_metadata()
            self._save_integrity_data()
            
            bt.logging.info(f"Successfully added benchmark dataset for {chain}: {version}")
            bt.logging.info(f"Dataset contains {len(dataset)} records, file size: {os.path.getsize(file_path)} bytes")
            
            return version
            
        except Exception as e:
            bt.logging.error(f"Failed to add benchmark dataset for {chain}: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return ""

    def get_latest_dataset(self, chain: str) -> Optional[Dict[str, Any]]:
        """
        Fetch the latest benchmark dataset and metadata for a chain.
        
        Args:
            chain: Chain identifier
            
        Returns:
            Dictionary containing dataset version, metadata, and data
        """
        try:
            meta = self.metadata.get(chain)
            if not meta:
                bt.logging.warning(f"No benchmark dataset found for chain {chain}")
                return None
            
            file_path = meta["file"]
            
            # Verify file exists
            if not os.path.exists(file_path):
                bt.logging.error(f"Dataset file not found: {file_path}")
                return None
            
            # Verify file integrity
            current_hash = self._calculate_file_hash(file_path)
            expected_hash = meta.get("hash", "")
            
            if current_hash != expected_hash:
                bt.logging.error(f"Dataset integrity check failed for {chain}")
                bt.logging.error(f"Expected hash: {expected_hash}, got: {current_hash}")
                return None
            
            # Load dataset
            with open(file_path, "r") as f:
                data = json.load(f)
            
            bt.logging.info(f"Loaded benchmark dataset for {chain}")
            bt.logging.info(f"Version: {meta['version']}, Records: {len(data)}")
            
            return {
                "version": meta["version"],
                "timestamp": meta["timestamp"],
                "description": meta.get("description", ""),
                "record_count": len(data),
                "file_size": meta.get("file_size", 0),
                "data": data
            }
            
        except Exception as e:
            bt.logging.error(f"Failed to load benchmark dataset for {chain}: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return None

    def get_dataset_version(self, chain: str) -> Optional[str]:
        """Get the current dataset version for a chain"""
        meta = self.metadata.get(chain)
        return meta["version"] if meta else None

    def is_dataset_outdated(self, chain: str) -> bool:
        """Check if dataset needs updating based on age"""
        meta = self.metadata.get(chain)
        if not meta:
            return True  # No dataset exists
        
        try:
            timestamp = datetime.fromisoformat(meta["timestamp"])
            age_hours = (datetime.utcnow() - timestamp).total_seconds() / 3600
            
            outdated = age_hours > self.auto_update_hours
            if outdated:
                bt.logging.info(f"Dataset for {chain} is {age_hours:.1f} hours old (threshold: {self.auto_update_hours})")
            
            return outdated
            
        except Exception as e:
            bt.logging.error(f"Error checking dataset age for {chain}: {e}")
            return True  # Assume outdated if we can't check

    def generate_sample_dataset(self, chain: str, size: int = 100) -> List[Dict[str, Any]]:
        """
        Generate sample dataset for testing and initial setup.
        
        Args:
            chain: Chain identifier
            size: Number of sample records to generate
            
        Returns:
            List of sample transaction records
        """
        import random
        import string
        
        bt.logging.info(f"Generating sample dataset for {chain} with {size} records")
        
        sample_data = []
        base_timestamp = datetime.utcnow() - timedelta(days=30)
        
        for i in range(size):
            # Generate random token address
            token_id = "0x" + "".join(random.choices(string.hexdigits.lower(), k=40))
            
            # Generate realistic-looking data
            liquidity = random.uniform(1000, 1000000)
            volume = random.uniform(100, 100000)
            market_cap = random.uniform(10000, 10000000)
            price = random.uniform(0.001, 1000)
            
            # Add timestamp progression
            timestamp = base_timestamp + timedelta(hours=i)
            
            record = {
                'tokenID': token_id,
                'poolAddress': "0x" + "".join(random.choices(string.hexdigits.lower(), k=40)),
                'liquidityPoolSize': liquidity,
                'volume24hUsd': volume,
                'marketCapUsd': market_cap,
                'price': price,
                'priceInUsd': price,
                'timestamp': timestamp.isoformat(),
                'walletID': "0x" + "".join(random.choices(string.hexdigits.lower(), k=40)),
                'buyAmount': random.uniform(0.1, 100),
                'buyValueEth': random.uniform(0.01, 10),
                'walletEthBalance': random.uniform(0.1, 100),
                # Add historical data for LSTM training
                'historicalLiquidity': [
                    {
                        'timestamp': (timestamp - timedelta(days=j)).isoformat(),
                        'value': liquidity * random.uniform(0.8, 1.2)
                    } for j in range(1, 13)
                ],
                'historicalVolume': [
                    {
                        'timestamp': (timestamp - timedelta(days=j)).isoformat(),
                        'value': volume * random.uniform(0.8, 1.2)
                    } for j in range(1, 13)
                ]
            }
            
            sample_data.append(record)
        
        bt.logging.info(f"Generated {len(sample_data)} sample records for {chain}")
        return sample_data

    async def auto_update_datasets(self, chains: List[str]):
        """
        Automatically update datasets that are outdated.
        This would typically fetch fresh data from external sources.
        """
        bt.logging.info("Starting automatic dataset update check")
        
        for chain in chains:
            try:
                if self.is_dataset_outdated(chain):
                    bt.logging.info(f"Updating outdated dataset for {chain}")
                    
                    # In production, this would fetch real data from APIs
                    # For now, generate fresh sample data
                    new_dataset = self.generate_sample_dataset(chain, size=150)
                    
                    version = self.add_dataset(
                        chain=chain,
                        dataset=new_dataset,
                        description=f"Auto-updated dataset for {chain}",
                        validate=True
                    )
                    
                    if version:
                        bt.logging.info(f"Successfully updated dataset for {chain}: {version}")
                    else:
                        bt.logging.error(f"Failed to update dataset for {chain}")
                else:
                    bt.logging.debug(f"Dataset for {chain} is up to date")
                    
            except Exception as e:
                bt.logging.error(f"Error updating dataset for {chain}: {e}")
        
        bt.logging.info("Automatic dataset update check completed")

    def get_all_datasets_info(self) -> Dict[str, Dict[str, Any]]:
        """Get summary information about all datasets"""
        return self.metadata.copy()

    def cleanup_old_datasets(self, keep_versions: int = 3):
        """
        Clean up old dataset files, keeping only the most recent versions.
        
        Args:
            keep_versions: Number of versions to keep per chain
        """
        try:
            bt.logging.info(f"Starting dataset cleanup, keeping {keep_versions} versions per chain")
            
            # Group files by chain
            files_by_chain = {}
            for filename in os.listdir(self.dataset_dir):
                if filename.endswith('.json') and filename != 'metadata.json' and filename != 'integrity.json':
                    parts = filename.replace('.json', '').split('_')
                    if len(parts) >= 2:
                        chain = parts[0]
                        files_by_chain.setdefault(chain, []).append(filename)
            
            # Sort and clean up old files
            deleted_count = 0
            for chain, files in files_by_chain.items():
                files.sort(reverse=True)  # Newest first
                
                if len(files) > keep_versions:
                    files_to_delete = files[keep_versions:]
                    
                    for filename in files_to_delete:
                        file_path = os.path.join(self.dataset_dir, filename)
                        try:
                            os.remove(file_path)
                            deleted_count += 1
                            bt.logging.debug(f"Deleted old dataset file: {filename}")
                        except Exception as e:
                            bt.logging.error(f"Failed to delete {filename}: {e}")
            
            bt.logging.info(f"Dataset cleanup completed: deleted {deleted_count} old files")
            
        except Exception as e:
            bt.logging.error(f"Error during dataset cleanup: {e}")