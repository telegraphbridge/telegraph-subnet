import os
import json
import bittensor as bt
from datetime import datetime
from typing import Dict, Any, Optional

class BenchmarkDatasetManager:
    def __init__(self, dataset_dir: str = "data/benchmark_datasets"):
        self.dataset_dir = dataset_dir
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.metadata_file = os.path.join(self.dataset_dir, "metadata.json")
        self._load_metadata()

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "r") as f:
                    self.metadata = json.load(f)
                bt.logging.info(f"Loaded benchmark dataset metadata: {self.metadata}")
            except Exception as e:
                bt.logging.error(f"Failed to load benchmark metadata: {e}")
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self):
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
            bt.logging.debug("Benchmark dataset metadata saved.")
        except Exception as e:
            bt.logging.error(f"Failed to save benchmark metadata: {e}")

    def add_dataset(self, chain: str, dataset: list, description: str = "") -> str:
        """Add a new benchmark dataset for a chain, returns version hash."""
        timestamp = datetime.utcnow().isoformat()
        version = f"{chain}_{timestamp.replace(':', '').replace('.', '')}"
        file_path = os.path.join(self.dataset_dir, f"{version}.json")
        try:
            with open(file_path, "w") as f:
                json.dump(dataset, f)
            self.metadata[chain] = {
                "version": version,
                "file": file_path,
                "timestamp": timestamp,
                "description": description
            }
            self._save_metadata()
            bt.logging.info(f"Added new benchmark dataset for {chain}: {version}")
            return version
        except Exception as e:
            bt.logging.error(f"Failed to add benchmark dataset: {e}")
            return ""

    def get_latest_dataset(self, chain: str) -> Optional[Dict[str, Any]]:
        """Fetch the latest benchmark dataset and metadata for a chain."""
        meta = self.metadata.get(chain)
        if not meta:
            bt.logging.warning(f"No benchmark dataset found for chain {chain}")
            return None
        try:
            with open(meta["file"], "r") as f:
                data = json.load(f)
            bt.logging.info(f"Loaded benchmark dataset for {chain}, version {meta['version']}")
            return {
                "version": meta["version"],
                "timestamp": meta["timestamp"],
                "description": meta.get("description", ""),
                "data": data
            }
        except Exception as e:
            bt.logging.error(f"Failed to load benchmark dataset for {chain}: {e}")
            return None

    def get_dataset_version(self, chain: str) -> Optional[str]:
        meta = self.metadata.get(chain)
        return meta["version"] if meta else None