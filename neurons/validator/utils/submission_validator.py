import hashlib
import time
from typing import Optional, Set
from datetime import datetime, timedelta
import bittensor as bt

from base.types import TokenPrediction, ChainType

class SubmissionValidator:
    """
    Production-ready validator for incoming miner submissions.
    Validates format, recency, uniqueness, and quality according to whitepaper standards.
    """
    
    def __init__(self, min_confidence: float = 0.0, max_submission_age_minutes: int = 30):
        self.min_confidence = min_confidence
        self.max_submission_age_minutes = max_submission_age_minutes
        
        # Track recent submissions for duplicate detection
        self.recent_submissions: Set[str] = set()
        self.submission_cleanup_interval = 3600  # Clean up every hour
        self.last_cleanup = time.time()
        
        # Validation limits based on whitepaper
        self.max_addresses_per_submission = 50
        self.min_addresses_per_submission = 1
        self.max_address_length = 42  # 0x + 40 hex chars
        self.min_address_length = 42
        
        bt.logging.info(f"SubmissionValidator initialized with min_confidence={min_confidence}")

    def validate(self, prediction: TokenPrediction) -> Optional[str]:
        """
        Validate a prediction submission for format, content, and quality.
        
        Args:
            prediction: TokenPrediction object to validate
            
        Returns:
            None if valid, error string if invalid
        """
        try:
            # Clean up old submissions periodically
            self._cleanup_old_submissions()
            
            # 1. Validate basic structure
            error = self._validate_basic_structure(prediction)
            if error:
                return error
            
            # 2. Validate addresses format
            error = self._validate_addresses(prediction.addresses)
            if error:
                return error
            
            # 3. Validate pair addresses if provided
            if prediction.pairAddresses:
                error = self._validate_addresses(prediction.pairAddresses, "pairAddresses")
                if error:
                    return error
            
            # 4. Validate confidence scores
            error = self._validate_confidence_scores(prediction)
            if error:
                return error
            
            # 5. Validate timestamp recency
            error = self._validate_timestamp(prediction.timestamp)
            if error:
                return error
            
            # 6. Validate chain support
            error = self._validate_chain(prediction.chain)
            if error:
                return error
            
            # 7. Validate submission quality
            error = self._validate_quality(prediction)
            if error:
                return error
            
            bt.logging.debug(f"Submission validation passed: {len(prediction.addresses)} addresses")
            return None
            
        except Exception as e:
            bt.logging.error(f"Error during submission validation: {str(e)}")
            return f"Validation error: {str(e)}"

    def _validate_basic_structure(self, prediction: TokenPrediction) -> Optional[str]:
        """Validate basic prediction structure"""
        if not prediction:
            return "Prediction object is None"
        
        if not hasattr(prediction, 'addresses') or prediction.addresses is None:
            return "Missing addresses field"
        
        if not hasattr(prediction, 'chain') or prediction.chain is None:
            return "Missing chain field"
        
        if not hasattr(prediction, 'timestamp') or prediction.timestamp is None:
            return "Missing timestamp field"
        
        return None

    def _validate_addresses(self, addresses: list, field_name: str = "addresses") -> Optional[str]:
        """Validate address list format and content"""
        # Check addresses type and basic format
        if not isinstance(addresses, list):
            return f"{field_name} must be a list"
        
        if len(addresses) == 0:
            return f"{field_name} cannot be empty"
        
        if len(addresses) > self.max_addresses_per_submission:
            return f"{field_name} contains too many addresses (max: {self.max_addresses_per_submission})"
        
        if len(addresses) < self.min_addresses_per_submission:
            return f"{field_name} contains too few addresses (min: {self.min_addresses_per_submission})"
        
        # Validate each address format
        seen_addresses = set()
        for i, addr in enumerate(addresses):
            if not isinstance(addr, str):
                return f"{field_name}[{i}] must be a string"
            
            if not addr.startswith("0x"):
                return f"{field_name}[{i}] must start with '0x'"
            
            if len(addr) != self.max_address_length:
                return f"{field_name}[{i}] must be exactly {self.max_address_length} characters"
            
            # Check for valid hex characters
            try:
                int(addr[2:], 16)  # Try to parse as hex
            except ValueError:
                return f"{field_name}[{i}] contains invalid hex characters"
            
            # Check for duplicate addresses within submission
            if addr.lower() in seen_addresses:
                return f"{field_name}[{i}] is a duplicate address"
            seen_addresses.add(addr.lower())
            
            # Check for obviously invalid addresses (all zeros, etc.)
            if addr.lower() == "0x" + "0" * 40:
                return f"{field_name}[{i}] is an invalid zero address"
        
        return None

    def _validate_confidence_scores(self, prediction: TokenPrediction) -> Optional[str]:
        """Validate confidence scores format and values"""
        if not hasattr(prediction, 'confidence_scores'):
            return "Missing confidence_scores field"
        
        confidence_scores = prediction.confidence_scores
        if not isinstance(confidence_scores, dict):
            return "confidence_scores must be a dictionary"
        
        # Confidence scores are optional, but if provided must be valid
        if confidence_scores:
            # Check that all addresses have confidence scores
            addresses_set = set(addr.lower() for addr in prediction.addresses)
            confidence_keys_set = set(addr.lower() for addr in confidence_scores.keys())
            
            # Allow partial confidence scores, but they should be for valid addresses
            invalid_keys = confidence_keys_set - addresses_set
            if invalid_keys:
                return f"confidence_scores contains addresses not in addresses list: {list(invalid_keys)[:3]}"
            
            # Validate confidence values
            for addr, confidence in confidence_scores.items():
                if not isinstance(confidence, (int, float)):
                    return f"confidence_scores[{addr}] must be a number"
                
                if not (0.0 <= confidence <= 1.0):
                    return f"confidence_scores[{addr}] must be between 0.0 and 1.0"
                
                if confidence < self.min_confidence:
                    return f"confidence_scores[{addr}] below minimum threshold ({self.min_confidence})"
        
        return None

    def _validate_timestamp(self, timestamp: datetime) -> Optional[str]:
        """Validate timestamp recency"""
        if not isinstance(timestamp, datetime):
            return "timestamp must be a datetime object"
        
        now = datetime.utcnow()
        age_minutes = (now - timestamp).total_seconds() / 60
        
        if age_minutes > self.max_submission_age_minutes:
            return f"Submission too old: {age_minutes:.1f} minutes (max: {self.max_submission_age_minutes})"
        
        # Check for future timestamps (allow small clock skew)
        if timestamp > now + timedelta(minutes=5):
            return "Submission timestamp is too far in the future"
        
        return None

    def _validate_chain(self, chain: ChainType) -> Optional[str]:
        """Validate chain support"""
        if not isinstance(chain, ChainType):
            return "chain must be a ChainType enum"
        
        # Currently only support BASE chain
        supported_chains = [ChainType.BASE]
        if chain not in supported_chains:
            return f"Unsupported chain: {chain.value} (supported: {[c.value for c in supported_chains]})"
        
        return None

    def _validate_quality(self, prediction: TokenPrediction) -> Optional[str]:
        """Validate submission quality based on whitepaper guidelines"""
        num_addresses = len(prediction.addresses)
        
        # Check for reasonable number of predictions (Netheril guidelines)
        if num_addresses > 25:
            return f"Too many predictions ({num_addresses}), max recommended: 25"
        
        # Check confidence score distribution if provided
        if prediction.confidence_scores:
            confidence_values = list(prediction.confidence_scores.values())
            
            # Check for unreasonably high confidence across all predictions
            high_confidence_count = sum(1 for c in confidence_values if c > 0.9)
            if high_confidence_count == len(confidence_values) and len(confidence_values) > 5:
                return "Suspiciously high confidence across all predictions"
            
            # Check for all identical confidence scores (likely generated)
            if len(set(confidence_values)) == 1 and len(confidence_values) > 3:
                return "All confidence scores are identical (likely invalid)"
        
        return None

    def is_duplicate(self, prediction: TokenPrediction, recent_hashes: Set[str]) -> bool:
        """
        Check if this submission is a duplicate of a recent submission.
        
        Args:
            prediction: TokenPrediction to check
            recent_hashes: Set of recent submission hashes to check against
            
        Returns:
            True if duplicate, False otherwise
        """
        try:
            # Create a hash of the submission content
            submission_hash = self._calculate_submission_hash(prediction)
            
            # Check against provided recent hashes
            if submission_hash in recent_hashes:
                bt.logging.warning(f"Duplicate submission detected: hash {submission_hash[:16]}...")
                return True
            
            # Check against our internal tracking
            if submission_hash in self.recent_submissions:
                bt.logging.warning(f"Duplicate submission detected in internal tracking: hash {submission_hash[:16]}...")
                return True
            
            # Add to our tracking
            self.recent_submissions.add(submission_hash)
            
            return False
            
        except Exception as e:
            bt.logging.error(f"Error checking for duplicate submission: {str(e)}")
            # In case of error, assume not duplicate to be safe
            return False

    def _calculate_submission_hash(self, prediction: TokenPrediction) -> str:
        """Calculate a hash for the submission to detect duplicates"""
        # Sort addresses for consistent hashing
        sorted_addresses = sorted([addr.lower() for addr in prediction.addresses])
        
        # Include key elements in hash
        hash_data = {
            'addresses': sorted_addresses,
            'chain': prediction.chain.value,
            'timestamp_hour': prediction.timestamp.strftime('%Y%m%d%H')  # Group by hour
        }
        
        # Include confidence scores if present
        if prediction.confidence_scores:
            sorted_confidence = sorted(prediction.confidence_scores.items())
            hash_data['confidence'] = sorted_confidence
        
        # Create hash
        hash_string = str(hash_data)
        return hashlib.sha256(hash_string.encode()).hexdigest()

    def _cleanup_old_submissions(self):
        """Clean up old submission hashes to prevent memory growth"""
        current_time = time.time()
        
        if current_time - self.last_cleanup > self.submission_cleanup_interval:
            # For production, you'd want to track timestamps with hashes
            # For now, just clear periodically
            old_size = len(self.recent_submissions)
            
            # Keep recent submissions by limiting size
            if len(self.recent_submissions) > 1000:
                # Convert to list, sort, and keep most recent half
                submission_list = list(self.recent_submissions)
                self.recent_submissions = set(submission_list[-500:])
            
            new_size = len(self.recent_submissions)
            if old_size != new_size:
                bt.logging.debug(f"Cleaned up submission tracking: {old_size} -> {new_size} entries")
            
            self.last_cleanup = current_time

    def get_validation_stats(self) -> dict:
        """Get validation statistics for monitoring"""
        return {
            'recent_submissions_tracked': len(self.recent_submissions),
            'max_submission_age_minutes': self.max_submission_age_minutes,
            'min_confidence': self.min_confidence,
            'max_addresses_per_submission': self.max_addresses_per_submission,
            'last_cleanup': self.last_cleanup
        }