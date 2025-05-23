import os
import json
import asyncio
import bittensor as bt
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from base.types import ChainType, TokenPrediction

class RoundState(Enum):
    REGISTRATION = "registration"
    SUBMISSION = "submission" 
    EVALUATION = "evaluation"
    RESULTS = "results"
    COMPLETED = "completed"

@dataclass
class CompetitionRound:
    round_id: str
    chain: ChainType
    state: RoundState
    start_time: datetime
    registration_deadline: datetime
    submission_deadline: datetime
    evaluation_deadline: datetime
    results_deadline: datetime
    participants: List[int]  # miner UIDs
    submissions: Dict[int, Dict[str, Any]]  # uid -> submission data
    results: Dict[int, float]  # uid -> performance score
    winner_uid: Optional[int] = None
    dataset_version: Optional[str] = None

class CompetitionRoundManager:
    """
    Manages the complete lifecycle of competition rounds according to the whitepaper.
    Handles registration, submission, evaluation, and results phases automatically.
    """
    
    def __init__(self, storage_path: str = "data/competition_rounds", round_duration_hours: int = 6):
        self.storage_path = storage_path
        self.round_duration_hours = round_duration_hours
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Load existing rounds
        self.current_rounds: Dict[str, CompetitionRound] = {}
        self.completed_rounds: List[CompetitionRound] = []
        self._load_state()
        
        # Round timing configuration (in minutes)
        self.phase_durations = {
            "registration": 30,      # 30 minutes for miners to register
            "submission": 120,       # 2 hours for model submissions
            "evaluation": 180,       # 3 hours for evaluation
            "results": 30           # 30 minutes for results processing
        }
        
        bt.logging.info(f"CompetitionRoundManager initialized with {len(self.current_rounds)} active rounds")

    def _load_state(self):
        """Load competition state from persistent storage"""
        try:
            # Load current rounds
            current_file = os.path.join(self.storage_path, "current_rounds.json")
            if os.path.exists(current_file):
                with open(current_file, 'r') as f:
                    data = json.load(f)
                    for round_id, round_data in data.items():
                        self.current_rounds[round_id] = self._deserialize_round(round_data)
                        
            # Load completed rounds (last 100)
            completed_file = os.path.join(self.storage_path, "completed_rounds.json")
            if os.path.exists(completed_file):
                with open(completed_file, 'r') as f:
                    data = json.load(f)
                    self.completed_rounds = [self._deserialize_round(r) for r in data[-100:]]
                    
            bt.logging.info(f"Loaded {len(self.current_rounds)} current and {len(self.completed_rounds)} completed rounds")
            
        except Exception as e:
            bt.logging.error(f"Failed to load competition state: {e}")
            self.current_rounds = {}
            self.completed_rounds = []

    def _save_state(self):
        """Save competition state to persistent storage"""
        try:
            # Save current rounds
            current_file = os.path.join(self.storage_path, "current_rounds.json")
            with open(current_file, 'w') as f:
                serialized = {rid: self._serialize_round(r) for rid, r in self.current_rounds.items()}
                json.dump(serialized, f, indent=2)
                
            # Save completed rounds (keep last 100)
            completed_file = os.path.join(self.storage_path, "completed_rounds.json")
            with open(completed_file, 'w') as f:
                serialized = [self._serialize_round(r) for r in self.completed_rounds[-100:]]
                json.dump(serialized, f, indent=2)
                
            bt.logging.debug("Competition state saved successfully")
            
        except Exception as e:
            bt.logging.error(f"Failed to save competition state: {e}")

    def _serialize_round(self, round_obj: CompetitionRound) -> Dict[str, Any]:
        """Convert CompetitionRound to JSON-serializable dict"""
        data = asdict(round_obj)
        # Convert datetime objects to ISO strings
        for field in ['start_time', 'registration_deadline', 'submission_deadline', 
                     'evaluation_deadline', 'results_deadline']:
            if data[field]:
                data[field] = data[field].isoformat()
        # Convert enums to strings
        data['state'] = data['state'].value
        data['chain'] = data['chain'].value
        return data

    def _deserialize_round(self, data: Dict[str, Any]) -> CompetitionRound:
        """Convert JSON dict back to CompetitionRound object"""
        # Convert ISO strings back to datetime objects
        for field in ['start_time', 'registration_deadline', 'submission_deadline',
                     'evaluation_deadline', 'results_deadline']:
            if data[field]:
                data[field] = datetime.fromisoformat(data[field])
        # Convert strings back to enums
        data['state'] = RoundState(data['state'])
        data['chain'] = ChainType(data['chain'])
        return CompetitionRound(**data)

    def start_new_round(self, chain: ChainType, dataset_version: str) -> str:
        """
        Start a new competition round for the specified chain.
        Returns the round_id of the created round.
        """
        now = datetime.utcnow()
        round_id = f"{chain.value}_{now.strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate phase deadlines
        registration_deadline = now + timedelta(minutes=self.phase_durations["registration"])
        submission_deadline = registration_deadline + timedelta(minutes=self.phase_durations["submission"])
        evaluation_deadline = submission_deadline + timedelta(minutes=self.phase_durations["evaluation"])
        results_deadline = evaluation_deadline + timedelta(minutes=self.phase_durations["results"])
        
        # Create new round
        new_round = CompetitionRound(
            round_id=round_id,
            chain=chain,
            state=RoundState.REGISTRATION,
            start_time=now,
            registration_deadline=registration_deadline,
            submission_deadline=submission_deadline,
            evaluation_deadline=evaluation_deadline,
            results_deadline=results_deadline,
            participants=[],
            submissions={},
            results={},
            dataset_version=dataset_version
        )
        
        self.current_rounds[round_id] = new_round
        self._save_state()
        
        bt.logging.info(f"Started new competition round {round_id} for {chain.value}")
        bt.logging.info(f"Registration deadline: {registration_deadline}")
        bt.logging.info(f"Submission deadline: {submission_deadline}")
        bt.logging.info(f"Evaluation deadline: {evaluation_deadline}")
        
        return round_id

    def register_participant(self, round_id: str, miner_uid: int) -> bool:
        """
        Register a miner for participation in a competition round.
        Returns True if registration successful, False otherwise.
        """
        if round_id not in self.current_rounds:
            bt.logging.warning(f"Cannot register for unknown round {round_id}")
            return False
            
        round_obj = self.current_rounds[round_id]
        
        # Check if registration is still open
        if round_obj.state != RoundState.REGISTRATION:
            bt.logging.warning(f"Registration closed for round {round_id}, current state: {round_obj.state}")
            return False
            
        # Check if deadline passed
        if datetime.utcnow() > round_obj.registration_deadline:
            bt.logging.warning(f"Registration deadline passed for round {round_id}")
            self._advance_round_state(round_id)
            return False
            
        # Check if already registered
        if miner_uid in round_obj.participants:
            bt.logging.debug(f"Miner {miner_uid} already registered for round {round_id}")
            return True
            
        # Register the miner
        round_obj.participants.append(miner_uid)
        self._save_state()
        
        bt.logging.info(f"Registered miner {miner_uid} for competition round {round_id}")
        return True

    def submit_model(self, round_id: str, miner_uid: int, model_data: Dict[str, Any]) -> bool:
        """
        Submit a model for evaluation in a competition round.
        Returns True if submission successful, False otherwise.
        """
        if round_id not in self.current_rounds:
            bt.logging.warning(f"Cannot submit to unknown round {round_id}")
            return False
            
        round_obj = self.current_rounds[round_id]
        
        # Check if submissions are open
        if round_obj.state != RoundState.SUBMISSION:
            bt.logging.warning(f"Submissions not open for round {round_id}, current state: {round_obj.state}")
            return False
            
        # Check if deadline passed
        if datetime.utcnow() > round_obj.submission_deadline:
            bt.logging.warning(f"Submission deadline passed for round {round_id}")
            self._advance_round_state(round_id)
            return False
            
        # Check if miner is registered
        if miner_uid not in round_obj.participants:
            bt.logging.warning(f"Miner {miner_uid} not registered for round {round_id}")
            return False
            
        # Validate submission data
        if not self._validate_model_submission(model_data):
            bt.logging.warning(f"Invalid model submission from miner {miner_uid}")
            return False
            
        # Store the submission
        round_obj.submissions[miner_uid] = {
            "submission_time": datetime.utcnow().isoformat(),
            "model_data": model_data
        }
        self._save_state()
        
        bt.logging.info(f"Received model submission from miner {miner_uid} for round {round_id}")
        return True

    def _validate_model_submission(self, model_data: Dict[str, Any]) -> bool:
        """Validate model submission format and content"""
        required_fields = ["addresses", "confidence_scores", "timestamp"]
        
        for field in required_fields:
            if field not in model_data:
                bt.logging.warning(f"Missing required field '{field}' in model submission")
                return False
                
        # Validate addresses format
        addresses = model_data.get("addresses", [])
        if not isinstance(addresses, list) or len(addresses) == 0:
            bt.logging.warning("Invalid addresses format in submission")
            return False
            
        # Validate confidence scores
        confidence_scores = model_data.get("confidence_scores", {})
        if not isinstance(confidence_scores, dict):
            bt.logging.warning("Invalid confidence_scores format in submission")
            return False
            
        return True

    def get_round_for_submission(self, chain: ChainType) -> Optional[str]:
        """
        Get the current round ID that accepts submissions for the given chain.
        Returns None if no round is accepting submissions.
        """
        now = datetime.utcnow()
        
        for round_id, round_obj in self.current_rounds.items():
            if (round_obj.chain == chain and 
                round_obj.state == RoundState.SUBMISSION and 
                now <= round_obj.submission_deadline):
                return round_id
                
        return None

    def get_rounds_for_evaluation(self) -> List[str]:
        """
        Get round IDs that are ready for evaluation.
        """
        ready_rounds = []
        now = datetime.utcnow()
        
        for round_id, round_obj in self.current_rounds.items():
            if (round_obj.state == RoundState.EVALUATION or
                (round_obj.state == RoundState.SUBMISSION and now > round_obj.submission_deadline)):
                ready_rounds.append(round_id)
                
        return ready_rounds

    def store_evaluation_results(self, round_id: str, results: Dict[int, float]) -> bool:
        """
        Store evaluation results for a competition round.
        """
        if round_id not in self.current_rounds:
            bt.logging.warning(f"Cannot store results for unknown round {round_id}")
            return False
            
        round_obj = self.current_rounds[round_id]
        round_obj.results = results
        
        # Determine winner (highest score)
        if results:
            winner_uid = max(results, key=results.get)
            round_obj.winner_uid = winner_uid
            bt.logging.info(f"Round {round_id} winner: miner {winner_uid} with score {results[winner_uid]:.4f}")
        
        self._save_state()
        return True

    def complete_round(self, round_id: str) -> bool:
        """
        Mark a round as completed and move it to completed rounds.
        """
        if round_id not in self.current_rounds:
            return False
            
        round_obj = self.current_rounds[round_id]
        round_obj.state = RoundState.COMPLETED
        
        # Move to completed rounds
        self.completed_rounds.append(round_obj)
        del self.current_rounds[round_id]
        
        self._save_state()
        bt.logging.info(f"Competition round {round_id} completed")
        return True

    def _advance_round_state(self, round_id: str):
        """Advance a round to the next state based on current time"""
        if round_id not in self.current_rounds:
            return
            
        round_obj = self.current_rounds[round_id]
        now = datetime.utcnow()
        
        if round_obj.state == RoundState.REGISTRATION and now > round_obj.registration_deadline:
            round_obj.state = RoundState.SUBMISSION
            bt.logging.info(f"Round {round_id} advanced to SUBMISSION phase")
            
        elif round_obj.state == RoundState.SUBMISSION and now > round_obj.submission_deadline:
            round_obj.state = RoundState.EVALUATION
            bt.logging.info(f"Round {round_id} advanced to EVALUATION phase")
            
        elif round_obj.state == RoundState.EVALUATION and now > round_obj.evaluation_deadline:
            round_obj.state = RoundState.RESULTS
            bt.logging.info(f"Round {round_id} advanced to RESULTS phase")
            
        elif round_obj.state == RoundState.RESULTS and now > round_obj.results_deadline:
            self.complete_round(round_id)
            
        self._save_state()

    def update_round_states(self):
        """Update all round states based on current time - call this periodically"""
        for round_id in list(self.current_rounds.keys()):
            self._advance_round_state(round_id)

    def get_round_info(self, round_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific round"""
        if round_id in self.current_rounds:
            return self._serialize_round(self.current_rounds[round_id])
        return None

    def get_active_rounds(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active rounds"""
        return {rid: self._serialize_round(r) for rid, r in self.current_rounds.items()}

    def should_start_new_round(self, chain: ChainType) -> bool:
        """
        Check if a new round should be started for the given chain.
        Returns True if no recent round exists for this chain.
        """
        now = datetime.utcnow()
        cutoff_time = now - timedelta(hours=self.round_duration_hours)
        
        # Check if there's already an active round for this chain
        for round_obj in self.current_rounds.values():
            if round_obj.chain == chain:
                return False  # Already have an active round
                
        # Check if we recently completed a round for this chain
        for round_obj in self.completed_rounds[-10:]:  # Check last 10 completed rounds
            if round_obj.chain == chain and round_obj.start_time > cutoff_time:
                return False  # Recently completed a round
                
        return True  # No recent round, should start new one

    def get_round_dataset(self, round_id: str) -> Optional[str]:
        """Get the dataset version used for a specific round"""
        if round_id in self.current_rounds:
            return self.current_rounds[round_id].dataset_version
        
        # Check completed rounds
        for round_obj in self.completed_rounds:
            if round_obj.round_id == round_id:
                return round_obj.dataset_version
        
        return None

    def get_rounds_using_dataset(self, dataset_version: str) -> List[str]:
        """Get all round IDs that use a specific dataset version"""
        round_ids = []
        
        # Check current rounds
        for round_id, round_obj in self.current_rounds.items():
            if round_obj.dataset_version == dataset_version:
                round_ids.append(round_id)
        
        # Check recent completed rounds
        for round_obj in self.completed_rounds[-20:]:  # Last 20 rounds
            if round_obj.dataset_version == dataset_version:
                round_ids.append(round_obj.round_id)
        
        return round_ids