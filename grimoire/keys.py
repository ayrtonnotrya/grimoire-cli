import time
import json
import threading
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from grimoire.config import config
from grimoire.logger import logger

class KeyManager:
    def __init__(self, state_file: Path = Path.home() / ".local/share/grimoire/key_state.json"):
        self.state_file = state_file
        self.lock = threading.Lock()
        self.keys = config.gemini_api_keys
        self.state = self._load_state()
        
        # Runtime tracking (not persisted, but could be if needed)
        self.runtime_locks = {k: threading.Lock() for k in self.keys}
        
        # In-memory sliding windows for rate limiting
        # key -> list of timestamps
        self.request_timestamps: Dict[str, List[float]] = {k: [] for k in self.keys} 
        # key -> list of (timestamp, tokens)
        self.token_timestamps: Dict[str, List[Tuple[float, int]]] = {k: [] for k in self.keys}

    def _load_state(self) -> Dict:
        if not self.state_file.exists():
            return {}
        try:
            with open(self.state_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load key state: {e}")
            return {}

    def _save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save key state: {e}")

    def _cleanup_timestamps(self, key: str, now: float):
        """Removes entries older than 60 seconds."""
        if key in self.request_timestamps:
            self.request_timestamps[key] = [t for t in self.request_timestamps[key] if t > now - 60]
        
        if key in self.token_timestamps:
            self.token_timestamps[key] = [(t, v) for t, v in self.token_timestamps[key] if t > now - 60]

    def get_best_key(self, estimated_tokens: int = 0) -> Optional[str]:
        """Selects the best available key based on usage and rate limits."""
        with self.lock:
            now = time.time()
            
            # Filter out exhausted keys
            available = [k for k in self.keys if k not in config.exhausted_keys]
            if not available:
                return None
            
            best_key = None
            lowest_load_score = float('inf')
            
            rpm_limit = config.rate_limits["rpm"]
            tpm_limit = config.rate_limits["tpm"]

            for key in available:
                # Ensure we have tracking structures
                if key not in self.request_timestamps:
                    self.request_timestamps[key] = []
                if key not in self.token_timestamps:
                    self.token_timestamps[key] = []

                self._cleanup_timestamps(key, now)
                
                current_rpm = len(self.request_timestamps[key])
                current_tpm = sum(t[1] for t in self.token_timestamps[key])
                
                # Calculate load percentage
                rpm_load = current_rpm / rpm_limit if rpm_limit > 0 else 1.0
                tpm_load = current_tpm / tpm_limit if tpm_limit > 0 else 1.0
                
                # If key is already over limit, penalize heavily
                if current_rpm >= rpm_limit or current_tpm + estimated_tokens > tpm_limit:
                    load_score = 1000 + rpm_load + tpm_load # High score = bad
                else:
                    # Score is max of loads (bottleneck principle)
                    load_score = max(rpm_load, tpm_load)
                
                if load_score < lowest_load_score:
                    lowest_load_score = load_score
                    best_key = key
            
            return best_key

    def acquire(self, key: str, estimated_tokens: int = 0):
        """Blocks until the key can be used (Rate Limiting)."""
        rpm_limit = config.rate_limits["rpm"]
        tpm_limit = config.rate_limits["tpm"]
        
        lock = self.runtime_locks.get(key)
        if not lock:
            lock = threading.Lock()
            self.runtime_locks[key] = lock
            
        with lock:
            now = time.time()
            
            # Ensure tracking structures exist
            if key not in self.request_timestamps:
                self.request_timestamps[key] = []
            if key not in self.token_timestamps:
                self.token_timestamps[key] = []

            # 1. RPM Check
            self._cleanup_timestamps(key, now)
            
            if len(self.request_timestamps[key]) >= rpm_limit:
                # Wait until oldest request expires
                if self.request_timestamps[key]:
                    wait_time = 60 - (now - self.request_timestamps[key][0])
                    if wait_time > 0:
                        logger.debug(f"RPM limit for key ...{key[-4:]}. Waiting {wait_time:.2f}s")
                        time.sleep(wait_time)
                        now = time.time()
                        self._cleanup_timestamps(key, now)
            
            # 2. TPM Check
            current_tpm = sum(t[1] for t in self.token_timestamps[key])
            if current_tpm + estimated_tokens > tpm_limit:
                 # Find wait time
                 needed = (current_tpm + estimated_tokens) - tpm_limit
                 freed = 0
                 wait_until = now
                 
                 # Sort by time just in case, though should be sorted
                 sorted_tokens = sorted(self.token_timestamps[key], key=lambda x: x[0])
                 
                 for ts, t in sorted_tokens:
                     freed += t
                     if freed >= needed:
                         wait_until = ts + 60
                         break
                 
                 wait_time = wait_until - now
                 if wait_time > 0:
                     logger.debug(f"TPM limit for key ...{key[-4:]}. Waiting {wait_time:.2f}s")
                     time.sleep(wait_time)
                     now = time.time()
                     self._cleanup_timestamps(key, now)

            # Record this request
            self.request_timestamps[key].append(now)
            self.token_timestamps[key].append((now, estimated_tokens))
            
            # Update persistent state
            if key not in self.state:
                self.state[key] = {"total_tokens": 0, "total_requests": 0, "last_used": 0}
            
            self.state[key]["total_tokens"] += estimated_tokens
            self.state[key]["total_requests"] += 1
            self.state[key]["last_used"] = now
            
            self._save_state()

    def mark_exhausted(self, key: str):
        config.exhausted_keys.add(key)
        logger.warning(f"Key ...{key[-4:]} marked as exhausted.")

key_manager = KeyManager()
