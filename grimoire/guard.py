import json
import time
from pathlib import Path
from datetime import datetime, date
from typing import Optional

class ImagenGuardError(Exception):
    """Base exception for ImagenGuard errors."""
    pass

class ImagenGuard:
    """
    Guardian of the Paid Satellite Project Key.
    Enforces strict rate limits and budget controls.
    """
    
    # Constants
    DATA_DIR = Path.home() / ".local" / "share" / "grimoire"
    KEY_FILE = DATA_DIR / "imagen_key.secret"
    USAGE_FILE = DATA_DIR / "imagen_usage.json"
    
    MAX_DAILY_REQUESTS = 100
    MIN_INTERVAL_SECONDS = 60

    def __init__(self):
        self._ensure_data_dir()

    def _ensure_data_dir(self):
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _load_usage(self) -> dict:
        if not self.USAGE_FILE.exists():
            return {"last_request_time": 0, "daily_count": 0, "date": str(date.today())}
        
        try:
            with open(self.USAGE_FILE, 'r') as f:
                data = json.load(f)
                # Reset if new day
                if data.get("date") != str(date.today()):
                    return {"last_request_time": data.get("last_request_time", 0), "daily_count": 0, "date": str(date.today())}
                return data
        except json.JSONDecodeError:
             return {"last_request_time": 0, "daily_count": 0, "date": str(date.today())}

    def _save_usage(self, usage: dict):
        with open(self.USAGE_FILE, 'w') as f:
            json.dump(usage, f)

    def get_key(self) -> str:
        """
        Retrieves the paid API key if all safety checks pass.
        """
        # 1. Check if key exists
        if not self.KEY_FILE.exists():
            raise ImagenGuardError(f"Imagen Key not found at {self.KEY_FILE}. Run 'grimoire set-imagen-key <KEY>' first.")

        # 2. Load Usage State
        usage = self._load_usage()
        current_time = time.time()
        
        # 3. Rate Limit Check (1 req/min)
        time_since_last = current_time - usage["last_request_time"]
        if time_since_last < self.MIN_INTERVAL_SECONDS:
            raise ImagenGuardError(f"Rate Limit Exceeded: Please wait {int(self.MIN_INTERVAL_SECONDS - time_since_last)} seconds.")

        # 4. Daily Budget Check (100 req/day)
        if usage["daily_count"] >= self.MAX_DAILY_REQUESTS:
            raise ImagenGuardError("Daily Budget Exceeded: Maximum 100 images per day reached.")

        # 5. Retrieve Key
        try:
            with open(self.KEY_FILE, 'r') as f:
                key = f.read().strip()
        except Exception as e:
            raise ImagenGuardError(f"Failed to read key file: {e}")

        if not key:
             raise ImagenGuardError("Imagen Key file is empty.")

        # 6. Update State (Commit usage)
        usage["last_request_time"] = current_time
        usage["daily_count"] += 1
        self._save_usage(usage)

        return key

    @classmethod
    def save_key(cls, key: str):
        """Saves the key to the secure location."""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(cls.KEY_FILE, 'w') as f:
            f.write(key.strip())
        # Set permissions to read/write only by user (600)
        cls.KEY_FILE.chmod(0o600)
