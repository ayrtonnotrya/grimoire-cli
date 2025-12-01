import os
import toml
from pathlib import Path
from typing import Optional, Dict, Any

APP_NAME = "grimoire"
CONFIG_DIR = Path.home() / ".config" / APP_NAME
CONFIG_FILE = CONFIG_DIR / "config.toml"

DEFAULT_CONFIG = {
    "gemini": {
        "api_key": "",
        "model_name": "gemini-2.5-flash"
    },
    "paths": {
        "library_root": "",
        "summaries_dir": "./summaries",
        "db_dir": "~/.local/share/grimoire/chroma_db"
    },
    # Default fallback if model not found
    "rate_limits": {
        "rpm": 10,
        "tpm": 250000
    }
}

# Free Tier Rate Limits (as of Nov 2025)
MODEL_RATE_LIMITS = {
    # Text-out models
    "gemini-2.5-pro": {"rpm": 2, "tpm": 125000},
    "gemini-2.5-flash": {"rpm": 10, "tpm": 250000},
    "gemini-2.5-flash-preview": {"rpm": 10, "tpm": 250000},
    "gemini-2.5-flash-lite": {"rpm": 15, "tpm": 250000},
    "gemini-2.5-flash-lite-preview": {"rpm": 15, "tpm": 250000},
    "gemini-2.0-flash": {"rpm": 15, "tpm": 1000000},
    "gemini-2.0-flash-lite": {"rpm": 30, "tpm": 1000000},
    
    # Live API (TPM only, RPM is *)
    "gemini-2.5-flash-live": {"rpm": 100, "tpm": 1000000}, # Heuristic RPM
    "gemini-2.5-flash-preview-native-audio": {"rpm": 100, "tpm": 500000}, # Heuristic RPM
    "gemini-2.0-flash-live": {"rpm": 100, "tpm": 1000000}, # Heuristic RPM
    
    # Multi-modal generation models
    "gemini-2.5-flash-preview-tts": {"rpm": 3, "tpm": 10000},
    "gemini-2.0-flash-preview-image-generation": {"rpm": 10, "tpm": 200000},
    
    # Other models
    "gemma-3": {"rpm": 30, "tpm": 15000},
    "gemma-3n": {"rpm": 30, "tpm": 15000},
    "gemini-embedding": {"rpm": 100, "tpm": 30000},
    "gemini-robotics-er-1.5-preview": {"rpm": 10, "tpm": 250000},
    
    # Deprecated models
    "gemini-1.5-flash": {"rpm": 15, "tpm": 250000},
    "gemini-1.5-flash-8b": {"rpm": 15, "tpm": 250000},
}

# Sigil Artificer Models
MODEL_PLANNING = "gemini-2.5-flash-lite"
MODEL_ALCHEMY = "gemini-2.5-pro"
MODEL_MATERIALIZATION = "imagen-4.0-ultra-generate-001"

class Config:
    def __init__(self):
        self._config = self._load_config()
        self.exhausted_keys = set()

    def _load_config(self) -> Dict[str, Any]:
        if not CONFIG_FILE.exists():
            return DEFAULT_CONFIG.copy()
        try:
            return toml.load(CONFIG_FILE)
        except Exception:
            return DEFAULT_CONFIG.copy()

    def save(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            toml.dump(self._config, f)

    @property
    def gemini_api_keys(self) -> list[str]:
        keys = self._config["gemini"]["api_key"]
        if not keys:
            return []
        return [k.strip() for k in keys.split("|") if k.strip()]

    @property
    def gemini_api_key(self) -> str:
        # Return the first key for backward compatibility or default usage
        keys = self.gemini_api_keys
        return keys[0] if keys else ""

    @gemini_api_key.setter
    def gemini_api_key(self, value: str):
        self._config["gemini"]["api_key"] = value

    @property
    def summaries_dir(self) -> Path:
        return Path(self._config["paths"]["summaries_dir"]).expanduser().resolve()

    @summaries_dir.setter
    def summaries_dir(self, value: str):
        self._config["paths"]["summaries_dir"] = str(Path(value).expanduser().resolve())

    @property
    def db_dir(self) -> Path:
        return Path(self._config["paths"]["db_dir"]).expanduser().resolve()
    
    @property
    def model_name(self) -> str:
        return self._config["gemini"].get("model_name", "gemini-2.5-flash")

    @property
    def rate_limits(self) -> Dict[str, int]:
        # 1. Check for user override in config file
        if "rate_limits" in self._config:
            return self._config["rate_limits"]
        
        # 2. Check for model-specific defaults
        model = self.model_name
        # Handle versioned models (e.g., gemini-2.5-flash-001 -> gemini-2.5-flash)
        # This is a simple heuristic, might need refinement
        base_model = model.split("-0")[0] 
        
        if model in MODEL_RATE_LIMITS:
            return MODEL_RATE_LIMITS[model]
        if base_model in MODEL_RATE_LIMITS:
             return MODEL_RATE_LIMITS[base_model]

        # 3. Fallback to safe default
        return DEFAULT_CONFIG["rate_limits"]

    @property
    def CONFIG_FILE(self) -> Path:
        return CONFIG_FILE

    @property
    def log_file(self) -> Path:
        from datetime import datetime
        log_dir = Path.home() / ".local" / "share" / "grimoire" / "logs"
        current_date = datetime.now().strftime("%d-%m-%Y")
        return log_dir / f"{current_date}.log"

    def get(self, section: str, key: str) -> Any:
        return self._config.get(section, {}).get(key)

    def set(self, section: str, key: str, value: Any):
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value

config = Config()
