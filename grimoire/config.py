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
    }
}

class Config:
    def __init__(self):
        self._config = self._load_config()

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
    def gemini_api_key(self) -> str:
        return self._config["gemini"]["api_key"]

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
    def CONFIG_FILE(self) -> Path:
        return CONFIG_FILE

    def get(self, section: str, key: str) -> Any:
        return self._config.get(section, {}).get(key)

    def set(self, section: str, key: str, value: Any):
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value

config = Config()
