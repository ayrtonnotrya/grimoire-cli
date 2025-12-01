import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from grimoire.keys import KeyManager
from pathlib import Path
import time
import json

def test_key_manager_initialization(tmp_path):
    state_file = tmp_path / "key_state.json"
    km = KeyManager(state_file=state_file)
    assert km.state == {}

def test_key_manager_persistence(tmp_path):
    state_file = tmp_path / "key_state.json"
    km = KeyManager(state_file=state_file)
    
    # Simulate usage
    with patch("grimoire.config.Config.gemini_api_keys", new_callable=PropertyMock) as mock_keys:
        mock_keys.return_value = ["k1"]
        with patch("grimoire.config.Config.rate_limits", new_callable=PropertyMock) as mock_limits:
            mock_limits.return_value = {"rpm": 10, "tpm": 1000}
            km.acquire("k1", estimated_tokens=100)
    
    # Check if state is updated in memory
    assert "k1" in km.state
    assert km.state["k1"]["total_tokens"] == 100
    
    # Check if file is created
    assert state_file.exists()
    
    # Load new manager
    km2 = KeyManager(state_file=state_file)
    assert "k1" in km2.state
    assert km2.state["k1"]["total_tokens"] == 100

def test_key_manager_get_best_key():
    with patch("grimoire.config.Config.gemini_api_keys", new_callable=PropertyMock) as mock_keys:
        mock_keys.return_value = ["k1", "k2"]
        with patch("grimoire.config.config.exhausted_keys", set()):
            km = KeyManager()
            key = km.get_best_key()
            # k2 is reserved, so only k1 should be returned
            assert key == "k1"

def test_key_manager_exhausted_key():
    with patch("grimoire.config.Config.gemini_api_keys", new_callable=PropertyMock) as mock_keys:
        # Use 3 keys so k3 is reserved, leaving k1 and k2 as candidates
        mock_keys.return_value = ["k1", "k2", "k3"]
        with patch("grimoire.config.config.exhausted_keys", {"k1"}):
            km = KeyManager()
            # k1 is exhausted, k3 is reserved. Should return k2.
            for _ in range(10):
                assert km.get_best_key() == "k2"
