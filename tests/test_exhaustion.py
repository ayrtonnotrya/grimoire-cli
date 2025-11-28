import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from grimoire.core import process_library
from grimoire.config import config

@pytest.fixture
def mock_config(monkeypatch):
    monkeypatch.setattr(config, "_config", {
        "gemini": {"api_key": "TEST_KEY"},
        "paths": {"summaries_dir": "/tmp/summaries", "db_dir": "/tmp/db"},
        "rate_limits": {"rpm": 10, "tpm": 1000}
    })
    return config

@patch("grimoire.core.generate_summary")
@patch("grimoire.core.parse_library_list")
@patch("grimoire.core.check_summary_exists")
def test_process_library_exhaustion_crash(mock_check_exists, mock_parse_list, mock_generate, mock_config, capsys):
    """
    Test that process_library handles 'All API Keys exhausted' gracefully 
    and accesses config.log_file without crashing.
    """
    # Setup
    mock_parse_list.return_value = [Path("/tmp/book1.pdf")]
    mock_check_exists.return_value = False
    
    # Simulate exhaustion response
    mock_generate.return_value = {
        "status": "skipped", 
        "file": "book1.pdf", 
        "message": "All API Keys exhausted (Daily Limit)"
    }

    # Run
    # This should NOT raise AttributeError: 'Config' object has no attribute 'log_file'
    process_library("/tmp/dummy_list.txt", verbose=True)

    # Verify output contains the log file path (which implies property access worked)
    captured = capsys.readouterr()
    assert "CRITICAL: All API Keys have been exhausted" in captured.out
    assert "Please check the logs for details" in captured.out
    
    # Verify log_file property exists and returns a Path
    assert isinstance(config.log_file, Path)
    assert "logs" in str(config.log_file)
