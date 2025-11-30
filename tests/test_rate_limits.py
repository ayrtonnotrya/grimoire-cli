import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path
from grimoire.core import generate_summary, Action
from grimoire.config import config
import time

# Define Mock Exceptions
class MockClientError(Exception):
    def __init__(self, message, code=None):
        self.message = message
        self.code = code
    def __str__(self):
        return self.message

@pytest.fixture
def mock_config(monkeypatch):
    monkeypatch.setattr(config, "_config", {
        "gemini": {"api_key": "TEST_KEY"},
        "paths": {"summaries_dir": "/tmp/summaries"},
        "rate_limits": {"rpm": 10, "tpm": 1000}
    })
    return config

@patch("grimoire.core.genai.Client")
@patch("grimoire.core.types")
@patch("grimoire.core.time.sleep")
@patch("pathlib.Path.mkdir")
def test_rate_limit_exponential_backoff(mock_mkdir, mock_sleep, mock_types, mock_client_cls, mock_config):
    """
    Test that generate_summary retries with exponential backoff when encountering
    a 429 Rate Limit error (not Quota Exhaustion).
    """
    # Setup Mock Client
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    # Simulate 429 Rate Limit Error (RPM/TPM)
    # It should fail 2 times and succeed on the 3rd
    rate_limit_error = MockClientError(
        message="429 ResourceExhausted: Rate limit exceeded for metric: RequestsPerMinute...",
        code=429
    )
    
    # Mock success response
    mock_response = MagicMock()
    mock_response.text = '{"header": {"title": "Test"}, "central_thesis": "Test"}'
    
    # Mock count_tokens to avoid fallback to stat()
    mock_client.models.count_tokens.return_value = MagicMock(total_tokens=100)

    # Side effect: Error -> Error -> Success
    mock_client.models.generate_content.side_effect = [
        rate_limit_error,
        rate_limit_error,
        mock_response
    ]

    # Mock file operations
    mock_file = MagicMock()
    mock_file.__enter__.return_value.read.return_value = b"fake pdf content"
    
    with patch("builtins.open", return_value=mock_file), \
         patch("pathlib.Path.exists", return_value=True), \
         patch("pathlib.Path.stat", return_value=MagicMock(st_size=1000)), \
         patch("grimoire.core.BookSummary", MagicMock()), \
         patch("json.loads", return_value={"header": {"title": "Test"}}):
         
        # Run
        result = generate_summary(Path("test.pdf"), ["TEST_KEY"])
    

    # Verify Success
    assert result["status"] == "success"
    
    # Verify Retries
    # generate_content called 3 times (2 fails + 1 success)
    assert mock_client.models.generate_content.call_count == 3
    
    # Verify Exponential Backoff
    # Expected sleeps: 10s (retry 1), 20s (retry 2)
    assert mock_sleep.call_count == 2
    mock_sleep.assert_has_calls([call(10), call(20)])
    
    print("\nExponential backoff verified: slept for 2s then 4s.")
