import pytest
from unittest.mock import MagicMock, patch
from grimoire.core import handle_api_error, Action, generate_summary
from grimoire.config import config
from google.genai import errors

class MockResponse:
    def __init__(self, text="", json_data=None):
        self.text = text
        self._json = json_data or {}
    
    def json(self):
        return self._json

def test_handle_api_error_quota_exceeded():
    """Test that explicit Quota Exceeded errors trigger key rotation."""
    error = Exception("429 Quota exceeded for some resource")
    action, msg = handle_api_error(error, "test_key", "test.pdf")
    assert action == Action.ROTATE_KEY
    assert "Cota diária excedida" in msg

def test_handle_api_error_daily_limit():
    """Test that Daily Limit errors trigger key rotation."""
    error = Exception("Daily Rate Limit Exceeded")
    action, msg = handle_api_error(error, "test_key", "test.pdf")
    assert action == Action.ROTATE_KEY
    assert "Cota diária excedida" in msg

def test_handle_api_error_transient_429():
    """Test that generic 429 errors trigger RETRY, not rotation."""
    error = Exception("429 Too Many Requests")
    action, msg = handle_api_error(error, "test_key", "test.pdf")
    assert action == Action.RETRY
    assert "Limite de taxa (429)" in msg

@patch("grimoire.core.genai.Client")
@patch("grimoire.core.key_manager")
@patch("time.sleep") # Mock sleep to speed up tests
def test_generate_summary_retries_on_429(mock_sleep, mock_key_manager, mock_client_cls):
    """Test that generate_summary retries on 429 and eventually fails without killing the key."""
    
    # Setup
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    # Mock generate_content to raise 429
    error_429 = Exception("429 Too Many Requests")
    mock_client.models.generate_content.side_effect = error_429
    
    # Mock key manager
    mock_key_manager.get_best_key.return_value = "test_key"
    
    # Run
    pdf_path = MagicMock()
    pdf_path.name = "test.pdf"
    pdf_path.stat.return_value.st_size = 1000
    
    # Ensure we have a key
    api_keys = ["test_key"]
    config.exhausted_keys = set()
    
    # We need to mock open() as well since generate_summary reads the file
    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = b"fake pdf content"
        
        result = generate_summary(pdf_path, api_keys)
    
    # Assertions
    assert result["status"] == "error"
    assert "Rate limit persistent" in result["message"]
    
    # Should have retried 3 times (initial + 3 retries = 4 calls)
    # Actually logic is: try -> catch -> retry loop (3 times)
    # So generate_content is called 1 (initial) + 3 (retries) = 4 times
    assert mock_client.models.generate_content.call_count == 4
    
    # Key should NOT be exhausted
    assert "test_key" not in config.exhausted_keys
    assert mock_key_manager.mark_exhausted.call_count == 0

@patch("grimoire.core.genai.Client")
@patch("grimoire.core.key_manager")
@patch("time.sleep")
def test_generate_summary_rotates_on_quota(mock_sleep, mock_key_manager, mock_client_cls):
    """Test that generate_summary rotates key on Quota Exceeded."""
    
    # Setup
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    
    # Mock generate_content to raise Quota Exceeded
    error_quota = Exception("429 Quota exceeded")
    mock_client.models.generate_content.side_effect = error_quota
    
    # Mock key manager
    mock_key_manager.get_best_key.side_effect = ["key1", "key2", None] # Return key1, then key2, then None
    
    # Run
    pdf_path = MagicMock()
    pdf_path.name = "test.pdf"
    pdf_path.stat.return_value.st_size = 1000
    
    api_keys = ["key1", "key2"]
    config.exhausted_keys = set()
    
    with patch("builtins.open", create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = b"fake pdf content"
        
        result = generate_summary(pdf_path, api_keys)
    
    # Should have tried key1, failed with quota, marked exhausted, tried key2, failed...
    # Actually, if key1 fails with Quota, it rotates immediately.
    # If key2 also fails, it rotates.
    # Eventually runs out of keys.
    
    assert "key1" in config.exhausted_keys
    assert "key2" in config.exhausted_keys
    assert result["status"] == "skipped" # All keys exhausted
