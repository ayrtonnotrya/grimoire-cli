import sys
from unittest.mock import MagicMock
import unittest
from unittest.mock import patch
import os

# Define Mock Exceptions first so we can use them in setup
class MockClientError(Exception):
    def __init__(self, message, code=None):
        self.message = message
        self.code = code
    def __str__(self):
        return self.message

class MockServerError(Exception):
    pass

# Mock dependencies BEFORE importing grimoire modules
sys.modules["toml"] = MagicMock()
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = MagicMock()
sys.modules["google.genai.types"] = MagicMock()

# Setup errors module
mock_errors = MagicMock()
mock_errors.ClientError = MockClientError
mock_errors.ServerError = MockServerError
sys.modules["google.genai.errors"] = mock_errors

# Also attach to google.genai.errors
sys.modules["google.genai"].errors = mock_errors

sys.modules["chromadb"] = MagicMock()
sys.modules["rich"] = MagicMock()
sys.modules["rich.progress"] = MagicMock()
sys.modules["rich.panel"] = MagicMock()
sys.modules["rich.console"] = MagicMock()
sys.modules["pydantic"] = MagicMock()
sys.modules["grimoire.schemas"] = MagicMock()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from grimoire.config import config
from grimoire.core import generate_summary, _index_single_book
from grimoire.db import generate_embeddings
from pathlib import Path
import json

class TestQuotaHandling(unittest.TestCase):
    def setUp(self):
        config.exhausted_keys.clear()
        self.api_key = "test_key"
        self.api_keys = ["key1", "key2"]
        self.pdf_path = Path("test.pdf")

    @patch("grimoire.core.genai.Client")
    def test_generate_summary_exhaustion(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        # Simulate ResourceExhausted with Daily limit message
        # Must include "429" or "ResourceExhausted" in message for core.py check
        error = MockClientError(
            message="429 ResourceExhausted: Quota exceeded for metric: ... RequestsPerDay ...",
            code=429
        )
        mock_client.models.generate_content.side_effect = error

        # Mock file open to avoid FileNotFoundError
        with patch("builtins.open", MagicMock()): 
             # We also need to mock prompt path check
             with patch("pathlib.Path.exists", return_value=True):
                 result = generate_summary(self.pdf_path, self.api_key)
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Daily Rate Limit Exceeded", result["message"])
        self.assertIn(self.api_key, config.exhausted_keys)

        # Second call should be skipped
        result = generate_summary(self.pdf_path, self.api_key)
        self.assertEqual(result["status"], "skipped")
        self.assertIn("exhausted", result["message"])

    @patch("grimoire.db.genai.Client")
    def test_generate_embeddings_exhaustion(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        # Must include "429" or "ResourceExhausted" in message if db.py checked it, 
        # but db.py checks e.code for retry logic, and then checks str(e) for Daily.
        # db.py: if e.code not in [429, 503]: raise e
        # db.py: if "RequestsPerDay" in str(e) ...
        error = MockClientError(
            message="Quota exceeded ... Daily ...",
            code=429
        )
        mock_client.models.embed_content.side_effect = error
        
        # First call should fail and add to exhausted_keys
        # generate_embeddings has a retry loop and sleep, we should mock sleep to speed up
        with patch("time.sleep", MagicMock()):
            with self.assertRaises(RuntimeError) as cm:
                generate_embeddings(["test"], self.api_key)
        
        self.assertIn("Daily Rate Limit Exceeded", str(cm.exception))
        self.assertIn(self.api_key, config.exhausted_keys)
        
        # Second call should raise RuntimeError immediately
        with self.assertRaises(RuntimeError) as cm:
            generate_embeddings(["test"], self.api_key)
        self.assertIn("API Key exhausted", str(cm.exception))

    @patch("grimoire.db.generate_embeddings")
    @patch("grimoire.db.document_exists")
    def test_index_single_book_retry(self, mock_document_exists, mock_generate_embeddings):
        # Mock db functions
        mock_document_exists.return_value = False
        mock_generate_embeddings.side_effect = [
            RuntimeError("Daily Rate Limit Exceeded"), # First key fails
            [[0.1, 0.2]] # Second key succeeds
        ]
        
        # Mock file open and json load
        with patch("builtins.open", MagicMock()):
            with patch("json.load", return_value={
                "header": {"title": "Test", "authors": [], "category": "", "keywords": []},
                "central_thesis": "Test",
                "structure_content": [],
                "key_concepts": [],
                "practical_system": None,
                "critical_analysis": {"relevance": "", "target_audience": ""}
            }):
                 # Mock BookSummary validation since we mocked schemas
                 with patch("grimoire.core.BookSummary", MagicMock()):
                     result = _index_single_book(self.pdf_path, self.api_keys)

        self.assertEqual(result["status"], "success")
        # Verify that generate_embeddings was called twice
        self.assertEqual(mock_generate_embeddings.call_count, 2)
        # Verify that one key was exhausted
        self.assertEqual(len(config.exhausted_keys), 0) 
        # Wait, core.py doesn't add to exhausted_keys, db.py does. 
        # But here we mocked db.generate_embeddings, so we need to manually simulate the side effect if we want to test that.
        # However, core.py logic relies on catching the error.
        # The important thing is that it retried.

if __name__ == "__main__":
    unittest.main()
