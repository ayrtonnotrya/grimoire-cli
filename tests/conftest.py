import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from grimoire.config import config
from grimoire.keys import KeyManager

@pytest.fixture
def mock_config():
    with patch("grimoire.config.config") as mock:
        mock.gemini_api_keys = ["fake_key_1", "fake_key_2"]
        mock.model_name = "gemini-2.5-flash"
        mock.rate_limits = {"rpm": 100, "tpm": 100000}
        mock.summaries_dir = Path("/tmp/summaries")
        mock.db_dir = Path("/tmp/db")
        mock.exhausted_keys = set()
        yield mock

@pytest.fixture
def mock_key_manager():
    with patch("grimoire.keys.key_manager") as mock:
        mock.get_best_key.return_value = "fake_key_1"
        mock.acquire.return_value = None
        yield mock

@pytest.fixture
def mock_genai_client():
    with patch("google.genai.Client") as mock:
        client_instance = MagicMock()
        mock.return_value = client_instance
        yield client_instance

@pytest.fixture
def mock_chroma_client():
    with patch("chromadb.PersistentClient") as mock:
        client_instance = MagicMock()
        mock.return_value = client_instance
        yield client_instance
