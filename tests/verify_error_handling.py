
import unittest
import sys
from unittest.mock import MagicMock, patch

# Mock google module before importing grimoire.core
mock_google = MagicMock()
mock_genai = MagicMock()
mock_google.genai = mock_genai
sys.modules["google"] = mock_google
sys.modules["google.genai"] = mock_genai
sys.modules["google.genai.types"] = MagicMock()
mock_toml = MagicMock()
mock_toml.load.return_value = {
    "gemini": {"api_key": "key1|key2", "model_name": "gemini-2.5-flash"},
    "rate_limits": {"rpm": 10, "tpm": 100000},
    "paths": {"summaries_dir": "/tmp/summaries", "db_dir": "/tmp/db"}
}
sys.modules["toml"] = mock_toml
sys.modules["rich"] = MagicMock()
sys.modules["rich.progress"] = MagicMock()
sys.modules["rich.panel"] = MagicMock()
sys.modules["rich.console"] = MagicMock()
sys.modules["pydantic"] = MagicMock()
sys.modules["grimoire.schemas"] = MagicMock()

from pathlib import Path
from grimoire.core import handle_api_error, Action, generate_summary
from grimoire.config import config

class TestErrorHandling(unittest.TestCase):
    def setUp(self):
        config.exhausted_keys = set()
        # Patch the property on the class or instance
        self.keys_patcher = patch.object(type(config), 'gemini_api_keys', new_callable=unittest.mock.PropertyMock)
        self.mock_keys = self.keys_patcher.start()
        self.mock_keys.return_value = ["key1", "key2"]

    def tearDown(self):
        self.keys_patcher.stop()

    def test_handle_api_error_403_suspended(self):
        error = Exception("403 PERMISSION_DENIED: Consumer 'api_key:...' has been suspended.")
        action, msg = handle_api_error(error, "key1", "test.pdf")
        self.assertEqual(action, Action.ROTATE_KEY)
        self.assertIn("Permissão negada/Chave suspensa (403)", msg)

    def test_handle_api_error_429_quota(self):
        error = Exception("429 RESOURCE_EXHAUSTED: Quota exceeded")
        action, msg = handle_api_error(error, "key1", "test.pdf")
        self.assertEqual(action, Action.ROTATE_KEY)
        self.assertIn("Cota diária excedida (Quota exceeded)", msg)

    def test_handle_api_error_daily_limit(self):
        error = Exception("Erro desconhecido: Daily Rate Limit Exceeded")
        action, msg = handle_api_error(error, "key1", "test.pdf")
        self.assertEqual(action, Action.ROTATE_KEY)
        self.assertIn("Cota diária excedida (Daily Rate Limit Exceeded)", msg)

    def test_handle_api_error_503_unavailable(self):
        error = Exception("503 UNAVAILABLE: Service unavailable")
        action, msg = handle_api_error(error, "key1", "test.pdf")
        self.assertEqual(action, Action.RETRY)
        self.assertIn("Serviço indisponível (503)", msg)

    def test_handle_api_error_400_invalid_arg(self):
        error = Exception("400 INVALID_ARGUMENT: Bad request")
        action, msg = handle_api_error(error, "key1", "test.pdf")
        self.assertEqual(action, Action.SKIP_FILE)
        self.assertIn("Erro na solicitação (400)", msg)

    @patch("grimoire.core.genai.Client")
    @patch("grimoire.core.time.sleep") # Mock sleep to speed up tests
    @patch("pathlib.Path.mkdir")
    @patch("grimoire.core.key_manager")
    def test_generate_summary_rotation(self, mock_key_manager, mock_mkdir, mock_sleep, MockClient):
        # Mock key_manager to return key1 then key2
        mock_key_manager.get_best_key.side_effect = ["key1", "key2", None]
        mock_key_manager.acquire.return_value = None
        
        # Setup mock client to fail with 403 on first key, succeed on second
        mock_instance_1 = MagicMock()
        mock_instance_1.models.generate_content.side_effect = Exception("403 PERMISSION_DENIED: Suspended")
        mock_instance_1.models.count_tokens.return_value.total_tokens = 100
        
        mock_instance_2 = MagicMock()
        mock_instance_2.models.generate_content.return_value.text = '{"header": {"title": "Test", "authors": [], "category": "", "keywords": []}, "central_thesis": "", "structure_content": [], "key_concepts": [], "practical_system": null, "critical_analysis": {"relevance": "", "target_audience": ""}}'
        mock_instance_2.models.count_tokens.return_value.total_tokens = 100
        
        # Side effect for Client constructor to return different mocks based on key
        def client_side_effect(api_key):
            if api_key == "key1":
                return mock_instance_1
            return mock_instance_2
            
        MockClient.side_effect = client_side_effect

        # Mock file operations
        with patch("builtins.open", unittest.mock.mock_open(read_data=b"pdf_data")) as mock_file:
             # We need to mock path existence for prompt
             with patch("pathlib.Path.exists", return_value=True):
                 # We need to mock stat for fallback token counting
                 with patch("pathlib.Path.stat") as mock_stat:
                     mock_stat.return_value.st_size = 100
                     
                     # Run generate_summary
                     result = generate_summary(Path("test.pdf"), ["key1", "key2"])
                     print(f"DEBUG: Result: {result}")
                     
                     self.assertEqual(result["status"], "success")
                     self.assertIn("key1", config.exhausted_keys)
                     self.assertNotIn("key2", config.exhausted_keys)

if __name__ == "__main__":
    unittest.main()
