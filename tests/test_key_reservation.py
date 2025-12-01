import unittest
from unittest.mock import MagicMock, patch
from grimoire.keys import KeyManager
from grimoire.config import config

class TestKeyReservation(unittest.TestCase):
    def setUp(self):
        self.key_manager = KeyManager()
        # Mock config to have multiple keys
        self.original_keys = config.gemini_api_keys
        self.original_exhausted = config.exhausted_keys
        
        config._config["gemini"]["api_key"] = "key1|key2|key3"
        config.exhausted_keys = set()
        self.key_manager.keys = ["key1", "key2", "key3"]

    def tearDown(self):
        config._config["gemini"]["api_key"] = "|".join(self.original_keys)
        config.exhausted_keys = self.original_exhausted

    def test_get_best_key_reserved_false(self):
        # Should return one of the first two keys, NEVER key3
        for _ in range(20):
            key = self.key_manager.get_best_key(reserved_only=False)
            self.assertIn(key, ["key1", "key2"])
            self.assertNotEqual(key, "key3")

    def test_get_best_key_reserved_true(self):
        # Should return ONLY key3
        for _ in range(20):
            key = self.key_manager.get_best_key(reserved_only=True)
            self.assertEqual(key, "key3")

    def test_get_best_key_single_key(self):
        # If only 1 key, it should be returned regardless of reserved_only
        self.key_manager.keys = ["key1"]
        
        key_false = self.key_manager.get_best_key(reserved_only=False)
        self.assertEqual(key_false, "key1")
        
        key_true = self.key_manager.get_best_key(reserved_only=True)
        self.assertEqual(key_true, "key1")

    def test_exhaustion_behavior(self):
        # Exhaust key1
        config.exhausted_keys.add("key1")
        
        # reserved=False should return key2
        key = self.key_manager.get_best_key(reserved_only=False)
        self.assertEqual(key, "key2")
        
        # Exhaust key2
        config.exhausted_keys.add("key2")
        
        # reserved=False should return None
        key = self.key_manager.get_best_key(reserved_only=False)
        self.assertIsNone(key)
        
        # reserved=True should still return key3
        key = self.key_manager.get_best_key(reserved_only=True)
        self.assertEqual(key, "key3")

if __name__ == '__main__':
    unittest.main()
