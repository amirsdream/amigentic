"""Tests for configuration module."""

import os
import unittest
from unittest.mock import patch

from src.config import Config


class TestConfig(unittest.TestCase):
    """Test cases for Config class."""

    def test_default_values(self):
        """Test default configuration values."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            self.assertEqual(config.ollama_model, "llama3.1")
            self.assertEqual(config.ollama_temperature, 0.0)
            self.assertEqual(config.phoenix_port, 6006)
            self.assertEqual(config.max_input_length, 1000)

    def test_environment_override(self):
        """Test configuration from environment variables."""
        env_vars = {
            "OLLAMA_MODEL": "llama2",
            "OLLAMA_TEMPERATURE": "0.5",
            "PHOENIX_PORT": "7007",
            "MAX_INPUT_LENGTH": "2000"
        }
        with patch.dict(os.environ, env_vars):
            config = Config()
            self.assertEqual(config.ollama_model, "llama2")
            self.assertEqual(config.ollama_temperature, 0.5)
            self.assertEqual(config.phoenix_port, 7007)
            self.assertEqual(config.max_input_length, 2000)

    def test_validate_valid_config(self):
        """Test validation with valid configuration."""
        config = Config()
        is_valid, error_msg = config.validate()
        self.assertTrue(is_valid)
        self.assertIsNone(error_msg)

    def test_validate_invalid_temperature(self):
        """Test validation with invalid temperature."""
        with patch.dict(os.environ, {"OLLAMA_TEMPERATURE": "3.0"}):
            config = Config()
            is_valid, error_msg = config.validate()
            self.assertFalse(is_valid)
            self.assertIn("OLLAMA_TEMPERATURE", error_msg)

    def test_validate_invalid_port(self):
        """Test validation with invalid port."""
        with patch.dict(os.environ, {"PHOENIX_PORT": "100"}):
            config = Config()
            is_valid, error_msg = config.validate()
            self.assertFalse(is_valid)
            self.assertIn("PHOENIX_PORT", error_msg)

    def test_validate_invalid_input_length(self):
        """Test validation with invalid input length."""
        with patch.dict(os.environ, {"MAX_INPUT_LENGTH": "-1"}):
            config = Config()
            is_valid, error_msg = config.validate()
            self.assertFalse(is_valid)
            self.assertIn("MAX_INPUT_LENGTH", error_msg)

    def test_repr(self):
        """Test string representation."""
        config = Config()
        repr_str = repr(config)
        self.assertIn("Config", repr_str)
        self.assertIn("llama3.1", repr_str)


if __name__ == "__main__":
    unittest.main()
