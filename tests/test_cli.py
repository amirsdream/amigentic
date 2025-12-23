"""Tests for CLI module."""

import unittest
from unittest.mock import MagicMock, patch
from io import StringIO

from src.cli import CLI
from src.config import Config
from src.agent import AgentManager


class TestCLI(unittest.TestCase):
    """Test cases for CLI class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Config()
        self.agent_manager = MagicMock(spec=AgentManager)
        self.cli = CLI(self.agent_manager, self.config)

    def test_initialization(self):
        """Test CLI initialization."""
        self.assertEqual(self.cli.query_count, 0)
        self.assertIsNotNone(self.cli.config)
        self.assertIsNotNone(self.cli.agent_manager)

    def test_validate_input_valid(self):
        """Test input validation with valid input."""
        result = self.cli.validate_input("test query")
        self.assertTrue(result)

    def test_validate_input_too_long(self):
        """Test input validation with too long input."""
        long_input = "a" * (self.config.max_input_length + 1)
        result = self.cli.validate_input(long_input)
        self.assertFalse(result)

    @patch('sys.stdout', new_callable=StringIO)
    def test_show_help(self, mock_stdout):
        """Test showing help message."""
        self.cli.show_help()
        output = mock_stdout.getvalue()
        self.assertIn("Help", output)
        self.assertIn("exit", output)
        self.assertIn(str(self.config.max_input_length), output)


if __name__ == "__main__":
    unittest.main()
