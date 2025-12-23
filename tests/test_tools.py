"""Tests for tools module."""

import unittest
from unittest.mock import patch, MagicMock

from src.tools import ToolManager


class TestToolManager(unittest.TestCase):
    """Test cases for ToolManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.tool_manager = ToolManager()

    def test_initialization(self):
        """Test tool manager initialization."""
        self.assertEqual(len(self.tool_manager.tools), 0)

    @patch('src.tools.DuckDuckGoSearchRun')
    def test_initialize_tools(self, mock_search):
        """Test tool initialization."""
        mock_tool = MagicMock()
        mock_tool.name = "duckduckgo_search"
        mock_search.return_value = mock_tool
        
        tools = self.tool_manager.initialize_tools()
        
        self.assertEqual(len(tools), 1)
        self.assertEqual(len(self.tool_manager.tools), 1)
        mock_search.assert_called_once()

    @patch('src.tools.DuckDuckGoSearchRun')
    def test_get_tool_names(self, mock_search):
        """Test getting tool names."""
        mock_tool = MagicMock()
        mock_tool.name = "duckduckgo_search"
        mock_search.return_value = mock_tool
        
        self.tool_manager.initialize_tools()
        names = self.tool_manager.get_tool_names()
        
        self.assertEqual(names, ["duckduckgo_search"])

    @patch('src.tools.DuckDuckGoSearchRun')
    def test_initialize_tools_failure(self, mock_search):
        """Test tool initialization failure."""
        mock_search.side_effect = Exception("Tool initialization failed")
        
        with self.assertRaises(Exception):
            self.tool_manager.initialize_tools()


if __name__ == "__main__":
    unittest.main()
