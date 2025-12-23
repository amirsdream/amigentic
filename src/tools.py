"""Tools module for agent capabilities."""

import logging
from typing import List

from langchain_core.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun

logger = logging.getLogger(__name__)


class ToolManager:
    """Manager for agent tools."""

    def __init__(self):
        """Initialize tool manager."""
        self.tools: List[BaseTool] = []

    def initialize_tools(self) -> List[BaseTool]:
        """Initialize and return all available tools.
        
        Returns:
            List of initialized tools.
        """
        try:
            # Initialize search tool
            search = DuckDuckGoSearchRun()
            self.tools.append(search)
            
            logger.info(f"Initialized {len(self.tools)} tool(s)")
            return self.tools
            
        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")
            raise

    def get_tool_names(self) -> List[str]:
        """Get names of all initialized tools.
        
        Returns:
            List of tool names.
        """
        return [tool.name for tool in self.tools]
