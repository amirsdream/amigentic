"""Command-line interface module."""

import logging
from typing import Optional

from .config import Config
from .agent import AgentManager

logger = logging.getLogger(__name__)


class CLI:
    """Interactive command-line interface for the agent."""

    def __init__(self, agent_manager: AgentManager, config: Config):
        """Initialize CLI.
        
        Args:
            agent_manager: The agent manager instance.
            config: Application configuration.
        """
        self.agent_manager = agent_manager
        self.config = config
        self.query_count = 0

    def validate_input(self, user_input: str) -> bool:
        """Validate user input.
        
        Args:
            user_input: The user's input string.
            
        Returns:
            True if input is valid, False otherwise.
        """
        if len(user_input) > self.config.max_input_length:
            logger.warning(f"Input too long: {len(user_input)} characters")
            print(
                f"❌ Input too long. "
                f"Maximum {self.config.max_input_length} characters."
            )
            return False
        return True

    def show_help(self) -> None:
        """Display help information."""
        print("\n📖 Help:")
        print("  - Type your question and press Enter")
        print("  - Type 'exit' or 'quit' to stop")
        print(f"  - Maximum input length: {self.config.max_input_length} characters")
        print(f"  - Current model: {self.config.ollama_model}")

    def run(self) -> int:
        """Run the interactive CLI loop.
        
        Returns:
            Exit code (0 for success).
        """
        print("\n--- 🚀 Agent Ready! ---")
        print("Commands: 'exit' or 'quit' to stop, 'help' for help\n")
        
        try:
            while True:
                try:
                    user_input = input("\nUser: ").strip()
                except EOFError:
                    logger.info("EOF received, exiting")
                    break
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ["exit", "quit"]:
                    logger.info(f"User requested exit after {self.query_count} queries")
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == "help":
                    self.show_help()
                    continue
                
                # Validate input
                if not self.validate_input(user_input):
                    continue
                
                # Process query
                self.query_count += 1
                logger.info(f"Processing query #{self.query_count}")
                self.agent_manager.process_query(user_input)
        
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
            print("\n\n👋 Interrupted. Goodbye!")
        except Exception as e:
            logger.error(f"Unexpected error in CLI loop: {e}", exc_info=True)
            print(f"\n❌ Unexpected error: {e}")
            return 1
        finally:
            logger.info(
                f"Application shutting down. "
                f"Total queries processed: {self.query_count}"
            )
            print(f"\n📊 Session stats: {self.query_count} queries processed")
        
        return 0
