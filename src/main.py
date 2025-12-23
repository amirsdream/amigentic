"""Main application entry point."""

import sys
import signal
import logging
from dotenv import load_dotenv

from .config import Config
from .observability import ObservabilityManager
from .tools import ToolManager
from .agent import AgentManager
from .cli import CLI

# Load environment variables
load_dotenv()


def setup_logging(config: Config) -> None:
    """Configure application logging.
    
    Args:
        config: Application configuration.
    """
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def handle_shutdown(signum, frame):
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    logger = logging.getLogger(__name__)
    logger.info("Shutdown signal received")
    print("\n\n👋 Shutting down gracefully...")
    sys.exit(0)


def main() -> int:
    """Main application entry point.
    
    Returns:
        Exit code (0 for success, non-zero for error).
    """
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Load and validate configuration
    config = Config()
    setup_logging(config)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting LangChain Agentic Application")
    
    print("\n" + "="*50)
    print("🤖 LangChain Agentic Application")
    print("="*50)
    
    # Validate configuration
    is_valid, error_msg = config.validate()
    if not is_valid:
        logger.error(f"Configuration validation failed: {error_msg}")
        print(f"❌ Configuration error: {error_msg}")
        return 1
    
    # Initialize components
    agent_manager = AgentManager(config)
    
    # Validate environment
    if not agent_manager.validate_ollama_connection():
        logger.error("Environment validation failed, exiting")
        return 1
    
    # Set up observability
    obs_manager = ObservabilityManager(config)
    if obs_manager.setup():
        print(f"\n👀 Observability: Phoenix UI is running at {obs_manager.get_url()}")
        print("   Open this URL in your browser to see traces.\n")
    else:
        print("\n⚠️  Running without observability (Phoenix setup failed)\n")
    
    # Initialize tools
    try:
        tool_manager = ToolManager()
        tools = tool_manager.initialize_tools()
    except Exception as e:
        logger.error(f"Failed to initialize tools: {e}", exc_info=True)
        print(f"❌ Failed to initialize tools: {e}")
        return 1
    
    # Create agent
    try:
        agent_manager.create_agent(tools)
    except Exception as e:
        logger.error(f"Failed to create agent: {e}", exc_info=True)
        print(f"❌ Failed to initialize agent: {e}")
        return 1
    
    # Run CLI
    cli = CLI(agent_manager, config)
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
