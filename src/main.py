"""Main application entry point."""

import sys
import logging
from dotenv import load_dotenv

from .config import Config
from .observability import ObservabilityManager
from .tools import ToolManager
from .meta_agent_system import MetaAgentSystem

# Load environment variables
load_dotenv()


def setup_logging(config: Config) -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main() -> int:
    """Main application entry point."""
    try:
        # Initialize configuration
        config = Config()
        setup_logging(config)
        
        logger = logging.getLogger(__name__)
        logger.info("Starting Meta-Agent System...")
        
        # Initialize observability
        obs_manager = ObservabilityManager(config)
        obs_manager.setup()
        
        # Initialize tools
        tool_manager = ToolManager()
        tools = tool_manager.initialize_tools()
        logger.info(f"Initialized {len(tools)} tool(s)")
        
        # Initialize meta-agent system
        logger.info("Initializing meta-agent system...")
        system = MetaAgentSystem(config, tools)
        
        # Simple interactive loop
        print("\n" + "="*60)
        print("🤖 META-AGENT SYSTEM")
        print("="*60)
        print("Dynamic agent creation based on your needs")
        print("Commands: 'quit', 'memory', 'clear', 'graph', 'show-memory'\n")
        
        while True:
            try:
                user_input = input("❓ Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'memory':
                    summary = system.get_memory_summary()
                    print(f"\n💾 Memory: {summary['exchanges']} exchanges ({summary['message_count']} messages)")
                    if summary['preview']:
                        print("Recent:")
                        for msg in summary['preview']:
                            role_label = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
                            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                            print(f"  {role_label}: {content}")
                    print()
                    continue
                
                if user_input.lower() == 'clear':
                    system.clear_memory()
                    print("✨ Memory cleared\n")
                    continue
                
                if user_input.lower() == 'show-memory':
                    system.show_memory_visualization()
                    continue
                
                if user_input.lower() == 'graph':
                    print("⚠️  Run a query first to generate an execution graph\n")
                    continue
                
                if not user_input:
                    continue
                
                # Process query
                result = system.process_query(user_input)
                
                # Display final answer
                print("\n" + "="*60)
                print("✅ FINAL ANSWER")
                print("="*60)
                print(result['final_answer'])
                
                # Offer to generate graph
                print("\n📊 Generate execution graph? (yes/no): ", end="")
                graph_choice = input().strip().lower()
                if graph_choice in ['yes', 'y']:
                    system.generate_execution_graph(result, auto_open=True)
                print("\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}", exc_info=True)
                print(f"\n❌ Error: {e}\n")
        
        return 0
        
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
