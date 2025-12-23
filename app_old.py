"""Production-grade LangChain Agentic Application with Local LLM and Observability.

This application provides an interactive agent powered by Ollama (local LLM)
with web search capabilities and full observability via Phoenix.
"""

import os
import sys
import signal
import logging
from typing import Optional, Any
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class Config:
    """Application configuration."""
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
    PHOENIX_PORT = int(os.getenv("PHOENIX_PORT", "6006"))
    MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "1000"))
    SYSTEM_PROMPT = os.getenv(
        "SYSTEM_PROMPT",
        "You are a helpful assistant that can search the web for information."
    )


def setup_observability() -> Optional[Any]:
    """Set up Phoenix observability.
    
    Returns:
        Phoenix session object or None if setup fails.
    """
    try:
        session = px.launch_app()
        endpoint = f"http://127.0.0.1:{Config.PHOENIX_PORT}/v1/traces"
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        )
        trace_api.set_tracer_provider(tracer_provider)
        LangChainInstrumentor().instrument()
        logger.info(f"Phoenix observability started at {session.url}")
        return session
    except Exception as e:
        logger.error(f"Failed to set up observability: {e}")
        return None


def validate_environment() -> bool:
    """Validate required environment and dependencies.
    
    Returns:
        True if environment is valid, False otherwise.
    """
    try:
        # Test Ollama connection
        llm = ChatOllama(model=Config.OLLAMA_MODEL, temperature=0)
        llm.invoke("test")
        logger.info("Ollama connection successful")
        return True
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        logger.error("Make sure Ollama is running and the model is installed")
        logger.error(f"Run: ollama pull {Config.OLLAMA_MODEL}")
        return False


def create_agent_executor():
    """Create and configure the agent.
    
    Returns:
        Configured agent executor.
    """
    # Initialize tools
    search = DuckDuckGoSearchRun()
    tools = [search]
    logger.info(f"Initialized {len(tools)} tool(s)")

    # Initialize LLM
    llm = ChatOllama(
        model=Config.OLLAMA_MODEL,
        temperature=Config.OLLAMA_TEMPERATURE,
    )
    logger.info(f"Initialized LLM: {Config.OLLAMA_MODEL}")

    # Create agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=Config.SYSTEM_PROMPT
    )
    logger.info("Agent created successfully")
    return agent


def validate_input(user_input: str) -> bool:
    """Validate user input.
    
    Args:
        user_input: The user's input string.
        
    Returns:
        True if input is valid, False otherwise.
    """
    if len(user_input) > Config.MAX_INPUT_LENGTH:
        logger.warning(f"Input too long: {len(user_input)} characters")
        print(f"❌ Input too long. Maximum {Config.MAX_INPUT_LENGTH} characters.")
        return False
    return True


def process_query(agent, user_input: str) -> None:
    """Process a user query with the agent.
    
    Args:
        agent: The agent executor.
        user_input: The user's query.
    """
    logger.info(f"Processing query: {user_input[:100]}...")
    
    inputs = {"messages": [{"role": "user", "content": user_input}]}
    final_message = None
    
    try:
        print("\n--- 🚀 Agent Execution Started ---")
        for chunk in agent.stream(inputs, stream_mode="updates"):
            for node, values in chunk.items():
                print(f"\n--- 🔄 Step: {node} ---")
                if "messages" in values:
                    messages = values["messages"]
                    if not isinstance(messages, list):
                        messages = [messages]
                    
                    for msg in messages:
                        if msg.type == "ai":
                            if msg.tool_calls:
                                print("🤖 [Agent] Decided to call tool(s):")
                                for tool in msg.tool_calls:
                                    print(f"   🛠️  Tool: {tool['name']}")
                                    print(f"   Arguments: {tool['args']}")
                                    logger.info(f"Tool call: {tool['name']} with args {tool['args']}")
                            else:
                                print("🤖 [Agent] Response:")
                                print(f"   {msg.content}")
                                final_message = msg
                        elif msg.type == "tool":
                            print("📦 [Tool] Output:")
                            content = msg.content
                            display_content = (content[:300] + '...') if len(content) > 300 else content
                            print(f"   {display_content}")
        
        if final_message:
            print("\n" + "="*30)
            print("✅ Final Answer:")
            print(final_message.content)
            print("="*30)
            logger.info("Query processed successfully")
        else:
            print("\n❌ No final answer received.")
            logger.warning("No final answer received")

    except KeyboardInterrupt:
        logger.info("Query interrupted by user")
        print("\n⚠️  Query interrupted.")
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        print(f"\n❌ An error occurred: {e}")
        print("Please check the logs for more details.")


def handle_shutdown(signum, frame):
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    logger.info("Shutdown signal received")
    print("\n\n👋 Shutting down gracefully...")
    sys.exit(0)


def main():
    """Main application entry point."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    logger.info("Starting LangChain Agentic Application")
    print("\n" + "="*50)
    print("🤖 LangChain Agentic Application")
    print("="*50)
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed, exiting")
        sys.exit(1)
    
    # Set up observability
    session = setup_observability()
    if session:
        print(f"\n👀 Observability: Phoenix UI is running at {session.url}")
        print("   Open this URL in your browser to see traces.\n")
    else:
        print("\n⚠️  Running without observability (Phoenix setup failed)\n")
    
    # Create agent
    try:
        agent = create_agent_executor()
    except Exception as e:
        logger.error(f"Failed to create agent: {e}", exc_info=True)
        print(f"❌ Failed to initialize agent: {e}")
        sys.exit(1)
    
    # Interactive loop
    print("\n--- 🚀 Agent Ready! ---")
    print("Commands: 'exit' or 'quit' to stop, 'help' for help\n")
    
    query_count = 0
    
    try:
        while True:
            try:
                user_input = input("\nUser: ").strip()
            except EOFError:
                # Handle Ctrl+D
                logger.info("EOF received, exiting")
                break
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ["exit", "quit"]:
                logger.info(f"User requested exit after {query_count} queries")
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == "help":
                print("\n📖 Help:")
                print("  - Type your question and press Enter")
                print("  - Type 'exit' or 'quit' to stop")
                print(f"  - Maximum input length: {Config.MAX_INPUT_LENGTH} characters")
                print(f"  - Current model: {Config.OLLAMA_MODEL}")
                continue
            
            # Validate input
            if not validate_input(user_input):
                continue
            
            # Process query
            query_count += 1
            logger.info(f"Processing query #{query_count}")
            process_query(agent, user_input)
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n\n👋 Interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        logger.info(f"Application shutting down. Total queries processed: {query_count}")
        print(f"\n📊 Session stats: {query_count} queries processed")


if __name__ == "__main__":
    main()
