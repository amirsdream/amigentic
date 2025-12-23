"""Agent creation and management module."""

import logging
from typing import Any, List

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.tools import BaseTool

from .config import Config

logger = logging.getLogger(__name__)


class AgentManager:
    """Manager for LangChain agent lifecycle."""

    def __init__(self, config: Config):
        """Initialize agent manager.
        
        Args:
            config: Application configuration.
        """
        self.config = config
        self.llm: Any = None
        self.agent: Any = None

    def validate_ollama_connection(self) -> bool:
        """Validate connection to Ollama.
        
        Returns:
            True if connection is successful, False otherwise.
        """
        try:
            llm = ChatOllama(
                model=self.config.ollama_model,
                temperature=0
            )
            llm.invoke("test")
            logger.info("Ollama connection successful")
            return True
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            logger.error("Make sure Ollama is running and the model is installed")
            logger.error(f"Run: ollama pull {self.config.ollama_model}")
            return False

    def create_llm(self) -> Any:
        """Create and configure the LLM.
        
        Returns:
            Configured LLM instance.
        """
        self.llm = ChatOllama(
            model=self.config.ollama_model,
            temperature=self.config.ollama_temperature,
        )
        logger.info(f"Initialized LLM: {self.config.ollama_model}")
        return self.llm

    def create_agent(self, tools: List[BaseTool]) -> Any:
        """Create the agent with tools.
        
        Args:
            tools: List of tools for the agent.
            
        Returns:
            Configured agent.
        """
        if not self.llm:
            self.create_llm()
        
        self.agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=self.config.system_prompt
        )
        
        logger.info("Agent created successfully")
        return self.agent

    def process_query(self, user_input: str) -> None:
        """Process a user query with the agent.
        
        Args:
            user_input: The user's query.
        """
        if not self.agent:
            raise RuntimeError("Agent not initialized")
        
        logger.info(f"Processing query: {user_input[:100]}...")
        
        inputs = {"messages": [{"role": "user", "content": user_input}]}
        final_message = None
        
        try:
            print("\n--- 🚀 Agent Execution Started ---")
            for chunk in self.agent.stream(inputs, stream_mode="updates"):
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
                                        logger.info(
                                            f"Tool call: {tool['name']} "
                                            f"with args {tool['args']}"
                                        )
                                else:
                                    print("🤖 [Agent] Response:")
                                    print(f"   {msg.content}")
                                    final_message = msg
                            elif msg.type == "tool":
                                print("📦 [Tool] Output:")
                                content = msg.content
                                display_content = (
                                    (content[:300] + '...')
                                    if len(content) > 300
                                    else content
                                )
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
