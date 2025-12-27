"""
Main entry point for Magentic with LangGraph integration.

This version uses LangGraph for state management, checkpointing, and message
passing while preserving the dynamic meta-agent behavior.
"""

import asyncio
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from .config import Config
from .tools import ToolManager
from .agents import MetaAgentSystem
from .langgraph_runner import LangGraphExecutor
from .observability import ObservabilityManager
from .services.rag import RAGService
from .services.mcp_client import MCPClient

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

console = Console()


async def main_async() -> int:
    """Main async entry point with LangGraph.

    Returns:
        Exit code (0 for success, 1 for error)
    """

    # Display banner
    console.print("\n")
    console.print(
        Panel.fit(
            "[bold cyan]Magentic[/bold cyan] - Magnetic Agent Networks\n"
            "[dim]Dynamic meta-agent system with LangGraph infrastructure[/dim]\n\n"
            "[dim]🔍 Observability: http://localhost:6006[/dim]",
            border_style="cyan",
        )
    )
    console.print("\n")

    # Load configuration
    config = Config()

    # Validate configuration
    is_valid, error_msg = config.validate()
    if not is_valid:
        console.print(f"[red]❌ Configuration error: {error_msg}[/red]")
        console.print("[yellow]Please check your .env file[/yellow]")
        sys.exit(1)

    logger.info(f"✓ Configuration loaded: {config}")

    # Show LLM provider info
    provider_info = {
        "ollama": f"[cyan]Ollama[/cyan] (local) - {config.ollama_model}",
        "openai": f"[cyan]OpenAI[/cyan] - {config.openai_model}",
        "claude": f"[cyan]Claude[/cyan] - {config.anthropic_model}",
    }.get(config.llm_provider, "Unknown")

    console.print(f"🤖 LLM Provider: {provider_info}")
    console.print(f"🌡️  Temperature: {config.llm_temperature}\n")

    # Setup observability (optional - system works without it)
    obs_manager = ObservabilityManager(config)
    if config.enable_observability:
        obs_success = obs_manager.setup()
        if obs_success:
            logger.info("✓ Observability configured")
            console.print(
                f"[green]✓ Phoenix dashboard: http://localhost:{config.phoenix_port}[/green]"
            )
        else:
            logger.info("⚠️  Running without observability")
            console.print(
                "[yellow]⚠️  Running without observability (Phoenix failed to start)[/yellow]"
            )
    else:
        logger.info("ℹ️  Observability disabled (ENABLE_OBSERVABILITY=false)")
        console.print("[dim]ℹ️  Observability disabled[/dim]")

    # Initialize RAG service (optional)
    rag_service = None
    if config.enable_rag:
        try:
            console.print("[dim]📚 Initializing RAG service...[/dim]", end="")
            rag_service = RAGService(
                persist_directory=config.rag_persist_directory,
                vector_store=config.rag_vector_store,
                qdrant_mode=config.rag_qdrant_mode,
                qdrant_url=config.rag_qdrant_url,
                qdrant_collection=config.rag_qdrant_collection,
                chunk_size=config.rag_chunk_size,
                chunk_overlap=config.rag_chunk_overlap,
                embedding_provider=config.rag_embedding_provider,
                embedding_model=config.rag_embedding_model,
                ollama_base_url=config.ollama_base_url,
            )
            stats = rag_service.get_stats()
            console.print(
                f"\r[green]✓ RAG service ready ({stats.get('vector_store', 'unknown')}): {stats.get('document_count', 0)} docs[/green]"
            )
            logger.info(f"✓ RAG service initialized: {stats}")
        except Exception as e:
            console.print(f"\r[yellow]⚠️  RAG service failed: {e}[/yellow]")
            logger.warning(f"RAG service initialization failed: {e}")
    else:
        console.print("[dim]ℹ️  RAG service disabled[/dim]")

    # Initialize MCP client (optional)
    mcp_client = None
    if config.enable_mcp:
        try:
            console.print("[dim]🔌 Initializing MCP client...[/dim]", end="")
            mcp_client = MCPClient(gateway_url=config.mcp_gateway_url)
            health = await mcp_client.health_check()
            if health.get("status") == "healthy":
                healthy_servers = health.get("healthy_servers", 0)
                total_servers = health.get("total_servers", 0)
                console.print(
                    f"\r[green]✓ MCP Gateway ready: {healthy_servers}/{total_servers} servers healthy[/green]"
                )
                logger.info(f"✓ MCP client connected to {config.mcp_gateway_url}")
            else:
                console.print(f"\r[yellow]⚠️  MCP Gateway unavailable[/yellow]")
                logger.warning("MCP Gateway health check failed")
                mcp_client = None
        except Exception as e:
            console.print(f"\r[yellow]⚠️  MCP client failed: {e}[/yellow]")
            logger.warning(f"MCP client initialization failed: {e}")
            mcp_client = None
    else:
        console.print(
            "[dim]ℹ️  MCP client disabled (set ENABLE_MCP=true and start: cd docker && ./start.sh)[/dim]"
        )

    # Initialize tools with RAG and MCP support
    tool_manager = ToolManager(rag_service=rag_service, mcp_client=mcp_client)
    tools = await tool_manager.initialize_tools()
    logger.info(f"✓ Loaded {len(tools)} tools")

    # Initialize meta-agent system with RAG for active retrieval and tool manager
    meta_system = MetaAgentSystem(config, tools, rag_service=rag_service, tool_manager=tool_manager)
    logger.info("✓ Meta-agent system initialized")

    # Pre-cache MCP tools for all roles to avoid async/sync issues during execution
    if mcp_client:
        console.print("[dim]🔧 Pre-caching role tools...[/dim]", end="")
        await meta_system.agent_executor.pre_cache_role_tools()
        console.print("\r[green]✓ Role tools pre-cached[/green]")

    # Initialize LangGraph executor
    langgraph_executor = LangGraphExecutor(meta_system)
    logger.info("✓ LangGraph executor ready")

    # Warm up the coordinator to reduce first query latency
    console.print("[dim]🔥 Warming up coordinator...[/dim]", end="")
    meta_system.coordinator.warmup()
    console.print("\r[green]✓ System ready! LangGraph infrastructure enabled.[/green]")
    console.print("[dim]Commands: 'quit', 'memory', 'clear', 'show-memory', 'help'[/dim]\n")

    # Interactive loop
    while True:
        try:
            # Get user query
            console.print("[bold cyan]❓ Your question:[/bold cyan]")
            query = input("> ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                console.print("\n[yellow]👋 Goodbye![/yellow]")
                return 0

            if query.lower() == "help":
                console.print("\n[bold]Available Commands:[/bold]")
                console.print("  [cyan]quit/exit[/cyan] - Exit the application")
                console.print("  [cyan]memory[/cyan] - Show conversation history summary")
                console.print("  [cyan]clear[/cyan] - Clear conversation memory")
                console.print("  [cyan]show-memory[/cyan] - Display detailed conversation table")
                console.print("  [cyan]help[/cyan] - Show this help message\n")
                continue

            if query.lower() == "memory":
                summary = meta_system.get_memory_summary()
                console.print(
                    f"\n[bold]💾 Memory:[/bold] {summary['exchanges']} exchanges ({summary['message_count']} messages)"
                )
                if summary["preview"]:
                    console.print("[dim]Recent:[/dim]")
                    for msg in summary["preview"]:
                        role_label = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
                        content = (
                            msg["content"][:100] + "..."
                            if len(msg["content"]) > 100
                            else msg["content"]
                        )
                        console.print(f"  {role_label}: {content}")
                console.print()
                continue

            if query.lower() == "clear":
                meta_system.clear_memory()
                console.print("\n[green]✨ Memory cleared[/green]\n")
                continue

            if query.lower() == "show-memory":
                meta_system.show_memory_visualization()
                console.print()
                continue

            if not query:
                continue

            # Execute with LangGraph
            console.print("\n[cyan]⚙️  Processing query...[/cyan]")
            result = await langgraph_executor.execute_query(query, stream=False)

            # Display result
            console.print("\n" + "=" * 70)
            console.print("[bold green]FINAL RESULT[/bold green]")
            console.print("=" * 70 + "\n")

            final_output = result.get("final_output", "No output generated")
            console.print(
                Panel(Markdown(final_output), title="[bold]Answer[/bold]", border_style="green")
            )

            # Display execution stats
            console.print(f"\n[dim]📊 Session ID: {result['session_id']}[/dim]")
            console.print(
                f"[dim]🤖 Agents used: {result['agent_count']} agents in {result['layer_count']} layers[/dim]"
            )
            console.print(f"[dim]📈 Execution trace: {len(result['execution_trace'])} events[/dim]")
            console.print(f"[dim]🔍 Phoenix dashboard: http://localhost:6006[/dim]\n")

        except KeyboardInterrupt:
            console.print("\n\n[yellow]👋 Goodbye![/yellow]")
            # Cleanup MCP client
            if mcp_client:
                await mcp_client.close()
            return 0
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")
            logger.exception("Error processing query")
            continue


def main() -> int:
    """Synchronous entry point.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        # Clean exit on Ctrl+C
        console.print("\n[yellow]👋 Goodbye![/yellow]")
        return 0
    except Exception as e:
        console.print(f"\n[red]Fatal error: {e}[/red]")
        logger.exception("Fatal error")
        return 1


if __name__ == "__main__":
    main()
