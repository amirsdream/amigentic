"""
Main entry point for Magentic with LangGraph integration.

This version uses LangGraph for state management, checkpointing, and message
passing while preserving the dynamic meta-agent behavior.
"""

import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from .config import Config
from .tools import ToolManager
from .meta_agent_system import MetaAgentSystem
from .langgraph_executor import LangGraphExecutor
from .observability import ObservabilityManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    console.print(Panel.fit(
        "[bold cyan]Magentic[/bold cyan] - Magnetic Agent Networks\n"
        "[dim]Dynamic meta-agent system with LangGraph infrastructure[/dim]\n\n"
        "[dim]🔍 Observability: http://localhost:6006[/dim]",
        border_style="cyan"
    ))
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
        "claude": f"[cyan]Claude[/cyan] - {config.anthropic_model}"
    }.get(config.llm_provider, "Unknown")
    
    console.print(f"🤖 LLM Provider: {provider_info}")
    console.print(f"🌡️  Temperature: {config.llm_temperature}\n")
    
    # Setup observability (optional - system works without it)
    obs_manager = ObservabilityManager(config)
    if config.enable_observability:
        obs_success = obs_manager.setup()
        if obs_success:
            logger.info("✓ Observability configured")
            console.print(f"[green]✓ Phoenix dashboard: http://localhost:{config.phoenix_port}[/green]")
        else:
            logger.info("⚠️  Running without observability")
            console.print("[yellow]⚠️  Running without observability (Phoenix failed to start)[/yellow]")
    else:
        logger.info("ℹ️  Observability disabled (ENABLE_OBSERVABILITY=false)")
        console.print("[dim]ℹ️  Observability disabled[/dim]")
    
    # Initialize tools
    tool_manager = ToolManager()
    tools = tool_manager.initialize_tools()
    logger.info(f"✓ Loaded {len(tools)} tools")
    
    # Initialize meta-agent system
    meta_system = MetaAgentSystem(config, tools)
    logger.info("✓ Meta-agent system initialized")
    
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
            
            if query.lower() in ['quit', 'exit', 'q']:
                console.print("\n[yellow]👋 Goodbye![/yellow]")
                return 0
            
            if query.lower() == 'help':
                console.print("\n[bold]Available Commands:[/bold]")
                console.print("  [cyan]quit/exit[/cyan] - Exit the application")
                console.print("  [cyan]memory[/cyan] - Show conversation history summary")
                console.print("  [cyan]clear[/cyan] - Clear conversation memory")
                console.print("  [cyan]show-memory[/cyan] - Display detailed conversation table")
                console.print("  [cyan]help[/cyan] - Show this help message\n")
                continue
            
            if query.lower() == 'memory':
                summary = meta_system.get_memory_summary()
                console.print(f"\n[bold]💾 Memory:[/bold] {summary['exchanges']} exchanges ({summary['message_count']} messages)")
                if summary['preview']:
                    console.print("[dim]Recent:[/dim]")
                    for msg in summary['preview']:
                        role_label = "👤 User" if msg["role"] == "user" else "🤖 Assistant"
                        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                        console.print(f"  {role_label}: {content}")
                console.print()
                continue
            
            if query.lower() == 'clear':
                meta_system.clear_memory()
                console.print("\n[green]✨ Memory cleared[/green]\n")
                continue
            
            if query.lower() == 'show-memory':
                meta_system.show_memory_visualization()
                console.print()
                continue
            
            if not query:
                continue
            
            # Execute with LangGraph
            console.print("\n[cyan]⚙️  Processing query...[/cyan]")
            result = await langgraph_executor.execute_query(query, stream=False)
            
            # Display result
            console.print("\n" + "="*70)
            console.print("[bold green]FINAL RESULT[/bold green]")
            console.print("="*70 + "\n")
            
            final_output = result.get("final_output", "No output generated")
            console.print(Panel(
                Markdown(final_output),
                title="[bold]Answer[/bold]",
                border_style="green"
            ))
            
            # Display execution stats
            console.print(f"\n[dim]📊 Session ID: {result['session_id']}[/dim]")
            console.print(f"[dim]🤖 Agents used: {result['agent_count']} agents in {result['layer_count']} layers[/dim]")
            console.print(f"[dim]📈 Execution trace: {len(result['execution_trace'])} events[/dim]")
            console.print(f"[dim]🔍 Phoenix dashboard: http://localhost:6006[/dim]\n")
            
        except KeyboardInterrupt:
            console.print("\n\n[yellow]👋 Goodbye![/yellow]")
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
