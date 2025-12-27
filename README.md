# Magentic

**Magnetic Agent Networks** — Dynamic multi-agent AI system with parallel execution.

## Why Magentic?

Traditional AI assistants use a single model for every task. Magentic takes a different approach: it acts as an **AI orchestrator** that understands your question, breaks it into components, and deploys the right combination of specialized agents to tackle each part. A research question might spawn a web researcher and an analyzer working in parallel; a coding task might use a planner, coder, and critic in sequence. This dynamic orchestration means you get expert-level responses without manually prompting different models—Magentic figures out the optimal strategy automatically and executes it efficiently using parallel processing.

## Quick Start

```bash
# 1. Setup
git clone <your-repo-url> && cd test_langchain
chmod +x magentic.sh && ./magentic.sh setup

# 2. Start all services
./magentic.sh start

# 3. Open http://localhost:3000
```

> **CLI mode:** Run `./magentic.sh cli` for interactive terminal mode.

## What is Magentic?

Magentic analyzes your questions and automatically creates an optimal network of specialized AI agents. Simple questions use one agent; complex questions spawn multiple agents working in parallel.

**Key Features:**
- 🤖 **Dynamic Planning** — AI creates optimal agent networks per query
- ⚡ **Parallel Execution** — Agents run simultaneously via LangGraph
- 🔍 **Web Search** — Agents can search the web for current info
- 📚 **RAG Support** — Query your own documents (optional)
- 🎨 **Modern Web UI** — Real-time execution visualization
- 🔐 **User Auth** — Accounts with history, or guest mode

**Supported LLMs:** Ollama (local/free), OpenAI, Claude

## Commands

```bash
./magentic.sh start     # Start all services
./magentic.sh stop      # Stop all services
./magentic.sh status    # Show service status
./magentic.sh cli       # Interactive CLI mode
./magentic.sh help      # All available commands
```

## Configuration

Edit `.env` to configure your LLM:

```bash
# Ollama (default, free, local)
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:1b

# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

# Claude
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

**Optional features:**
```bash
ENABLE_RAG=true          # Enable document retrieval
ENABLE_MCP=true          # Enable MCP tools (requires Docker)
DEBUG_STATE=true         # Enable debug visualization
```

## Agent Roles

| Role | Description |
|------|-------------|
| **Researcher** | Searches the web for current information |
| **Retriever** | Searches your knowledge base (RAG) |
| **Analyzer** | Analyzes data and explains concepts |
| **Planner** | Creates step-by-step plans |
| **Writer** | Writes articles and documentation |
| **Coder** | Generates and explains code |
| **Critic** | Reviews and improves content |
| **Synthesizer** | Combines multiple inputs into final output |

## How It Works

```
Query → Meta-Coordinator → Execution Plan → LangGraph DAG
              ↓
Layer 0: [researcher_0, researcher_1] (parallel)
              ↓
Layer 1: [analyzer_2] (waits for layer 0)
              ↓
Layer 2: [synthesizer_3] (final answer)
              ↓
WebSocket Stream → React UI
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System design and components |
| [Authentication](docs/AUTHENTICATION.md) | User auth and security |
| [RAG & Tools](docs/RAG_AND_TOOLS.md) | RAG setup and MCP integration |
| [Changelog](CHANGELOG.md) | Version history |

## Tech Stack

**Backend:** FastAPI, SQLAlchemy, LangGraph, SQLite  
**Frontend:** React 18, Vite, Tailwind CSS  
**LLMs:** Ollama, OpenAI, Claude

## License

MIT License - see [LICENSE](LICENSE)
