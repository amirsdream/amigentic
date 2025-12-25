# Magentic

**Magnetic Agent Networks** - AI-powered meta-agent system with dynamic topology generation and parallel execution.

## Features

- 🎯 **Dynamic Planning**: AI generates unique agent topologies per query
- ⚡ **Parallel Execution**: DAG-based layer execution with LangGraph
- �� **State Management**: Checkpointing and crash recovery
- 🔍 **Web Search**: DuckDuckGo integration for research agents
- 📊 **Observability**: Phoenix dashboard for real-time tracing
- 🎨 **8 Agent Roles**: Researcher, Analyzer, Planner, Writer, Coder, Critic, Synthesizer, Coordinator

## Quick Start

### 1. Install Dependencies

```bash
# Option A: Local Ollama (free, private)
# Install Ollama from https://ollama.com
ollama pull llama3.2:1b

# Option B: OpenAI (requires API key)
# Get API key from https://platform.openai.com

# Option C: Claude (requires API key)
# Get API key from https://console.anthropic.com

# Install Python packages
pip install -r requirements.txt
```

### 2. Configure

Copy `.env.example` to `.env` and configure your LLM:

```bash
# For Ollama (default, runs locally)
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:1b

# For OpenAI
# LLM_PROVIDER=openai
# OPENAI_API_KEY=your-key-here
# OPENAI_MODEL=gpt-4o

# For Claude
# LLM_PROVIDER=claude
# ANTHROPIC_API_KEY=your-key-here
# ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

### 3. Run

```bash
python app.py
```

### 3. Phoenix Dashboard (Optional)

Open http://localhost:6006 for real-time LLM tracing

## Usage

```
❓ Your question: Compare Python and Rust for web development

Building dynamic graph with 4 agents in 3 layers
  Layer 0: [researcher_0, researcher_1] (parallel)
  Layer 1: [analyzer_2]
  Layer 2: [synthesizer_3]
  
✓ Execution Complete!
```

## Commands

- `help` - Show commands
- `memory` - View conversation history
- `clear` - Clear memory
- `quit` - Exit

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Configuration

Edit `.env` file (copy from `.env.example`):

**LLM Provider:**
- `LLM_PROVIDER` - Choose: `ollama` (local), `openai`, or `claude`
- `LLM_TEMPERATURE` - 0.0 (precise) to 2.0 (creative)

**Ollama (local, free):**
- `OLLAMA_MODEL` - e.g., `llama3.2:1b`, `llama3.1`, `mistral`
- `OLLAMA_BASE_URL` - Default: `http://localhost:11434`

**OpenAI (cloud, API key required):**
- `OPENAI_API_KEY` - Your API key
- `OPENAI_MODEL` - e.g., `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`

**Claude (cloud, API key required):**
- `ANTHROPIC_API_KEY` - Your API key
- `ANTHROPIC_MODEL` - e.g., `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`

**System:**
- `MAX_PARALLEL_AGENTS` - Concurrent agents (2-5 recommended)

## Tech Stack

- **LangChain** - Agent orchestration
- **LangGraph** - State management & checkpointing
- **LLMs** - Ollama (local), OpenAI, or Claude
- **Phoenix** - Observability
- **Rich** - Terminal UI

## License

MIT License - see [LICENSE](LICENSE)
