# Installation Guide

> **Recommended:** Use `./magentic.sh setup` for automated installation. This guide is for manual setup.

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** (for frontend)
- **Docker** (optional, for MCP services)

## Manual Installation

### 1. Clone and Setup Environment

```bash
git clone <your-repo-url>
cd test_langchain

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install frontend
cd frontend && npm install && cd ..
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your LLM provider:

```bash
# Ollama (free, local)
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.2:1b

# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# Claude
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Initialize Database

```bash
alembic upgrade head
# Optional: Create demo user (demo/demo123)
python init_db.py --demo
```

### 4. Start Services

**Terminal 1 (Backend):**
```bash
source .venv/bin/activate
python -m src.run_api
```

**Terminal 2 (Frontend):**
```bash
cd frontend && npm run dev
```

Open http://localhost:3000

## LLM Provider Setup

### Ollama (Recommended for local)

```bash
# Install Ollama: https://ollama.com
ollama pull llama3.2:1b
ollama pull nomic-embed-text  # For RAG embeddings
```

### OpenAI

Get API key from https://platform.openai.com

### Claude

Get API key from https://console.anthropic.com

## Optional Features

### RAG (Knowledge Base)

```bash
ENABLE_RAG=true
RAG_VECTOR_STORE=qdrant
```

See [RAG & Tools Guide](docs/RAG_AND_TOOLS.md) for details.

### MCP (Model Context Protocol)

Requires Docker:
```bash
./magentic.sh mcp
```

See [RAG & Tools Guide](docs/RAG_AND_TOOLS.md) for MCP setup.

## Troubleshooting

**Port already in use:**
```bash
./magentic.sh stop
# or manually: lsof -i :8000 | kill
```

**Database issues:**
```bash
./magentic.sh db-reset
```

**Missing dependencies:**
```bash
pip install -r requirements.txt
```
