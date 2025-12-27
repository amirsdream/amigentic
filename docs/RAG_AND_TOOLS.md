# RAG & Tools Guide

## RAG (Retrieval-Augmented Generation)

### Overview

Magentic provides **two modes** of RAG integration:

1. **Active RAG** (Automatic): Knowledge base context is automatically injected into the planning phase
2. **Passive RAG** (Tool-based): Agents can explicitly search via `search_knowledge_base` tool

### Active RAG

When RAG is enabled, every query automatically searches the knowledge base and injects relevant context before planning:

```
Query → RAG Search → Enriched Query → MetaCoordinator → Better Plan
```

**Benefits:**
- Coordinator makes more informed decisions about which agents to spawn
- Relevant context is available from the start (not discovered mid-execution)
- No explicit tool call required

**Configuration:**
```bash
# .env
ENABLE_RAG=true
# Active RAG uses: top 3 documents with relevance score >= 0.5
```

### Setup

```bash
# Install dependencies
pip install qdrant-client chromadb sentence-transformers

# Start Qdrant (optional, uses in-memory by default)
docker run -p 6333:6333 qdrant/qdrant
```

### Configuration

```bash
# .env
ENABLE_RAG=true
RAG_VECTOR_STORE=qdrant          # or "chroma"
RAG_EMBEDDING_MODEL=all-MiniLM-L6-v2
QDRANT_URL=http://localhost:6333  # for Qdrant
CHROMA_PATH=./data/chroma         # for ChromaDB
```

### Adding Documents

```python
from src.services import RAGService

rag = RAGService()
rag.add_documents([
    {"content": "Your text here", "metadata": {"source": "doc1"}},
])
```

### Passive RAG (Tool)

Agents can also explicitly call the `search_knowledge_base` tool during execution to search for specific information.

---

## MCP (Model Context Protocol)

### Setup

```bash
# Start MCP Gateway
cd docker && docker-compose up -d mcp-gateway
```

### Configuration

```bash
# .env
MCP_ENABLED=true
MCP_GATEWAY_URL=http://localhost:3100
```

### Available MCP Services

| Service | Tools |
|---------|-------|
| filesystem | read_file, write_file, list_directory |
| fetch | fetch_url |
| memory | store, retrieve |

### Adding Custom MCP Servers

Edit `docker/mcp-gateway/config.json`:

```json
{
  "servers": {
    "my-server": {
      "command": "node",
      "args": ["path/to/server.js"]
    }
  }
}
```

---

## Tool Manager

Tools are initialized in `ToolManager`:

```python
# Built-in tools
- DuckDuckGoSearchRun    # Web search
- search_knowledge_base  # RAG retrieval (if enabled)
- MCP tools              # Dynamic from MCP gateway (if enabled)
```

Agents automatically receive appropriate tools based on their role.
