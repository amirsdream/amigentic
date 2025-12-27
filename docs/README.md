# Documentation

## Guides

| Document | Description |
|----------|-------------|
| [Architecture](ARCHITECTURE.md) | System design, execution flow, components |
| [Authentication](AUTHENTICATION.md) | User auth, API endpoints, security |
| [RAG & Tools](RAG_AND_TOOLS.md) | RAG setup, MCP integration |

## Quick Reference

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | System status |
| `WS /ws` | WebSocket for queries |
| `POST /register` | Create user |
| `POST /login` | Authenticate |

### WebSocket Protocol

**Send query:**
```json
{"query": "What is Python?", "conversation_id": "uuid"}
```

**Events received:**
- `plan` — Execution plan with agents
- `agent_start` — Agent began execution
- `agent_complete` — Agent finished with output
- `complete` — Final synthesized response
- `error` — Error occurred

### Environment Variables

```bash
# LLM Provider (required)
LLM_PROVIDER=ollama          # ollama, openai, or claude
OLLAMA_MODEL=llama3.2:1b

# Optional Features
ENABLE_RAG=false
ENABLE_MCP=false
DEBUG_STATE=false
```

See main [README](../README.md) for quick start.
