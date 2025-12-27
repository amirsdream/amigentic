# Changelog

## [1.1.0] - 2024-12-25

### Added
- **Token Usage Tracking** — Track prompt/completion tokens per agent and total
- **Layer Barrier Synchronization** — Ensures agents complete before next layer starts
- **Conversation History** — Agents receive context from previous steps
- **Debug State Visualization** — Set `DEBUG_STATE=true` to see execution flow

### Fixed
- Multi-agent context passing when 2-3 agents feed into next layer
- Chat persistence not loading on page refresh
- WebSocket event ordering for agent completion

### Configuration
```bash
DEBUG_STATE=true         # Enable state visualization
UI_DISPLAY_LIMIT=200     # Character limit for output display
```

## [1.0.0] - 2024-12-20

### Initial Release
- Dynamic meta-agent system with LangGraph
- Parallel agent execution in layers
- Web UI with real-time WebSocket updates
- User authentication with conversation history
- RAG support with Qdrant vector store
- MCP tool integration
- Support for Ollama, OpenAI, and Claude
