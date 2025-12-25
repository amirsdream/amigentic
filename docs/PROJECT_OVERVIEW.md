# Magentic - Project Overview

## Introduction

Magentic (Magnetic Agent Networks) is an advanced AI-powered meta-agent system that dynamically generates and executes specialized agent networks to solve complex queries. Unlike traditional fixed-pipeline systems, Magentic uses AI to analyze each query and create a unique topology of agents optimized for that specific task.

## What Makes Magentic Unique?

### 1. Dynamic Agent Topology
Instead of using a pre-defined set of agents for every query, Magentic's meta-coordinator (itself an LLM) analyzes each query and decides:
- Which roles are needed (researcher, analyzer, writer, coder, etc.)
- How many agents of each type
- What specific task each agent should perform
- How agents should depend on each other (execution order)

**Example**: 
- "Hi" → 1 analyzer agent
- "Explain quantum computing" → 1 analyzer agent  
- "Compare Python vs Rust for web development" → 2 researchers + 1 synthesizer (3 agents, 2 layers)

### 2. Intelligent Complexity Scaling
The system automatically scales based on query complexity:

**Simple Queries** (greetings, basic questions):
- Uses single agent
- Fast response
- Minimal LLM calls

**Medium Queries** (explanations, single-topic analysis):
- 1-2 agents
- Focused analysis
- Efficient resource usage

**Complex Queries** (comparisons, research, multi-step):
- Multiple specialized agents
- Parallel execution
- Synthesizer for final integration

### 3. Parallel Execution with LangGraph
Magentic builds a DAG (Directed Acyclic Graph) from the agent plan and executes independent agents in parallel:

```
Layer 0: [researcher_0, researcher_1] → run simultaneously
Layer 1: [analyzer_2] → waits for both researchers
Layer 2: [synthesizer_3] → final synthesis
```

This dramatically reduces execution time for complex queries.

### 4. Real-Time Web UI
Beautiful React-based interface with:
- Live progress updates via WebSocket
- Animated indicators showing execution status
- Expandable agent details with timing information
- Markdown rendering with syntax-highlighted code blocks
- Toggle to show/hide execution details

## System Components

### Frontend (React + Vite)

**Location**: `/frontend`

**Key Files**:
- `src/App.jsx` - Main application with WebSocket client
- `src/index.css` - Tailwind styling + markdown styles
- `package.json` - Dependencies (react, react-markdown, lucide-react, etc.)

**Features**:
- Real-time WebSocket connection to backend
- Message history with user/assistant bubbles
- Execution visualization with progress indicators
- Expandable agent steps
- Markdown rendering for beautiful formatted responses
- Syntax highlighting for code blocks (Tokyo Night theme)

### Backend (FastAPI + LangGraph)

**Location**: `/src`

**Key Files**:
- `api.py` - FastAPI server with WebSocket endpoint
- `run_api.py` - Server runner
- `meta_coordinator.py` - AI planning engine
- `meta_agent_system.py` - LangGraph execution
- `role_library.py` - Agent role definitions
- `config.py` - Configuration and LLM setup
- `tools.py` - DuckDuckGo search integration
- `observability.py` - Phoenix tracing setup

**API Endpoints**:
- `GET /` - Health check and system info
- `GET /health` - Health status
- `POST /query` - Synchronous query processing
- `GET /memory` - Retrieve conversation history
- `WS /ws` - WebSocket for real-time streaming

### WebSocket Message Flow

**Client → Server**:
```json
{
  "query": "Compare Python and Rust"
}
```

**Server → Client** (streaming):
```json
{"type": "status", "message": "Processing query..."}
{"type": "stage", "stage": "planning", "message": "AI analyzing..."}
{"type": "plan", "data": {
  "agents": [...],
  "total_agents": 3,
  "total_layers": 2
}}
{"type": "agent_start", "data": {
  "agent_id": "researcher_0",
  "role": "researcher",
  "task": "Research Python"
}}
{"type": "agent_complete", "data": {
  "agent_id": "researcher_0",
  "output_length": 1500
}}
{"type": "complete", "data": {
  "output": "Final synthesized response..."
}}
```

## Agent Roles

### 1. **Researcher** 🔍
- **Purpose**: Gather current information from the web
- **Tools**: DuckDuckGo search
- **Use Cases**: Latest news, current facts, real-time data
- **Example**: "What are the latest developments in AI?"

### 2. **Analyzer** 📊
- **Purpose**: Deep analysis, explanations, breakdowns
- **Tools**: None (pure reasoning)
- **Use Cases**: Explain concepts, analyze situations, compare options
- **Example**: "Explain how blockchain works"

### 3. **Planner** 📋
- **Purpose**: Create step-by-step plans and strategies
- **Tools**: None
- **Use Cases**: Project plans, roadmaps, sequential processes
- **Example**: "Create a plan to learn web development"

### 4. **Writer** ✍️
- **Purpose**: Create articles, stories, documentation
- **Tools**: None
- **Use Cases**: Long-form content, creative writing, documentation
- **Example**: "Write a blog post about renewable energy"

### 5. **Coder** 💻
- **Purpose**: Generate, explain, or debug code
- **Tools**: None
- **Use Cases**: Programming tasks, code examples, technical solutions
- **Example**: "Write a Python function to sort a list"

### 6. **Critic** 🎯
- **Purpose**: Review and improve existing content
- **Tools**: None
- **Use Cases**: Quality assurance, content improvement, error detection
- **Example**: Used internally to improve outputs

### 7. **Synthesizer** 🔄
- **Purpose**: Combine outputs from multiple agents into coherent response
- **Tools**: None
- **Use Cases**: Final integration for multi-agent queries
- **Example**: Always used as final agent when 2+ agents execute

### 8. **Coordinator** 🎭
- **Purpose**: Meta-planning and delegation
- **Tools**: None
- **Use Cases**: Very complex queries requiring sub-planning
- **Example**: Rare - used for extremely complex multi-stage tasks

## Execution Example

### Query: "Compare Python and Rust for web development"

**Step 1 - Meta-Coordinator Analysis**:
```
Complexity: COMPLEX (comparison task)
Decision: Need research on both, then synthesis
```

**Step 2 - Plan Generation**:
```json
{
  "description": "Research and compare Python vs Rust",
  "agents": [
    {
      "role": "researcher",
      "task": "Research Python for web development",
      "depends_on": []
    },
    {
      "role": "researcher", 
      "task": "Research Rust for web development",
      "depends_on": []
    },
    {
      "role": "synthesizer",
      "task": "Compare Python and Rust based on research",
      "depends_on": [0, 1]
    }
  ]
}
```

**Step 3 - DAG Construction**:
```
Layer 0: [researcher_0, researcher_1] (parallel)
Layer 1: [synthesizer_2] (waits for both)
```

**Step 4 - Execution**:
1. Both researchers execute simultaneously
2. Each searches DuckDuckGo for current information
3. Synthesizer receives both outputs
4. Creates comprehensive comparison
5. Final output delivered to user

**WebSocket Updates**:
- Plan notification (3 agents, 2 layers)
- researcher_0 start → complete (with timing)
- researcher_1 start → complete (with timing)
- synthesizer_2 start → complete
- Final message with markdown formatting

## Configuration

### Environment Variables

**Required**:
```bash
LLM_PROVIDER=ollama  # or openai, claude
```

**LLM-Specific**:
```bash
# Ollama
OLLAMA_MODEL=llama3.2:1b
OLLAMA_BASE_URL=http://localhost:11434

# OpenAI  
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

# Claude
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

**Optional**:
```bash
LLM_TEMPERATURE=0.7
MAX_PARALLEL_AGENTS=3
PHOENIX_ENABLED=true
```

### LLM Provider Comparison

| Feature | Ollama | OpenAI | Claude |
|---------|--------|--------|--------|
| Cost | Free | Pay per token | Pay per token |
| Privacy | 100% local | Cloud | Cloud |
| Speed | Depends on hardware | Fast | Fast |
| Quality | Good (llama3.2) | Excellent (GPT-4) | Excellent (Claude 3.5) |
| Setup | Requires local install | API key only | API key only |
| Recommended For | Development, privacy | Production, best quality | Production, long context |

## Development Workflow

### Running Locally

**1. Backend**:
```bash
.venv/bin/python -m src.run_api
# Listens on http://localhost:8000
```

**2. Frontend**:
```bash
cd frontend
npm install  # first time only
npm run dev
# Listens on http://localhost:3000
```

**3. Access**:
- Web UI: http://localhost:3000
- API: http://localhost:8000
- Phoenix: http://localhost:6006

### Adding New Agent Roles

1. Define role in `role_library.py`:
```python
{
    "name": "translator",
    "description": "Translates text between languages",
    "system_prompt": "You are a professional translator...",
    "capabilities": ["translation", "localization"]
}
```

2. Update meta-coordinator prompt in `meta_coordinator.py`
3. Role automatically available in agent pool

### Monitoring and Debugging

**Phoenix Dashboard**:
- Visit http://localhost:6006
- View LLM calls, latencies, token usage
- Trace execution flow through agents
- Debug prompt/response issues

**Browser Console**:
- WebSocket message flow
- React component state
- Execution data preservation

**Terminal Logs**:
- Agent execution progress
- Layer-by-layer execution
- Timing information
- Error messages

## Performance Characteristics

### Simple Query ("Hi"):
- Agents: 1
- Layers: 1  
- Time: ~2-3 seconds (Ollama) / ~1-2 seconds (OpenAI/Claude)
- LLM Calls: 2 (meta-coordinator + analyzer)

### Medium Query ("Explain X"):
- Agents: 1
- Layers: 1
- Time: ~3-5 seconds (Ollama) / ~2-3 seconds (OpenAI/Claude)  
- LLM Calls: 2 (meta-coordinator + analyzer)

### Complex Query ("Compare X vs Y"):
- Agents: 2-4
- Layers: 2-3
- Time: ~10-20 seconds (Ollama) / ~5-10 seconds (OpenAI/Claude)
- LLM Calls: 4-6 (meta-coordinator + agents + synthesizer)

**Parallelization Benefit**: 
Without parallel execution, complex queries would take 2-3x longer as agents would run sequentially.

## Future Enhancements

### Potential Features
- [ ] Agent result caching for common queries
- [ ] Custom agent role creation via UI
- [ ] Execution graph visualization
- [ ] Export conversation as markdown/PDF
- [ ] Voice input/output
- [ ] Multi-user support with sessions
- [ ] Agent performance analytics
- [ ] Custom tool integration
- [ ] Streaming token-by-token output
- [ ] Mobile-responsive UI improvements

### Scalability Improvements
- [ ] Redis for distributed state management
- [ ] Agent execution rate limiting
- [ ] Query queue for high traffic
- [ ] Response caching layer
- [ ] Horizontal scaling with load balancer

## Technical Decisions

### Why LangGraph?
- Built-in state management
- Native checkpointing support
- Graph-based execution model perfect for DAGs
- Excellent observability integration
- Production-ready with error handling

### Why FastAPI?
- Native async/await support
- WebSocket support out of the box
- Automatic API documentation
- Fast performance
- Type hints and validation

### Why React + Vite?
- Fast development with HMR
- Modern build tooling
- Great developer experience
- Large ecosystem
- TypeScript support (can be added)

### Why WebSocket over HTTP polling?
- True real-time updates
- Lower latency
- Reduced server load
- Bidirectional communication
- Better user experience

## Troubleshooting

### Common Issues

**Frontend won't connect**:
- Check backend is running on port 8000
- Verify WebSocket URL in App.jsx
- Check browser console for errors

**Agents not executing in parallel**:
- Check LangGraph execution logs
- Verify dependencies in plan are correct
- Ensure MAX_PARALLEL_AGENTS > 1

**Slow execution with Ollama**:
- Ollama depends on local hardware
- Consider using smaller model (llama3.2:1b)
- Or switch to OpenAI/Claude for production

**WebSocket disconnects**:
- Check for network issues
- Verify no firewall blocking WebSocket
- Check server logs for errors

## Contributing

### Code Structure
```
/
├── src/                  # Backend Python code
│   ├── api.py           # FastAPI server
│   ├── meta_coordinator.py
│   ├── meta_agent_system.py
│   └── ...
├── frontend/            # React frontend
│   ├── src/
│   │   ├── App.jsx
│   │   └── index.css
│   └── package.json
├── docs/               # Documentation
├── requirements.txt    # Python dependencies
└── README.md          # Main readme
```

### Development Guidelines
1. Keep agents stateless and focused
2. Use proper error handling in WebSocket code
3. Add loading states for better UX
4. Log important events for debugging
5. Follow existing code style
6. Test with different LLM providers

## License

MIT License - See LICENSE file for details

## Acknowledgments

Built with:
- LangChain & LangGraph for agent orchestration
- FastAPI for high-performance web serving
- React for modern UI development
- Arize Phoenix for observability
- The open-source AI community

---

**Magentic** - Where AI agents organize themselves dynamically to solve your queries efficiently.
