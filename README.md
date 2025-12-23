# LangChain Agentic App (Production-Grade)

A production-ready LangChain application with an **agentic workflow** using a **local LLM (Ollama)** and complete observability via **Arize Phoenix**.

## Features

- 🤖 **Local LLM**: Uses Ollama with Llama 3.1 (no external API calls)
- 🔍 **Web Search**: DuckDuckGo integration for real-time information
- 📊 **Observability**: Full tracing with Arize Phoenix (local, no cloud)
- 📝 **Logging**: Comprehensive logging to file and console
- ⚙️ **Configuration**: Environment-based configuration
- 🛡️ **Error Handling**: Robust error handling and validation
- 🎯 **Interactive**: Chat-style CLI interface with commands

## Prerequisites

1.  **Ollama**: Download and install from [ollama.com](https://ollama.com)
2.  **Python 3.8+**: Ensure Python is installed

## Setup

1.  **Install Ollama and pull the model**:
    ```bash
    ollama pull llama3.1
    ```

2.  **Install Dependencies**:
    ```bash
    pip install langchain langchain-ollama langchain-community duckduckgo-search ddgs python-dotenv arize-phoenix openinference-instrumentation-langchain
    ```

3.  **Configure** (optional):
    Edit `.env` to customize settings:
    - `OLLAMA_MODEL`: Model name (default: llama3.1)
    - `OLLAMA_TEMPERATURE`: Model temperature (default: 0)
    - `PHOENIX_PORT`: Phoenix UI port (default: 6006)
    - `MAX_INPUT_LENGTH`: Maximum input length (default: 1000)
    - `SYSTEM_PROMPT`: Custom system prompt

## Running the Application

```bash
python app.py
```

## Usage

### Interactive Commands

- Type your question and press Enter
- `help` - Show help information
- `exit` or `quit` - Exit the application
- `Ctrl+C` - Graceful shutdown
- `Ctrl+D` - EOF exit

### Observability Dashboard

1.  When the app starts, you'll see: `Phoenix UI is running at http://localhost:6006`
2.  Open that URL in your browser
3.  Navigate to the **Traces** tab
4.  Watch real-time execution traces showing:
    - User inputs
    - Agent reasoning
    - Tool calls (searches)
    - Tool outputs
    - Final responses

### Logs

- Console logs: Real-time output with INFO level
- File logs: `agent.log` with detailed execution history

## Architecture

### Components

1.  **Agent**: LangGraph-based agent using `create_agent`
2.  **LLM**: ChatOllama connecting to local Ollama instance
3.  **Tools**: DuckDuckGo search for web queries
4.  **Observability**: OpenTelemetry + Phoenix for tracing
5.  **Logging**: Python logging to file and console

### Workflow

1.  User inputs query
2.  Agent analyzes query
3.  Agent decides if tools are needed
4.  Tools execute (e.g., web search)
5.  Agent synthesizes final answer
6.  All steps traced in Phoenix

## Production Features

### Error Handling
- Environment validation on startup
- Graceful shutdown on signals (SIGINT/SIGTERM)
- Input validation
- Comprehensive exception handling
- Detailed error logging

### Configuration Management
- Environment-based configuration
- Type-safe config class
- Sensible defaults

### Logging
- Dual logging (file + console)
- Structured log format
- Different log levels
- Rotating logs for production

### Monitoring
- Query counting
- Session statistics
- Full trace capture
- Error tracking

## Troubleshooting

### "Environment validation failed"
- Ensure Ollama is running: `ollama serve`
- Verify model is installed: `ollama list`
- Pull model if needed: `ollama pull llama3.1`

### "Phoenix setup failed"
- Port 6006 might be in use
- Change `PHOENIX_PORT` in `.env`
- Check firewall settings

### No traces in Phoenix
- Ensure Phoenix UI is open in browser
- Refresh the Traces tab
- Check console for errors

## Project Structure

```
test_langchain/
├── src/                      # Source code
│   ├── __init__.py          # Package initialization
│   ├── main.py              # Application entry point
│   ├── config.py            # Configuration management
│   ├── agent.py             # Agent creation and management
│   ├── observability.py     # Phoenix observability setup
│   ├── tools.py             # Tool definitions and management
│   └── cli.py               # Command-line interface
├── tests/                    # Test suite
│   ├── __init__.py
│   ├── test_config.py       # Configuration tests
│   ├── test_tools.py        # Tool tests
│   └── test_cli.py          # CLI tests
├── app.py                    # Main entry point script
├── requirements.txt          # Python dependencies
├── .env                      # Environment configuration
├── .gitignore               # Git ignore rules
├── agent.log                # Application logs (generated)
└── README.md                # This file
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_config.py -v
```

- `app.py`: The main application script with the agentic workflow.
- `.env`: Environment variables (API keys).
- `.venv/`: Python virtual environment.

## Observability (Local with Arize Phoenix)

To see exactly what the model is asking and the tools it calls **without sending data externally**, we use **Arize Phoenix**.

1.  **Install Dependencies**:
    ```bash
    pip install arize-phoenix openinference-instrumentation-langchain
    ```

2.  **Run the App**:
    ```bash
    python app.py
    ```

3.  **View Traces**:
    - The app will print a URL (usually `http://localhost:6006`).
    - Open this URL in your browser.
    - You will see a dashboard with all the traces, inputs, outputs, and tool calls.
