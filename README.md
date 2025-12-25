# Magentic

**Magnetic Agent Networks** - AI-powered meta-agent system with dynamic topology generation and parallel execution.

![Magentic UI](assets/magentic.png)

## 🌟 New: Beautiful Web UI with Authentication

Magentic now includes a stunning React-based interface with user authentication and profiles!

### UI Features
- 💬 **Interactive Chat**: Clean, modern chat interface with gradient design
- 🔐 **User Authentication**: Login/Register system with bcrypt password hashing
- 👤 **User Profiles**: Personal profiles with avatars, stats, and conversation history
- 🎭 **Guest Mode**: Continue as guest without registration
- 💾 **Persistent History**: All conversations saved to SQLite database
- 🔴 **Live Progress**: Animated rotating indicator showing real-time execution status
- 📊 **Expandable Steps**: Click any agent to see detailed execution info and timing
- ⚡ **Real-time Updates**: WebSocket-based live progress streaming
- 🎨 **Markdown Support**: Beautiful rendering of formatted text, code blocks with syntax highlighting
- 👁️ **Toggle Details**: Show/hide execution details to focus on answers
- ✨ **Beautiful Design**: Purple gradient UI with Magentic branding

### Tech Stack
**Frontend:**
- React 18 + Vite for fast development
- Tailwind CSS with custom animations
- WebSocket for real-time updates
- react-markdown with GitHub-flavored markdown
- Lucide React for icons

**Backend:**
- FastAPI with WebSocket support
- SQLAlchemy ORM with SQLite database
- Alembic for database migrations
- Passlib + bcrypt for password security
- LangGraph for agent orchestration

**Database:**
- SQLite with proper connection pooling
- User profiles with authentication
- Conversation history with full metadata
- Automatic schema migrations with Alembic

## Features

- 🎯 **Dynamic Planning**: AI analyzes query complexity and generates optimal agent topology
- ⚡ **Parallel Execution**: DAG-based layer execution with LangGraph for maximum efficiency
- � **Layer Synchronization**: Barrier nodes ensure all agents in a layer complete before next layer starts
- 💾 **State Management**: Checkpointing and crash recovery with resumable execution
- 🗨️ **Conversation History**: Each agent receives context from previous conversation steps
- 🔍 **Web Search**: DuckDuckGo integration for research agents to fetch current information
- 📊 **Observability**: Phoenix dashboard for real-time LLM tracing and debugging
- 🐛 **Debug Mode**: Optional state visualization showing agent outputs and execution flow
- 🎨 **8 Specialized Roles**: Researcher, Analyzer, Planner, Writer, Coder, Critic, Synthesizer, Coordinator
- 🖥️ **Modern Web UI**: React-based interface with WebSocket streaming and markdown rendering
- 🧠 **Smart Scaling**: Automatically uses minimal agents for simple queries, scales up for complex tasks
- 🔄 **Conversation Memory**: Maintains context across multiple queries
- 🎭 **Multi-LLM Support**: Works with Ollama (local), OpenAI, and Claude
- ⚙️ **Configurable Output**: Set UI display limits to control agent response length
- � **Authentication**: Secure login/register with bcrypt password hashing
- 👤 **User Profiles**: Personal profiles with avatars, stats, and persistent conversation history
- 💾 **SQLite Database**: Local database with proper migrations for profile and history storage

## Quick Start

### Automated Setup (Recommended)

Run the setup script to install everything automatically:

```bash
./setup.sh
```

This will:
- Create virtual environment
- Install all Python dependencies
- Set up the database with migrations
- Install frontend dependencies
- Create .env file from template

After setup completes, configure your LLM in `.env` and start the servers.

### Manual Setup

<details>
<summary>Click to expand manual setup instructions</summary>

#### 1. Install Dependencies

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

#### 2. Setup Database

```bash
# Run migrations to create database
alembic upgrade head

# Optional: Create a demo user (username: demo, password: demo123)
python init_db.py --demo
```

#### 3. Configure LLM

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

**LLM Options:**
- **Ollama** (default, free, private) - Install from https://ollama.com then run `ollama pull llama3.2:1b`
- **OpenAI** - Get API key from https://platform.openai.com
- **Claude** - Get API key from https://console.anthropic.com

</details>

### Start the Application

**Terminal 1 - Backend:**
```bash
python -m src.run_api
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

**Open Browser:** `http://localhost:3000`

On first visit, you'll see a login modal with options to:
- **Register** - Create a new account
- **Login** - Use existing account
- **Continue as Guest** - No registration needed

### Database Management

```bash
# Initialize database
alembic upgrade head

# Create demo user (username: demo, password: demo123)
python init_db.py --demo

# Reset database (⚠️ DELETES ALL DATA!)
python reset_db.py              # With confirmation prompt
python reset_db.py --demo       # Reset and create demo user
python reset_db.py --confirm    # Skip confirmation

# Check database status
alembic current

# Create new migration after model changes
alembic revision --autogenerate -m "Description"
```

## CLI Mode

For command-line usage without the web UI:

```bash
python app.py
```

## Example Queries

Try these to see the agent system in action:

- **Simple**: "What is Python?" (uses 1 agent)
- **Medium**: "Explain how neural networks work" (uses 1-2 agents)
- **Complex**: "Compare React vs Vue and recommend which to use for a large e-commerce site" (uses multiple agents with synthesis)

## Features in Detail

### 🎯 Dynamic Agent Planning

The meta-coordinator analyzes your query and creates an optimal agent network:
- **Simple queries** → Single agent
- **Complex queries** → Multiple specialized agents in parallel
- **Research tasks** → Includes web search capabilities

### 🔐 Authentication & Profiles

- Register with just username/password (no email needed)
- Guest mode for trying without account
- Personal conversation history
- User statistics (queries, agents used)
- Avatar customization

### 📊 Real-time Visualization

Watch your query being processed in real-time:
- Live agent execution status
- Expandable agent details
- Execution timing
- Beautiful markdown rendering with syntax highlighting

## Tech Stack

**Backend:**
- **FastAPI** - Modern async web framework
- **SQLAlchemy** - Database ORM with migrations
- **LangGraph** - Agent orchestration
- **Passlib + bcrypt** - Secure password hashing
- **WebSockets** - Real-time updates

**Frontend:**
- **React 18** - Modern UI library
- **Vite** - Lightning-fast build tool with HMR
- **Tailwind CSS** - Utility-first styling
- **react-markdown** - Markdown rendering with GitHub-flavored markdown
- **rehype-highlight** - Syntax highlighting for code blocks
- **Lucide React** - Beautiful icon library

**LLM Support:**
- **Ollama** - Local, private, free
- **OpenAI** - GPT-4, GPT-3.5
- **Claude** - Claude 3.5 Sonnet, Opus

## Documentation

- [Authentication Guide](docs/AUTHENTICATION.md) - User registration, login, and profiles
- [Architecture Guide](docs/ARCHITECTURE.md) - Detailed system architecture
- [Project Overview](docs/PROJECT_OVERVIEW.md) - Comprehensive project documentation
- [Web UI Guide](frontend/README.md) - Frontend documentation
- [Database Migrations](alembic/README.md) - Alembic migration guide

## Advanced Configuration

Edit `.env` file for custom settings:

**System:**
- `MAX_PARALLEL_AGENTS` - Concurrent agents (2-5 recommended)
- `LLM_TEMPERATURE` - 0.0 (precise) to 2.0 (creative)

**Ollama Settings:**
- `OLLAMA_BASE_URL` - Default: `http://localhost:11434`
- `OLLAMA_MODEL` - Models: `llama3.2:1b`, `llama3.1`, `mistral`

**OpenAI Settings:**
- `OPENAI_MODEL` - Models: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`

**Claude Settings:**
- `ANTHROPIC_MODEL` - Models: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`

**Debug & Display:**
- `DEBUG_STATE` - Set to `true` to enable state visualization showing agent outputs (default: false)
- `UI_DISPLAY_LIMIT` - Character limit for agent output display (default: 200, minimum: 50)

## Phoenix Dashboard (Optional)

For real-time LLM tracing and debugging, Phoenix is automatically available at:
- **URL**: http://localhost:6006
- **Features**: Request tracing, token usage, performance metrics

## License

MIT License - see [LICENSE](LICENSE)
