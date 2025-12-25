# Magentic UI

Beautiful React frontend for the Magentic multi-agent system.

## Features

- 🎨 Modern, responsive UI with Tailwind CSS
- 🔴 Animated red ball indicator for active processing
- 📊 Expandable execution steps
- 💬 Real-time chat interface
- ⚡ WebSocket-based live updates
- 🎯 Agent execution visualization
- ✨ Magentic branding with gradient icons

## Quick Start

### 🚀 One Command (Recommended)

From the project root:
```bash
python start_magentic.py
```

Or:
```bash
./start_magentic.sh
```

This automatically:
- Installs dependencies if needed
- Starts backend API (port 8000)
- Starts frontend UI (port 3000)
- Handles both processes

Then open `http://localhost:3000` in your browser!

### Manual Setup (Alternative)

If you prefer to run components separately:

**1. Install Dependencies:**
```bash
cd frontend
npm install
cd ..
pip install fastapi uvicorn websockets
```

**2. Start Backend (Terminal 1):**
```bash
python -m src.run_api
```

**3. Start Frontend (Terminal 2):**
```bash
cd frontend
npm run dev
```

**4. Open Browser:**
Navigate to `http://localhost:3000`

## Usage

1. Open the UI in your browser
2. Type your query in the chat input
3. Watch as the AI Coordinator creates an execution plan
4. See the red rotating ball indicator while processing
5. Click on any agent step to expand and see details
6. View the final result in the chat

## Architecture

### Backend (`src/api.py`)
- FastAPI server with WebSocket support
- Real-time progress updates
- RESTful endpoints for health checks and memory management

### Frontend (`frontend/src/`)
- React with Vite for fast development
- Tailwind CSS for styling
- WebSocket client for real-time updates
- Lucide React icons

### Real-time Communication Flow

```
User Query → WebSocket → Backend Processing → Progress Updates → UI Visualization
```

Events sent via WebSocket:
- `status`: Initial acknowledgment
- `stage`: Planning/executing stages
- `plan`: Execution plan details
- `agent_start`: Agent begins work
- `agent_complete`: Agent finishes
- `complete`: Final result
- `error`: Error messages

## Customization

### Colors
Edit `frontend/tailwind.config.js` to customize the color scheme:

```javascript
colors: {
  'magentic': {
    primary: '#8B5CF6',  // Purple
    secondary: '#EC4899', // Pink
    dark: '#1F2937',
  }
}
```

### Animations
The red ball animation is defined in `frontend/src/index.css`:

```css
@keyframes pulse-red {
  0%, 100% {
    opacity: 1;
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.6);
  }
  50% {
    opacity: 0.8;
    box-shadow: 0 0 30px rgba(239, 68, 68, 0.8);
  }
}
```

## Development

### Backend Development
The API uses FastAPI with auto-reload enabled. Changes to `src/api.py` will automatically restart the server.

### Frontend Development
Vite provides hot module replacement. Changes to React components will update instantly in the browser.

## Production Build

### Frontend
```bash
cd frontend
npm run build
```

The production build will be in `frontend/dist/`

### Backend
Run with production settings:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Or use gunicorn for production:
```bash
gunicorn src.api:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Troubleshooting

### WebSocket Connection Failed
- Ensure the backend is running on port 8000
- Check CORS settings in `src/api.py`
- Verify no firewall blocking WebSocket connections

### Frontend Not Loading
- Clear browser cache
- Check console for errors
- Ensure Node.js version is 18+

### Port Already in Use
Backend (8000):
```bash
lsof -ti:8000 | xargs kill -9
```

Frontend (3000):
```bash
lsof -ti:3000 | xargs kill -9
```
