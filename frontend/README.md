# Magentic UI

React frontend for the Magentic multi-agent system.

## Quick Start

From project root:
```bash
./magentic.sh start
```

Or manually:
```bash
cd frontend
npm install
npm run dev
```

Opens at http://localhost:3000

## Features

- Real-time WebSocket updates
- Agent execution visualization
- Expandable step details with token usage
- User auth (login/register/guest)
- Chat history persistence
- Markdown rendering with syntax highlighting

## Tech Stack

- **React 18** + Vite
- **Tailwind CSS** with custom animations
- **Zustand** for state management
- **Framer Motion** for animations
- **react-markdown** + rehype-highlight

## Structure

```
src/
├── components/     # UI components
│   ├── Chat.jsx
│   ├── AgentStep.jsx
│   ├── Sidebar.jsx
│   └── ...
├── hooks/          # Custom hooks
│   └── useWebSocket.js
├── store/          # Zustand stores
│   └── conversationStore.js
└── App.jsx
```

## Environment

Create `.env`:
```bash
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
```
