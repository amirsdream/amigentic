"""
Run the Magentic API server.

Usage:
    python -m src.run_api
"""

import uvicorn
from src.api import app

if __name__ == "__main__":
    print("🚀 Starting Magentic API server...")
    print("📡 WebSocket endpoint: ws://localhost:8000/ws")
    print("📊 Health check: http://localhost:8000/health")
    print("")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
