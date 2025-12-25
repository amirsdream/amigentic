"""FastAPI backend for Magentic UI with WebSocket support for real-time updates."""

import asyncio
import logging
from typing import Dict, Any, List
from datetime import datetime
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .config import Config
from .tools import ToolManager
from .meta_agent_system import MetaAgentSystem
from .langgraph_executor import LangGraphExecutor
from .database import (
    get_db, get_or_create_user, save_conversation, get_user_conversations, 
    update_user_activity, UserProfile, create_user, authenticate_user
)

logger = logging.getLogger(__name__)


class RegisterRequest(BaseModel):
    """Register request model."""
    username: str
    password: str


class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str

app = FastAPI(title="Magentic API", version="2.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
config: Config = None  # type: ignore
meta_system: MetaAgentSystem = None  # type: ignore
executor: LangGraphExecutor = None  # type: ignore

# Active WebSocket connections
active_connections: List[WebSocket] = []


class QueryRequest(BaseModel):
    """Query request model."""
    query: str


class QueryResponse(BaseModel):
    """Query response model."""
    success: bool
    message: str
    session_id: str = ""


@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup."""
    global config, meta_system, executor
    
    logger.info("🚀 Starting Magentic API...")
    
    # Load configuration
    config = Config()
    is_valid, error_msg = config.validate()
    if not is_valid:
        raise RuntimeError(f"Invalid configuration: {error_msg}")
    
    # Initialize tools
    tool_manager = ToolManager()
    tools = tool_manager.initialize_tools()
    logger.info(f"✓ Loaded {len(tools)} tools")
    
    # Initialize meta-agent system
    meta_system = MetaAgentSystem(config, tools)
    logger.info("✓ Meta-agent system initialized")
    
    # Initialize executor
    executor = LangGraphExecutor(meta_system)
    logger.info("✓ LangGraph executor ready")
    
    # Warm up
    meta_system.coordinator.warmup()
    logger.info("✓ System warmed up")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Magentic API",
        "version": "2.0.0",
        "status": "ready",
        "llm_provider": config.llm_provider if config else "unknown"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "llm_provider": config.llm_provider if config else "unknown"
    }


@app.get("/profile/{username}")
async def get_profile(username: str, db: Session = Depends(get_db)):
    """Get user profile."""
    user = get_or_create_user(db, username, is_guest=True)
    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
        "avatar_emoji": user.avatar_emoji,
        "is_guest": user.is_guest,
        "created_at": user.created_at.isoformat(),
        "last_active": user.last_active.isoformat(),
        "theme": user.theme,
        "show_execution_details": bool(user.show_execution_details),
        "stats": {
            "total_queries": user.total_queries,
            "total_agents_executed": user.total_agents_executed
        }
    }


@app.post("/register")
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new user."""
    if len(request.username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    
    if len(request.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    
    user, error = create_user(db, request.username, request.password)
    
    if error or not user:
        raise HTTPException(status_code=400, detail=error or "Failed to create user")
    
    return {
        "success": True,
        "message": "User registered successfully",
        "user": {
            "id": user.id,  # type: ignore
            "username": user.username,  # type: ignore
            "display_name": user.display_name,  # type: ignore
            "avatar_emoji": user.avatar_emoji  # type: ignore
        }
    }


@app.post("/login")
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Login user."""
    user, error = authenticate_user(db, request.username, request.password)
    
    if error or not user:
        raise HTTPException(status_code=401, detail=error or "Authentication failed")
    
    return {
        "success": True,
        "message": "Login successful",
        "user": {
            "id": user.id,  # type: ignore
            "username": user.username,  # type: ignore
            "display_name": user.display_name,  # type: ignore
            "avatar_emoji": user.avatar_emoji,  # type: ignore
            "is_guest": user.is_guest  # type: ignore
        }
    }


@app.put("/profile/{username}")
async def update_profile(username: str, updates: dict, db: Session = Depends(get_db)):
    """Update user profile."""
    user = get_or_create_user(db, username)
    
    if "display_name" in updates:
        user.display_name = updates["display_name"]
    if "avatar_emoji" in updates:
        user.avatar_emoji = updates["avatar_emoji"]
    if "theme" in updates:
        user.theme = updates["theme"]
    if "show_execution_details" in updates:
        setattr(user, "show_execution_details", 1 if updates["show_execution_details"] else 0)
    
    update_user_activity(db, user.id)  # type: ignore
    
    return {"success": True, "message": "Profile updated"}


@app.get("/history/{username}")
async def get_history(username: str, limit: int = 50, db: Session = Depends(get_db)):
    """Get conversation history for user."""
    user = get_or_create_user(db, username)
    conversations = get_user_conversations(db, user.id, limit)  # type: ignore
    
    return {
        "username": username,
        "total": len(conversations),
        "conversations": [
            {
                "id": conv.id,
                "timestamp": conv.timestamp.isoformat(),
                "query": conv.query,
                "response": conv.response,
                "agents_used": conv.agents_used,
                "execution_time": conv.execution_time
            }
            for conv in conversations
        ]
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query (non-streaming endpoint)."""
    try:
        if not executor:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        result = await executor.execute_query(request.query)
        
        return QueryResponse(
            success=True,
            message=result.get("final_output", "No output generated"),
            session_id=result.get("session_id", "")
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time query processing."""
    await websocket.accept()
    active_connections.append(websocket)
    
    # Get username from query params or default to guest
    username = websocket.query_params.get("username", "guest")
    
    try:
        while True:
            # Receive query from client
            data = await websocket.receive_text()
            query_data = json.loads(data)
            query = query_data.get("query", "")
            
            if not query:
                await websocket.send_json({
                    "type": "error",
                    "message": "Empty query"
                })
                continue
            
            # Send acknowledgment
            await websocket.send_json({
                "type": "status",
                "message": "Processing query...",
                "stage": "received"
            })
            
            try:
                # Process query with real-time updates
                await process_query_with_updates(websocket, query, username)
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
    
    except WebSocketDisconnect:
        active_connections.remove(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


async def process_query_with_updates(websocket: WebSocket, query: str, username: str = "guest"):
    """Process query and send real-time updates via WebSocket."""
    try:
        # Send planning stage
        await websocket.send_json({
            "type": "stage",
            "stage": "planning",
            "message": "AI Coordinator analyzing query..."
        })
        
        # Create execution plan
        plan = meta_system.coordinator.create_execution_plan(query)
        
        # Convert plan to dict for database storage
        plan_dict = {
            "description": plan.description,
            "agents": plan.agents,
            "total_agents": len(plan.agents),
            "total_layers": len(plan.get_execution_layers())
        }
        
        # Send plan
        await websocket.send_json({
            "type": "plan",
            "data": {
                "description": plan.description,
                "agents": [
                    {
                        "agent_id": f"{agent.get('role')}_{idx}",
                        "role": agent.get("role"),
                        "task": agent.get("task"),
                        "layer": agent.get("layer", 0)
                    }
                    for idx, agent in enumerate(plan.agents)
                ],
                "total_agents": len(plan.agents),
                "total_layers": len(plan.get_execution_layers())
            }
        })
        
        # Send execution stage
        await websocket.send_json({
            "type": "stage",
            "stage": "executing",
            "message": f"Executing {len(plan.agents)} agents..."
        })
        
        # Execute with custom callback for progress
        result = await execute_with_progress(websocket, query)
        
        final_output = result.get("final_output", "")
        session_id = result.get("session_id", "")
        
        # Save conversation to database (only for registered users, not guests)
        try:
            from .database import SessionLocal
            db = SessionLocal()
            try:
                user = get_or_create_user(db, username, is_guest=username.startswith('guest_'))
                
                # Only save conversations for registered users
                if not user.is_guest:  # type: ignore
                    save_conversation(db, user.id, query, final_output, plan_dict, session_id)  # type: ignore
                    logger.info(f"Saved conversation for user {username}")
                else:
                    logger.info(f"Skipped saving conversation for guest user {username}")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
        
        # Send completion
        await websocket.send_json({
            "type": "complete",
            "data": {
                "output": final_output,
                "session_id": session_id,
                "execution_time": result.get("execution_time", 0)
            }
        })
        
    except Exception as e:
        logger.error(f"Error in process_query_with_updates: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })


async def execute_with_progress(websocket: WebSocket, query: str) -> Dict[str, Any]:
    """Execute query and send progress updates."""
    
    # Monkey-patch the meta_system to send updates
    original_execute = meta_system.execute_agent_for_langgraph
    
    async def execute_with_notification(*args, **kwargs):
        agent_id = kwargs.get("agent_id", args[0] if args else "unknown")
        role = kwargs.get("role", args[1] if len(args) > 1 else "unknown")
        task = kwargs.get("task", args[2] if len(args) > 2 else "")
        
        # Send agent start
        await websocket.send_json({
            "type": "agent_start",
            "data": {
                "agent_id": agent_id,
                "role": role,
                "task": task
            }
        })
        
        # Execute agent
        result = await original_execute(*args, **kwargs)
        
        # Extract tool calls if available
        tool_calls: List[Any] = []
        try:
            if hasattr(result, 'tool_calls') and result.tool_calls:  # type: ignore
                tool_calls = result.tool_calls  # type: ignore
            elif isinstance(result, dict):
                if 'tool_calls' in result:
                    tool_calls = result['tool_calls']  # type: ignore
        except Exception:
            # Ignore errors extracting tool calls
            pass
        
        # Send agent complete with full output
        await websocket.send_json({
            "type": "agent_complete",
            "data": {
                "agent_id": agent_id,
                "role": role,
                "output": str(result)[:1000],  # Limit to first 1000 chars
                "output_length": len(str(result)),
                "tool_calls": tool_calls
            }
        })
        
        return result
    
    # Temporarily replace the method
    meta_system.execute_agent_for_langgraph = execute_with_notification
    
    try:
        result = await executor.execute_query(query)
        return result
    finally:
        # Restore original method
        meta_system.execute_agent_for_langgraph = original_execute


@app.get("/memory")
async def get_memory():
    """Get conversation memory."""
    if not meta_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "history": meta_system.conversation_history,
        "count": len(meta_system.conversation_history)
    }


@app.post("/memory/clear")
async def clear_memory():
    """Clear conversation memory."""
    if not meta_system:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    meta_system.clear_memory()
    return {"message": "Memory cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
