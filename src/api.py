"""FastAPI backend for Magentic UI with WebSocket support for real-time updates."""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .config import Config
from .tools import ToolManager
from .agents import MetaAgentSystem
from .agents.token_tracker import reset_tracker, get_tracker
from .langgraph_runner import LangGraphExecutor
from .database import (
    get_db,
    get_or_create_user,
    save_conversation,
    get_user_conversations,
    update_user_activity,
    UserProfile,
    create_user,
    authenticate_user,
    create_chat_session,
    get_chat_session,
    get_user_chat_sessions,
    update_chat_session_title,
    delete_chat_session,
    add_chat_message,
    get_chat_messages,
)
from .services.rag import RAGService
from .services.mcp_client import MCPClient

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

    # Initialize RAG service (optional)
    rag_service = None
    if config.enable_rag:
        try:
            rag_service = RAGService(
                persist_directory=config.rag_persist_directory,
                vector_store=config.rag_vector_store,
                qdrant_mode=config.rag_qdrant_mode,
                qdrant_url=config.rag_qdrant_url,
                qdrant_collection=config.rag_qdrant_collection,
                chunk_size=config.rag_chunk_size,
                chunk_overlap=config.rag_chunk_overlap,
                embedding_provider=config.rag_embedding_provider,
                embedding_model=config.rag_embedding_model,
                ollama_base_url=config.ollama_base_url,
            )
            logger.info("✓ RAG service initialized")
        except Exception as e:
            logger.warning(f"RAG service initialization failed: {e}")

    # Initialize MCP client (optional)
    mcp_client = None
    if config.enable_mcp:
        try:
            mcp_client = MCPClient(gateway_url=config.mcp_gateway_url)
            health = await mcp_client.health_check()
            if health.get("status") == "healthy":
                healthy_servers = health.get("healthy_servers", 0)
                total_servers = health.get("total_servers", 0)
                logger.info(
                    f"✓ MCP Gateway ready: {healthy_servers}/{total_servers} servers healthy"
                )
            else:
                logger.warning("MCP Gateway health check failed")
                mcp_client = None
        except Exception as e:
            logger.warning(f"MCP client initialization failed: {e}")
            mcp_client = None

    # Initialize tools with RAG and MCP support
    tool_manager = ToolManager(rag_service=rag_service, mcp_client=mcp_client)
    tools = await tool_manager.initialize_tools()
    logger.info(f"✓ Loaded {len(tools)} tools")

    # Initialize meta-agent system with RAG for active retrieval and tool manager
    meta_system = MetaAgentSystem(config, tools, rag_service=rag_service, tool_manager=tool_manager)
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
        "llm_provider": config.llm_provider if config else "unknown",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "llm_provider": config.llm_provider if config else "unknown",
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
            "total_agents_executed": user.total_agents_executed,
        },
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
            "avatar_emoji": user.avatar_emoji,  # type: ignore
        },
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
            "is_guest": user.is_guest,  # type: ignore
        },
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
                "execution_time": conv.execution_time,
            }
            for conv in conversations
        ],
    }


# ============== Chat Session Endpoints ==============


class CreateChatRequest(BaseModel):
    """Create chat session request."""

    username: str
    title: str = "New Chat"


class UpdateChatTitleRequest(BaseModel):
    """Update chat title request."""

    title: str


class AddMessageRequest(BaseModel):
    """Add message to chat request."""

    role: str  # 'user' or 'assistant'
    content: str
    execution_data: Optional[dict] = None


@app.post("/chats")
async def create_chat(request: CreateChatRequest, db: Session = Depends(get_db)):
    """Create a new chat session."""
    user = get_or_create_user(db, request.username)
    user_id = int(user.id)  # type: ignore[arg-type]
    session_id = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"

    session = create_chat_session(db, user_id, session_id, request.title)

    return {
        "success": True,
        "chat": {
            "id": session.session_id,
            "title": session.title,
            "createdAt": session.created_at.isoformat(),
            "updatedAt": session.updated_at.isoformat(),
            "messages": [],
        },
    }


@app.get("/chats/{username}")
async def get_user_chats(username: str, limit: int = 50, db: Session = Depends(get_db)):
    """Get all chat sessions for a user."""
    user = get_or_create_user(db, username)
    sessions = get_user_chat_sessions(db, int(user.id), limit)  # type: ignore[arg-type]

    return {
        "username": username,
        "total": len(sessions),
        "chats": [
            {
                "id": s.session_id,
                "title": s.title,
                "createdAt": s.created_at.isoformat(),
                "updatedAt": s.updated_at.isoformat(),
                "messageCount": len(s.messages),
            }
            for s in sessions
        ],
    }


@app.get("/chats/{username}/{session_id}")
async def get_chat(username: str, session_id: str, db: Session = Depends(get_db)):
    """Get a specific chat session with all messages."""
    session = get_chat_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Chat not found")

    messages = get_chat_messages(db, session_id)

    return {
        "id": session.session_id,
        "title": session.title,
        "createdAt": session.created_at.isoformat(),
        "updatedAt": session.updated_at.isoformat(),
        "messages": [
            {
                "id": str(m.id),
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat(),
                "executionData": m.execution_data,
            }
            for m in messages
        ],
    }


@app.patch("/chats/{session_id}")
async def update_chat(
    session_id: str, request: UpdateChatTitleRequest, db: Session = Depends(get_db)
):
    """Update chat session title."""
    session = update_chat_session_title(db, session_id, request.title)
    if not session:
        raise HTTPException(status_code=404, detail="Chat not found")

    return {"success": True, "title": session.title}


@app.delete("/chats/{session_id}")
async def delete_chat(session_id: str, db: Session = Depends(get_db)):
    """Delete a chat session."""
    success = delete_chat_session(db, session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Chat not found")

    return {"success": True}


@app.post("/chats/{session_id}/messages")
async def add_message(session_id: str, request: AddMessageRequest, db: Session = Depends(get_db)):
    """Add a message to a chat session."""
    message = add_chat_message(
        db, session_id, request.role, request.content, request.execution_data
    )
    if not message:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Get updated session for new title
    session = get_chat_session(db, session_id)

    return {
        "success": True,
        "message": {
            "id": str(message.id),
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp.isoformat(),
        },
        "chatTitle": session.title if session else "New Chat",
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
            session_id=result.get("session_id", ""),
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
                await websocket.send_json({"type": "error", "message": "Empty query"})
                continue

            # Send acknowledgment
            await websocket.send_json(
                {"type": "status", "message": "Processing query...", "stage": "received"}
            )

            try:
                # Process query with real-time updates
                await process_query_with_updates(websocket, query, username)
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})

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
        # Reset token tracker for this execution
        reset_tracker()

        # Send planning stage
        await websocket.send_json(
            {"type": "stage", "stage": "planning", "message": "AI Coordinator analyzing query..."}
        )

        # Create execution plan
        plan = meta_system.coordinator.create_execution_plan(query)

        # Compute layers for each agent
        layers = plan.get_execution_layers()
        agent_to_layer = {}
        for layer_idx, layer_agents in enumerate(layers):
            for agent_idx in layer_agents:
                agent_to_layer[agent_idx] = layer_idx

        # Convert plan to dict for database storage
        plan_dict = {
            "description": plan.description,
            "agents": plan.agents,
            "total_agents": len(plan.agents),
            "total_layers": len(layers),
        }

        # Build agents data with agent_ids
        agents_data = [
            {
                "agent_id": f"{agent.get('role')}_{idx}",
                "role": agent.get("role"),
                "task": agent.get("task"),
                "layer": agent_to_layer.get(idx, 0),
            }
            for idx, agent in enumerate(plan.agents)
        ]

        # Log agent IDs being sent
        logger.info(f"Plan agent IDs: {[a['agent_id'] for a in agents_data]}")

        # Send plan
        await websocket.send_json(
            {
                "type": "plan",
                "data": {
                    "description": plan.description,
                    "agents": agents_data,
                    "total_agents": len(plan.agents),
                    "total_layers": len(layers),
                },
            }
        )

        # Send execution stage
        await websocket.send_json(
            {
                "type": "stage",
                "stage": "executing",
                "message": f"Executing {len(plan.agents)} agents...",
            }
        )

        # Execute with custom callback for progress - pass the SAME plan
        result = await execute_with_progress(websocket, query, plan)

        # Wait to ensure all agent_complete events are sent before the final complete event
        await asyncio.sleep(0.2)

        final_output = result.get("final_output", "")
        session_id = result.get("session_id", "")

        # Save conversation to database (only for registered users, not guests)
        try:
            from .database import SessionLocal

            db = SessionLocal()
            try:
                user = get_or_create_user(db, username, is_guest=username.startswith("guest_"))

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

        # Send completion with token usage
        tracker = get_tracker()
        token_summary = tracker.get_summary()

        await websocket.send_json(
            {
                "type": "complete",
                "data": {
                    "output": final_output,
                    "session_id": session_id,
                    "execution_time": result.get("execution_time", 0),
                    "token_usage": token_summary,
                },
            }
        )

    except Exception as e:
        logger.error(f"Error in process_query_with_updates: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})


async def execute_with_progress(websocket: WebSocket, query: str, plan) -> Dict[str, Any]:
    """Execute query and send progress updates.

    Args:
        websocket: WebSocket connection for sending updates
        query: User query to process
        plan: ExecutionPlan to use (same plan sent to frontend)
    """

    # Monkey-patch the meta_system to send updates
    original_execute = meta_system.execute_agent_for_langgraph

    async def execute_with_notification(*args, **kwargs):
        agent_id = kwargs.get("agent_id", args[0] if args else "unknown")
        role = kwargs.get("role", args[1] if len(args) > 1 else "unknown")
        task = kwargs.get("task", args[2] if len(args) > 2 else "")
        context = kwargs.get("context", args[3] if len(args) > 3 else "")

        # Send agent start
        logger.info(f"Sending agent_start event for {agent_id} ({role})")
        try:
            await websocket.send_json(
                {
                    "type": "agent_start",
                    "data": {
                        "agent_id": agent_id,
                        "role": role,
                        "task": task,
                        "input": (
                            context[: config.ui_display_limit]
                            if context
                            else "(No previous agent outputs)"
                        ),
                    },
                }
            )
        except Exception as e:
            logger.error(f"Failed to send agent_start event: {e}")
            # Continue anyway

        # Execute agent
        try:
            result = await original_execute(*args, **kwargs)
        except Exception as e:
            logger.error(f"Agent {agent_id} execution failed: {e}", exc_info=True)
            # Send error event
            try:
                await websocket.send_json(
                    {
                        "type": "agent_complete",
                        "data": {
                            "agent_id": agent_id,
                            "role": role,
                            "input": (
                                context[: config.ui_display_limit]
                                if context
                                else "(No previous agent outputs)"
                            ),
                            "output": f"[ERROR: {str(e)[:config.ui_display_limit]}]",
                            "output_length": len(str(e)),
                            "tool_calls": [],
                            "error": True,
                        },
                    }
                )
            except Exception as ws_error:
                logger.error(f"Failed to send error event: {ws_error}")
            # Return error as dict to maintain consistency
            return {"content": f"[ERROR: {str(e)}]", "tool_calls": []}

        # Debug logging
        logger.info(f"Agent {agent_id} result type: {type(result)}")

        # Extract content and tool calls from result dict
        if isinstance(result, dict):
            output_str = result.get("content", str(result))
            tool_calls = result.get("tool_calls", [])
            logger.info(f"Agent {agent_id} has {len(tool_calls)} tool calls")
        else:
            # Fallback for string results
            output_str = str(result)
            tool_calls = []
            logger.warning(f"Agent {agent_id} returned non-dict result: {type(result)}")

        logger.info(f"Agent {agent_id} output: {output_str[:500]}")

        # Send agent complete with full output and token usage
        logger.info(
            f"Sending agent_complete event for {agent_id} with {len(tool_calls)} tool calls"
        )
        try:
            # Get token usage for this agent
            tracker = get_tracker()
            agent_tokens = tracker.get_agent_summary(agent_id)

            await websocket.send_json(
                {
                    "type": "agent_complete",
                    "data": {
                        "agent_id": agent_id,
                        "role": role,
                        "input": (
                            context[: config.ui_display_limit]
                            if context
                            else "(No previous agent outputs)"
                        ),
                        "output": output_str[: config.ui_display_limit],
                        "output_length": len(output_str),
                        "tool_calls": tool_calls,
                        "token_usage": agent_tokens,
                    },
                }
            )
            logger.info(f"✓ Successfully sent agent_complete for {agent_id}")
        except Exception as e:
            logger.error(f"Failed to send agent_complete event: {e}")
            # Continue anyway

        return result

    # Temporarily replace the method
    meta_system.execute_agent_for_langgraph = execute_with_notification

    try:
        # Pass the SAME plan to executor to ensure consistent agent IDs
        result = await executor.execute_query(query, plan=plan)
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
        "count": len(meta_system.conversation_history),
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
