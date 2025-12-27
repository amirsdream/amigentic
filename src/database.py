"""Database models and setup for user profiles and conversation history."""

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Text,
    JSON,
    Boolean,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import warnings
import logging

# Suppress bcrypt version warning from passlib
warnings.filterwarnings("ignore", message=".*bcrypt.*")
logging.getLogger("passlib").setLevel(logging.ERROR)

# Fix passlib + bcrypt compatibility issue with Python 3.13
# Patch bcrypt's __about__ module if needed
try:
    import bcrypt

    if not hasattr(bcrypt, "__about__"):

        class _About:
            __version__ = getattr(bcrypt, "__version__", "4.0.0")

        bcrypt.__about__ = _About()  # type: ignore[attr-defined]
except ImportError:
    pass

from passlib.context import CryptContext

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Database setup
DB_PATH = Path(__file__).parent.parent / "data" / "magentic.db"
DB_PATH.parent.mkdir(exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}, pool_pre_ping=True, pool_recycle=3600
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


class UserProfile(Base):
    """User profile with preferences and settings."""

    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=True)  # Null for guest users
    is_guest = Column(Boolean, default=False)
    display_name = Column(String, nullable=True)
    avatar_emoji = Column(String, default="👤")
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)

    # Preferences
    theme = Column(String, default="dark")
    show_execution_details = Column(Integer, default=1)  # 1=true, 0=false
    preferred_llm = Column(String, nullable=True)

    # Stats
    total_queries = Column(Integer, default=0)
    total_agents_executed = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)


class ChatSession(Base):
    """Chat session to group conversations."""

    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("user_profiles.id"), index=True, nullable=False)
    title = Column(String, default="New Chat")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
    """Individual chat messages within a session."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), index=True, nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Execution metadata (for assistant messages)
    execution_data = Column(JSON, nullable=True)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")


class Conversation(Base):
    """Conversation history per user (legacy - kept for backward compatibility)."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True, nullable=False)
    session_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Message data
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)

    # Execution metadata
    execution_plan = Column(JSON, nullable=True)  # Store plan as JSON
    agents_used = Column(Integer, default=0)
    execution_time = Column(Integer, default=0)  # milliseconds


# Tables are created via Alembic migrations
# Do not create tables here - use: alembic upgrade head


def run_migrations():
    """
    Run database migrations on startup.
    This ensures the database schema is up-to-date after checkout.
    """
    from sqlalchemy import inspect, text
    
    logger = logging.getLogger(__name__)
    
    # Create tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    # Get existing columns for user_profiles table
    inspector = inspect(engine)
    
    # Define migrations: (table_name, column_name, column_type, default)
    migrations = [
        ("user_profiles", "total_tokens_used", "INTEGER", "0"),
        ("user_profiles", "total_cost", "REAL", "0.0"),
    ]
    
    with engine.connect() as conn:
        for table_name, column_name, column_type, default in migrations:
            try:
                existing_columns = [col["name"] for col in inspector.get_columns(table_name)]
                
                if column_name not in existing_columns:
                    logger.info(f"Migration: Adding column {column_name} to {table_name}")
                    conn.execute(text(
                        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type} DEFAULT {default}"
                    ))
                    conn.commit()
            except Exception as e:
                logger.warning(f"Migration warning for {table_name}.{column_name}: {e}")


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_or_create_user(db, username: str, is_guest: bool = True) -> UserProfile:
    """Get existing user or create new one."""
    user = db.query(UserProfile).filter(UserProfile.username == username).first()
    if not user:
        user = UserProfile(username=username, display_name=username, is_guest=is_guest)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def create_user(db, username: str, password: str) -> tuple[Optional[UserProfile], Optional[str]]:
    """Create a new registered user."""
    # Check if user exists
    existing = db.query(UserProfile).filter(UserProfile.username == username).first()
    if existing:
        return None, "Username already exists"

    # Create user
    user = UserProfile(
        username=username,
        password_hash=get_password_hash(password),
        display_name=username,
        is_guest=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user, None


def authenticate_user(
    db, username: str, password: str
) -> tuple[Optional[UserProfile], Optional[str]]:
    """Authenticate a user."""
    user = db.query(UserProfile).filter(UserProfile.username == username).first()

    if not user:
        return None, "User not found"

    if user.is_guest:
        return None, "Cannot login to guest account"

    if not verify_password(password, user.password_hash):
        return None, "Incorrect password"

    # Update last active
    user.last_active = datetime.utcnow()
    db.commit()

    return user, None


def update_user_activity(db, user_id: int):
    """Update user's last active timestamp."""
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if user:
        user.last_active = datetime.utcnow()
        db.commit()


def save_conversation(
    db,
    user_id: int,
    query: str,
    response: str,
    execution_plan: Optional[dict] = None,
    session_id: Optional[str] = None,
    token_usage: Optional[dict] = None,
):
    """Save conversation to database."""
    conversation = Conversation(
        user_id=user_id,
        session_id=session_id,
        query=query,
        response=response,
        execution_plan=execution_plan,
        agents_used=len(execution_plan.get("agents", [])) if execution_plan else 0,
    )
    db.add(conversation)

    # Update user stats
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if user:
        user.total_queries += 1
        if execution_plan:
            user.total_agents_executed += len(execution_plan.get("agents", []))
        
        # Update token and cost stats
        if token_usage:
            total_tokens = token_usage.get("total_tokens", 0)
            total_cost = token_usage.get("total_cost", 0.0)
            user.total_tokens_used = (user.total_tokens_used or 0) + total_tokens
            user.total_cost = (user.total_cost or 0.0) + total_cost

    db.commit()
    return conversation


def get_user_conversations(db, user_id: int, limit: int = 50):
    """Get recent conversations for a user."""
    return (
        db.query(Conversation)
        .filter(Conversation.user_id == user_id)
        .order_by(Conversation.timestamp.desc())
        .limit(limit)
        .all()
    )


# ============== Chat Session Functions ==============


def create_chat_session(db, user_id: int, session_id: str, title: str = "New Chat") -> ChatSession:
    """Create a new chat session."""
    session = ChatSession(session_id=session_id, user_id=user_id, title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_chat_session(db, session_id: str) -> Optional[ChatSession]:
    """Get a chat session by session_id."""
    return db.query(ChatSession).filter(ChatSession.session_id == session_id).first()


def get_user_chat_sessions(db, user_id: int, limit: int = 50) -> List[ChatSession]:
    """Get all chat sessions for a user."""
    return (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
        .limit(limit)
        .all()
    )


def update_chat_session_title(db, session_id: str, title: str) -> Optional[ChatSession]:
    """Update the title of a chat session."""
    session = get_chat_session(db, session_id)
    if session:
        session.title = title
        session.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(session)
    return session


def delete_chat_session(db, session_id: str) -> bool:
    """Delete a chat session and all its messages."""
    session = get_chat_session(db, session_id)
    if session:
        db.delete(session)
        db.commit()
        return True
    return False


def add_chat_message(
    db, session_id: str, role: str, content: str, execution_data: Optional[dict] = None
) -> Optional[ChatMessage]:
    """Add a message to a chat session."""
    session = get_chat_session(db, session_id)
    if not session:
        return None

    message = ChatMessage(
        session_id=session.id, role=role, content=content, execution_data=execution_data
    )
    db.add(message)

    # Update session timestamp
    session.updated_at = datetime.utcnow()

    # Update title if this is the first user message
    if role == "user" and session.title == "New Chat":
        # Use first 50 chars of first message as title
        session.title = content[:50] + ("..." if len(content) > 50 else "")

    db.commit()
    db.refresh(message)
    return message


def get_chat_messages(db, session_id: str) -> List[ChatMessage]:
    """Get all messages for a chat session."""
    session = get_chat_session(db, session_id)
    if not session:
        return []

    return (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.timestamp.asc())
        .all()
    )
