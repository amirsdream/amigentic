"""Database models and setup for user profiles and conversation history."""

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from passlib.context import CryptContext
from pathlib import Path
from typing import Optional

# Suppress bcrypt version warning from passlib
import warnings
warnings.filterwarnings("ignore", message=".*bcrypt.*")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Database setup
DB_PATH = Path(__file__).parent.parent / "data" / "magentic.db"
DB_PATH.parent.mkdir(exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
    pool_recycle=3600
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


class Conversation(Base):
    """Conversation history per user."""
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
        user = UserProfile(
            username=username, 
            display_name=username,
            is_guest=is_guest
        )
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
        is_guest=False
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user, None


def authenticate_user(db, username: str, password: str) -> tuple[Optional[UserProfile], Optional[str]]:
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


def save_conversation(db, user_id: int, query: str, response: str, 
                     execution_plan: Optional[dict] = None, session_id: Optional[str] = None):
    """Save conversation to database."""
    conversation = Conversation(
        user_id=user_id,
        session_id=session_id,
        query=query,
        response=response,
        execution_plan=execution_plan,
        agents_used=len(execution_plan.get("agents", [])) if execution_plan else 0
    )
    db.add(conversation)
    
    # Update user stats
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if user:
        user.total_queries += 1
        if execution_plan:
            user.total_agents_executed += len(execution_plan.get("agents", []))
    
    db.commit()
    return conversation


def get_user_conversations(db, user_id: int, limit: int = 50):
    """Get recent conversations for a user."""
    return db.query(Conversation)\
        .filter(Conversation.user_id == user_id)\
        .order_by(Conversation.timestamp.desc())\
        .limit(limit)\
        .all()
