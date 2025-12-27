"""
FastAPI Users configuration - Clean, minimal auth setup.
Uses battle-tested fastapi-users library instead of custom code.
"""

from typing import AsyncGenerator, Optional
from datetime import datetime
import uuid

from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers, IntegerIDMixin
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy,
)
from fastapi_users.db import SQLAlchemyUserDatabase
from fastapi_users_db_sqlalchemy import SQLAlchemyBaseUserTable
from sqlalchemy import Column, String, Integer, DateTime, Boolean, Float
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from pydantic import BaseModel
from pathlib import Path
import secrets

# ============== Configuration ==============

import os

# JWT Secret: Use environment variable, or generate a random one for development
# In production, ALWAYS set JWT_SECRET environment variable!
_jwt_secret_env = os.environ.get("JWT_SECRET")
if not _jwt_secret_env:
    import warnings
    _jwt_secret_env = secrets.token_hex(32)
    warnings.warn(
        "JWT_SECRET not set in environment. Using random secret. "
        "Sessions will be invalidated on restart. Set JWT_SECRET for production!",
        RuntimeWarning
    )
JWT_SECRET: str = _jwt_secret_env

# Database setup
DB_PATH = Path(__file__).parent.parent.parent / "data" / "magentic.db"
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


# ============== Database Models ==============

class Base(DeclarativeBase):
    pass


class User(SQLAlchemyBaseUserTable[int], Base):
    """User model with custom fields."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)  # type: ignore
    
    # Custom fields
    display_name = Column(String(100), nullable=True)
    avatar_emoji = Column(String(10), default="👤")
    theme = Column(String(20), default="dark")
    is_guest = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Stats
    total_queries = Column(Integer, default=0)
    total_agents_executed = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)


# ============== Pydantic Schemas ==============

from fastapi_users import schemas


class UserRead(schemas.BaseUser[int]):
    """User data returned to client."""
    display_name: Optional[str] = None
    avatar_emoji: str = "👤"
    theme: str = "dark"
    is_guest: bool = False
    total_queries: int = 0
    total_agents_executed: int = 0
    total_tokens_used: int = 0
    total_cost: float = 0.0


class UserCreate(schemas.BaseUserCreate):
    """User registration data."""
    display_name: Optional[str] = None
    avatar_emoji: str = "👤"
    is_guest: bool = False


class UserUpdate(schemas.BaseUserUpdate):
    """User update data."""
    display_name: Optional[str] = None
    avatar_emoji: Optional[str] = None
    theme: Optional[str] = None


# ============== Database Session ==============

async def create_db_and_tables():
    """Create database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session."""
    async with async_session_maker() as session:
        yield session


async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    """Get user database adapter."""
    yield SQLAlchemyUserDatabase(session, User)


# ============== User Manager ==============

class UserManager(IntegerIDMixin, BaseUserManager[User, int]):
    """Custom user manager with additional logic."""
    
    reset_password_token_secret = JWT_SECRET
    verification_token_secret = JWT_SECRET

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        """Called after successful registration."""
        print(f"User {user.email} registered")

    async def on_after_login(
        self, user: User, request: Optional[Request] = None, response=None
    ):
        """Called after successful login."""
        # Update last_active timestamp
        user.last_active = datetime.utcnow()  # type: ignore

    async def on_after_forgot_password(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        """Called after password reset request."""
        print(f"Password reset requested for {user.email}. Token: {token}")

    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        """Called after email verification request."""
        print(f"Verification requested for {user.email}. Token: {token}")


async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    """Get user manager instance."""
    yield UserManager(user_db)


# ============== Authentication Backend ==============

# Bearer token transport (for JWT in Authorization header)
bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")


def get_jwt_strategy() -> JWTStrategy:
    """JWT strategy with configurable lifetime."""
    return JWTStrategy(secret=JWT_SECRET, lifetime_seconds=3600 * 24 * 7)  # 7 days


# Authentication backend combining transport and strategy
auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)


# ============== FastAPI Users Instance ==============

fastapi_users = FastAPIUsers[User, int](get_user_manager, [auth_backend])

# Dependency to get current user
current_active_user = fastapi_users.current_user(active=True)
current_superuser = fastapi_users.current_user(active=True, superuser=True)
optional_current_user = fastapi_users.current_user(active=True, optional=True)


# ============== Guest User Support ==============

async def create_guest_user(session: AsyncSession) -> tuple[User, str]:
    """Create a temporary guest user and return user + token."""
    guest_id = str(uuid.uuid4())[:8]
    # Use a valid email format for guest users
    guest_email = f"guest_{guest_id}@magentic.app"
    
    user = User(
        email=guest_email,
        hashed_password="",  # No password for guests
        display_name=f"Guest {guest_id}",
        avatar_emoji="👻",
        is_guest=True,
        is_active=True,
        is_verified=True,  # Guests are auto-verified
    )
    
    session.add(user)
    await session.commit()
    await session.refresh(user)
    
    # Generate JWT token for guest
    strategy = get_jwt_strategy()
    token = await strategy.write_token(user)
    
    return user, token
