"""Authentication Module using fastapi-users.

Simple, clean auth using the battle-tested fastapi-users library.

Components:
- users: User model, schemas, and fastapi-users configuration
- router: FastAPI router with auth endpoints

Features:
- JWT-based authentication
- User registration and login
- Guest user support
- Profile management
- Password reset (email verification ready to enable)
"""

from .users import (
    User,
    UserRead,
    UserCreate,
    UserUpdate,
    fastapi_users,
    auth_backend,
    current_active_user,
    current_superuser,
    optional_current_user,
    get_async_session,
    create_db_and_tables,
    create_guest_user,
)
from .router import router

__all__ = [
    # Models & Schemas
    "User",
    "UserRead",
    "UserCreate",
    "UserUpdate",
    # FastAPI Users
    "fastapi_users",
    "auth_backend",
    "current_active_user",
    "current_superuser",
    "optional_current_user",
    # Database
    "get_async_session",
    "create_db_and_tables",
    # Utilities
    "create_guest_user",
    # Router
    "router",
]
