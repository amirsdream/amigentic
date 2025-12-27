"""Authentication API routes for Magentic.

Simplified authentication using existing database models.
"""

from datetime import datetime, timedelta
from typing import Optional, Any
import logging
import secrets

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import jwt

from ..database import (
    get_db,
    UserProfile,
    create_user,
    authenticate_user,
    get_or_create_user,
    get_password_hash,
    verify_password,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer(auto_error=False)

# JWT Configuration - use environment variable or generate random for dev
import os
JWT_SECRET = os.environ.get("JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7


# ============== Request/Response Models ==============

class RegisterRequest(BaseModel):
    """User registration request."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, max_length=100)
    email: Optional[str] = None


class LoginRequest(BaseModel):
    """User login request."""
    username: str
    password: str
    remember_me: bool = False


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str


class UpdateProfileRequest(BaseModel):
    """Profile update request."""
    display_name: Optional[str] = None
    avatar_emoji: Optional[str] = None
    theme: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    """Password change request."""
    current_password: str
    new_password: str = Field(..., min_length=6, max_length=100)


# ============== JWT Helper Functions ==============

def create_access_token(user_id: Any, username: Any, is_guest: Any = False) -> str:
    """Create JWT access token."""
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(user_id),
        "username": str(username),
        "is_guest": is_guest,
        "exp": expire,
        "type": "access"
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id: Any, username: Any) -> str:
    """Create JWT refresh token."""
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": str(user_id),
        "username": str(username),
        "exp": expire,
        "type": "refresh"
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str, token_type: str = "access") -> Optional[dict]:
    """Verify JWT token and return payload."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != token_type:
            return None
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# ============== Auth Dependencies ==============

async def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    """Get current user from Authorization header."""
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    
    token = auth_header.split(" ")[1]
    payload = verify_token(token, "access")
    if not payload:
        return None
    
    user_id = int(payload["sub"])
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    return user


async def require_auth(
    request: Request,
    db: Session = Depends(get_db),
) -> Any:
    """Require authentication."""
    user = await get_current_user(request, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


# ============== Registration & Login ==============

@router.post("/register", response_model=dict)
async def register(
    request_data: RegisterRequest,
    db: Session = Depends(get_db),
):
    """Register a new user account."""
    user, error = create_user(db, request_data.username, request_data.password)
    
    if error or not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error or "Registration failed",
        )
    
    # Create tokens
    access_token = create_access_token(user.id, user.username)
    refresh_token = create_refresh_token(user.id, user.username)
    
    return {
        "success": True,
        "message": "Registration successful",
        "user": {
            "id": user.id,
            "username": user.username,
            "display_name": user.display_name,
            "avatar_emoji": user.avatar_emoji,
            "is_guest": user.is_guest,
        },
        "tokens": {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        },
    }


@router.post("/login", response_model=dict)
async def login(
    request_data: LoginRequest,
    db: Session = Depends(get_db),
):
    """Login with username and password."""
    user, error = authenticate_user(db, request_data.username, request_data.password)
    
    if error or not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error or "Authentication failed",
        )
    
    # Create tokens
    access_token = create_access_token(user.id, user.username)
    refresh_token = create_refresh_token(user.id, user.username)
    
    return {
        "success": True,
        "message": "Login successful",
        "user": {
            "id": user.id,
            "username": user.username,
            "display_name": user.display_name,
            "avatar_emoji": user.avatar_emoji,
            "is_guest": user.is_guest,
            "theme": user.theme,
        },
        "tokens": {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        },
    }


@router.post("/guest", response_model=dict)
async def login_as_guest(
    db: Session = Depends(get_db),
):
    """Create a guest session (limited features)."""
    # Create or get guest user
    guest = get_or_create_user(db, f"guest_{secrets.token_hex(4)}", is_guest=True)
    
    if not guest:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create guest session",
        )
    
    # Create tokens
    access_token = create_access_token(guest.id, guest.username, is_guest=True)
    refresh_token = create_refresh_token(guest.id, guest.username)
    
    return {
        "success": True,
        "message": "Guest session created",
        "user": {
            "id": guest.id,
            "username": guest.username,
            "display_name": guest.display_name or "Guest",
            "avatar_emoji": guest.avatar_emoji,
            "is_guest": True,
        },
        "tokens": {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        },
    }


@router.post("/refresh", response_model=dict)
async def refresh_tokens(
    request: Request,
    request_data: Optional[RefreshTokenRequest] = None,
    db: Session = Depends(get_db),
):
    """Refresh access token using refresh token."""
    # Try to get token from body first, then header
    token = None
    if request_data and request_data.refresh_token:
        token = request_data.refresh_token
    else:
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
    
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token required",
        )
    
    payload = verify_token(token, "refresh")
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    
    user_id = int(payload["sub"])
    username = payload["username"]
    
    # Verify user still exists
    user = db.query(UserProfile).filter(UserProfile.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    # Create new tokens
    access_token = create_access_token(user.id, user.username, user.is_guest)
    new_refresh_token = create_refresh_token(user.id, user.username)
    
    return {
        "success": True,
        "tokens": {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        },
    }


@router.post("/logout")
async def logout():
    """Logout (client should discard tokens)."""
    return {"success": True, "message": "Logged out successfully"}


# ============== Profile Management ==============

@router.get("/me", response_model=dict)
async def get_current_user_profile(
    user: UserProfile = Depends(require_auth),
):
    """Get current user profile."""
    return {
        "user": {
            "id": user.id,
            "username": user.username,
            "display_name": user.display_name,
            "avatar_emoji": user.avatar_emoji,
            "is_guest": user.is_guest,
            "theme": user.theme,
            "created_at": user.created_at.isoformat() if user.created_at else None,  # type: ignore
            "total_queries": user.total_queries,
            "total_agents_executed": user.total_agents_executed,
            "total_tokens_used": user.total_tokens_used or 0,
            "total_cost": user.total_cost or 0.0,
        },
    }


@router.put("/me", response_model=dict)
async def update_profile(
    request_data: UpdateProfileRequest,
    user: UserProfile = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Update current user profile."""
    if user.is_guest:  # type: ignore
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Guests cannot update profile",
        )
    
    # Update fields
    if request_data.display_name is not None:
        user.display_name = request_data.display_name  # type: ignore
    if request_data.avatar_emoji is not None:
        user.avatar_emoji = request_data.avatar_emoji  # type: ignore
    if request_data.theme is not None:
        user.theme = request_data.theme  # type: ignore
    
    db.commit()
    
    return {"success": True, "message": "Profile updated"}


@router.post("/me/password", response_model=dict)
async def change_password(
    request_data: ChangePasswordRequest,
    user: UserProfile = Depends(require_auth),
    db: Session = Depends(get_db),
):
    """Change user password."""
    if user.is_guest:  # type: ignore
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Guests cannot change password",
        )
    
    # Verify current password
    if not verify_password(request_data.current_password, user.password_hash):  # type: ignore
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )
    
    # Update password
    user.password_hash = get_password_hash(request_data.new_password)  # type: ignore
    db.commit()
    
    return {"success": True, "message": "Password changed successfully"}


# ============== Analytics (simplified) ==============

@router.get("/analytics", response_model=dict)
async def get_user_analytics(
    user: UserProfile = Depends(require_auth),
):
    """Get usage analytics for current user."""
    return {
        "stats": {
            "total_queries": user.total_queries,
            "total_agents_executed": user.total_agents_executed,
            "total_tokens_used": user.total_tokens_used or 0,
            "total_cost": user.total_cost or 0.0,
        },
    }
