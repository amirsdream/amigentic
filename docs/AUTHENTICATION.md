# Authentication System

## Overview

Magentic includes a complete authentication system with user registration, login, and guest mode. All user data is stored securely in a local SQLite database with bcrypt password hashing.

## Features

### 1. User Registration
- Simple username/password registration (no email required)
- Minimum 3 characters for username
- Minimum 6 characters for password
- Bcrypt password hashing for security
- Unique username enforcement

### 2. User Login
- Secure authentication with password verification
- Session persistence via localStorage
- Automatic reconnection on page refresh
- Last active timestamp tracking

### 3. Guest Mode
- Continue without registration
- Auto-generated guest username (`guest_<timestamp>`)
- Guest users marked with `is_guest` flag
- No password required for guests
- Guest data persisted in database

### 4. User Profiles
- Display name and avatar emoji customization
- User statistics (total queries, agents executed)
- Theme preferences
- Conversation history
- Profile editing (display name, avatar)

## Database Schema

### UserProfile Table
```sql
CREATE TABLE user_profiles (
    id INTEGER PRIMARY KEY,
    username VARCHAR UNIQUE NOT NULL,
    password_hash VARCHAR,              -- NULL for guest users
    is_guest BOOLEAN DEFAULT FALSE,
    display_name VARCHAR,
    avatar_emoji VARCHAR DEFAULT '👤',
    created_at DATETIME,
    last_active DATETIME,
    theme VARCHAR DEFAULT 'dark',
    show_execution_details INTEGER DEFAULT 1,
    preferred_llm VARCHAR,
    total_queries INTEGER DEFAULT 0,
    total_agents_executed INTEGER DEFAULT 0
);
```

### Conversation Table
```sql
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES user_profiles(id),
    session_id VARCHAR,
    timestamp DATETIME,
    query TEXT,
    response TEXT,
    execution_plan JSON,
    agents_used INTEGER,
    execution_time INTEGER
);
```

## API Endpoints

### POST /register
Register a new user.

**Request:**
```json
{
  "username": "myusername",
  "password": "mypassword"
}
```

**Response:**
```json
{
  "success": true,
  "message": "User registered successfully",
  "user": {
    "id": 1,
    "username": "myusername",
    "display_name": "myusername",
    "avatar_emoji": "👤"
  }
}
```

**Errors:**
- 400: Username must be at least 3 characters
- 400: Password must be at least 6 characters
- 400: Username already exists

### POST /login
Authenticate a user.

**Request:**
```json
{
  "username": "myusername",
  "password": "mypassword"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Login successful",
  "user": {
    "id": 1,
    "username": "myusername",
    "display_name": "myusername",
    "avatar_emoji": "👤",
    "is_guest": false
  }
}
```

**Errors:**
- 401: User not found
- 401: Cannot login to guest account
- 401: Incorrect password

### GET /profile/{username}
Get user profile information.

**Response:**
```json
{
  "id": 1,
  "username": "myusername",
  "display_name": "My Display Name",
  "avatar_emoji": "🚀",
  "is_guest": false,
  "created_at": "2024-12-25T10:00:00",
  "last_active": "2024-12-25T12:30:00",
  "theme": "dark",
  "show_execution_details": true,
  "stats": {
    "total_queries": 42,
    "total_agents_executed": 156
  }
}
```

### PUT /profile/{username}
Update user profile.

**Request:**
```json
{
  "display_name": "New Name",
  "avatar_emoji": "🎨",
  "theme": "dark"
}
```

### GET /history/{username}
Get conversation history (last 50 conversations).

**Response:**
```json
{
  "conversations": [
    {
      "id": 1,
      "timestamp": "2024-12-25T12:00:00",
      "query": "What is Python?",
      "response": "Python is...",
      "agents_used": 3,
      "execution_time": 2500
    }
  ]
}
```

## Frontend Integration

### Login Modal
The `LoginModal` component provides:
- Login/Register mode toggle
- Form validation
- Error handling
- Guest continue button
- Responsive design

### Profile Modal
The `ProfileModal` component provides:
- Avatar emoji selection
- Display name editing
- User statistics display
- Logout button (for registered users)
- Profile editing workflow

### WebSocket Connection
The WebSocket connection includes the username:
```javascript
const websocket = new WebSocket(`ws://localhost:8000/ws?username=${username}`);
```

### Session Persistence
Username is stored in localStorage:
```javascript
localStorage.setItem('magentic_username', username);
const username = localStorage.getItem('magentic_username');
```

## Security

### Password Hashing
- Uses Passlib with bcrypt scheme
- Automatic salt generation
- Secure password verification
- Default cost factor (12 rounds)

### Guest Users
- No password storage for guests
- Marked with `is_guest=true` flag
- Cannot login to guest accounts
- Guest data persisted but not secure

### Database
- SQLite database at `data/magentic.db`
- Connection pooling enabled
- Pre-ping to detect stale connections
- Connection recycling (1 hour)

## Database Migrations

### Initialize Database
```bash
# Create initial schema
alembic upgrade head
```

### Create New Migration
```bash
# Auto-generate from model changes
alembic revision --autogenerate -m "Description"

# Apply migration
alembic upgrade head
```

### Check Status
```bash
# View current version
alembic current

# View migration history
alembic history
```

## Usage Flow

### New User Registration
1. User opens app → sees Login Modal
2. Clicks "Don't have an account? Register"
3. Enters username and password (min 3/6 chars)
4. Clicks "Register"
5. Backend creates user with hashed password
6. Frontend stores username in localStorage
7. WebSocket connects with username
8. User can start chatting

### Returning User Login
1. User opens app
2. Username found in localStorage
3. Auto-authenticates if session valid
4. Connects WebSocket with username
5. Loads conversation history

### Guest Mode
1. User opens app → sees Login Modal
2. Clicks "Continue as Guest"
3. Auto-generates guest username
4. Marks user as guest (no password)
5. User can chat (but not login later)

### Logout
1. User clicks profile button
2. Opens ProfileModal
3. Clicks "Logout" button (if not guest)
4. Clears localStorage
5. Closes WebSocket
6. Shows Login Modal again

## Best Practices

### For Users
- Use strong passwords (6+ characters minimum)
- Don't share credentials
- Guest mode for temporary sessions
- Register for persistent history

### For Developers
- Never store plain text passwords
- Always hash passwords with bcrypt
- Validate input on both frontend and backend
- Use parameterized queries (SQLAlchemy ORM)
- Handle database connections properly (session management)
- Log authentication attempts for security monitoring
- Use HTTPS in production

## Future Enhancements

Potential improvements:
- Email verification
- Password reset functionality
- Two-factor authentication
- OAuth integration (Google, GitHub)
- Session tokens with expiration
- Rate limiting on auth endpoints
- Account deletion
- Export conversation history
- Shared conversations/workspaces
- Role-based access control
