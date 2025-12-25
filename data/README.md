# Data Directory

This directory contains the SQLite database for user profiles and conversation history.

## Files

- `magentic.db` - SQLite database file (auto-created on first run)

## Database Tables

### user_profiles
- User information, preferences, and statistics
- Fields: username, display_name, avatar_emoji, theme, stats

### conversations  
- Conversation history per user
- Fields: query, response, execution_plan, timestamp, user_id

## Note

The database file is created automatically when the backend starts.
