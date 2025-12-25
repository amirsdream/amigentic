#!/usr/bin/env python3
"""
Magentic Database Initialization Script

This script initializes the database and can optionally create a demo user.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.database import SessionLocal, Base, engine, create_user, get_or_create_user


def init_database(create_demo: bool = False):
    """Initialize the database and optionally create demo user."""
    
    print("🗄️  Initializing Magentic Database")
    print("=" * 50)
    
    # Create all tables
    print("\n📋 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created")
    
    # Create a session
    db = SessionLocal()
    
    try:
        if create_demo:
            print("\n👤 Creating demo user...")
            
            # Check if demo user exists
            from sqlalchemy import select
            from src.database import UserProfile
            
            existing = db.query(UserProfile).filter(
                UserProfile.username == "demo"
            ).first()
            
            if existing:
                print("⚠️  Demo user already exists")
            else:
                user, error = create_user(db, "demo", "demo123")
                
                if error:
                    print(f"❌ Failed to create demo user: {error}")
                else:
                    print("✓ Demo user created")
                    print("  Username: demo")
                    print("  Password: demo123")
        
        print("\n" + "=" * 50)
        print("✅ Database initialization complete!")
        print("")
        
        if create_demo:
            print("You can now login with:")
            print("  Username: demo")
            print("  Password: demo123")
            print("")
        
    except Exception as e:
        print(f"\n❌ Error during initialization: {e}")
        db.rollback()
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Initialize Magentic database"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Create a demo user (username: demo, password: demo123)"
    )
    
    args = parser.parse_args()
    
    init_database(create_demo=args.demo)
