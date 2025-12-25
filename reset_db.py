#!/usr/bin/env python3
"""
Magentic Database Reset Script

This script completely resets the database to a clean state.
Use this for development when you want to start fresh.

WARNING: This will DELETE ALL DATA!

Usage:
    python reset_db.py              # Reset and create empty DB
    python reset_db.py --demo       # Reset and create demo user
    python reset_db.py --confirm    # Skip confirmation prompt
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.database import SessionLocal, create_user, DB_PATH


def reset_database(create_demo: bool = False, skip_confirm: bool = False):
    """Reset the database to a clean state."""
    
    print("⚠️  MAGENTIC DATABASE RESET")
    print("=" * 50)
    print("⚠️  WARNING: This will DELETE ALL DATA!")
    print(f"   Database: {DB_PATH}")
    print("=" * 50)
    
    if not skip_confirm:
        response = input("\nAre you sure you want to continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("❌ Reset cancelled")
            return
    
    print("\n🗑️  Deleting database file...")
    try:
        if DB_PATH.exists():
            DB_PATH.unlink()
            print(f"✓ Deleted {DB_PATH}")
        else:
            print(f"ℹ️  Database file doesn't exist (already clean)")
        
        # Verify deletion
        if DB_PATH.exists():
            print(f"❌ Failed to delete database file!")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error deleting database: {e}")
        sys.exit(1)
    
    # Small delay to ensure file system sync
    import time
    time.sleep(0.5)
    
    print("\n📋 Running Alembic migrations...")
    try:
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True
        )
        print("✓ Database recreated with migrations")
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"   {line}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running migrations: {e}")
        if e.stderr:
            print(e.stderr)
        sys.exit(1)
    
    # Create a session
    db = SessionLocal()
    
    try:
        if create_demo:
            print("\n👤 Creating demo user...")
            user, error = create_user(db, "demo", "demo123")
            
            if error:
                print(f"❌ Failed to create demo user: {error}")
            else:
                print(f"✓ Demo user created!")
                print(f"   Username: demo")
                print(f"   Password: demo123")
        
        print("\n✅ Database reset complete!")
        print("   All tables recreated from migrations")
        
        if not create_demo:
            print("\n💡 Tip: Run with --demo flag to create a demo user")
        
    except Exception as e:
        print(f"\n❌ Error during reset: {e}")
        db.rollback()
        sys.exit(1)
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(
        description="Reset Magentic database to clean state"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Create demo user (username: demo, password: demo123)"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    args = parser.parse_args()
    
    reset_database(create_demo=args.demo, skip_confirm=args.confirm)


if __name__ == "__main__":
    main()
