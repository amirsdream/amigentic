# Magentic Database Migrations

This directory contains Alembic migrations for the Magentic database.

## Creating a Migration

After making changes to the database models in `src/database.py`, create a new migration:

```bash
# Auto-generate migration from model changes
alembic revision --autogenerate -m "Description of changes"

# Or create an empty migration
alembic revision -m "Description of changes"
```

## Running Migrations

```bash
# Upgrade to latest version
alembic upgrade head

# Downgrade one version
alembic downgrade -1

# Check current version
alembic current

# View migration history
alembic history
```

## Initial Setup

If you're starting with an existing database, you can mark it as current:

```bash
# Mark database as being at the latest migration without running it
alembic stamp head
```

## Migration Files

Migrations are stored in `alembic/versions/` with auto-generated names.
Each migration has `upgrade()` and `downgrade()` functions.
