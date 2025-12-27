"""
MCP Server: Database Operations
Provides SQLite database query and management capabilities
"""

import os
import logging
import re
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
import aiosqlite  # type: ignore[import-untyped]
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MCP Database Server")

# Configuration
DB_DIR = Path(os.getenv("DB_DIR", "/data"))
MAX_RESULTS = int(os.getenv("MAX_RESULTS", 1000))
ALLOWED_WRITE_OPS = os.getenv("ALLOW_WRITE_OPS", "true").lower() == "true"

# Ensure DB directory exists
DB_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# Prometheus Metrics
# ============================================
DB_REQUESTS = Counter(
    'db_requests_total', 
    'Total database requests',
    ['database', 'operation', 'status']
)
DB_QUERY_DURATION = Histogram(
    'db_query_duration_seconds',
    'Database query duration',
    ['database', 'operation'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
)
DB_ROWS_RETURNED = Histogram(
    'db_rows_returned',
    'Rows returned per query',
    ['database'],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
)
DB_SIZE_BYTES = Gauge(
    'db_size_bytes',
    'Database size in bytes',
    ['database']
)
DB_ACTIVE_CONNECTIONS = Gauge(
    'db_active_connections',
    'Active database connections',
    ['database']
)

# ============================================
# Models
# ============================================

class QueryRequest(BaseModel):
    database: str  # Database name (without .db extension)
    sql: str
    params: Optional[List[Any]] = None
    max_results: Optional[int] = None


class CreateTableRequest(BaseModel):
    database: str
    table_name: str
    columns: Dict[str, str]  # column_name: column_type


class InsertRequest(BaseModel):
    database: str
    table_name: str
    data: Dict[str, Any]


class ListTablesRequest(BaseModel):
    database: str


# ============================================
# Helpers
# ============================================

def get_db_path(name: str) -> Path:
    """Get safe database path."""
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '', name)
    if not safe_name:
        raise ValueError("Invalid database name")
    return DB_DIR / f"{safe_name}.db"


def is_read_only(sql: str) -> bool:
    """Check if SQL is read-only."""
    sql_upper = sql.strip().upper()
    return sql_upper.startswith(('SELECT', 'PRAGMA', 'EXPLAIN'))


async def execute_query(db_path: Path, sql: str, params: Optional[List[Any]] = None, max_results: Optional[int] = None):
    """Execute SQL query and return results."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        
        async with db.execute(sql, params or []) as cursor:
            if sql.strip().upper().startswith('SELECT') or sql.strip().upper().startswith('PRAGMA'):
                rows = list(await cursor.fetchmany(max_results or MAX_RESULTS))
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                return {
                    "columns": columns,
                    "rows": [dict(zip(columns, row)) for row in rows],
                    "row_count": len(rows)
                }
            else:
                await db.commit()
                return {
                    "rows_affected": cursor.rowcount,
                    "last_row_id": cursor.lastrowid
                }


# ============================================
# API Endpoints
# ============================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    databases = list(DB_DIR.glob("*.db"))
    
    # Update database size metrics
    for db_path in databases:
        DB_SIZE_BYTES.labels(database=db_path.stem).set(db_path.stat().st_size)
    
    return {
        "status": "healthy",
        "db_dir": str(DB_DIR),
        "databases": [db.stem for db in databases],
        "write_ops_enabled": ALLOWED_WRITE_OPS
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/tools")
async def list_tools():
    """List available tools."""
    tools = [
        {
            "name": "query",
            "description": "Execute SQL query (SELECT, PRAGMA)",
            "parameters": {
                "database": {"type": "string", "description": "Database name"},
                "sql": {"type": "string", "description": "SQL query"},
                "params": {"type": "array", "description": "Query parameters"},
                "max_results": {"type": "integer", "description": "Max rows to return"}
            }
        },
        {
            "name": "list_tables",
            "description": "List all tables in database",
            "parameters": {
                "database": {"type": "string", "description": "Database name"}
            }
        },
        {
            "name": "describe_table",
            "description": "Get table schema",
            "parameters": {
                "database": {"type": "string", "description": "Database name"},
                "table_name": {"type": "string", "description": "Table name"}
            }
        },
        {
            "name": "list_databases",
            "description": "List all available databases",
            "parameters": {}
        }
    ]
    
    if ALLOWED_WRITE_OPS:
        tools.extend([
            {
                "name": "execute",
                "description": "Execute SQL (INSERT, UPDATE, DELETE, CREATE)",
                "parameters": {
                    "database": {"type": "string", "description": "Database name"},
                    "sql": {"type": "string", "description": "SQL statement"},
                    "params": {"type": "array", "description": "Query parameters"}
                }
            },
            {
                "name": "create_table",
                "description": "Create a new table",
                "parameters": {
                    "database": {"type": "string", "description": "Database name"},
                    "table_name": {"type": "string", "description": "Table name"},
                    "columns": {"type": "object", "description": "Column definitions {name: type}"}
                }
            },
            {
                "name": "insert",
                "description": "Insert a row into table",
                "parameters": {
                    "database": {"type": "string", "description": "Database name"},
                    "table_name": {"type": "string", "description": "Table name"},
                    "data": {"type": "object", "description": "Row data {column: value}"}
                }
            }
        ])
    
    return tools


@app.post("/tools/list_databases")
async def list_databases():
    """List all databases."""
    databases = []
    for db_path in DB_DIR.glob("*.db"):
        stat = db_path.stat()
        databases.append({
            "name": db_path.stem,
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime
        })
    return {"databases": databases}


@app.post("/tools/query")
async def query(request: QueryRequest):
    """Execute read-only SQL query."""
    start_time = time.perf_counter()
    try:
        db_path = get_db_path(request.database)
        
        if not db_path.exists():
            DB_REQUESTS.labels(database=request.database, operation='query', status='not_found').inc()
            raise HTTPException(status_code=404, detail=f"Database '{request.database}' not found")
        
        if not is_read_only(request.sql):
            DB_REQUESTS.labels(database=request.database, operation='query', status='invalid').inc()
            raise HTTPException(
                status_code=400, 
                detail="Only SELECT and PRAGMA queries allowed. Use /tools/execute for write operations."
            )
        
        DB_ACTIVE_CONNECTIONS.labels(database=request.database).inc()
        try:
            result = await execute_query(
                db_path, 
                request.sql, 
                request.params,
                request.max_results
            )
        finally:
            DB_ACTIVE_CONNECTIONS.labels(database=request.database).dec()
        
        # Record metrics
        row_count = len(result.get('rows', []))
        DB_REQUESTS.labels(database=request.database, operation='query', status='success').inc()
        DB_QUERY_DURATION.labels(database=request.database, operation='query').observe(time.perf_counter() - start_time)
        DB_ROWS_RETURNED.labels(database=request.database).observe(row_count)
        
        logger.info(f"✓ Query on {request.database}: {row_count} rows")
        return result
        
    except ValueError as e:
        DB_REQUESTS.labels(database=request.database, operation='query', status='error').inc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/execute")
async def execute_sql(request: QueryRequest):
    """Execute write SQL (INSERT, UPDATE, DELETE, CREATE)."""
    if not ALLOWED_WRITE_OPS:
        raise HTTPException(status_code=403, detail="Write operations are disabled")
    
    try:
        db_path = get_db_path(request.database)
        
        # Create database if it doesn't exist
        db_path.touch()
        
        result = await execute_query(db_path, request.sql, request.params)
        
        logger.info(f"✓ Execute on {request.database}: {result.get('rows_affected', 0)} rows affected")
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Execute error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/list_tables")
async def list_tables(request: ListTablesRequest):
    """List all tables in database."""
    try:
        db_path = get_db_path(request.database)
        
        if not db_path.exists():
            raise HTTPException(status_code=404, detail=f"Database '{request.database}' not found")
        
        result = await execute_query(
            db_path,
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        
        tables = [row["name"] for row in result["rows"]]
        return {"database": request.database, "tables": tables}
        
    except Exception as e:
        logger.error(f"List tables error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class DescribeTableRequest(BaseModel):
    database: str
    table_name: str


@app.post("/tools/describe_table")
async def describe_table(request: DescribeTableRequest):
    """Get table schema."""
    try:
        db_path = get_db_path(request.database)
        
        if not db_path.exists():
            raise HTTPException(status_code=404, detail=f"Database '{request.database}' not found")
        
        result = await execute_query(
            db_path,
            f"PRAGMA table_info({request.table_name})"
        )
        
        columns = [
            {
                "name": row["name"],
                "type": row["type"],
                "nullable": not row["notnull"],
                "default": row["dflt_value"],
                "primary_key": bool(row["pk"])
            }
            for row in result["rows"]
        ]
        
        return {
            "database": request.database,
            "table": request.table_name,
            "columns": columns
        }
        
    except Exception as e:
        logger.error(f"Describe table error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/create_table")
async def create_table(request: CreateTableRequest):
    """Create a new table."""
    if not ALLOWED_WRITE_OPS:
        raise HTTPException(status_code=403, detail="Write operations are disabled")
    
    try:
        db_path = get_db_path(request.database)
        db_path.touch()
        
        # Build CREATE TABLE statement
        columns_sql = ", ".join(
            f"{col} {dtype}" 
            for col, dtype in request.columns.items()
        )
        sql = f"CREATE TABLE IF NOT EXISTS {request.table_name} ({columns_sql})"
        
        await execute_query(db_path, sql)
        
        logger.info(f"✓ Created table {request.table_name} in {request.database}")
        return {
            "success": True,
            "database": request.database,
            "table": request.table_name
        }
        
    except Exception as e:
        logger.error(f"Create table error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/insert")
async def insert_row(request: InsertRequest):
    """Insert a row into table."""
    if not ALLOWED_WRITE_OPS:
        raise HTTPException(status_code=403, detail="Write operations are disabled")
    
    try:
        db_path = get_db_path(request.database)
        
        if not db_path.exists():
            raise HTTPException(status_code=404, detail=f"Database '{request.database}' not found")
        
        columns = list(request.data.keys())
        placeholders = ", ".join("?" for _ in columns)
        sql = f"INSERT INTO {request.table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        
        result = await execute_query(db_path, sql, list(request.data.values()))
        
        logger.info(f"✓ Inserted row into {request.table_name}")
        return {
            "success": True,
            "database": request.database,
            "table": request.table_name,
            "last_row_id": result.get("last_row_id")
        }
        
    except Exception as e:
        logger.error(f"Insert error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
