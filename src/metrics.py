"""
Prometheus Metrics module for Magentic.
Provides metrics collection for monitoring the application.
Optional - enabled via ENABLE_METRICS=true environment variable.
"""

import os
import time
import logging
import asyncio
from functools import wraps
from typing import Callable, Any, Optional, TYPE_CHECKING
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Check if metrics are enabled
METRICS_ENABLED = os.getenv("ENABLE_METRICS", "false").lower() == "true"

# Try to import prometheus_client
_prometheus_client = None
_Counter = None
_Histogram = None
_Gauge = None
_Info = None
_generate_latest = None
_CONTENT_TYPE_LATEST = None
_BaseHTTPMiddleware = None

if METRICS_ENABLED:
    try:
        import prometheus_client
        _prometheus_client = prometheus_client
        _Counter = prometheus_client.Counter
        _Histogram = prometheus_client.Histogram
        _Gauge = prometheus_client.Gauge
        _Info = prometheus_client.Info
        _generate_latest = prometheus_client.generate_latest
        _CONTENT_TYPE_LATEST = prometheus_client.CONTENT_TYPE_LATEST
        
        from starlette.middleware.base import BaseHTTPMiddleware as _BHM
        _BaseHTTPMiddleware = _BHM
        
        logger.info("✓ Prometheus metrics enabled")
    except ImportError:
        logger.warning("prometheus-client not installed. Install with: pip install prometheus-client")
        METRICS_ENABLED = False

PROMETHEUS_AVAILABLE = METRICS_ENABLED and _prometheus_client is not None


# ============================================
# Metric instances (None if not available)
# ============================================

APP_INFO: Any = None
HTTP_REQUESTS_TOTAL: Any = None
HTTP_REQUEST_DURATION: Any = None
HTTP_REQUESTS_IN_PROGRESS: Any = None
LLM_REQUESTS_TOTAL: Any = None
LLM_REQUEST_DURATION: Any = None
LLM_TOKENS_TOTAL: Any = None
LLM_COST_TOTAL: Any = None
AGENT_EXECUTIONS_TOTAL: Any = None
AGENT_EXECUTION_DURATION: Any = None
AGENTS_IN_PROGRESS: Any = None
TOOL_CALLS_TOTAL: Any = None
TOOL_CALL_DURATION: Any = None
QUERIES_TOTAL: Any = None
QUERY_DURATION: Any = None
WEBSOCKET_CONNECTIONS: Any = None
WEBSOCKET_MESSAGES_TOTAL: Any = None
RAG_QUERIES_TOTAL: Any = None
RAG_DOCUMENTS_INDEXED: Any = None
MCP_REQUESTS_TOTAL: Any = None
MCP_REQUEST_DURATION: Any = None
ERRORS_TOTAL: Any = None

if PROMETHEUS_AVAILABLE and _Counter and _Histogram and _Gauge and _Info:
    # Application info
    APP_INFO = _Info('magentic_app', 'Application information')
    
    # HTTP metrics
    HTTP_REQUESTS_TOTAL = _Counter(
        'http_requests_total',
        'Total HTTP requests',
        ['method', 'endpoint', 'status']
    )
    HTTP_REQUEST_DURATION = _Histogram(
        'http_request_duration_seconds',
        'HTTP request latency',
        ['method', 'endpoint'],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30]
    )
    HTTP_REQUESTS_IN_PROGRESS = _Gauge(
        'http_requests_in_progress',
        'HTTP requests currently being processed'
    )
    
    # LLM metrics
    LLM_REQUESTS_TOTAL = _Counter(
        'llm_requests_total',
        'Total LLM API calls',
        ['provider', 'model', 'status']
    )
    LLM_REQUEST_DURATION = _Histogram(
        'llm_request_duration_seconds',
        'LLM request latency',
        ['provider', 'model'],
        buckets=[0.5, 1, 2.5, 5, 10, 30, 60, 120]
    )
    LLM_TOKENS_TOTAL = _Counter(
        'llm_tokens_total',
        'Total tokens used',
        ['provider', 'model', 'type']
    )
    LLM_COST_TOTAL = _Counter(
        'llm_cost_dollars_total',
        'Total cost in dollars',
        ['provider', 'model']
    )
    
    # Agent metrics
    AGENT_EXECUTIONS_TOTAL = _Counter(
        'agent_executions_total',
        'Total agent executions',
        ['agent_type', 'status']
    )
    AGENT_EXECUTION_DURATION = _Histogram(
        'agent_execution_duration_seconds',
        'Agent execution time',
        ['agent_type'],
        buckets=[1, 5, 10, 30, 60, 120, 300, 600]
    )
    AGENTS_IN_PROGRESS = _Gauge(
        'agents_in_progress',
        'Number of agents currently executing'
    )
    
    # Tool metrics
    TOOL_CALLS_TOTAL = _Counter(
        'tool_calls_total',
        'Total tool invocations',
        ['tool_name', 'status']
    )
    TOOL_CALL_DURATION = _Histogram(
        'tool_call_duration_seconds',
        'Tool call latency',
        ['tool_name'],
        buckets=[0.1, 0.5, 1, 2.5, 5, 10, 30]
    )
    
    # Query metrics
    QUERIES_TOTAL = _Counter(
        'queries_total',
        'Total user queries',
        ['status']
    )
    QUERY_DURATION = _Histogram(
        'query_duration_seconds',
        'End-to-end query processing time',
        buckets=[1, 5, 10, 30, 60, 120, 300]
    )
    
    # WebSocket metrics
    WEBSOCKET_CONNECTIONS = _Gauge(
        'websocket_connections_active',
        'Active WebSocket connections'
    )
    WEBSOCKET_MESSAGES_TOTAL = _Counter(
        'websocket_messages_total',
        'Total WebSocket messages',
        ['direction']
    )
    
    # RAG metrics
    RAG_QUERIES_TOTAL = _Counter(
        'rag_queries_total',
        'Total RAG queries',
        ['status']
    )
    RAG_DOCUMENTS_INDEXED = _Gauge(
        'rag_documents_indexed',
        'Number of documents in RAG index'
    )
    
    # MCP metrics
    MCP_REQUESTS_TOTAL = _Counter(
        'mcp_requests_total',
        'Total MCP server requests',
        ['server', 'tool', 'status']
    )
    MCP_REQUEST_DURATION = _Histogram(
        'mcp_request_duration_seconds',
        'MCP request latency',
        ['server', 'tool'],
        buckets=[0.1, 0.5, 1, 2.5, 5, 10]
    )
    
    # Error tracking
    ERRORS_TOTAL = _Counter(
        'errors_total',
        'Total errors',
        ['type', 'component']
    )


# ============================================
# Decorators for instrumenting code
# ============================================

def track_llm_call(provider: str, model: str) -> Callable:
    """Decorator to track LLM API calls."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not PROMETHEUS_AVAILABLE:
                return await func(*args, **kwargs)
            
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                if LLM_REQUESTS_TOTAL:
                    LLM_REQUESTS_TOTAL.labels(provider=provider, model=model, status='success').inc()
                return result
            except Exception as e:
                if LLM_REQUESTS_TOTAL:
                    LLM_REQUESTS_TOTAL.labels(provider=provider, model=model, status='error').inc()
                if ERRORS_TOTAL:
                    ERRORS_TOTAL.labels(type=type(e).__name__, component='llm').inc()
                raise
            finally:
                duration = time.perf_counter() - start_time
                if LLM_REQUEST_DURATION:
                    LLM_REQUEST_DURATION.labels(provider=provider, model=model).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not PROMETHEUS_AVAILABLE:
                return func(*args, **kwargs)
            
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                if LLM_REQUESTS_TOTAL:
                    LLM_REQUESTS_TOTAL.labels(provider=provider, model=model, status='success').inc()
                return result
            except Exception as e:
                if LLM_REQUESTS_TOTAL:
                    LLM_REQUESTS_TOTAL.labels(provider=provider, model=model, status='error').inc()
                if ERRORS_TOTAL:
                    ERRORS_TOTAL.labels(type=type(e).__name__, component='llm').inc()
                raise
            finally:
                duration = time.perf_counter() - start_time
                if LLM_REQUEST_DURATION:
                    LLM_REQUEST_DURATION.labels(provider=provider, model=model).observe(duration)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def track_agent_execution(agent_type: str) -> Callable:
    """Decorator to track agent executions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not PROMETHEUS_AVAILABLE:
                return await func(*args, **kwargs)
            
            if AGENTS_IN_PROGRESS:
                AGENTS_IN_PROGRESS.inc()
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                if AGENT_EXECUTIONS_TOTAL:
                    AGENT_EXECUTIONS_TOTAL.labels(agent_type=agent_type, status='success').inc()
                return result
            except Exception as e:
                if AGENT_EXECUTIONS_TOTAL:
                    AGENT_EXECUTIONS_TOTAL.labels(agent_type=agent_type, status='error').inc()
                if ERRORS_TOTAL:
                    ERRORS_TOTAL.labels(type=type(e).__name__, component='agent').inc()
                raise
            finally:
                if AGENTS_IN_PROGRESS:
                    AGENTS_IN_PROGRESS.dec()
                duration = time.perf_counter() - start_time
                if AGENT_EXECUTION_DURATION:
                    AGENT_EXECUTION_DURATION.labels(agent_type=agent_type).observe(duration)
        return wrapper
    return decorator


def track_tool_call(tool_name: str) -> Callable:
    """Decorator to track tool calls."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not PROMETHEUS_AVAILABLE:
                return await func(*args, **kwargs)
            
            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                if TOOL_CALLS_TOTAL:
                    TOOL_CALLS_TOTAL.labels(tool_name=tool_name, status='success').inc()
                return result
            except Exception as e:
                if TOOL_CALLS_TOTAL:
                    TOOL_CALLS_TOTAL.labels(tool_name=tool_name, status='error').inc()
                if ERRORS_TOTAL:
                    ERRORS_TOTAL.labels(type=type(e).__name__, component='tool').inc()
                raise
            finally:
                duration = time.perf_counter() - start_time
                if TOOL_CALL_DURATION:
                    TOOL_CALL_DURATION.labels(tool_name=tool_name).observe(duration)
        
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not PROMETHEUS_AVAILABLE:
                return func(*args, **kwargs)
            
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                if TOOL_CALLS_TOTAL:
                    TOOL_CALLS_TOTAL.labels(tool_name=tool_name, status='success').inc()
                return result
            except Exception as e:
                if TOOL_CALLS_TOTAL:
                    TOOL_CALLS_TOTAL.labels(tool_name=tool_name, status='error').inc()
                if ERRORS_TOTAL:
                    ERRORS_TOTAL.labels(type=type(e).__name__, component='tool').inc()
                raise
            finally:
                duration = time.perf_counter() - start_time
                if TOOL_CALL_DURATION:
                    TOOL_CALL_DURATION.labels(tool_name=tool_name).observe(duration)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


# ============================================
# Context managers for manual instrumentation
# ============================================

@contextmanager
def track_query():
    """Context manager to track query processing."""
    if not PROMETHEUS_AVAILABLE:
        yield
        return
    
    start_time = time.perf_counter()
    try:
        yield
        if QUERIES_TOTAL:
            QUERIES_TOTAL.labels(status='success').inc()
    except Exception as e:
        if QUERIES_TOTAL:
            QUERIES_TOTAL.labels(status='error').inc()
        if ERRORS_TOTAL:
            ERRORS_TOTAL.labels(type=type(e).__name__, component='query').inc()
        raise
    finally:
        duration = time.perf_counter() - start_time
        if QUERY_DURATION:
            QUERY_DURATION.observe(duration)


@contextmanager  
def track_mcp_request(server: str, tool: str):
    """Context manager to track MCP requests."""
    if not PROMETHEUS_AVAILABLE:
        yield
        return
    
    start_time = time.perf_counter()
    try:
        yield
        if MCP_REQUESTS_TOTAL:
            MCP_REQUESTS_TOTAL.labels(server=server, tool=tool, status='success').inc()
    except Exception:
        if MCP_REQUESTS_TOTAL:
            MCP_REQUESTS_TOTAL.labels(server=server, tool=tool, status='error').inc()
        raise
    finally:
        duration = time.perf_counter() - start_time
        if MCP_REQUEST_DURATION:
            MCP_REQUEST_DURATION.labels(server=server, tool=tool).observe(duration)


# ============================================
# Helper functions for recording metrics
# ============================================

def record_tokens(provider: str, model: str, input_tokens: int, output_tokens: int) -> None:
    """Record token usage."""
    if not PROMETHEUS_AVAILABLE or not LLM_TOKENS_TOTAL:
        return
    LLM_TOKENS_TOTAL.labels(provider=provider, model=model, type='input').inc(input_tokens)
    LLM_TOKENS_TOTAL.labels(provider=provider, model=model, type='output').inc(output_tokens)


def record_llm_request(provider: str, model: str, duration: float, success: bool) -> None:
    """Record LLM request."""
    if not PROMETHEUS_AVAILABLE:
        return
    if LLM_REQUESTS_TOTAL:
        status = 'success' if success else 'error'
        LLM_REQUESTS_TOTAL.labels(provider=provider, model=model, status=status).inc()
    if LLM_REQUEST_DURATION:
        LLM_REQUEST_DURATION.labels(provider=provider, model=model).observe(duration)


def record_cost(provider: str, model: str, cost: float) -> None:
    """Record LLM cost."""
    if not PROMETHEUS_AVAILABLE or not LLM_COST_TOTAL:
        return
    LLM_COST_TOTAL.labels(provider=provider, model=model).inc(cost)


def record_websocket_connect() -> None:
    """Record WebSocket connection."""
    if PROMETHEUS_AVAILABLE and WEBSOCKET_CONNECTIONS:
        WEBSOCKET_CONNECTIONS.inc()


def record_websocket_disconnect() -> None:
    """Record WebSocket disconnection."""
    if PROMETHEUS_AVAILABLE and WEBSOCKET_CONNECTIONS:
        WEBSOCKET_CONNECTIONS.dec()


def record_websocket_message(direction: str) -> None:
    """Record WebSocket message (direction: 'sent' or 'received')."""
    if PROMETHEUS_AVAILABLE and WEBSOCKET_MESSAGES_TOTAL:
        WEBSOCKET_MESSAGES_TOTAL.labels(direction=direction).inc()


def record_rag_query(success: bool) -> None:
    """Record RAG query."""
    if PROMETHEUS_AVAILABLE and RAG_QUERIES_TOTAL:
        RAG_QUERIES_TOTAL.labels(status='success' if success else 'error').inc()


def set_rag_documents_count(count: int) -> None:
    """Set RAG documents count."""
    if PROMETHEUS_AVAILABLE and RAG_DOCUMENTS_INDEXED:
        RAG_DOCUMENTS_INDEXED.set(count)


def record_error(error_type: str, component: str) -> None:
    """Record an error."""
    if PROMETHEUS_AVAILABLE and ERRORS_TOTAL:
        ERRORS_TOTAL.labels(type=error_type, component=component).inc()


# ============================================
# Metrics Middleware (must be added BEFORE app starts)
# ============================================

def get_metrics_middleware() -> Optional[type]:
    """Get the MetricsMiddleware class if metrics are enabled.
    
    Must be called BEFORE app.startup() - add middleware at module load time.
    Returns None if metrics are disabled.
    """
    if not PROMETHEUS_AVAILABLE:
        return None
    
    if not (_BaseHTTPMiddleware and HTTP_REQUESTS_IN_PROGRESS and HTTP_REQUEST_DURATION and HTTP_REQUESTS_TOTAL):
        return None
    
    import re
    from starlette.middleware.base import BaseHTTPMiddleware
    from fastapi import Request, Response
    
    class MetricsMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next: Callable) -> Response:
            if request.url.path == '/metrics':
                return await call_next(request)
            
            method = request.method
            path = self._normalize_path(request.url.path)
            
            HTTP_REQUESTS_IN_PROGRESS.inc()
            start_time = time.perf_counter()
            status = 500
            
            try:
                response = await call_next(request)
                status = response.status_code
                return response
            finally:
                duration = time.perf_counter() - start_time
                HTTP_REQUEST_DURATION.labels(method=method, endpoint=path).observe(duration)
                HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=path, status=status).inc()
                HTTP_REQUESTS_IN_PROGRESS.dec()
        
        def _normalize_path(self, path: str) -> str:
            path = path.rstrip('/')
            path = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '{id}', path)
            path = re.sub(r'/\d+(?=/|$)', '/{id}', path)
            return path or '/'
    
    return MetricsMiddleware


def add_metrics_endpoint(app: Any) -> None:
    """Add /metrics endpoint to the app.
    
    Can be called at module load time (before app starts).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    
    if _generate_latest is not None and _CONTENT_TYPE_LATEST is not None:
        from fastapi import Response as FastAPIResponse
        generate_fn = _generate_latest
        content_type = _CONTENT_TYPE_LATEST
        
        @app.get('/metrics', include_in_schema=False)
        async def metrics_endpoint() -> FastAPIResponse:
            return FastAPIResponse(
                content=generate_fn(),
                media_type=content_type
            )
        
        logger.info("✓ Prometheus metrics endpoint registered at /metrics")


# ============================================
# Metrics setup (call during startup for app info only)
# ============================================

def setup_metrics(app: Any, version: str = "2.0.0") -> bool:
    """Set app info for metrics. Call during startup event.
    
    Note: Middleware and endpoint should be added BEFORE app starts
    using get_metrics_middleware() and add_metrics_endpoint().
    """
    if not PROMETHEUS_AVAILABLE:
        logger.info("Metrics disabled (ENABLE_METRICS=false or prometheus-client not installed)")
        return False
    
    # Set app info
    if APP_INFO:
        APP_INFO.info({
            'version': version,
            'name': 'magentic'
        })
        logger.info("✓ Prometheus metrics initialized")
    
    return True
