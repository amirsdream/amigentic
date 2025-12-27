#!/usr/bin/env bash
#
# Magentic - Unified Management Script
# Start, stop, and manage all services (MCP, Database, Application)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Paths
DOCKER_DIR="$SCRIPT_DIR/docker"
VENV_DIR="$SCRIPT_DIR/.venv"
DATA_DIR="$SCRIPT_DIR/data"
DB_PATH="$DATA_DIR/magentic.db"
PID_FILE="$DATA_DIR/.magentic.pid"
API_PID_FILE="$DATA_DIR/.magentic-api.pid"

# Configuration
DEFAULT_PORT=8000
API_PORT=${MAGENTIC_API_PORT:-$DEFAULT_PORT}
MCP_GATEWAY_PORT=9000

# Preferred Python version (3.13 has best package compatibility)
PREFERRED_PYTHON_VERSION="3.13"

# pyenv paths
PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
PYENV_BIN="$PYENV_ROOT/bin/pyenv"

# ============================================
# Utility Functions
# ============================================

log_info() { echo -e "${BLUE}ℹ${NC} $1"; }
log_success() { echo -e "${GREEN}✓${NC} $1"; }
log_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
log_error() { echo -e "${RED}✗${NC} $1"; }

# Find the best Python interpreter (prefer 3.13 for compatibility)
# Uses pyenv if available, otherwise falls back to system Python
find_best_python() {
    local python_cmd=""
    
    # Check pyenv first (preferred)
    if [[ -x "$PYENV_BIN" ]]; then
        # Check if Python 3.13 is installed via pyenv
        for ver in "3.13" "3.12" "3.11" "3.10"; do
            local pyenv_version=$($PYENV_BIN versions --bare 2>/dev/null | grep "^$ver" | sort -V | tail -1)
            if [[ -n "$pyenv_version" ]]; then
                local pyenv_python="$PYENV_ROOT/versions/$pyenv_version/bin/python"
                if [[ -x "$pyenv_python" ]]; then
                    python_cmd="$pyenv_python"
                    break
                fi
            fi
        done
    fi
    
    # If pyenv didn't find a good version, check system paths
    if [[ -z "$python_cmd" ]]; then
        local system_paths=("/usr/bin" "/usr/local/bin" "/opt/homebrew/bin")
        
        for ver in "3.13" "3.12" "3.11" "3.10"; do
            for path in "${system_paths[@]}"; do
                if [[ -x "$path/python$ver" ]]; then
                    python_cmd="$path/python$ver"
                    break 2
                fi
            done
        done
    fi
    
    # Final fallback to generic python3
    if [[ -z "$python_cmd" ]]; then
        for path in "/usr/bin" "/usr/local/bin"; do
            if [[ -x "$path/python3" ]]; then
                python_cmd="$path/python3"
                break
            fi
        done
    fi
    
    echo "$python_cmd"
}

# Check if pyenv is installed
check_pyenv() {
    if [[ -x "$PYENV_BIN" ]]; then
        return 0
    fi
    return 1
}

# Install Python version via pyenv
install_python_via_pyenv() {
    local version=$1
    
    if ! check_pyenv; then
        log_error "pyenv is not installed"
        return 1
    fi
    
    log_info "Installing Python $version via pyenv (this may take a few minutes)..."
    
    # Find the latest patch version
    local full_version=$($PYENV_BIN install --list 2>/dev/null | grep "^  $version" | grep -v "dev\|rc\|a\|b" | tail -1 | tr -d ' ')
    
    if [[ -z "$full_version" ]]; then
        log_error "Python $version not found in pyenv"
        return 1
    fi
    
    echo -e "  ${BLUE}ℹ${NC} Installing Python $full_version..."
    
    if $PYENV_BIN install "$full_version" 2>&1 | while read line; do
        echo -ne "\r  ${BLUE}⠋${NC} $line                    \r"
    done; then
        log_success "Python $full_version installed via pyenv"
        return 0
    else
        log_error "Failed to install Python $full_version"
        return 1
    fi
}

print_banner() {
    echo -e "${MAGENTA}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════╗
║   __  __                        _   _                 ║
║  |  \/  | __ _  __ _  ___ _ __ | |_(_) ___           ║
║  | |\/| |/ _` |/ _` |/ _ \ '_ \| __| |/ __|          ║
║  | |  | | (_| | (_| |  __/ | | | |_| | (__           ║
║  |_|  |_|\__,_|\__, |\___|_| |_|\__|_|\___|          ║
║                |___/                                  ║
║                                                       ║
║          Magnetic Agent Networks                      ║
╚═══════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

check_python() {
    if [[ ! -d "$VENV_DIR" ]]; then
        log_error "Virtual environment not found. Run: ./magentic.sh setup"
        exit 1
    fi
    
    # Check if core dependencies are installed
    local python="$VENV_DIR/bin/python"
    if ! $python -c "import dotenv, langchain, fastapi" 2>/dev/null; then
        log_error "Dependencies not installed. Run: ./magentic.sh setup"
        exit 1
    fi
}

# Softer check for use during setup (returns status instead of exit)
check_python_soft() {
    if [[ ! -d "$VENV_DIR" ]]; then
        return 1
    fi
    
    local python="$VENV_DIR/bin/python"
    if ! $python -c "import dotenv" 2>/dev/null; then
        return 1
    fi
    return 0
}

get_python() {
    echo "$VENV_DIR/bin/python"
}

# Silent check for Docker availability (no warnings)
check_docker_silent() {
    if ! command -v docker &> /dev/null; then
        return 1
    fi
    
    if ! docker info &> /dev/null 2>&1; then
        return 1
    fi
    
    return 0
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_warning "Docker is not installed - MCP services will be unavailable"
        return 1
    fi
    
    if ! docker info &> /dev/null 2>&1; then
        log_warning "Docker daemon is not running - MCP services will be unavailable"
        return 1
    fi
    
    return 0
}

get_compose_cmd() {
    if docker compose version &> /dev/null 2>&1; then
        echo "docker compose"
    else
        echo "docker-compose"
    fi
}

# Check if MCP containers are running (docker level)
is_mcp_containers_running() {
    if check_docker_silent; then
        local compose_cmd=$(get_compose_cmd)
        cd "$DOCKER_DIR" 2>/dev/null || return 1
        local running=$($compose_cmd ps --services --filter "status=running" 2>/dev/null | wc -l)
        cd "$SCRIPT_DIR"
        [[ $running -gt 0 ]]
    else
        return 1
    fi
}

# Check if MCP Gateway is actually responding (health check)
is_mcp_running() {
    # First check if gateway health endpoint responds
    if curl -sf "http://localhost:$MCP_GATEWAY_PORT/health" > /dev/null 2>&1; then
        return 0
    fi
    return 1
}

is_app_running() {
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

is_api_running() {
    if [[ -f "$API_PID_FILE" ]]; then
        local pid=$(cat "$API_PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=${3:-30}
    local attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -sf "$url" > /dev/null 2>&1; then
            return 0
        fi
        ((attempt++))
        sleep 1
    done
    return 1
}

# ============================================
# Database Management
# ============================================

init_database() {
    local during_setup=${1:-false}  # Pass "true" when called from setup
    
    echo
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}           Database Setup${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    
    # Check if database already exists
    local db_exists=false
    if [[ -f "$DB_PATH" ]] || [[ -f "$DATA_DIR/magentic.db" ]]; then
        db_exists=true
        echo -e "  ${BLUE}ℹ${NC} Database already exists"
        
        if [[ "$during_setup" == "true" ]]; then
            read -p "  Re-initialize database? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_info "Keeping existing database"
                return 0
            fi
        fi
    else
        echo -e "  ${BLUE}ℹ${NC} No existing database found"
    fi
    
    # During setup, use soft check; otherwise use strict check
    if [[ "$during_setup" == "true" ]]; then
        if ! check_python_soft; then
            log_warning "Python environment not ready, skipping database initialization"
            echo -e "  ${YELLOW}  ${NC} Run './magentic.sh db-init' after setup completes"
            return 0
        fi
    else
        check_python
    fi
    
    mkdir -p "$DATA_DIR"
    echo -e "  ${GREEN}✓${NC} Data directory ready: $DATA_DIR"
    
    local alembic_bin="$VENV_DIR/bin/alembic"
    
    # Run Alembic migrations
    if [[ -f "$SCRIPT_DIR/alembic.ini" ]]; then
        cd "$SCRIPT_DIR"
        if [[ -x "$alembic_bin" ]]; then
            echo -ne "  ${BLUE}⠋${NC} Running database migrations..."
            if $alembic_bin upgrade head 2>&1 | tail -3; then
                echo -e "  ${GREEN}✓${NC} Database migrations applied"
            else
                echo -e "  ${YELLOW}⚠${NC} Migration had issues (may be OK if DB already exists)"
            fi
        else
            log_warning "Alembic not installed"
            echo -e "  ${YELLOW}  ${NC} Database migrations skipped"
        fi
    else
        log_info "No alembic.ini found - using default SQLite database"
        # Create a simple SQLite database if needed
        local python="$VENV_DIR/bin/python"
        if [[ -x "$python" ]]; then
            $python -c "
import sqlite3
import os
db_path = os.path.join('$DATA_DIR', 'magentic.db')
conn = sqlite3.connect(db_path)
conn.execute('CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)')
conn.commit()
conn.close()
print('  ✓ SQLite database initialized:', db_path)
" 2>/dev/null && echo -e "  ${GREEN}✓${NC} Database initialized"
        fi
    fi
    
    cd "$SCRIPT_DIR"
    return 0
}

reset_database() {
    log_warning "This will delete all data!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [[ -f "$DB_PATH" ]]; then
            rm "$DB_PATH"
            log_success "Database deleted"
        fi
        
        # Re-initialize
        init_database
    else
        log_info "Cancelled"
    fi
}

# ============================================
# MCP Docker Management
# ============================================

start_mcp() {
    if ! check_docker; then
        log_warning "Skipping MCP services (Docker not available)"
        return 1
    fi
    
    log_info "Starting MCP services..."
    
    cd "$DOCKER_DIR"
    
    # Setup environment
    if [[ ! -f ".env" ]] && [[ -f ".env.template" ]]; then
        cp ".env.template" ".env"
        log_success "Created MCP .env from template"
    fi
    
    # Create directories
    mkdir -p shared-workspace data
    
    local compose_cmd=$(get_compose_cmd)
    
    # Build and start
    $compose_cmd build --quiet
    $compose_cmd up -d
    
    log_info "Waiting for MCP Gateway..."
    
    if wait_for_service "http://localhost:$MCP_GATEWAY_PORT/health" "MCP Gateway" 30; then
        log_success "MCP Gateway is ready"
        
        # Show server status
        local health=$(curl -sf "http://localhost:$MCP_GATEWAY_PORT/health" 2>/dev/null)
        if [[ -n "$health" ]]; then
            local healthy=$(echo "$health" | grep -o '"healthy_servers":[0-9]*' | cut -d: -f2)
            local total=$(echo "$health" | grep -o '"total_servers":[0-9]*' | cut -d: -f2)
            log_success "MCP Servers: $healthy/$total healthy"
        fi
    else
        log_error "MCP Gateway failed to start"
        $compose_cmd logs --tail=20 mcp-gateway
        cd "$SCRIPT_DIR"
        return 1
    fi
    
    cd "$SCRIPT_DIR"
    return 0
}

stop_mcp() {
    if ! check_docker; then
        return 0
    fi
    
    log_info "Stopping MCP services..."
    
    cd "$DOCKER_DIR"
    local compose_cmd=$(get_compose_cmd)
    $compose_cmd down
    cd "$SCRIPT_DIR"
    
    log_success "MCP services stopped"
}

remove_mcp() {
    if ! check_docker; then
        return 0
    fi
    
    log_warning "This will remove MCP containers, volumes, and data!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$DOCKER_DIR"
        local compose_cmd=$(get_compose_cmd)
        
        # Stop and remove with volumes
        $compose_cmd down -v --remove-orphans
        
        # Remove data
        if [[ -d "data" ]]; then
            rm -rf data/*
            log_success "MCP data cleared"
        fi
        
        cd "$SCRIPT_DIR"
        log_success "MCP resources removed"
    else
        log_info "Cancelled"
    fi
}

# ============================================
# Application Management
# ============================================

start_cli() {
    log_info "Starting Magentic CLI..."
    check_python
    
    local python=$(get_python)
    
    # Ask user about MCP
    echo
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}           CLI Configuration${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    
    local enable_mcp=false
    
    # Check if MCP Gateway is responding
    if is_mcp_running; then
        echo -e "  ${GREEN}✓${NC} MCP Gateway is running (http://localhost:$MCP_GATEWAY_PORT)"
        read -p "  Enable MCP integration? (Y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            enable_mcp=true
        fi
    else
        # Gateway not responding - check why
        if is_mcp_containers_running; then
            echo -e "  ${YELLOW}⚠${NC} MCP containers running but Gateway not responding"
            echo -e "  ${BLUE}ℹ${NC} Gateway may still be starting up..."
            read -p "  Wait and retry, or restart MCP? (w=wait/r=restart/n=skip) [w]: " -n 1 -r
            echo
            case $REPLY in
                r|R)
                    echo
                    echo -e "  ${BLUE}ℹ${NC} Restarting MCP services..."
                    stop_mcp
                    sleep 2
                    if start_mcp && is_mcp_running; then
                        enable_mcp=true
                        echo -e "  ${GREEN}✓${NC} MCP Gateway restarted successfully"
                    else
                        echo -e "  ${RED}✗${NC} MCP Gateway failed to start"
                    fi
                    ;;
                n|N)
                    echo -e "  ${BLUE}ℹ${NC} Skipping MCP"
                    ;;
                *)
                    # Wait and retry
                    echo -ne "  ${BLUE}⠋${NC} Waiting for MCP Gateway..."
                    local attempts=0
                    while [[ $attempts -lt 15 ]]; do
                        if is_mcp_running; then
                            echo -e "\r  ${GREEN}✓${NC} MCP Gateway is now responding"
                            enable_mcp=true
                            break
                        fi
                        sleep 2
                        ((attempts++))
                        echo -ne "\r  ${BLUE}⠋${NC} Waiting for MCP Gateway... ($attempts/15)"
                    done
                    if [[ "$enable_mcp" != true ]]; then
                        echo -e "\r  ${RED}✗${NC} MCP Gateway still not responding    "
                    fi
                    ;;
            esac
        elif check_docker_silent; then
            echo -e "  ${YELLOW}⚠${NC} MCP Gateway is not running"
            read -p "  Enable MCP? This will start Docker services (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo
                echo -e "  ${BLUE}ℹ${NC} Starting MCP Gateway..."
                echo
                if start_mcp; then
                    # Wait a moment for gateway to be fully ready
                    sleep 2
                    if is_mcp_running; then
                        enable_mcp=true
                        echo
                        echo -e "  ${GREEN}✓${NC} MCP Gateway started successfully"
                    else
                        echo -e "  ${RED}✗${NC} MCP Gateway failed to respond"
                    fi
                else
                    echo -e "  ${RED}✗${NC} Failed to start MCP services"
                fi
            fi
        else
            if ! command -v docker &> /dev/null; then
                echo -e "  ${BLUE}ℹ${NC} Docker not installed - MCP services unavailable"
            else
                echo -e "  ${BLUE}ℹ${NC} Docker daemon not running - MCP services unavailable"
            fi
            read -p "  Continue without MCP? (Y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Nn]$ ]]; then
                log_info "Cancelled"
                return 0
            fi
        fi
    fi
    
    echo
    
    # Set MCP environment
    if [[ "$enable_mcp" == true ]]; then
        export ENABLE_MCP=true
        export MCP_GATEWAY_URL="http://localhost:$MCP_GATEWAY_PORT"
        log_success "MCP integration enabled"
    else
        export ENABLE_MCP=false
        log_info "Running without MCP"
    fi
    
    echo
    log_info "Starting interactive mode..."
    echo
    
    # Run the CLI
    cd "$SCRIPT_DIR"
    exec $python -m src.main
}

start_api() {
    log_info "Starting Magentic API server..."
    check_python
    
    if is_api_running; then
        log_warning "API server is already running"
        return 0
    fi
    
    local python=$(get_python)
    
    # Set environment
    if is_mcp_running; then
        export ENABLE_MCP=true
        export MCP_GATEWAY_URL="http://localhost:$MCP_GATEWAY_PORT"
        log_info "MCP integration enabled"
    else
        export ENABLE_MCP=false
        log_warning "Running without MCP (Docker not running)"
    fi
    
    # Check if metrics should be enabled (from .env or default to true)
    if [[ -f "$SCRIPT_DIR/.env" ]]; then
        source "$SCRIPT_DIR/.env" 2>/dev/null || true
    fi
    export ENABLE_METRICS=${ENABLE_METRICS:-true}
    if [[ "$ENABLE_METRICS" == "true" ]]; then
        log_info "Prometheus metrics enabled (/metrics endpoint)"
    fi
    
    mkdir -p "$DATA_DIR"
    
    # Start API in background
    cd "$SCRIPT_DIR"
    nohup $python -m uvicorn src.api:app --host 0.0.0.0 --port $API_PORT > "$DATA_DIR/api.log" 2>&1 &
    echo $! > "$API_PID_FILE"
    
    log_info "Waiting for API server..."
    
    if wait_for_service "http://localhost:$API_PORT/health" "API" 30; then
        log_success "API server running at http://localhost:$API_PORT"
    else
        log_error "API server failed to start"
        cat "$DATA_DIR/api.log" | tail -20
        return 1
    fi
}

stop_api() {
    if is_api_running; then
        local pid=$(cat "$API_PID_FILE")
        kill "$pid" 2>/dev/null || true
        rm -f "$API_PID_FILE"
        log_success "API server stopped"
    else
        log_info "API server is not running"
    fi
}

start_frontend() {
    log_info "Starting frontend dev server..."
    
    if [[ ! -d "$SCRIPT_DIR/frontend" ]]; then
        log_error "Frontend directory not found"
        return 1
    fi
    
    cd "$SCRIPT_DIR/frontend"
    
    if [[ ! -d "node_modules" ]]; then
        log_info "Installing frontend dependencies..."
        npm install
    fi
    
    # Start in background
    nohup npm run dev > "$DATA_DIR/frontend.log" 2>&1 &
    echo $! > "$DATA_DIR/.frontend.pid"
    
    sleep 3
    log_success "Frontend running at http://localhost:8081"
    cd "$SCRIPT_DIR"
}

stop_frontend() {
    if [[ -f "$DATA_DIR/.frontend.pid" ]]; then
        local pid=$(cat "$DATA_DIR/.frontend.pid")
        kill "$pid" 2>/dev/null || true
        rm -f "$DATA_DIR/.frontend.pid"
        log_success "Frontend stopped"
    fi
}

# ============================================
# Observability Stack (Prometheus, Grafana, Loki)
# ============================================

start_observability() {
    if ! check_docker; then
        log_error "Docker is required for observability stack"
        return 1
    fi
    
    echo
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}           Observability Stack${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    
    local compose_cmd=$(get_compose_cmd)
    cd "$DOCKER_DIR"
    
    log_info "Starting Prometheus, Grafana, Loki, Promtail, cAdvisor..."
    echo
    
    # Start services with visible output
    echo -e "  ${BLUE}⠋${NC} Pulling/starting containers..."
    if $compose_cmd --profile observability up -d prometheus loki promtail grafana cadvisor 2>&1; then
        echo
        
        # Wait for Prometheus
        echo -ne "  ${BLUE}⠋${NC} Waiting for Prometheus..."
        local prometheus_ready=false
        for i in {1..30}; do
            if curl -sf "http://localhost:9090/-/healthy" > /dev/null 2>&1; then
                prometheus_ready=true
                echo -e "\r  ${GREEN}✓${NC} Prometheus ready              "
                break
            fi
            echo -ne "\r  ${BLUE}⠋${NC} Waiting for Prometheus... ($i/30)"
            sleep 1
        done
        if ! $prometheus_ready; then
            echo -e "\r  ${YELLOW}⚠${NC} Prometheus still starting...     "
        fi
        
        # Wait for Grafana
        echo -ne "  ${BLUE}⠋${NC} Waiting for Grafana..."
        local grafana_ready=false
        for i in {1..30}; do
            if curl -sf "http://localhost:3000/api/health" > /dev/null 2>&1; then
                grafana_ready=true
                echo -e "\r  ${GREEN}✓${NC} Grafana ready                  "
                break
            fi
            echo -ne "\r  ${BLUE}⠋${NC} Waiting for Grafana... ($i/30)"
            sleep 1
        done
        if ! $grafana_ready; then
            echo -e "\r  ${YELLOW}⚠${NC} Grafana still starting...       "
        fi
        
        # Wait for Loki
        echo -ne "  ${BLUE}⠋${NC} Waiting for Loki..."
        local loki_ready=false
        for i in {1..20}; do
            if curl -sf "http://localhost:3100/ready" > /dev/null 2>&1; then
                loki_ready=true
                echo -e "\r  ${GREEN}✓${NC} Loki ready                     "
                break
            fi
            echo -ne "\r  ${BLUE}⠋${NC} Waiting for Loki... ($i/20)"
            sleep 1
        done
        if ! $loki_ready; then
            echo -e "\r  ${YELLOW}⚠${NC} Loki still starting...          "
        fi
        
        echo
        if $prometheus_ready && $grafana_ready; then
            log_success "Observability stack is running!"
            echo
            echo -e "  ${CYAN}Prometheus:${NC} http://localhost:9090"
            echo -e "  ${CYAN}Grafana:${NC}    http://localhost:3000 (admin/magentic123)"
            echo -e "  ${CYAN}Loki:${NC}       http://localhost:3100"
            echo
            echo -e "  ${BLUE}ℹ${NC} Metrics endpoint: http://localhost:$API_PORT/metrics"
            echo -e "  ${BLUE}ℹ${NC} Data persisted to Docker volumes"
        else
            log_warning "Some services still starting - check ./magentic.sh metrics-status"
        fi
        echo
    else
        log_error "Failed to start observability stack"
        cd "$SCRIPT_DIR"
        return 1
    fi
    
    cd "$SCRIPT_DIR"
    return 0
}

stop_observability() {
    if ! check_docker_silent; then
        return 0
    fi
    
    log_info "Stopping observability stack..."
    
    local compose_cmd=$(get_compose_cmd)
    local docker_cmd="${compose_cmd%% *}"  # Extract docker command
    cd "$DOCKER_DIR"
    
    # Use 'down' instead of 'stop' to properly remove containers and network references
    echo -ne "  ${BLUE}⠋${NC} Stopping and removing observability containers..."
    $compose_cmd --profile observability down --remove-orphans 2>/dev/null
    echo -e "\r  ${GREEN}✓${NC} Observability containers removed              "
    
    # Prune any orphaned networks
    echo -ne "  ${BLUE}⠋${NC} Cleaning up networks..."
    $docker_cmd network prune -f > /dev/null 2>&1 || true
    echo -e "\r  ${GREEN}✓${NC} Networks cleaned up                          "
    
    echo
    log_success "Observability stack stopped"
    echo -e "  ${BLUE}ℹ${NC} Data is preserved in Docker volumes"
    cd "$SCRIPT_DIR"
}

observability_status() {
    echo
    echo -e "${CYAN}Observability Stack Status${NC}"
    echo
    
    echo -n "  Prometheus: "
    if curl -sf "http://localhost:9090/-/healthy" > /dev/null 2>&1; then
        echo -e "${GREEN}Running${NC} (http://localhost:9090)"
    else
        echo -e "${YELLOW}Not running${NC}"
    fi
    
    echo -n "  Grafana:    "
    if curl -sf "http://localhost:3000/api/health" > /dev/null 2>&1; then
        echo -e "${GREEN}Running${NC} (http://localhost:3000)"
    else
        echo -e "${YELLOW}Not running${NC}"
    fi
    
    echo -n "  Loki:       "
    if curl -sf "http://localhost:3100/ready" > /dev/null 2>&1; then
        echo -e "${GREEN}Running${NC} (http://localhost:3100)"
    else
        echo -e "${YELLOW}Not running${NC}"
    fi
    
    echo -n "  cAdvisor:   "
    if curl -sf "http://localhost:8081/healthz" > /dev/null 2>&1; then
        echo -e "${GREEN}Running${NC} (http://localhost:8081)"
    else
        echo -e "${YELLOW}Not running${NC}"
    fi
    
    echo
    echo -n "  Metrics Endpoint: "
    if curl -sf "http://localhost:$API_PORT/metrics" > /dev/null 2>&1; then
        echo -e "${GREEN}Available${NC} (http://localhost:$API_PORT/metrics)"
    else
        echo -e "${YELLOW}Unavailable${NC} - set ENABLE_METRICS=true and restart API"
    fi
    
    echo
    
    # Show volume info
    if check_docker_silent; then
        echo "  Data Volumes:"
        for vol in prometheus-data grafana-data loki-data; do
            if docker volume inspect "$vol" > /dev/null 2>&1; then
                local size=$(docker system df -v 2>/dev/null | grep "$vol" | awk '{print $3}' || echo "N/A")
                echo -e "    - ${vol}: ${GREEN}exists${NC}"
            fi
        done
    fi
    echo
}

# ============================================
# Full Stack Management
# ============================================

start_all() {
    print_banner
    log_info "Starting all Magentic services..."
    echo
    
    # Load .env to check settings
    if [[ -f "$SCRIPT_DIR/.env" ]]; then
        source "$SCRIPT_DIR/.env" 2>/dev/null || true
    fi
    
    # 1. Initialize database
    init_database
    echo
    
    # 2. Start MCP (optional, continues if fails)
    start_mcp || true
    echo
    
    # 3. Start Observability stack if ENABLE_METRICS=true and Docker available
    if [[ "${ENABLE_METRICS:-false}" == "true" ]] && check_docker_silent; then
        log_info "Starting observability stack (ENABLE_METRICS=true)..."
        start_observability || true
        echo
    fi
    
    # 4. Start API
    start_api
    echo
    
    # 5. Start Frontend (optional)
    if [[ -d "$SCRIPT_DIR/frontend" ]]; then
        start_frontend
        echo
    fi
    
    show_status
    
    echo
    log_success "Magentic is ready!"
    echo
    echo -e "  ${CYAN}API:${NC}      http://localhost:$API_PORT"
    echo -e "  ${CYAN}Frontend:${NC} http://localhost:8081"
    if is_mcp_running; then
        echo -e "  ${CYAN}MCP:${NC}      http://localhost:$MCP_GATEWAY_PORT"
    fi
    echo -e "  ${CYAN}Metrics:${NC}    http://localhost:$API_PORT/metrics"
    echo
    echo -e "  ${CYAN}Observability:${NC}"
    if curl -sf "http://localhost:9090/-/healthy" > /dev/null 2>&1; then
        echo -e "    Prometheus: http://localhost:9090 ${GREEN}(running)${NC}"
        echo -e "    Grafana:    http://localhost:3000 ${GREEN}(running)${NC} - admin/magentic123"
    else
        echo -e "    Prometheus: http://localhost:9090 ${YELLOW}(not running)${NC}"
        echo -e "    Grafana:    http://localhost:3000 ${YELLOW}(not running)${NC} - admin/magentic123"
        echo -e "    ${BLUE}ℹ${NC} Run ${YELLOW}./magentic.sh metrics${NC} to start"
    fi
    echo
    echo -e "  Run ${YELLOW}./magentic.sh cli${NC} for interactive mode"
    echo -e "  Run ${YELLOW}./magentic.sh stop${NC} to stop all services"
    echo
}

stop_all() {
    log_info "Stopping all Magentic services..."
    echo
    
    stop_frontend
    stop_api
    stop_observability 2>/dev/null || true
    stop_mcp
    
    echo
    log_success "All services stopped"
}

remove_all() {
    log_warning "This will remove ALL Magentic data and resources!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Stop everything first
        stop_frontend 2>/dev/null || true
        stop_api 2>/dev/null || true
        
        # Remove MCP with volumes
        if check_docker; then
            cd "$DOCKER_DIR"
            local compose_cmd=$(get_compose_cmd)
            $compose_cmd down -v --remove-orphans 2>/dev/null || true
            rm -rf data/* 2>/dev/null || true
            cd "$SCRIPT_DIR"
        fi
        
        # Remove database
        rm -f "$DB_PATH" 2>/dev/null || true
        
        # Remove logs and pid files
        rm -f "$DATA_DIR"/*.log 2>/dev/null || true
        rm -f "$DATA_DIR"/.*pid 2>/dev/null || true
        
        # Remove execution graphs
        rm -f "$SCRIPT_DIR/execution_graphs"/*.html 2>/dev/null || true
        
        log_success "All resources removed"
    else
        log_info "Cancelled"
    fi
}

# ============================================
# Status and Info
# ============================================

show_status() {
    echo
    echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}                  Service Status                   ${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
    echo
    
    # Database
    if [[ -f "$DB_PATH" ]]; then
        local db_size=$(du -h "$DB_PATH" 2>/dev/null | cut -f1)
        echo -e "  Database:  ${GREEN}●${NC} Ready ($db_size)"
    else
        echo -e "  Database:  ${YELLOW}○${NC} Not initialized"
    fi
    
    # MCP Gateway
    if is_mcp_running; then
        local health=$(curl -sf "http://localhost:$MCP_GATEWAY_PORT/health" 2>/dev/null)
        if [[ -n "$health" ]]; then
            local healthy=$(echo "$health" | grep -o '"healthy_servers":[0-9]*' | cut -d: -f2)
            local total=$(echo "$health" | grep -o '"total_servers":[0-9]*' | cut -d: -f2)
            echo -e "  MCP:       ${GREEN}●${NC} Running ($healthy/$total servers)"
        else
            echo -e "  MCP:       ${YELLOW}●${NC} Starting..."
        fi
    else
        echo -e "  MCP:       ${RED}○${NC} Stopped"
    fi
    
    # API Server
    if is_api_running; then
        echo -e "  API:       ${GREEN}●${NC} Running (port $API_PORT)"
    else
        echo -e "  API:       ${RED}○${NC} Stopped"
    fi
    
    # Frontend
    if [[ -f "$DATA_DIR/.frontend.pid" ]]; then
        local pid=$(cat "$DATA_DIR/.frontend.pid")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo -e "  Frontend:  ${GREEN}●${NC} Running (port 8081)"
        else
            echo -e "  Frontend:  ${RED}○${NC} Stopped"
        fi
    else
        echo -e "  Frontend:  ${RED}○${NC} Stopped"
    fi
    
    echo
    echo -e "${CYAN}═══════════════════════════════════════════════════${NC}"
}

show_logs() {
    local service=${1:-all}
    
    case $service in
        mcp)
            if check_docker; then
                cd "$DOCKER_DIR"
                $(get_compose_cmd) logs -f --tail=100
                cd "$SCRIPT_DIR"
            fi
            ;;
        api)
            if [[ -f "$DATA_DIR/api.log" ]]; then
                tail -f "$DATA_DIR/api.log"
            else
                log_error "API log not found"
            fi
            ;;
        frontend)
            if [[ -f "$DATA_DIR/frontend.log" ]]; then
                tail -f "$DATA_DIR/frontend.log"
            else
                log_error "Frontend log not found"
            fi
            ;;
        all|*)
            log_info "Use: ./magentic.sh logs [mcp|api|frontend]"
            ;;
    esac
}

# ============================================
# Setup / Installation
# ============================================

# Spinner characters
SPINNER_CHARS="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

spinner() {
    local pid=$1
    local msg=$2
    local i=0
    local len=${#SPINNER_CHARS}
    
    while kill -0 $pid 2>/dev/null; do
        local char="${SPINNER_CHARS:$i:1}"
        printf "\r${BLUE}${char}${NC} ${msg}"
        i=$(( (i + 1) % len ))
        sleep 0.1
    done
    printf "\r"
}

progress_bar() {
    local current=$1
    local total=$2
    local width=40
    local percent=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))
    
    printf "\r  ["
    printf "%${filled}s" | tr ' ' '█'
    printf "%${empty}s" | tr ' ' '░'
    printf "] %3d%% " "$percent"
}

install_dependencies() {
    echo
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}           Installing Dependencies${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    
    local pip="$VENV_DIR/bin/pip"
    local use_uv=false
    local uv_path=""
    
    # Check if uv is available and working (10-100x faster than pip)
    # Check common locations for uv
    if [[ -x "$HOME/.local/bin/uv" ]]; then
        uv_path="$HOME/.local/bin/uv"
    elif [[ -x "$HOME/.cargo/bin/uv" ]]; then
        uv_path="$HOME/.cargo/bin/uv"
    elif command -v uv &> /dev/null; then
        uv_path=$(command -v uv)
    fi
    
    # Verify uv actually works
    if [[ -n "$uv_path" ]] && "$uv_path" --version &> /dev/null; then
        use_uv=true
        local uv_ver=$("$uv_path" --version 2>/dev/null | head -1)
        echo -e "  ${GREEN}⚡${NC} Using ${CYAN}uv${NC} ($uv_ver) - 10-100x faster than pip"
        echo
    else
        echo -e "  ${YELLOW}💡${NC} Tip: Install ${CYAN}uv${NC} for 10-100x faster installs:"
        echo -e "     ${YELLOW}curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
        echo
        # Upgrade pip if using pip
        echo -e "  ${BLUE}⠋${NC} Upgrading pip, wheel, setuptools..."
        $pip install --upgrade pip wheel setuptools -q 2>/dev/null
        if [[ $? -eq 0 ]]; then
            echo -e "  ${GREEN}✓${NC} pip, wheel, setuptools upgraded"
        else
            echo -e "  ${RED}✗${NC} Failed to upgrade pip"
            return 1
        fi
    fi
    
    # Read requirements and count packages
    if [[ ! -f "$SCRIPT_DIR/requirements.txt" ]]; then
        log_error "requirements.txt not found"
        return 1
    fi
    
    # Install all requirements
    echo
    if [[ "$use_uv" == true ]]; then
        # Get the Python version from the venv
        local venv_python="$VENV_DIR/bin/python"
        local venv_python_ver=$($venv_python --version 2>/dev/null | cut -d' ' -f2)
        local venv_python_minor=$(echo "$venv_python_ver" | cut -d'.' -f1-2)
        
        echo -e "  ${BLUE}⚡${NC} Installing packages with uv..."
        echo -e "  ${BLUE}ℹ${NC} Target: Python $venv_python_ver (from venv)"
        echo -e "  ${YELLOW}  (this is much faster than pip)${NC}"
        echo
        
        # Run uv pip install with explicit Python path AND version constraint
        # --python specifies interpreter, --python-version ensures resolution for correct version
        "$uv_path" pip install -r "$SCRIPT_DIR/requirements.txt" \
            --python "$venv_python" \
            --python-version "$venv_python_minor"
        local uv_exit=$?
        
        if [[ $uv_exit -ne 0 ]]; then
            echo -e "  ${YELLOW}⚠${NC} uv failed, falling back to pip..."
            echo
            $pip install -r "$SCRIPT_DIR/requirements.txt" 2>&1 | while read -r line; do
                if [[ "$line" =~ ^Collecting ]]; then
                    echo -e "    ${BLUE}↓${NC} $line"
                elif [[ "$line" =~ ^Successfully ]]; then
                    echo -e "    ${GREEN}✓${NC} $line"
                fi
            done
        fi
    else
        echo -e "  ${BLUE}⠋${NC} Installing packages from requirements.txt..."
        echo -e "  ${YELLOW}  (this may take a few minutes)${NC}"
        echo
        
        # Install with visible progress
        $pip install -r "$SCRIPT_DIR/requirements.txt" 2>&1 | while read -r line; do
            if [[ "$line" =~ ^Collecting ]]; then
                echo -e "    ${BLUE}↓${NC} $line"
            elif [[ "$line" =~ ^Successfully ]]; then
                echo -e "    ${GREEN}✓${NC} $line"
            fi
        done
    fi
    
    local install_exit=${PIPESTATUS[0]}
    
    echo
    echo -e "  ${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Verify key packages
    echo -e "  ${BLUE}ℹ${NC} Verifying installation..."
    local verify_failed=0
    
    for pkg in langchain langgraph qdrant-client fastapi rich python-dotenv alembic; do
        if $pip show "$pkg" &>/dev/null; then
            local ver=$($pip show "$pkg" 2>/dev/null | grep "^Version:" | cut -d' ' -f2)
            echo -e "    ${GREEN}✓${NC} $pkg ($ver)"
        else
            echo -e "    ${RED}✗${NC} $pkg missing"
            ((verify_failed++))
        fi
    done
    
    echo
    if [[ $verify_failed -eq 0 ]]; then
        log_success "All dependencies installed successfully!"
    else
        log_warning "Some packages may need manual installation"
    fi
    
    return 0
}

check_prerequisites() {
    echo
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}           Checking Prerequisites${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    
    local has_errors=0
    local has_warnings=0
    
    # Check pyenv (REQUIRED)
    echo -e "  ${BLUE}pyenv${NC} (required for Python management)"
    if check_pyenv; then
        local pyenv_version=$($PYENV_BIN --version 2>/dev/null | awk '{print $2}')
        echo -e "    ${GREEN}✓${NC} pyenv $pyenv_version"
        echo -e "    ${GREEN}✓${NC} Location: $PYENV_ROOT"
        
        # List installed Python versions
        local installed_versions=$($PYENV_BIN versions --bare 2>/dev/null | tr '\n' ' ')
        if [[ -n "$installed_versions" ]]; then
            echo -e "    ${GREEN}✓${NC} Installed: $installed_versions"
        else
            echo -e "    ${YELLOW}⚠${NC} No Python versions installed via pyenv"
        fi
    else
        echo -e "    ${RED}✗${NC} pyenv not found"
        echo -e "    ${RED}  ${NC} Expected at: $PYENV_ROOT"
        has_errors=1
    fi
    
    # Check Python
    echo -e "  ${BLUE}Python${NC}"
    local best_python=$(find_best_python)
    local available_pythons=""
    local has_python_313=0
    
    # Check pyenv versions first
    if check_pyenv; then
        for ver in "3.13" "3.12" "3.11" "3.10"; do
            local pyenv_ver=$($PYENV_BIN versions --bare 2>/dev/null | grep "^$ver" | sort -V | tail -1)
            if [[ -n "$pyenv_ver" ]]; then
                if [[ "$ver" == "3.13" ]]; then
                    available_pythons="$available_pythons ${GREEN}$pyenv_ver (pyenv, recommended)${NC}"
                    has_python_313=1
                else
                    available_pythons="$available_pythons $pyenv_ver (pyenv)"
                fi
            fi
        done
    fi
    
    # Also check system Python
    for ver in "3.13" "3.12" "3.11" "3.10"; do
        if [[ -x "/usr/bin/python$ver" ]]; then
            local full_ver=$(/usr/bin/python$ver --version 2>/dev/null | cut -d' ' -f2)
            if [[ "$ver" == "3.13" ]]; then
                available_pythons="$available_pythons ${GREEN}$full_ver (system)${NC}"
                has_python_313=1
            else
                available_pythons="$available_pythons $full_ver (system)"
            fi
        fi
    done
    
    if [[ -n "$best_python" ]]; then
        local best_ver=$($best_python --version 2>/dev/null | cut -d' ' -f2)
        local best_minor=$(echo $best_ver | cut -d'.' -f2)
        
        if [[ "$best_minor" -ge 10 ]]; then
            echo -e "    ${GREEN}✓${NC} Available:$available_pythons"
            echo -e "    ${GREEN}✓${NC} Will use: $best_python ($best_ver)"
            
            # Warn if not using 3.13
            if [[ "$best_minor" != "13" && "$has_python_313" -eq 0 ]]; then
                echo -e "    ${YELLOW}⚠${NC} Python 3.13 recommended for best compatibility"
                echo -e "    ${YELLOW}  ${NC} Install with: $PYENV_BIN install 3.13"
            fi
            
            # Warn if using Python 3.14+ (bleeding edge)
            if [[ "$best_minor" -ge 14 ]]; then
                echo -e "    ${YELLOW}⚠${NC} Python 3.14+ detected - some packages may not have wheels"
                echo -e "    ${YELLOW}  ${NC} Install Python 3.13: $PYENV_BIN install 3.13"
                has_warnings=1
            fi
        else
            echo -e "    ${RED}✗${NC} Python $best_ver (requires 3.10+)"
            has_errors=1
        fi
    else
        echo -e "    ${RED}✗${NC} Python 3.10+ not found"
        if check_pyenv; then
            echo -e "    ${YELLOW}  ${NC} Install with: $PYENV_BIN install 3.13"
        fi
        has_errors=1
    fi
    
    # Check pip
    echo -e "  ${BLUE}pip${NC}"
    if command -v pip3 &> /dev/null || command -v pip &> /dev/null; then
        local pip_version=$(pip3 --version 2>/dev/null || pip --version 2>/dev/null | awk '{print $2}')
        echo -e "    ${GREEN}✓${NC} pip $pip_version"
    else
        echo -e "    ${YELLOW}⚠${NC} pip not found (will be installed with venv)"
    fi
    
    # Check Node.js and npm
    echo -e "  ${BLUE}Node.js & npm${NC} (for frontend)"
    if command -v node &> /dev/null; then
        local node_version=$(node --version)
        echo -e "    ${GREEN}✓${NC} Node.js $node_version"
    else
        echo -e "    ${YELLOW}⚠${NC} Node.js not found (frontend will be unavailable)"
        has_warnings=1
    fi
    
    if command -v npm &> /dev/null; then
        local npm_version=$(npm --version)
        echo -e "    ${GREEN}✓${NC} npm $npm_version"
    else
        echo -e "    ${YELLOW}⚠${NC} npm not found (frontend will be unavailable)"
        has_warnings=1
    fi
    
    # Check Docker
    echo -e "  ${BLUE}Docker${NC} (for MCP services)"
    if command -v docker &> /dev/null; then
        local docker_version=$(docker --version | grep -oP '\d+\.\d+\.\d+' | head -1)
        echo -e "    ${GREEN}✓${NC} Docker $docker_version"
        
        if docker info &> /dev/null 2>&1; then
            echo -e "    ${GREEN}✓${NC} Docker daemon running"
        else
            echo -e "    ${YELLOW}⚠${NC} Docker daemon not running"
            has_warnings=1
        fi
    else
        echo -e "    ${YELLOW}⚠${NC} Docker not found (MCP services will be unavailable)"
        has_warnings=1
    fi
    
    # Check docker-compose
    if command -v docker &> /dev/null; then
        if docker compose version &> /dev/null 2>&1; then
            local compose_version=$(docker compose version | grep -oP '\d+\.\d+\.\d+' | head -1)
            echo -e "    ${GREEN}✓${NC} Docker Compose $compose_version"
        elif command -v docker-compose &> /dev/null; then
            local compose_version=$(docker-compose --version | grep -oP '\d+\.\d+\.\d+' | head -1)
            echo -e "    ${GREEN}✓${NC} docker-compose $compose_version"
        else
            echo -e "    ${YELLOW}⚠${NC} Docker Compose not found"
            has_warnings=1
        fi
    fi
    
    # Check Git
    echo -e "  ${BLUE}Git${NC}"
    if command -v git &> /dev/null; then
        local git_version=$(git --version | awk '{print $3}')
        echo -e "    ${GREEN}✓${NC} Git $git_version"
    else
        echo -e "    ${YELLOW}⚠${NC} Git not found (optional)"
    fi
    
    # Check curl
    echo -e "  ${BLUE}curl${NC}"
    if command -v curl &> /dev/null; then
        echo -e "    ${GREEN}✓${NC} curl available"
    else
        echo -e "    ${YELLOW}⚠${NC} curl not found (needed for health checks)"
        has_warnings=1
    fi
    
    echo
    echo -e "  ${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [[ $has_errors -eq 1 ]]; then
        echo
        log_error "Missing required prerequisites. Please install them first."
        echo
        if ! check_pyenv; then
            echo -e "  ${YELLOW}Install pyenv:${NC}"
            echo "    curl https://pyenv.run | bash"
            echo ""
            echo "  Then add to your ~/.bashrc or ~/.zshrc:"
            echo '    export PYENV_ROOT="$HOME/.pyenv"'
            echo '    export PATH="$PYENV_ROOT/bin:$PATH"'
            echo '    eval "$(pyenv init -)"'
            echo ""
            echo "  After installing pyenv, install Python 3.13:"
            echo "    pyenv install 3.13"
            echo ""
        else
            echo -e "  ${YELLOW}Install Python 3.13 via pyenv:${NC}"
            echo "    $PYENV_BIN install 3.13"
            echo ""
        fi
        return 1
    fi
    
    if [[ $has_warnings -eq 1 ]]; then
        echo
        log_warning "Some optional tools are missing. The system will work with reduced features."
        echo
        echo "  To enable all features, install:"
        echo "    ${YELLOW}Node.js:${NC} https://nodejs.org (for frontend)"
        echo "    ${YELLOW}Docker:${NC}  https://docker.com (for MCP services)"
        echo
        read -p "Continue anyway? (Y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            return 1
        fi
    fi
    
    return 0
}

setup_frontend() {
    echo
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}           Setting Up Frontend${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    
    if [[ ! -d "$SCRIPT_DIR/frontend" ]]; then
        log_warning "Frontend directory not found"
        return 1
    fi
    
    if ! command -v npm &> /dev/null; then
        log_warning "npm not found - skipping frontend setup"
        echo "  Install Node.js from https://nodejs.org to enable frontend"
        return 1
    fi
    
    cd "$SCRIPT_DIR/frontend"
    
    if [[ -d "node_modules" ]]; then
        log_info "Frontend dependencies already installed"
        read -p "  Reinstall? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            cd "$SCRIPT_DIR"
            return 0
        fi
        rm -rf node_modules package-lock.json
    fi
    
    echo -ne "  ${BLUE}⠋${NC} Installing npm packages..."
    npm install 2>&1 | tail -1 &
    spinner $! "Installing npm packages..."
    wait $!
    
    if [[ -d "node_modules" ]]; then
        local pkg_count=$(find node_modules -maxdepth 1 -type d | wc -l)
        echo -e "  ${GREEN}✓${NC} Frontend ready ($pkg_count packages installed)"
    else
        echo -e "  ${RED}✗${NC} npm install failed"
        cd "$SCRIPT_DIR"
        return 1
    fi
    
    cd "$SCRIPT_DIR"
    return 0
}

setup_docker_mcp() {
    echo
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}           Setting Up MCP (Docker)${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    
    if ! command -v docker &> /dev/null; then
        log_warning "Docker not installed - skipping MCP setup"
        echo "  Install Docker from https://docker.com to enable MCP services"
        return 1
    fi
    
    if ! docker info &> /dev/null 2>&1; then
        log_warning "Docker daemon not running - skipping MCP setup"
        echo "  Start Docker and run: ./magentic.sh mcp"
        return 1
    fi
    
    if [[ ! -d "$DOCKER_DIR" ]]; then
        log_warning "Docker directory not found"
        return 1
    fi
    
    cd "$DOCKER_DIR"
    
    # Setup .env
    if [[ ! -f ".env" ]] && [[ -f ".env.template" ]]; then
        cp ".env.template" ".env"
        log_success "Created MCP .env from template"
    fi
    
    # Create directories
    mkdir -p shared-workspace data
    
    echo
    read -p "  Build MCP Docker images now? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        local compose_cmd=$(get_compose_cmd)
        
        echo -ne "  ${BLUE}⠋${NC} Building MCP images (this may take a while)..."
        $compose_cmd build 2>&1 | tail -1 &
        spinner $! "Building MCP images..."
        wait $!
        
        if [[ $? -eq 0 ]]; then
            echo -e "  ${GREEN}✓${NC} MCP images built"
            echo
            echo "  Start MCP services with: ${YELLOW}./magentic.sh mcp${NC}"
        else
            echo -e "  ${YELLOW}⚠${NC} MCP build had issues - try: cd docker && docker compose build"
        fi
    else
        log_info "Skipped MCP build. Run later: ./magentic.sh mcp"
    fi
    
    cd "$SCRIPT_DIR"
    return 0
}

run_setup() {
    print_banner
    echo -e "${CYAN}First-Time Setup${NC}"
    
    # Step 1: Check all prerequisites
    if ! check_prerequisites; then
        exit 1
    fi
    
    # Step 2: Create virtual environment
    echo
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}           Python Virtual Environment${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    
    local need_install_deps=true
    local best_python=$(find_best_python)
    local python_ver=""
    local best_minor=""
    
    if [[ -n "$best_python" ]]; then
        python_ver=$($best_python --version 2>/dev/null | cut -d' ' -f2)
        best_minor=$(echo "$python_ver" | cut -d'.' -f2)
    fi
    
    # Check if we need to install Python 3.13
    if [[ -z "$best_python" ]] || [[ "$best_minor" -lt 10 ]] || [[ "$best_minor" -ge 14 ]]; then
        if check_pyenv; then
            echo -e "  ${YELLOW}⚠${NC} Python 3.13 not found (best for package compatibility)"
            echo
            read -p "  Install Python 3.13 via pyenv? (Y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                if install_python_via_pyenv "3.13"; then
                    best_python=$(find_best_python)
                    python_ver=$($best_python --version 2>/dev/null | cut -d' ' -f2)
                    best_minor=$(echo "$python_ver" | cut -d'.' -f2)
                else
                    log_error "Failed to install Python 3.13"
                    exit 1
                fi
            fi
        fi
    fi
    
    if [[ -z "$best_python" ]]; then
        log_error "No suitable Python found. Install Python 3.13 via pyenv:"
        echo "    $PYENV_BIN install 3.13"
        exit 1
    fi
    
    echo -e "  ${BLUE}ℹ${NC} Using: ${CYAN}$best_python${NC} ($python_ver)"
    echo
    
    if [[ -d "$VENV_DIR" ]]; then
        # Check current venv Python version
        local venv_python_ver=""
        if [[ -f "$VENV_DIR/bin/python" ]]; then
            venv_python_ver=$("$VENV_DIR/bin/python" --version 2>/dev/null | cut -d' ' -f2)
        fi
        
        log_warning "Virtual environment already exists (Python $venv_python_ver)"
        
        # Check if venv is using a different Python version
        local venv_minor=$(echo "$venv_python_ver" | cut -d'.' -f2)
        local best_minor=$(echo "$python_ver" | cut -d'.' -f2)
        
        if [[ "$venv_minor" != "$best_minor" ]]; then
            echo -e "  ${YELLOW}⚠${NC} Current venv uses Python 3.$venv_minor, but 3.$best_minor is available"
            echo -e "  ${YELLOW}  ${NC} Recreating with Python 3.$best_minor is recommended for better compatibility"
        fi
        
        read -p "  Recreate it with $best_python? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
            $best_python -m venv "$VENV_DIR"
            log_success "Virtual environment recreated with Python $python_ver"
        else
            log_info "Using existing virtual environment"
            # Check if deps already installed
            local python="$VENV_DIR/bin/python"
            if $python -c "import dotenv, langchain, fastapi" 2>/dev/null; then
                log_info "Dependencies already installed"
                read -p "  Reinstall dependencies? (y/N): " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    need_install_deps=false
                fi
            fi
        fi
    else
        echo -ne "  ${BLUE}⠋${NC} Creating virtual environment with $best_python..."
        $best_python -m venv "$VENV_DIR" &
        spinner $! "Creating virtual environment..."
        wait $!
        log_success "Virtual environment created with Python $python_ver"
    fi
    
    # Step 3: Install Python dependencies
    if [[ "$need_install_deps" == true ]]; then
        install_dependencies
    fi
    
    # Step 4: Configure LLM Provider and .env
    echo
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}           LLM Configuration${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    
    local need_llm_config=true
    local llm_provider="ollama"
    local openai_key=""
    local anthropic_key=""
    
    # Check if .env already exists FIRST
    if [[ -f "$SCRIPT_DIR/.env" ]]; then
        # Show current configuration
        local current_provider=$(grep "^LLM_PROVIDER=" "$SCRIPT_DIR/.env" 2>/dev/null | cut -d'=' -f2)
        if [[ -n "$current_provider" ]]; then
            echo -e "  ${BLUE}ℹ${NC} Current configuration:"
            echo -e "    Provider: ${CYAN}$current_provider${NC}"
        fi
        
        log_warning ".env file already exists"
        read -p "  Reconfigure LLM settings? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Keeping existing .env configuration"
            need_llm_config=false
            # Read provider from existing .env for ollama setup check
            llm_provider="$current_provider"
        fi
    fi
    
    # Only ask for model/key if user wants to configure
    if [[ "$need_llm_config" == true ]]; then
        echo
        echo "Select your LLM provider:"
        echo -e "  ${GREEN}1)${NC} Ollama (Local, Free) ${YELLOW}[Recommended]${NC}"
        echo -e "  ${GREEN}2)${NC} OpenAI (Requires API key)"
        echo -e "  ${GREEN}3)${NC} Claude/Anthropic (Requires API key)"
        echo
        read -p "Enter choice [1-3]: " provider_choice
        
        case $provider_choice in
            1)
                llm_provider="ollama"
                log_info "Selected: Ollama (local)"
                ;;
            2)
                llm_provider="openai"
                log_info "Selected: OpenAI"
                read -p "Enter your OpenAI API key: " openai_key
                ;;
            3)
                llm_provider="claude"
                log_info "Selected: Claude/Anthropic"
                read -p "Enter your Anthropic API key: " anthropic_key
                ;;
            *)
                log_warning "Invalid choice. Defaulting to Ollama."
                ;;
        esac
        
        # Create/overwrite .env file
        create_env_file "$llm_provider" "$openai_key" "$anthropic_key"
    fi
    
    # Step 5: Setup Ollama if selected
    if [[ "$llm_provider" == "ollama" ]]; then
        setup_ollama
    fi
    
    # Step 6: Create directories
    echo
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}              Creating Directories${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    mkdir -p "$SCRIPT_DIR/rag_data" "$SCRIPT_DIR/execution_graphs" "$DATA_DIR"
    echo -e "  ${GREEN}✓${NC} rag_data/"
    echo -e "  ${GREEN}✓${NC} execution_graphs/"
    echo -e "  ${GREEN}✓${NC} data/"
    log_success "Created required directories"
    
    # Step 7: Initialize database
    init_database true
    
    # Step 8: Setup Frontend (if npm is available)
    setup_frontend
    
    # Step 9: Setup Docker MCP (if Docker is available)
    setup_docker_mcp
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Setup Complete!
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    echo
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}           ✓ Setup Complete!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    echo -e "  ${CYAN}Quick Start:${NC}"
    echo -e "    ${GREEN}./magentic.sh start${NC}     Start all services"
    echo -e "    ${GREEN}./magentic.sh cli${NC}       Run interactive CLI"
    echo
    echo -e "  ${CYAN}Individual Services:${NC}"
    echo -e "    ${GREEN}./magentic.sh api${NC}       Start API server only"
    echo -e "    ${GREEN}./magentic.sh frontend${NC}  Start frontend only"
    echo -e "    ${GREEN}./magentic.sh mcp${NC}       Start MCP services"
    echo
    echo -e "  ${CYAN}Management:${NC}"
    echo -e "    ${GREEN}./magentic.sh status${NC}    Check service status"
    echo -e "    ${GREEN}./magentic.sh logs${NC}      View service logs"
    echo -e "    ${GREEN}./magentic.sh stop${NC}      Stop all services"
    echo
    echo -e "  ${CYAN}Documentation:${NC}"
    echo -e "    README.md, QUICKSTART.md, INSTALL.md"
    echo
}

create_env_file() {
    local provider=$1
    local openai_key=${2:-your-openai-api-key-here}
    local anthropic_key=${3:-your-anthropic-api-key-here}
    
    cat > "$SCRIPT_DIR/.env" << EOF
# Magentic Configuration
LLM_PROVIDER=$provider
LLM_TEMPERATURE=0.7

# Ollama Configuration
OLLAMA_MODEL=llama3.2:1b
OLLAMA_BASE_URL=http://localhost:11434

# OpenAI Configuration
OPENAI_API_KEY=$openai_key
OPENAI_MODEL=gpt-4o

# Anthropic/Claude Configuration
ANTHROPIC_API_KEY=$anthropic_key
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929

# Observability & Metrics
PHOENIX_PORT=6006
ENABLE_OBSERVABILITY=false
ENABLE_METRICS=true
MAX_INPUT_LENGTH=1000

# Multi-Agent Configuration
MAX_PARALLEL_AGENTS=3
UI_DISPLAY_LIMIT=200
DEBUG_STATE=false

# RAG Configuration
ENABLE_RAG=true
RAG_VECTOR_STORE=qdrant
RAG_QDRANT_MODE=memory
RAG_QDRANT_URL=http://localhost:6333
RAG_QDRANT_COLLECTION=knowledge_base
RAG_PERSIST_DIRECTORY=./rag_data
RAG_CHUNK_SIZE=1000
RAG_CHUNK_OVERLAP=200
RAG_TOP_K=4

# Embeddings
RAG_EMBEDDING_PROVIDER=ollama
RAG_EMBEDDING_MODEL=nomic-embed-text

# MCP Configuration
ENABLE_MCP=false
MCP_GATEWAY_URL=http://localhost:9000
EOF
    log_success ".env file created"
}

setup_ollama() {
    if ! command -v ollama &> /dev/null; then
        log_warning "Ollama is not installed"
        echo
        echo "Install Ollama from: https://ollama.ai/download"
        echo "  Linux/Mac: curl -fsSL https://ollama.ai/install.sh | sh"
        echo
        read -p "Install Ollama now? (Linux/Mac only) (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
                log_info "Installing Ollama..."
                curl -fsSL https://ollama.ai/install.sh | sh
                log_success "Ollama installed"
            else
                log_error "Automatic installation not supported on $OSTYPE"
                return 1
            fi
        else
            log_warning "Skipping Ollama installation"
            return 0
        fi
    else
        log_success "Ollama is installed"
    fi
    
    # Check if Ollama is running
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        log_success "Ollama service is running"
        
        # Pull models
        log_info "Pulling LLM model (llama3.2:1b)..."
        ollama pull llama3.2:1b 2>&1 | tail -1
        
        log_info "Pulling embedding model (nomic-embed-text)..."
        ollama pull nomic-embed-text 2>&1 | tail -1
        
        log_success "Models ready"
    else
        log_warning "Ollama is not running"
        echo "  Start it with: ollama serve"
        echo "  Then pull models: ollama pull llama3.2:1b && ollama pull nomic-embed-text"
    fi
}

# ============================================
# Help
# ============================================

show_help() {
    print_banner
    echo "Usage: ./magentic.sh <command> [options]"
    echo
    echo -e "${CYAN}Setup:${NC}"
    echo "  setup             First-time setup (venv, deps, config)"
    echo
    echo -e "${CYAN}Full Stack Commands:${NC}"
    echo "  start             Start all services (MCP, API, Frontend)"
    echo "  stop              Stop all services"
    echo "  restart           Restart all services"
    echo "  status            Show status of all services"
    echo "  remove            Remove all resources and data"
    echo
    echo -e "${CYAN}Individual Services:${NC}"
    echo "  cli               Run interactive CLI mode"
    echo "  api               Start API server only"
    echo "  api-stop          Stop API server"
    echo "  mcp               Start MCP Docker services"
    echo "  mcp-stop          Stop MCP services"
    echo "  mcp-remove        Remove MCP containers and data"
    echo "  frontend          Start frontend dev server"
    echo "  frontend-stop     Stop frontend"
    echo
    echo -e "${CYAN}Observability (Optional):${NC}"
    echo "  metrics           Start observability stack (Prometheus, Grafana, Loki)"
    echo "  metrics-stop      Stop observability stack"
    echo "  metrics-status    Show observability status"
    echo
    echo -e "${CYAN}Database:${NC}"
    echo "  db-init           Initialize/migrate database"
    echo "  db-reset          Reset database (deletes all data)"
    echo
    echo -e "${CYAN}Utilities:${NC}"
    echo "  logs [service]    Show logs (mcp|api|frontend)"
    echo "  health            Check health of all services"
    echo "  help              Show this help message"
    echo
    echo -e "${CYAN}Environment Variables:${NC}"
    echo "  MAGENTIC_API_PORT  API server port (default: 8000)"
    echo
    echo -e "${CYAN}Examples:${NC}"
    echo "  ./magentic.sh setup        # First-time setup"
    echo "  ./magentic.sh start        # Start everything"
    echo "  ./magentic.sh cli          # Interactive mode"
    echo "  ./magentic.sh stop         # Stop all services"
    echo "  ./magentic.sh logs mcp     # View MCP logs"
    echo
}

health_check() {
    echo -e "${CYAN}Health Check${NC}"
    echo
    
    # Database
    echo -n "Database: "
    if [[ -f "$DB_PATH" ]]; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${YELLOW}Not initialized${NC}"
    fi
    
    # MCP
    echo -n "MCP Gateway: "
    if curl -sf "http://localhost:$MCP_GATEWAY_PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}Unavailable${NC}"
    fi
    
    # API
    echo -n "API Server: "
    if curl -sf "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}Unavailable${NC}"
    fi
    
    # Frontend
    echo -n "Frontend: "
    if curl -sf "http://localhost:8081" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}Unavailable${NC}"
    fi
    
    # Metrics endpoint
    echo -n "Metrics: "
    if curl -sf "http://localhost:$API_PORT/metrics" > /dev/null 2>&1; then
        echo -e "${GREEN}OK (http://localhost:$API_PORT/metrics)${NC}"
    else
        echo -e "${YELLOW}Disabled${NC} (set ENABLE_METRICS=true)"
    fi
    
    # Observability stack
    echo -n "Prometheus: "
    if curl -sf "http://localhost:9090/-/healthy" > /dev/null 2>&1; then
        echo -e "${GREEN}OK (http://localhost:9090)${NC}"
    else
        echo -e "${YELLOW}Not running${NC} (run: ./magentic.sh metrics)"
    fi
    
    echo -n "Grafana: "
    if curl -sf "http://localhost:3000/api/health" > /dev/null 2>&1; then
        echo -e "${GREEN}OK (http://localhost:3000)${NC}"
    else
        echo -e "${YELLOW}Not running${NC} (run: ./magentic.sh metrics)"
    fi
    
    echo
}

# ============================================
# Main
# ============================================

main() {
    local command=${1:-help}
    shift 2>/dev/null || true
    
    case $command in
        # Setup
        setup|install|init)
            run_setup
            ;;
        
        # Full stack
        start)
            start_all
            ;;
        stop)
            stop_all
            ;;
        restart)
            stop_all
            sleep 2
            start_all
            ;;
        status)
            show_status
            ;;
        remove|clean)
            remove_all
            ;;
        
        # Individual services
        cli|run)
            start_cli
            ;;
        api)
            start_api
            ;;
        api-stop)
            stop_api
            ;;
        mcp)
            start_mcp
            ;;
        mcp-stop)
            stop_mcp
            ;;
        mcp-remove)
            remove_mcp
            ;;
        frontend|ui)
            start_frontend
            ;;
        frontend-stop|ui-stop)
            stop_frontend
            ;;
        
        # Observability
        metrics|observability)
            start_observability
            ;;
        metrics-stop|observability-stop)
            stop_observability
            ;;
        metrics-status|observability-status)
            observability_status
            ;;
        
        # Database
        db-init|db|migrate)
            init_database
            ;;
        db-reset)
            reset_database
            ;;
        
        # Utilities
        logs)
            show_logs "$@"
            ;;
        health)
            health_check
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            echo "Run './magentic.sh help' for usage"
            exit 1
            ;;
    esac
}

main "$@"
