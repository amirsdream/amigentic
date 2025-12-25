#!/bin/bash

# Magentic Setup Script
# This script sets up the entire project for first-time use

set -e  # Exit on error

echo "🚀 Magentic Setup Script"
echo "======================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python found: $(python3 --version)${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo ""
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt
echo -e "${GREEN}✓ Python dependencies installed${NC}"

# Create data directory if it doesn't exist
if [ ! -d "data" ]; then
    echo ""
    echo "📁 Creating data directory..."
    mkdir -p data
    echo -e "${GREEN}✓ Data directory created${NC}"
fi

# Check if database exists
if [ -f "data/magentic.db" ]; then
    echo ""
    echo -e "${YELLOW}⚠ Database already exists at data/magentic.db${NC}"
    read -p "Do you want to reset the database? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing old database..."
        rm data/magentic.db
        echo -e "${GREEN}✓ Old database removed${NC}"
    fi
fi

# Run database migrations
echo ""
echo "🗄️  Setting up database..."
if [ -f "data/magentic.db" ]; then
    # Database exists, just run migrations
    alembic upgrade head
    echo -e "${GREEN}✓ Database migrations applied${NC}"
else
    # New database, create and migrate
    alembic upgrade head
    echo -e "${GREEN}✓ Database created and initialized${NC}"
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo ""
        echo "⚙️  Creating .env file from .env.example..."
        cp .env.example .env
        echo -e "${GREEN}✓ .env file created${NC}"
        echo -e "${YELLOW}⚠ Please edit .env file to configure your LLM settings${NC}"
    else
        echo ""
        echo -e "${YELLOW}⚠ No .env.example file found. Please create .env manually${NC}"
    fi
else
    echo ""
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# Install frontend dependencies
if [ -d "frontend" ]; then
    echo ""
    echo "📦 Installing frontend dependencies..."
    cd frontend
    
    # Check if npm is installed
    if ! command -v npm &> /dev/null; then
        echo -e "${YELLOW}⚠ npm is not installed. Skipping frontend setup.${NC}"
        echo -e "${YELLOW}  Please install Node.js to use the web UI.${NC}"
    else
        npm install
        echo -e "${GREEN}✓ Frontend dependencies installed${NC}"
    fi
    
    cd ..
fi

# Summary
echo ""
echo "================================================"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Configure your LLM in .env file:"
echo "   - For Ollama (local): Set LLM_PROVIDER=ollama"
echo "   - For OpenAI: Set LLM_PROVIDER=openai and add API key"
echo "   - For Claude: Set LLM_PROVIDER=claude and add API key"
echo ""
echo "2. Start the backend:"
echo "   python -m src.run_api"
echo ""
echo "3. In a new terminal, start the frontend:"
echo "   cd frontend && npm run dev"
echo ""
echo "4. Open your browser to:"
echo "   http://localhost:3000"
echo ""
echo -e "${GREEN}Happy coding! 🎉${NC}"
echo ""
