#!/bin/bash
# End-to-end test runner for m2vdb
# Tests all API endpoints and functionality

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}m2vdb End-to-End Test Suite${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🧹 Cleaning up...${NC}"
    docker compose down
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Start docker-compose
echo -e "${CYAN}📦 Starting docker-compose...${NC}"
docker compose up -d

# Give it a moment to start
sleep 2

# Check if container is running
if ! docker compose ps | grep -q "Up"; then
    echo -e "${RED}✗ Failed to start docker-compose${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker services started${NC}"
echo ""

# Run main end-to-end tests
echo -e "${CYAN}🧪 Running end-to-end tests...${NC}"
echo -e "${CYAN}========================================${NC}"
if uv run python tests/test_e2e.py; then
    echo ""
    echo -e "${GREEN}✓ End-to-end tests passed${NC}"
else
    echo ""
    echo -e "${RED}✗ End-to-end tests failed${NC}"
    exit 1
fi

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${GREEN}✅ All tests completed successfully!${NC}"
echo -e "${CYAN}========================================${NC}"
