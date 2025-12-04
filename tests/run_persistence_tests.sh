#!/bin/bash
# Persistence test runner for m2vdb
# Tests that data persists across container restarts

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}m2vdb Persistence Test Suite${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}🧹 Final cleanup...${NC}"
    docker-compose down
    echo -e "${GREEN}✓ Cleaned up${NC}"
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Phase 1: Create data
echo -e "${CYAN}Phase 1: Creating persistent data${NC}"
echo -e "${CYAN}========================================${NC}"

echo -e "${CYAN}📦 Starting docker-compose...${NC}"
docker-compose up -d
sleep 2

if ! docker-compose ps | grep -q "Up"; then
    echo -e "${RED}✗ Failed to start docker-compose${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker services started${NC}"
echo ""

echo -e "${CYAN}📝 Creating indexes with data...${NC}"
if uv run python tests/test_e2e_persistence.py --create; then
    echo -e "${GREEN}✓ Data created successfully${NC}"
else
    echo -e "${RED}✗ Failed to create data${NC}"
    exit 1
fi

# Phase 2: Restart and verify
echo ""
echo -e "${CYAN}Phase 2: Restarting container${NC}"
echo -e "${CYAN}========================================${NC}"

echo -e "${YELLOW}🔄 Stopping docker-compose...${NC}"
docker-compose down
sleep 2
echo -e "${GREEN}✓ Container stopped${NC}"

echo -e "${CYAN}📦 Starting docker-compose again...${NC}"
docker-compose up -d
sleep 3

if ! docker-compose ps | grep -q "Up"; then
    echo -e "${RED}✗ Failed to restart docker-compose${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker services restarted${NC}"
echo ""

# Phase 3: Verify persistence
echo -e "${CYAN}Phase 3: Verifying data persistence${NC}"
echo -e "${CYAN}========================================${NC}"

echo -e "${CYAN}🔍 Checking if data persisted...${NC}"
if uv run python tests/test_persistence_verify.py; then
    echo -e "${GREEN}✓ Persistence verification passed${NC}"
else
    echo -e "${RED}✗ Persistence verification failed${NC}"
    exit 1
fi

# Phase 4: Cleanup
echo ""
echo -e "${CYAN}Phase 4: Cleanup${NC}"
echo -e "${CYAN}========================================${NC}"

echo -e "${CYAN}🧹 Cleaning up test data...${NC}"
if uv run python tests/test_e2e_persistence.py --cleanup; then
    echo -e "${GREEN}✓ Test data cleaned up${NC}"
else
    echo -e "${YELLOW}⚠ Cleanup had issues (non-fatal)${NC}"
fi

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${GREEN}✅ Persistence tests completed successfully!${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo -e "${GREEN}Summary:${NC}"
echo -e "  ✓ Created indexes with data"
echo -e "  ✓ Stopped and restarted container"
echo -e "  ✓ Verified data persisted"
echo -e "  ✓ Cleaned up test data"
