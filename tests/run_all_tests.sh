#!/bin/bash
# Master test runner for m2vdb
# Runs all end-to-end tests including persistence verification

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo ""
echo -e "${BOLD}${CYAN}╔════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║   m2vdb Complete Test Suite            ║${NC}"
echo -e "${BOLD}${CYAN}╚════════════════════════════════════════╝${NC}"
echo ""

# Parse arguments
RUN_E2E=true
RUN_PERSISTENCE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --e2e-only)
            RUN_PERSISTENCE=false
            shift
            ;;
        --persistence-only)
            RUN_E2E=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --e2e-only          Run only end-to-end tests (no persistence)"
            echo "  --persistence-only  Run only persistence tests"
            echo "  --help             Show this help message"
            echo ""
            echo "Default: Run both test suites"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Track overall status
OVERALL_STATUS=0

# Run end-to-end tests
if [ "$RUN_E2E" = true ]; then
    echo -e "${CYAN}${BOLD}[1/2] Running End-to-End Tests${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    if ./tests/run_e2e_tests.sh; then
        echo ""
        echo -e "${GREEN}✓ End-to-End Tests: PASSED${NC}"
    else
        echo ""
        echo -e "${RED}✗ End-to-End Tests: FAILED${NC}"
        OVERALL_STATUS=1
        
        if [ "$RUN_PERSISTENCE" = false ]; then
            exit $OVERALL_STATUS
        fi
    fi
    
    echo ""
    echo ""
fi

# Run persistence tests
if [ "$RUN_PERSISTENCE" = true ]; then
    if [ "$RUN_E2E" = true ]; then
        echo -e "${CYAN}${BOLD}[2/2] Running Persistence Tests${NC}"
    else
        echo -e "${CYAN}${BOLD}[1/1] Running Persistence Tests${NC}"
    fi
    echo -e "${CYAN}========================================${NC}"
    echo ""
    
    if ./tests/run_persistence_tests.sh; then
        echo ""
        echo -e "${GREEN}✓ Persistence Tests: PASSED${NC}"
    else
        echo ""
        echo -e "${RED}✗ Persistence Tests: FAILED${NC}"
        OVERALL_STATUS=1
    fi
    
    echo ""
fi

# Final summary
echo ""
echo -e "${CYAN}========================================${NC}"
if [ $OVERALL_STATUS -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✅ ALL TESTS PASSED${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    echo -e "${GREEN}Test Summary:${NC}"
    if [ "$RUN_E2E" = true ]; then
        echo -e "  ✓ End-to-end API tests"
    fi
    if [ "$RUN_PERSISTENCE" = true ]; then
        echo -e "  ✓ Data persistence verification"
    fi
else
    echo -e "${RED}${BOLD}❌ SOME TESTS FAILED${NC}"
    echo -e "${CYAN}========================================${NC}"
    echo ""
    echo -e "${RED}Please check the output above for details${NC}"
fi

echo ""
exit $OVERALL_STATUS
