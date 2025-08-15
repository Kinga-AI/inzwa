#!/bin/bash
# Test runner for Inzwa MVP - phase by phase

echo "==================================="
echo "Inzwa MVP Test Runner"
echo "==================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to run phase tests
run_phase() {
    phase=$1
    echo -e "\n${GREEN}Testing Phase $phase${NC}"
    poetry run pytest tests/test_phase${phase}_*.py -v
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Phase $phase tests passed${NC}"
    else
        echo -e "${RED}✗ Phase $phase tests failed${NC}"
        exit 1
    fi
}

# Install dependencies if needed
if [ ! -d ".venv" ]; then
    echo "Installing dependencies..."
    poetry install
fi

# Run tests phase by phase
if [ "$1" == "1" ] || [ "$1" == "" ]; then
    run_phase 1
fi

if [ "$1" == "2" ] || [ "$1" == "" ]; then
    # run_phase 2  # Uncomment when Phase 2 is ready
    echo "Phase 2 tests not yet implemented"
fi

# Run linting
echo -e "\n${GREEN}Running code quality checks${NC}"
poetry run black --check src/
poetry run ruff check src/

echo -e "\n${GREEN}All tests completed!${NC}"
