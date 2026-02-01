#!/bin/bash
# Auto-start script for neo_model training dashboard
# This script is called automatically by the training pipeline

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DASHBOARD_DIR="$PROJECT_ROOT/dashboard"

echo "========================================="
echo "Starting neo_model Training Dashboard"
echo "========================================="

# Check if dashboard directory exists
if [ ! -d "$DASHBOARD_DIR" ]; then
    echo "Error: Dashboard directory not found at $DASHBOARD_DIR"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/venv" ]; then
    echo "Activating virtual environment..."
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "Flask not found. Installing Flask..."
    pip install flask
fi

# Check if dashboard is already running
DASHBOARD_PID=$(pgrep -f "python.*dashboard/app.py" || echo "")
if [ -n "$DASHBOARD_PID" ]; then
    echo "Dashboard already running (PID: $DASHBOARD_PID)"
    echo "Access at: http://$(curl -s ifconfig.me):8080"
    exit 0
fi

# Start dashboard server in background
cd "$DASHBOARD_DIR"
echo "Starting Flask server on port 8080..."

# Get public IP
PUBLIC_IP=$(curl -s ifconfig.me || echo "localhost")

nohup python app.py > dashboard.log 2>&1 &
DASHBOARD_PID=$!

# Wait a moment for server to start
sleep 2

# Verify it started
if ps -p $DASHBOARD_PID > /dev/null; then
    echo "✓ Dashboard started successfully (PID: $DASHBOARD_PID)"
    echo ""
    echo "==========================================="
    echo "Dashboard Access:"
    echo "  Public URL: http://$PUBLIC_IP:8080"
    echo "  Local URL:  http://localhost:8080"
    echo "==========================================="
    echo ""
    echo "Dashboard will auto-refresh every 30 seconds"
    echo "Logs available at: $DASHBOARD_DIR/dashboard.log"
else
    echo "✗ Dashboard failed to start"
    echo "Check logs at: $DASHBOARD_DIR/dashboard.log"
    exit 1
fi
