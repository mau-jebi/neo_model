#!/bin/bash
# Start script for neo_model training dashboard

set -e

cd "$(dirname "$0")"

echo "Starting neo_model training dashboard..."
echo "========================================="

# Activate virtual environment if it exists
if [ -d "../venv" ]; then
    echo "Activating virtual environment..."
    source ../venv/bin/activate
fi

# Check if Flask is installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "Flask not found. Installing..."
    pip install flask
fi

# Check if training.log exists
LOG_PATH="$HOME/neo_model/training.log"
if [ ! -f "$LOG_PATH" ]; then
    echo "Warning: training.log not found at $LOG_PATH"
fi

# Start Flask app
echo "Starting Flask server on port 8080..."
echo "Access dashboard at: http://209.20.157.204:8080"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

python3 app.py
