#!/bin/bash
set -e

VENV_DIR=".venv"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "[NeuroStream] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"

# Install / update dependencies
echo "[NeuroStream] Checking dependencies..."
pip install -q -r requirements.txt

# Launch
echo "[NeuroStream] Starting..."
python main.py
