#!/bin/bash

echo "==============================="
echo " PPE Detection System Starting "
echo "==============================="

# Step 1: Create virtual environment (if not exists)
if [ ! -d "env" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv env
fi

# Step 2: Activate environment
echo "[INFO] Activating environment..."
source env/bin/activate

# Step 3: Install dependencies
echo "[INFO] Installing requirements..."
pip install -r requirements.txt

# Step 4: Create required directories
echo "[INFO] Setting up folders..."
mkdir -p logs outputs

# Step 5: Run detection script
echo "[INFO] Running PPE detection..."
python main.py > logs/output.log 2>&1

echo "[SUCCESS] System is running!"
echo "Check logs in /logs folder"