#!/usr/bin/env bash

echo "Creating virtual environment..."
python3 -m venv projEnv

echo "Activating environment..."
source projEnv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Done! Environment ready."