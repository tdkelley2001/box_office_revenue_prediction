#!/bin/bash

echo "Checking for virtual environment."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists."
fi

# Activate the virtual environment
echo "Activating virtual environment."
source venv/bin/activate

# Install requirements
echo "Installing Python packages from requirements.txt."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete. Your environment is ready."
