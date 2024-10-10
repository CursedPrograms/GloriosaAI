#!/bin/bash

VENV_DIR="psdenv"

# Check if the virtual environment directory exists
if [ ! -d "$VENV_DIR" ]; then
    # Create the virtual environment
    python -m venv "$VENV_DIR"
fi

# Activate the virtual environment and run the Python script
source "$VENV_DIR/bin/activate"
python main.py

# Pause for user input before closing (optional)
read -p "Press Enter to continue..."