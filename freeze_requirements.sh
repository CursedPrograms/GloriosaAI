#!/bin/bash

echo "📦 Generating requirements.txt from current environment..."

# Activate venv if it exists
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Generate requirements.txt
pip freeze > requirements.txt

echo "✅ requirements.txt created successfully!"
