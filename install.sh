#!/bin/bash

echo "========================================="
echo "  AI based Performance Enhancement"
echo "  Quick Start Setup"
echo "========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org"
    exit 1
fi

echo "[1] Installing backend dependencies..."
cd backend
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo ""
echo "========================================="
echo "  Setup Complete!"
echo "========================================="
echo ""
echo "To run the project:"
echo ""
echo "1. Start Backend (in one terminal):"
echo "   cd backend"
echo "   python app.py"
echo ""
echo "2. Start Frontend (in another terminal):"
echo "   cd frontend"
echo "   python -m http.server 8000"
echo ""
echo "3. Open browser and go to:"
echo "   http://localhost:8000"
echo ""
echo "Backend API: http://localhost:5000"
echo ""
