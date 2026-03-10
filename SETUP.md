# Backend run instructions
cd backend
pip install -r requirements.txt
python app.py

# Frontend - open in browser or use local server
# Option 1: Open directly
frontend/index.html

# Option 2: Use Python server
cd frontend
python -m http.server 8000
# Then visit http://localhost:8000

# API runs on http://localhost:5000
