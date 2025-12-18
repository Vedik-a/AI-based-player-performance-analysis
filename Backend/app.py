from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from werkzeug.utils import secure_filename
from pathlib import Path
from pose_classifier import PoseClassifier

# Serve frontend from the `frontend` directory so the app and UI share the same origin.
app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend'), static_url_path='/')
# Allow CORS for API routes (for cases where frontend is served separately)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize classifier
classifier = PoseClassifier()
# Ensure pre-trained model is loaded on startup (models exist in backend/models/)
try:
    classifier.load_model()
except Exception:
    pass

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_loaded = classifier.model is not None
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict cricket shot from uploaded image"""
    try:
        # Note: model is loaded lazily in `classifier.predict` if available on disk.
        
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No image provided'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'message': 'Invalid file format. Only PNG, JPG, JPEG allowed.'
            }), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result, error = classifier.predict(filepath)

        if error:
            return jsonify({
                'success': False,
                'message': error
            }), 400

        # Optionally generate player tips (without exposing image quality metrics)
        player_tips = None
        try:
            # Use lightweight player tips generator (no image quality metrics)
            player_tips = classifier._generate_player_tips_from_image(filepath)
        except Exception:
            player_tips = None

        # Generate injury risk assessment (BEFORE deleting file)
        injury_risk = None
        try:
            injury_risk = classifier.generate_injury_risk_from_image(filepath)
        except Exception:
            injury_risk = None

        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass

        response_payload = {
            'success': True,
            'prediction': result['shot'],
            'confidence': round(result['confidence'] * 100, 2),
            'all_predictions': {
                name: round(prob * 100, 2)
                for name, prob in result['all_predictions'].items()
            }
        }

        if player_tips is not None:
            response_payload['player_tips'] = player_tips

        if injury_risk is not None:
            response_payload['injury_risk'] = injury_risk

        return jsonify(response_payload)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/shots', methods=['GET'])
def get_shots():
    """Get list of all shot types"""
    return jsonify({
        'shots': list(classifier.label_map.values())
    })


# Serve the single-page frontend (index.html) at root when requested from the Flask server.
@app.route('/', methods=['GET'])
def serve_frontend():
    return app.send_static_file('index.html')

# Note: Image quality analysis endpoint removed. Player tips are returned by /api/predict.

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'success': False,
        'message': 'File too large. Maximum size is 10 MB.'
    }), 413

if __name__ == '__main__':
    print("Starting AI based Performance Enhancement API...")
    print("Dataset path: ../archive/CricketCoachingDataSet")
    app.run(debug=False, port=5000, host='0.0.0.0')
