# AI based Performance Enhancement

A simple yet effective AI-powered web application that identifies cricket shots using pose detection. Built with Flask, MediaPipe, OpenCV, and a modern web UI.

## 🎯 Features

- **AI-Powered Shot Detection**: Uses MediaPipe for pose detection and Random Forest classifier
- **6 Cricket Shot Types**:
  - Cut Shot
  - Cover Drive
  - Straight Drive
  - Pull Shot
  - Leg Glance Shot
  - Scoop Shot
- **Beautiful Web UI**: Modern, responsive interface with gradient design
- **Real-time Predictions**: Upload an image and get instant shot identification
- **Confidence Scores**: See detailed confidence breakdown for all shot types
- **Model Training**: Train the model using your own cricket dataset

## 📁 Project Structure

```
cricket/
├── archive/
│   └── CricketCoachingDataSet/
│       ├── 0. Cut Shot/
│       ├── 1. Cover Drive/
│       ├── 2. Straight Drive/
│       ├── 3. Pull Shot/
│       ├── 4. Leg Glance Shot/
│       └── 5. Scoop Shot/
├── backend/
│   ├── app.py                 # Flask API
│   ├── pose_classifier.py     # ML model
│   ├── models/                # Trained models
│   ├── uploads/               # Temporary uploads
│   └── requirements.txt       # Python dependencies
└── frontend/
    ├── index.html             # Main page
    ├── styles.css             # Modern UI styles
    └── script.js              # Frontend logic
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 2: Start the Backend

```bash
cd backend
python app.py
```

The API will start at `http://localhost:5000`

### Step 3: Open the Frontend

Open `frontend/index.html` in your web browser (or use a local server):

```bash
# Using Python's built-in server
cd frontend
python -m http.server 8000
```

Then visit `http://localhost:8000`

## 💻 Usage

1. **Train the Model**:
   - Click "Train Model" button
   - The backend will process images from the dataset and train the classifier
   - Status will update when training is complete

2. **Make Predictions**:
   - Click the upload area or drag-and-drop an image
   - Click "Predict Shot"
   - View results with confidence scores

## 🔧 API Endpoints

### Health Check
```
GET /api/health
```
Returns model status

### Train Model
```
POST /api/train
```
Request body:
```json
{
  "dataset_path": "../archive/CricketCoachingDataSet"
}
```

### Make Prediction
```
POST /api/predict
```
Multipart form data with image file

### Get Shot Types
```
GET /api/shots
```
Returns list of all shot types

## 🎨 UI Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Gradient Backgrounds**: Modern purple/blue gradient theme
- **Real-time Status**: Live model status indicator
- **Drag & Drop**: Easy image upload
- **Confidence Visualization**: Bar charts showing model confidence
- **Smooth Animations**: Professional transitions and effects

## 🧠 Technical Details

### Backend Stack
- **Framework**: Flask with CORS support
- **Computer Vision**: OpenCV, MediaPipe
- **ML**: Scikit-learn Random Forest Classifier
- **Data Processing**: NumPy, Pandas

### Frontend Stack
- **Markup**: HTML5
- **Styling**: CSS3 with CSS Grid and Flexbox
- **Scripting**: Vanilla JavaScript (no dependencies!)
- **API**: Fetch API for communication

## 📊 Model Details

- **Pose Detection**: MediaPipe's full-body pose estimation
- **Features**: 132 landmarks (33 keypoints × 4 values: x, y, z, visibility)
- **Classifier**: Random Forest (100 estimators)
- **Data Scaling**: StandardScaler normalization
- **Training Data**: Up to 50 images per shot type

## 🔒 Security

- File size limit: 10 MB
- Allowed formats: PNG, JPG, JPEG
- Uploaded files are deleted after processing
- CORS enabled for local development

## 📝 Configuration

Edit these settings in `backend/app.py`:
- `UPLOAD_FOLDER`: Where uploaded files are stored
- `ALLOWED_EXTENSIONS`: Supported image formats
- `MAX_FILE_SIZE`: Maximum upload size
- `app.run()`: Port and host settings

## 🐛 Troubleshooting

**Model not training?**
- Check dataset path is correct
- Ensure images exist in the shot directories
- Check console for error messages

**Predictions not working?**
- Make sure the model is trained first
- Verify image contains a human pose
- Check image format and size

**CORS errors?**
- Flask-CORS should be handling this
- Clear browser cache
- Check frontend URL matches backend CORS settings

## 🎓 How It Works

1. **Image Input**: User uploads cricket shot image
2. **Pose Detection**: MediaPipe detects human pose (33 keypoints)
3. **Feature Extraction**: Extracts 132 features from landmarks
4. **Normalization**: StandardScaler normalizes features
5. **Prediction**: Random Forest classifier predicts shot type
6. **Confidence**: Returns probability scores for all classes

## 📈 Performance

- Training time: ~2-5 minutes (depends on dataset size)
- Prediction time: ~500ms per image
- Accuracy: Varies based on dataset quality and shot diversity

## 🚀 Future Enhancements

- [ ] Video support for real-time detection
- [ ] Model accuracy metrics dashboard
- [ ] Data augmentation for better training
- [ ] Batch predictions
- [ ] Model versioning and deployment
- [ ] REST API documentation (Swagger)

## 📄 License

This project is free to use and modify.

## 👨‍💻 Author

Built with ❤️ for cricket enthusiasts and AI learners!
