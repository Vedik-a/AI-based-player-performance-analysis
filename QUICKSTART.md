# 🏏 AI based Performance Enhancement - Getting Started

Your project is ready! Here's how to run it:

## ⚡ Quick Start (Windows)

1. **Install Dependencies** (one-time setup):
   ```bash
   install.bat
   ```

2. **Terminal 1 - Start Backend**:
   ```bash
   cd backend
   python app.py
   ```
   You should see: `Running on http://127.0.0.1:5000`

3. **Terminal 2 - Start Frontend**:
   ```bash
   cd frontend
   python -m http.server 8000
   ```

4. **Open Browser**:
   - Go to `http://localhost:8000`

## ⚡ Quick Start (Mac/Linux)

1. **Install Dependencies** (one-time setup):
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

2. **Terminal 1 - Start Backend**:
   ```bash
   cd backend
   python3 app.py
   ```

3. **Terminal 2 - Start Frontend**:
   ```bash
   cd frontend
   python3 -m http.server 8000
   ```

4. **Open Browser**:
   - Go to `http://localhost:8000`

## 🎮 Using the App

### Step 1: Train the Model
- Click the **"Train Model"** button
- Wait for training to complete (2-5 minutes)
- You'll see a success message when done
- The status dot will turn green

### Step 2: Make Predictions
- Click the upload area or drag-and-drop an image
- Click **"Predict Shot"**
- See the results with confidence scores!

## 📊 What Each Page Shows

- **Status**: Shows if the model is trained and ready
- **Train Model**: Train the AI on your cricket dataset
- **Predict Shot**: Upload images and get predictions
- **Shot Types**: List of all 6 cricket shots the model recognizes

## 🎨 Beautiful UI Features

✨ Modern gradient design with blue/purple theme
✨ Responsive layout works on phone, tablet, desktop
✨ Real-time confidence breakdown with bar charts
✨ Smooth animations and transitions
✨ Drag-and-drop file upload
✨ Clean, professional interface

## 🔧 Project Structure

```
cricket/
├── backend/
│   ├── app.py                 ← Flask API server
│   ├── pose_classifier.py     ← AI model & training
│   ├── requirements.txt       ← Python packages
│   ├── models/                ← Trained model files
│   └── uploads/               ← Temp image storage
├── frontend/
│   ├── index.html             ← Main page
│   ├── styles.css             ← Beautiful styling
│   └── script.js              ← Frontend logic
├── archive/
│   └── CricketCoachingDataSet/ ← Your cricket images
├── README.md                   ← Full documentation
└── SETUP.md                    ← Setup guide
```

## 📝 How It Works

1. **You upload an image** → Frontend sends to backend
2. **MediaPipe detects pose** → Finds 33 body keypoints
3. **AI model analyzes pose** → Compares to trained shots
4. **Model predicts shot** → Returns with confidence scores
5. **Results displayed** → Beautiful visualization on UI

## 🚨 Troubleshooting

**Model won't train?**
- Check that `archive/CricketCoachingDataSet/` folder exists
- Make sure shot folders (0-5) contain images
- Check backend console for error messages

**Images won't upload?**
- Use PNG or JPG format
- Keep file size under 10 MB
- Make sure it's a valid image

**Connection error?**
- Make sure backend is running on port 5000
- Make sure frontend is running on port 8000
- Clear browser cache

**Python not found?**
- Add Python to your system PATH
- Or use full path: `C:\Python\python.exe app.py`

## 💡 Tips

- Use clear, well-lit cricket shot images for best results
- Train with diverse images from different angles
- If accuracy is low, train with more images per shot
- Backend takes 2-5 minutes to train on first run

## 🎯 Next Steps

After running successfully:
1. Upload a cricket shot image
2. Click "Predict Shot"
3. Watch the AI identify the shot! 🎉

Happy predicting! 🏏✨
