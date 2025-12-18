#!/usr/bin/env python
"""
Script to train the pose classifier model on the cricket dataset.
Run this once to generate pose_model.pkl and scaler.pkl
"""

import sys
import os
from pose_classifier import PoseClassifier

if __name__ == '__main__':
    print("Starting model training...")
    print("="*70)
    
    classifier = PoseClassifier()
    dataset_path = os.path.join(os.path.dirname(__file__), '..', 'archive', 'CricketCoachingDataSet')
    
    # Ensure path exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        sys.exit(1)
    
    # Train the model
    success = classifier.train_model(dataset_path)
    
    if success:
        print("="*70)
        print("SUCCESS! Model training complete.")
        print(f"Model file: models/pose_model.pkl")
        print(f"Scaler file: models/scaler.pkl")
        print("="*70)
        print("\nYou can now run the Flask app with pre-trained model:")
        print("  python app.py")
        sys.exit(0)
    else:
        print("ERROR: Model training failed")
        sys.exit(1)
