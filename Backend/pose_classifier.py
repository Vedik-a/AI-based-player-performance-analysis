import os
import cv2
import numpy as np
import pandas as pd
import pickle
import mediapipe as mp
# Defer heavy sklearn imports until needed (lazy import)
from pathlib import Path

class PoseClassifier:
    def __init__(self, model_path='models/pose_model.pkl', scaler_path='models/scaler.pkl'):
        self.mp_pose = mp.solutions.pose
        # Create a lightweight default pose object; more specific contexts
        # will be created during detection attempts when necessary.
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=1, smooth_landmarks=False)
        # Normalize model and scaler paths to absolute paths relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = model_path if os.path.isabs(model_path) else os.path.join(base_dir, model_path)
        self.scaler_path = scaler_path if os.path.isabs(scaler_path) else os.path.join(base_dir, scaler_path)
        self.model = None
        self.scaler = None
        self.label_map = {
            0: 'Cut Shot',
            1: 'Cover Drive',
            2: 'Straight Drive',
            3: 'Pull Shot',
            4: 'Leg Glance Shot',
            5: 'Scoop Shot'
        }
        # Do not eagerly load sklearn/model at import time to avoid slow startup.
        # Model will be loaded on demand (lazy) when `predict` or `train_model` is called.

    def _preprocess_image(self, image):
        """Apply preprocessing to improve pose detection (contrast/brightness enhancement)"""
        # Convert to LAB color space for better contrast adjustment
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge back and convert to BGR
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced

    def _gamma_correction(self, image, gamma=1.2):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype('uint8')
        return cv2.LUT(image, table)

    def _center_crop(self, image, target_ratio=0.75):
        h, w = image.shape[:2]
        new_w = int(w * target_ratio)
        new_h = int(h * target_ratio)
        x1 = max(0, (w - new_w) // 2)
        y1 = max(0, (h - new_h) // 2)
        return image[y1:y1 + new_h, x1:x1 + new_w]

    def _try_pose_on_image(self, image, model_complexity=1, min_detection_confidence=0.5):
        """Run MediaPipe Pose on provided image with given params. Returns results or None."""
        try:
            # Use a short-lived Pose context so parameters can vary between attempts
            with self.mp_pose.Pose(static_image_mode=True,
                                   model_complexity=model_complexity,
                                   min_detection_confidence=min_detection_confidence,
                                   smooth_landmarks=False) as pose:
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if results and results.pose_landmarks:
                    return results
        except Exception:
            return None
        return None

    def extract_pose_landmarks(self, image):
        """Extract pose landmarks from an image with multiple detection attempts."""
        # Build a set of transformed candidate images to try
        candidates = []
        try:
            candidates.append(image)
            candidates.append(self._preprocess_image(image))
            candidates.append(cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC))
            candidates.append(self._preprocess_image(cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)))
            candidates.append(self._gamma_correction(image, gamma=1.3))
            candidates.append(self._center_crop(image, target_ratio=0.85))
            # Rotations
            rot90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            rot270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            candidates.append(rot90)
            candidates.append(rot270)
        except Exception:
            candidates = [image]

        # Try a few model_complexity and confidence settings to improve detection
        for img_try in candidates:
            for model_complexity in (1, 2, 0):
                for det_conf in (0.3, 0.5):
                    results = self._try_pose_on_image(img_try, model_complexity=model_complexity, min_detection_confidence=det_conf)
                    if results and results.pose_landmarks:
                        landmarks = []
                        for landmark in results.pose_landmarks.landmark:
                            landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                        return np.array(landmarks)

        return None

    def train_model(self, dataset_path):
        """Train the pose classifier using the dataset"""
        print("Starting model training...")
        
        X = []
        y = []
        
        # Load images and extract landmarks
        for shot_idx, shot_dir in enumerate(sorted(os.listdir(dataset_path))):
            shot_path = os.path.join(dataset_path, shot_dir)
            if not os.path.isdir(shot_path):
                continue
            
            image_count = 0
            for img_file in os.listdir(shot_path):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                img_path = os.path.join(shot_path, img_file)
                image = cv2.imread(img_path)
                
                if image is None:
                    continue
                
                # Resize image for faster processing
                image = cv2.resize(image, (640, 480))
                
                landmarks = self.extract_pose_landmarks(image)
                if landmarks is not None:
                    X.append(landmarks)
                    y.append(shot_idx)
                    image_count += 1
                
                if image_count >= 50:  # Limit images per class for faster training
                    break
            
            print(f"  {shot_dir}: {image_count} images processed")
        
        if len(X) == 0:
            print("No landmarks extracted. Check dataset path.")
            return False
        
        X = np.array(X)
        y = np.array(y)
        
        # Standardize features (import locally to avoid global sklearn import)
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X_scaled, y)
        
        # Save model and scaler
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Model training complete! Saved to {self.model_path}")
        return True

    def load_model(self):
        """Load trained model and scaler"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("Model and scaler loaded successfully!")
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
        return False

    def predict(self, image_path):
        """Predict cricket shot from image"""
        # Lazy-load model/scaler if available on disk
        if self.model is None or self.scaler is None:
            loaded = self.load_model()
            if not loaded:
                return None, "Model not trained. Please train the model first."
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None, "Failed to load image"
            
            # Resize image
            image = cv2.resize(image, (640, 480))
            
            landmarks = self.extract_pose_landmarks(image)
            if landmarks is None:
                return None, "No human pose detected in the image"
            
            # Scale landmarks
            landmarks_scaled = self.scaler.transform([landmarks])
            
            # Get prediction and confidence
            prediction = self.model.predict(landmarks_scaled)[0]
            confidence = self.model.predict_proba(landmarks_scaled)[0][prediction]
            
            shot_name = self.label_map.get(prediction, "Unknown")
            
            return {
                'shot': shot_name,
                'confidence': float(confidence),
                'all_predictions': {
                    self.label_map[i]: float(prob)
                    for i, prob in enumerate(self.model.predict_proba(landmarks_scaled)[0])
                }
            }, None
        
        except Exception as e:
            return None, str(e)

    def cleanup(self):
        """Cleanup resources"""
        self.pose.close()

    def analyze_image(self, image_path):
        """DEPRECATED: Heavy image analysis removed.
        Use `_generate_player_tips_from_image(image_path)` for lightweight player tips generation.
        """
        return None, "analyze_image is deprecated. Use _generate_player_tips_from_image()"

    def _generate_player_tips(self, image, pose_results):
        """Generate cricket-specific improvement tips for players"""
        tips = []
        
        if not pose_results.pose_landmarks:
            tips.append({
                'category': 'Stance & Position',
                'tip': 'Position your entire body in the frame for proper technique analysis. A complete stance (head to feet) is essential for accurate shot classification.',
                'priority': 'Critical'
            })
            return tips
        
        # Extract key landmarks
        landmarks = pose_results.pose_landmarks.landmark
        
        # Left and right shoulder (indices 11, 12)
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        # Left and right hip (indices 23, 24)
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Left and right elbow (indices 13, 14)
        left_elbow = landmarks[13]
        right_elbow = landmarks[14]
        
        # Spine angle (shoulder to hip alignment)
        shoulder_y_diff = abs(left_shoulder.y - right_shoulder.y)
        hip_y_diff = abs(left_hip.y - right_hip.y)
        
        # Check stance balance
        if shoulder_y_diff > 0.1 or hip_y_diff > 0.1:
            tips.append({
                'category': 'Balance & Stance',
                'tip': 'Your stance appears unbalanced. Keep shoulders and hips level for better shot execution and consistency. Balance is key to power and control.',
                'priority': 'High'
            })
        else:
            tips.append({
                'category': 'Balance & Stance',
                'tip': '✓ Good stance balance! Maintain this level shoulder and hip position for consistent shot execution.',
                'priority': 'Success'
            })
        
        # Check arm position and bend
        # Elbow visibility and position
        if left_elbow.visibility > 0.5 and right_elbow.visibility > 0.5:
            left_elbow_bend = abs(left_elbow.y - left_shoulder.y) / max(abs(left_shoulder.y - left_hip.y), 0.01)
            right_elbow_bend = abs(right_elbow.y - right_shoulder.y) / max(abs(right_shoulder.y - right_hip.y), 0.01)
            
            if left_elbow_bend < 0.2 or right_elbow_bend < 0.2:
                tips.append({
                    'category': 'Arm Positioning',
                    'tip': 'Bend your elbows more during the shot. Proper arm bend provides better control and power generation.',
                    'priority': 'High'
                })
            else:
                tips.append({
                    'category': 'Arm Positioning',
                    'tip': '✓ Good arm bend! This provides excellent control and power generation for your shots.',
                    'priority': 'Success'
                })
        else:
            tips.append({
                'category': 'Arm Positioning',
                'tip': 'Keep your arms visible and properly positioned. Arms are crucial for shot technique and balance.',
                'priority': 'Medium'
            })
        
        # Check body rotation
        shoulder_x_spread = abs(left_shoulder.x - right_shoulder.x)
        hip_x_spread = abs(left_hip.x - right_hip.x)
        
        if shoulder_x_spread < 0.15 or hip_x_spread < 0.15:
            tips.append({
                'category': 'Body Rotation',
                'tip': 'Increase your body rotation and side-on positioning. A proper side-on stance provides more power and stability.',
                'priority': 'High'
            })
        else:
            tips.append({
                'category': 'Body Rotation',
                'tip': '✓ Good side-on positioning! Your body rotation is well-aligned for effective shot execution.',
                'priority': 'Success'
            })
        
        # Check head position
        head = landmarks[0]
        if head.visibility > 0.5:
            head_shoulder_alignment = abs(head.x - (left_shoulder.x + right_shoulder.x) / 2)
            
            if head_shoulder_alignment > 0.15:
                tips.append({
                    'category': 'Head Position',
                    'tip': 'Keep your head steady and aligned with your shoulders. A still head improves balance and shot accuracy.',
                    'priority': 'Medium'
                })
            else:
                tips.append({
                    'category': 'Head Position',
                    'tip': '✓ Excellent head alignment! This steady position helps with balance and shot accuracy.',
                    'priority': 'Success'
                })
        
        # Check knee bend
        left_knee = landmarks[25]
        right_knee = landmarks[26]
        
        if left_knee.visibility > 0.5 and right_knee.visibility > 0.5:
            left_knee_bend = abs(left_knee.y - left_hip.y) / max(abs(left_hip.y - landmarks[31].y), 0.01)
            right_knee_bend = abs(right_knee.y - right_hip.y) / max(abs(right_hip.y - landmarks[32].y), 0.01)
            
            avg_knee_bend = (left_knee_bend + right_knee_bend) / 2
            
            if avg_knee_bend < 0.3:
                tips.append({
                    'category': 'Leg & Knee Position',
                    'tip': 'Bend your knees more during the shot. Proper knee bend lowers your center of gravity for stability and power.',
                    'priority': 'High'
                })
            else:
                tips.append({
                    'category': 'Leg & Knee Position',
                    'tip': '✓ Great knee bend! This lowers your center of gravity for better stability and shot execution.',
                    'priority': 'Success'
                })
        
        # General tips based on quality score
        if not tips:  # Fallback
            tips.append({
                'category': 'Overall Technique',
                'tip': 'Focus on maintaining proper stance, balance, and body alignment for consistent shot execution.',
                'priority': 'Medium'
            })
        
        return tips

    def _generate_player_tips_from_image(self, image_path):
        """Lightweight helper: load image, run pose detection, and return player tips.

        This avoids computing brightness/sharpness/saturation and only runs the
        pose detector followed by the existing `_generate_player_tips` logic.
        Returns a list of tip dicts or None on failure.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            image = cv2.resize(image, (640, 480))

            # Build candidate images similar to extract_pose_landmarks
            candidates = [image, self._preprocess_image(image), self._gamma_correction(image, 1.3)]
            candidates.append(cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC))
            candidates.append(self._center_crop(image, 0.85))

            results = None
            for img_try in candidates:
                results = self._try_pose_on_image(img_try, model_complexity=1, min_detection_confidence=0.3)
                if results and results.pose_landmarks:
                    break

            if not results:
                return None

            tips = self._generate_player_tips(image, results)
            return tips
        except Exception:
            return None

    def _assess_injury_risk_from_pose_results(self, pose_results):
        """Assess injury risk from MediaPipe pose results.

        Returns a dict: { 'score': float(0-100), 'level': 'Low'|'Moderate'|'High', 'reasons': [str,...] }
        """
        reasons = []
        score = 0.0

        if not pose_results.pose_landmarks:
            return {
                'score': None,
                'level': 'Unknown',
                'reasons': ['No pose detected to assess injury risk']
            }

        landmarks = pose_results.pose_landmarks.landmark

        # Basic checks: shoulder/hip asymmetry, knee extension, head tilt, spine curvature
        try:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_knee = landmarks[25]
            right_knee = landmarks[26]
            left_ankle = landmarks[27]
            right_ankle = landmarks[28]
            head = landmarks[0]

            # Shoulder/hip level asymmetry (higher differences imply compensation)
            shoulder_y_diff = abs(left_shoulder.y - right_shoulder.y)
            hip_y_diff = abs(left_hip.y - right_hip.y)
            asymmetry = max(shoulder_y_diff, hip_y_diff)
            if asymmetry > 0.12:
                score += 35
                reasons.append('Marked shoulder/hip asymmetry — may indicate imbalance or compensation')
            elif asymmetry > 0.06:
                score += 15
                reasons.append('Moderate asymmetry between left and right sides')

            # Knee extension: very straight knees during a dynamic stance can increase joint load
            knee_extension_left = abs(left_knee.y - left_hip.y)
            knee_extension_right = abs(right_knee.y - right_hip.y)
            avg_knee_ext = (knee_extension_left + knee_extension_right) / 2
            # Smaller values (knees close to hips vertically) imply less bend => higher risk
            if avg_knee_ext < 0.12:
                score += 25
                reasons.append('Low knee bend — increases risk to knee and lower-back under load')
            elif avg_knee_ext < 0.2:
                score += 10

            # Head alignment
            shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
            head_offset = abs(head.x - shoulder_mid_x)
            if head_offset > 0.15:
                score += 10
                reasons.append('Head significantly off center — may reduce balance and increase fall risk')

            # Ankle visibility / stance width
            ankle_spread = abs(left_ankle.x - right_ankle.x)
            if ankle_spread < 0.12:
                score += 10
                reasons.append('Narrow stance width — may reduce stability')

        except Exception:
            # if landmark indices not present for some reason, return unknown
            return {
                'score': None,
                'level': 'Unknown',
                'reasons': ['Insufficient landmarks to assess injury risk']
            }

        # Cap score to 100
        score = min(100.0, score)

        if score >= 50:
            level = 'High'
        elif score >= 20:
            level = 'Moderate'
        else:
            level = 'Low'

        if not reasons:
            reasons.append('No major issues detected; maintain current technique and conditioning')

        return {
            'score': round(score, 1),
            'level': level,
            'reasons': reasons
        }

    def generate_injury_risk_from_image(self, image_path):
        """Load image, run pose detection and return injury risk assessment dict or None."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None

            image = cv2.resize(image, (640, 480))

            # Try multiple candidate transforms to improve pose detection
            candidates = [image, self._preprocess_image(image), self._gamma_correction(image, 1.3)]
            candidates.append(cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC))
            candidates.append(self._center_crop(image, 0.85))

            results = None
            for img_try in candidates:
                results = self._try_pose_on_image(img_try, model_complexity=1, min_detection_confidence=0.3)
                if results and results.pose_landmarks:
                    break

            if not results:
                return None

            return self._assess_injury_risk_from_pose_results(results)
        except Exception:
            return None
