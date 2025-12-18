#!/usr/bin/env python3
"""Quick test to verify injury_risk is returned by /api/predict"""

import requests
import json
import os

# Find a test image
test_images = [
    "c:\\Users\\vedik\\Downloads\\cricket\\archive\\CricketCoachingDataSet\\5. Scoop Shot\\53_scoop_main.jpg",
    "c:\\Users\\vedik\\Downloads\\cricket\\archive\\CricketCoachingDataSet\\0. Cut Shot\\0_cut_main.jpg",
]

api_url = "http://127.0.0.1:5000/api/predict"

for img_path in test_images:
    if not os.path.exists(img_path):
        print(f"⚠️  Image not found: {img_path}")
        continue

    print(f"\n📸 Testing: {os.path.basename(img_path)}")
    print(f"   Path: {img_path}")

    try:
        with open(img_path, 'rb') as f:
            files = {'image': f}
            resp = requests.post(api_url, files=files)

        data = resp.json()
        print(f"   Status: {resp.status_code}")
        print(f"   Success: {data.get('success', False)}")

        if data.get('success'):
            print(f"   Prediction: {data.get('prediction')}")
            print(f"   Confidence: {data.get('confidence')}%")
            
            if 'injury_risk' in data:
                print(f"   ✓ Injury Risk FOUND:")
                print(f"      Level: {data['injury_risk'].get('level')}")
                print(f"      Score: {data['injury_risk'].get('score')}")
                print(f"      Reasons: {data['injury_risk'].get('reasons')}")
            else:
                print(f"   ✗ Injury Risk NOT in response (keys: {list(data.keys())})")
        else:
            print(f"   Error: {data.get('message')}")

    except Exception as e:
        print(f"   Exception: {e}")
