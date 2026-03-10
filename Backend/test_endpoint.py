import requests
import json

testfile = r'C:\Users\vedik\Downloads\cricket\archive\CricketCoachingDataSet\0. Cut Shot\100_cut_-30.jpg'

try:
    with open(testfile, 'rb') as f:
        files = {'image': f}
        r = requests.post('http://localhost:5000/api/predict', files=files, timeout=30)
        data = r.json()
        print("Status:", data.get('success'))
        if data.get('success'):
            print("Prediction:", data.get('prediction'))
            player_tips = data.get('player_tips', [])
            print(f"Player Tips Count: {len(player_tips)}")
            if player_tips:
                print("\nFirst player tip:")
                print(json.dumps(player_tips[0], indent=2))
        else:
            print("Error:", data.get('message'))
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
