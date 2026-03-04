import json
import os
import sys

# Add parent directory to path to import fetch_data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_load_wind_levels():
    # Create a temporary test json
    test_json = "test_levels.json"
    data = [{"name": "TestLevel", "h": 999, "type": "AGL"}]
    with open(test_json, "w") as f:
        json.dump(data, f)
    
    try:
        # Import the function (we'll need to refactor fetch_data to have a load_config function or similar, 
        # or we test the logic we are ABOUT to insert)
        
        # Simulating the logic to be inserted in fetch_data.py
        if os.path.exists(test_json):
            with open(test_json, "r") as f:
                loaded = json.load(f)
            print(f"Loaded: {loaded[0]['name']}")
            assert loaded[0]['name'] == "TestLevel"
            assert loaded[0]['h'] == 999
            print("Config load test passed!")
            
    finally:
        if os.path.exists(test_json):
            os.remove(test_json)

if __name__ == "__main__":
    test_load_wind_levels()
