import requests
import json
import sys

def test_api():
    url = "http://127.0.0.1:8000/api/v2/chat"
    payload = {
        "messages": [
            {"role": "user", "content": "Hello, this is a test message. Please tell me who you are."}
        ]
    }
    
    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        print("Response stream:")
        for line in response.iter_lines():
            if line:
                print(line.decode('utf-8'))
                
        print("\nTest passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_api()
