import requests
import json

def test_ai_connection():
    url = "http://127.0.0.1:8000/analyze-history"
    payload = {
        "vitals": [{"readingType": "Blood Pressure", "value": 120, "value2": 80}],
        "surgeries": [],
        "medications": [],
        "allergies": [],
        "chronic_diseases": []
    }
    
    print(f"Connecting to AI server at {url}...")
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print("[OK] AI Server is ALIVE!")
            print("Response Preview:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            if "analysis_en" in data and "analysis_ar" in data:
                print("\n[OK] Bilingual support confirmed!")
            else:
                print("\n[WARN] Missing one of the languages in response.")
                
            if "#" in data.get("analysis_en", "") or "*" in data.get("analysis_en", ""):
                print("[WARN] Markdown detected (Headers or Bold). Cleaning might be needed.")
            else:
                print("[OK] Clean text output confirmed (No Markdown).")
        else:
            print(f"[ERR] Server returned error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[ERR] Could not connect to AI server: {e}")
        print("Make sure you ran 'python server.py' first!")

if __name__ == "__main__":
    test_ai_connection()
