import requests

# URL direct către Agentul Audio (Port 8004)
url = "http://localhost:8004/analyze"

# Calea ta către fișierul fake
file_path = r"D:\deepfake-detector-structure\agents\audio\dataset\Train1\gen_1.wav"

try:
    print(f"🔊 Testing Audio Agent directly with: {file_path}")
    with open(file_path, "rb") as f:
        # Trimitem fișierul
        response = requests.post(url, files={"file": f})
        
    print(f"✅ Status Code: {response.status_code}")
    print("📊 Response:", response.json())

except Exception as e:
    print(f"❌ EROARE: {e}")