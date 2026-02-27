import requests

# URL-ul Creierului
url = "http://localhost:8000/orchestrate"

# Un fișier de test (folosește un wav real dacă ai, sau un mp4)
# Dacă nu ai, creăm unul gol doar de test
file_path = r"D:\deepfake-detector-structure\agents\audio\dataset\Real\wavs\LJ001-0003.wav" 
# ⚠️ ATENȚIE: Modifică calea de mai sus către un fișier wav REAL din dataset-ul tău!
# De exemplu: D:\deepfake-detector-structure\agents\audio\dataset\Real\wavs\ceva.wav

try:
    print("🚀 Sending mission to NASA Brain...")
    
    with open(file_path, "rb") as f:
        # Trimitem un request "Multipart" (simulăm browser-ul)
        response = requests.post(
            url,
            files={"file": f},
            data={
                "file_type": "audio",  # Spunem că e audio
                "instruction": "analyze for deepfake artifacts"
            }
        )
    
    print("\n🔍 --- MISSION REPORT ---")
    print(response.json())

except FileNotFoundError:
    print(f"❌ Nu găsesc fișierul de test la: {file_path}")
    print("Te rog pune calea către un fișier .wav existent din dataset-ul tău în script.")