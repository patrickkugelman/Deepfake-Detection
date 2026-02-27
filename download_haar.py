import requests
import os

url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
target_path = "agents/bio/haarcascade_frontalface_default.xml"

try:
    print(f"Downloading {url}...")
    response = requests.get(url)
    response.raise_for_status()
    
    with open(target_path, "wb") as f:
        f.write(response.content)
    
    print(f"✅ Successfully downloaded to {target_path}")
    print(f"Size: {len(response.content)} bytes")

except Exception as e:
    print(f"❌ Failed to download: {e}")
