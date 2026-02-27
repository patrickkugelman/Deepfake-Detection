import uvicorn
from api import app

# Acesta este punctul de intrare (Entry Point)
if __name__ == "__main__":
    print("🚀 Starting NASA Coordinator Brain...")
    # Pornim serverul pe portul 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)