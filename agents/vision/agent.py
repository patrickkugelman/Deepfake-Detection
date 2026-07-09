from fastapi import FastAPI, HTTPException
import torch
import torch.nn.functional as F
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from models.xception import XceptionNet
from utils.preprocessing import (
    extract_frames, 
    detect_faces, 
    crop_face
)

app = FastAPI(title="DeepGuard Vision Agent - Deepfake Detection")

# Global model
model = None
device = None

# Transformări inline pentru inferență (înlocuiește nevoia de fișiere temporare)
eval_transforms = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- MCP REGISTRATION ---
import requests
import time
import threading

def register_with_coordinator():
    """Register this agent with the Coordinator via MCP"""
    mcp_payload = {
        "name": "vision",
        "description": "Analyzes images for visual artifacts, face warping, and inconsistencies using XceptionNet",
        "input_schema": {"type": "image", "formats": ["jpg", "png", "frames"]},
        "output_schema": {"score": "float", "prediction": "str", "faces_detected": "int"},
        "capabilities": ["visual_artifact_detection", "face_analysis", "xception_net"],
        "endpoint": "http://deepguard_vision:8002/analyze"
    }
    
    coordinator_base = os.getenv("COORDINATOR_URL", "http://deepguard_brain:8000")
    coordinator_url = f"{coordinator_base.rstrip('/')}/mcp/register"
    max_retries = 5
    
    for i in range(max_retries):
        try:
            print(f"Attempting to register with Coordinator (Attempt {i+1})...")
            response = requests.post(coordinator_url, json=mcp_payload, timeout=5)
            if response.status_code == 200:
                print("✅ Successfully registered with Coordinator!")
                return
            else:
                print(f"Registration rejected: {response.text}")
        except Exception as e:
            print(f"Coordinator not ready yet: {e}")
        
        time.sleep(5)
    
    print("❌ Failed to register with Coordinator after retries")

@app.on_event("startup")
async def load_model():
    """Load XceptionNet model on startup and register MCP"""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Initialize model
    model = XceptionNet(num_classes=2)
    
    # Încărcarea ponderilor salvate din antrenament
    weights_path = "models/best_vision_model.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("✅ Ponderi încărcate cu succes din best_vision_model.pth!")
    else:
        print("⚠️ Fișierul cu ponderi nu a fost găsit. Folosesc random weights.")
    
    model.to(device)
    model.eval()
    
    print("✅ Vision Agent ready!")
    
    # Register MCP in background
    threading.Thread(target=register_with_coordinator, daemon=True).start()

@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "vision", "device": str(device)}

@app.post("/analyze")
async def analyze_media(request: dict):
    """
    Analyze media for visual deepfake artifacts
    """
    file_path = request.get('file_path')
    
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    try:
        ext = Path(file_path).suffix.lower()
        
        if ext in ['.jpg', '.jpeg', '.png']:
            result = await analyze_image(file_path)
        elif ext in ['.mp4', '.avi', '.mov']:
            result = await analyze_video(file_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def process_and_predict(image_rgb_array):
    """Procesează imaginea direct în RAM și face predicția"""
    global model, device
    
    # 1. Rezolvă eroarea de 'None' spunându-i lui Pylance că modelul este asigurat
    if model is None or device is None:
        raise RuntimeError("Modelul sau device-ul nu au fost inițializate!")

    pil_img = Image.fromarray(image_rgb_array)
    
    # 2. Rezolvă eroarea cu 'unsqueeze' ajutând Pylance să înțeleagă tipul
    tensor_img = eval_transforms(pil_img)
    
    # Verificare pentru linter: îi confirmăm că a devenit Tensor
    if not isinstance(tensor_img, torch.Tensor):
        tensor_img = torch.tensor(tensor_img)
        
    tensor = tensor_img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor) # Returnează logits
        probs = F.softmax(outputs, dim=1)
        fake_prob = probs[0, 1].item() # Indexul 1 este clasa "Fake"
        
    return fake_prob

async def analyze_image(image_path: str):
    """Analyze single image"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not fully initialized")
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not read image")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = detect_faces(img_rgb)
    
    if len(faces) == 0:
        # No face detected, analyze full image
        score = process_and_predict(img_rgb)
    else:
        # Analyze each face and average
        scores = []
        for face_box in faces[:3]:  # Max 3 faces
            try:
                face_img = crop_face(img_rgb, face_box)
                face_score = process_and_predict(face_img)
                scores.append(face_score)
            except Exception as e:
                print(f"Error analyzing face: {e}")
                continue
        
        score = np.mean(scores) if scores else 0.5
    
    prediction = "FAKE" if score > 0.5 else "REAL"
    
    return {
        "score": float(score),
        "prediction": prediction,
        "confidence": float(score) if score > 0.5 else float(1 - score),
        "faces_detected": len(faces),
        "model": "XceptionNet",
        "frames_analyzed": 1
    }

async def analyze_video(video_path: str):
    """Analyze video (extract frames and average)"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not fully initialized")
    
    try:
        frames = extract_frames(video_path, num_frames=10)
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        scores = []
        faces_detected = 0
        
        for frame in frames: # frame is already an RGB array
            faces = detect_faces(frame)
            faces_detected += len(faces)
            
            if len(faces) > 0:
                face_img = crop_face(frame, faces[0])
            else:
                face_img = frame
            
            frame_score = process_and_predict(face_img)
            scores.append(frame_score)
        
        score = np.mean(scores)
        prediction = "FAKE" if score > 0.5 else "REAL"
        
        return {
            "score": float(score),
            "prediction": prediction,
            "confidence": float(score) if score > 0.5 else float(1 - score),
            "faces_detected": faces_detected // len(frames),
            "model": "XceptionNet",
            "frames_analyzed": len(frames)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")