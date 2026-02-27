from fastapi import FastAPI, HTTPException
import torch
import os
from pathlib import Path
import cv2
import numpy as np

from models.xception import XceptionNet
from utils.preprocessing import (
    preprocess_image, 
    extract_frames, 
    detect_faces, 
    crop_face
)

app = FastAPI(title="DeepGuard Vision Agent - Deepfake Detection")

# Global model
model = None
device = None

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
            print(f"📝 Attempting to register with Coordinator (Attempt {i+1})...")
            response = requests.post(coordinator_url, json=mcp_payload, timeout=5)
            if response.status_code == 200:
                print("✅ Successfully registered with Coordinator!")
                return
            else:
                print(f"⚠️ Registration rejected: {response.text}")
        except Exception as e:
            print(f"⚠️ Coordinator not ready yet: {e}")
        
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
    
    # For now, use random weights (we'll add pre-trained weights later)
    print("⚠️  Using untrained model (random weights)")
    print("📝 In production, load pre-trained weights here")
    
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
    
    Request:
        {
            "file_path": "/path/to/media.jpg",
            "generate_heatmap": true
        }
    
    Response:
        {
            "score": 0.73,
            "prediction": "FAKE",
            "confidence": 0.73,
            "faces_detected": 1,
            "model": "XceptionNet"
        }
    """
    file_path = request.get('file_path')
    
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    try:
        # Determine media type
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

async def analyze_image(image_path: str):
    """Analyze single image"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not fully initialized")
    
    # Read image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    faces = detect_faces(img_rgb)
    
    if len(faces) == 0:
        # No face detected, analyze full image
        tensor = preprocess_image(image_path)
        tensor = tensor.to(device)
        
        with torch.no_grad():
            score = model.predict(tensor)
            score = score.item()
    else:
        # Analyze each face and average
        scores = []
        
        for face_box in faces[:3]:  # Max 3 faces
            try:
                # Crop face
                face_img = crop_face(img_rgb, face_box)
                
                # Save temporarily
                temp_path = "/tmp/temp_face.jpg"
                cv2.imwrite(temp_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                
                # Preprocess
                tensor = preprocess_image(temp_path)
                tensor = tensor.to(device)
                
                # Predict
                with torch.no_grad():
                    face_score = model.predict(tensor)
                    scores.append(face_score.item())
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                print(f"Error analyzing face: {e}")
                continue
        
        # Average scores
        score = np.mean(scores) if scores else 0.5
    
    # Determine prediction
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
        # Extract frames
        frames = extract_frames(video_path, num_frames=10)
        
        if not frames:
            raise ValueError("No frames extracted from video")
        
        scores = []
        faces_detected = 0
        
        for frame in frames:
            # Detect faces
            faces = detect_faces(frame)
            faces_detected += len(faces)
            
            if len(faces) > 0:
                # Use first face
                face_img = crop_face(frame, faces[0])
            else:
                # Use full frame
                face_img = frame
            
            # Save temporarily
            temp_path = "/tmp/temp_frame.jpg"
            cv2.imwrite(temp_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            
            # Preprocess
            tensor = preprocess_image(temp_path)
            tensor = tensor.to(device)
            
            # Predict
            with torch.no_grad():
                frame_score = model.predict(tensor)
                scores.append(frame_score.item())
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        # Average scores
        score = np.mean(scores)
        prediction = "FAKE" if score > 0.5 else "REAL"
        
        return {
            "score": float(score),
            "prediction": prediction,
            "confidence": float(score) if score > 0.5 else float(1 - score),
            "faces_detected": faces_detected // len(frames),  # Average per frame
            "model": "XceptionNet",
            "frames_analyzed": len(frames)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")