"""
DeepGuard Video Agent - I3D Implementation
==========================================
Alternative video agent using I3D (Inflated 3D ConvNet) for temporal analysis.
"""

import os
import time
import logging
import threading
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
import requests

# Custom imports (ensure these exist in your project)
try:
    from models.i3d import I3D
    from utils.video_preprocessing import (
        extract_frames,
        detect_temporal_inconsistencies
    )
except ImportError as e:
    logging.warning(f"⚠️ Custom modules not found: {e}")
    I3D = None
    extract_frames = None
    detect_temporal_inconsistencies = None

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DeepGuardVideoI3D")

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="DeepGuard Video Agent (I3D)",
    version="2.0",
    description="Video deepfake detection using I3D temporal analysis"
)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
WEIGHTS_PATH = "/app/i3d_best.pth"
DEFAULT_NUM_FRAMES = 16

# Global model variables
model: Optional[Any] = None
device: Optional[torch.device] = None

# ============================================================================
# MCP REGISTRATION (Fixed to use deepguard_brain)
# ============================================================================
def register_with_coordinator() -> None:
    """
    Register this agent with the Coordinator via MCP Protocol.
    
    FIXED: Now uses 'deepguard_brain' as hostname.
    """
    mcp_payload = {
        "tool": {
            "name": "video",
            "display_name": "Video Agent (I3D)",
            "description": "Analyzes video for temporal inconsistencies, frame dropping, and deepfake artifacts using I3D (Inflated 3D ConvNet)",
            "version": "2.0.0",
            "agent_type": "neural",
            "modality": "video",
            "capabilities": [
                {
                    "name": "temporal_consistency",
                    "description": "Detects temporal anomalies using I3D",
                    "confidence_threshold": 0.8
                },
                {
                    "name": "frame_interpolation_detection",
                    "description": "Identifies fake or interpolated frames",
                    "confidence_threshold": 0.75
                },
                {
                    "name": "motion_analysis",
                    "description": "Analyzes motion patterns across time",
                    "confidence_threshold": 0.7
                }
            ],
            "input_schema": {
                "modality": "video",
                "formats": ["mp4", "avi", "mov", "mkv"],
                "max_size_mb": 500
            },
            "output_schema": {
                "verdict_field": "prediction",
                "confidence_field": "confidence",
                "details_field": "details"
            },
            "endpoint": "http://deepguard_video:8003/analyze",
            "health_endpoint": "http://deepguard_video:8003/health"
        }
    }
    
    # FIXED: Use deepguard_brain instead of 'coordinator'
    coordinator_url = os.getenv(
        "COORDINATOR_URL",
        "http://deepguard_brain:8000/mcp/register"
    )
    max_retries = 5
    retry_delay = 5
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"📝 Attempting to register with Coordinator (Attempt {attempt}/{max_retries})...")
            
            response = requests.post(
                coordinator_url,
                json=mcp_payload,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                logger.info("✅ Successfully registered with Coordinator!")
                logger.info(f"   Response: {response.json()}")
                return
            else:
                logger.warning(f"⚠️ Registration rejected (status {response.status_code}): {response.text}")
        
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"⚠️ Coordinator not reachable (attempt {attempt}): {e}")
        except requests.exceptions.Timeout:
            logger.warning(f"⚠️ Registration timeout (attempt {attempt})")
        except Exception as e:
            logger.warning(f"⚠️ Registration error (attempt {attempt}): {e}")
        
        if attempt < max_retries:
            logger.info(f"⏳ Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    logger.error("❌ Failed to register with Coordinator after all retries")
    logger.info("ℹ️  Agent will continue running, but won't appear in MCP registry")

# ============================================================================
# MODEL LOADING (Background Thread)
# ============================================================================
def load_model_background() -> None:
    """
    Load I3D model and weights in background.
    """
    global model, device
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧 Using device: {device}")
        
        # Check if I3D class is available
        if I3D is None:
            logger.error("❌ I3D model class not available (import failed)")
            logger.error("   Make sure models/i3d.py exists and is correct")
            return
        
        # Initialize model
        logger.info("🏗️ Initializing I3D model...")
        model = I3D(num_classes=2, dropout_prob=0.5)
        
        # Try to load trained weights
        if os.path.exists(WEIGHTS_PATH):
            logger.info(f"📥 Loading trained weights from {WEIGHTS_PATH}...")
            try:
                state_dict = torch.load(WEIGHTS_PATH, map_location=device)
                model.load_state_dict(state_dict)
                logger.info("✅ Loaded trained I3D weights!")
            except Exception as e:
                logger.error(f"❌ Failed to load weights: {e}")
                logger.warning("⚠️ Using random initialization")
        else:
            logger.warning(f"⚠️ No trained weights found at {WEIGHTS_PATH}")
            logger.warning("⚠️ Using random initialization (placeholder)")
        
        model.to(device)
        model.eval()
        
        logger.info("✅ Video Agent (I3D) ready!")
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        model = None
        device = None

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint for Docker"""
    return {
        "status": "healthy",
        "agent": "video_i3d",
        "device": str(device) if device else "not_initialized",
        "model_loaded": model is not None,
        "mode": "i3d"
    }

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with agent info"""
    return {
        "agent": "DeepGuard Video Agent (I3D)",
        "version": "2.0",
        "model": "I3D (Inflated 3D ConvNet)",
        "status": "operational" if model is not None else "loading"
    }

@app.post("/analyze")
async def analyze_video(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze video for temporal deepfake artifacts using I3D.
    
    Request:
        {
            "file_path": "/path/to/video.mp4",
            "num_frames": 16
        }
    
    Response:
        {
            "score": 0.73,
            "prediction": "FAKE",
            "confidence": 0.73,
            "temporal_inconsistency": 0.15,
            "model": "I3D",
            "frames_analyzed": 16
        }
    """
    # Validate model is loaded
    if model is None or device is None:
        raise HTTPException(
            status_code=503,
            detail="Model not fully initialized. Please try again in a few seconds."
        )
    
    # Check if required utility functions are available
    if extract_frames is None or detect_temporal_inconsistencies is None:
        raise HTTPException(
            status_code=500,
            detail="Video preprocessing utilities not available"
        )
    
    # Extract parameters
    file_path = request.get('file_path')
    num_frames = request.get('num_frames', DEFAULT_NUM_FRAMES)
    
    if not file_path:
        raise HTTPException(status_code=400, detail="Missing 'file_path' in request")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
    
    try:
        # Check if it's a video file
        ext = Path(file_path).suffix.lower()
        if ext not in ['.mp4', '.avi', '.mov', '.mkv']:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported video format: {ext}"
            )
        
        # Extract frames and analyze
        logger.info(f"🎥 Analyzing video: {file_path}")
        logger.info(f"   Extracting {num_frames} frames...")
        
        # Get video tensor
        video_tensor = extract_frames(file_path, num_frames=num_frames)
        
        if video_tensor is None:
            raise HTTPException(
                status_code=400,
                detail="Could not extract frames from video"
            )
        
        # Convert to tensor if needed
        if not isinstance(video_tensor, torch.Tensor):
            video_tensor = torch.from_numpy(video_tensor)
        
        video_tensor = video_tensor.to(device)
        
        # Predict with I3D
        logger.info("   Running I3D inference...")
        with torch.no_grad():
            score = model.predict(video_tensor)
            if isinstance(score, torch.Tensor):
                score = score.item()
        
        # Detect temporal inconsistencies
        logger.info("   Analyzing temporal consistency...")
        temporal_inconsistency = detect_temporal_inconsistencies(file_path)
        
        # Adjust score based on temporal analysis
        # High inconsistency = more likely fake
        adjusted_score = float(score * 0.7 + temporal_inconsistency * 0.3)
        
        # Determine prediction
        prediction = "FAKE" if adjusted_score > 0.5 else "REAL"
        confidence = adjusted_score if adjusted_score > 0.5 else (1 - adjusted_score)
        
        result = {
            "score": round(adjusted_score, 4),
            "prediction": prediction,
            "confidence": round(float(confidence), 4),
            "temporal_inconsistency": round(float(temporal_inconsistency), 4),
            "model": "I3D",
            "frames_analyzed": num_frames,
            "details": {
                "i3d_score": round(float(score), 4),
                "temporal_score": round(float(temporal_inconsistency), 4),
                "fusion_method": "weighted_average (0.7 I3D + 0.3 temporal)"
            }
        }
        
        logger.info(f"✅ Analysis complete:")
        logger.info(f"   Prediction: {prediction}")
        logger.info(f"   Confidence: {confidence:.2%}")
        logger.info(f"   Temporal inconsistency: {temporal_inconsistency:.2%}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Analysis failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed: {str(e)[:200]}"
        )

@app.post("/extract-frames")
async def extract_video_frames(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract frames from video for preview/testing.
    
    Request:
        {
            "file_path": "/path/to/video.mp4",
            "num_frames": 5
        }
    
    Response:
        {
            "frames": 5,
            "tensor_shape": [5, 3, 224, 224],
            "message": "Frames extracted successfully"
        }
    """
    if extract_frames is None:
        raise HTTPException(
            status_code=500,
            detail="Frame extraction utility not available"
        )
    
    file_path = request.get('file_path')
    num_frames = request.get('num_frames', 5)
    
    if not file_path:
        raise HTTPException(status_code=400, detail="Missing 'file_path' in request")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
    
    try:
        # Extract frames (for testing)
        video_tensor = extract_frames(file_path, num_frames=num_frames)
        
        if video_tensor is None:
            raise HTTPException(
                status_code=400,
                detail="Could not extract frames from video"
            )
        
        # Get shape
        if isinstance(video_tensor, torch.Tensor):
            shape = list(video_tensor.shape)
        elif isinstance(video_tensor, np.ndarray):
            shape = list(video_tensor.shape)
        else:
            shape = []
        
        return {
            "frames": num_frames,
            "tensor_shape": shape,
            "message": "Frames extracted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Frame extraction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Frame extraction failed: {str(e)[:200]}"
        )

# ============================================================================
# STARTUP EVENT
# ============================================================================
@app.on_event("startup")
async def startup_event() -> None:
    """
    Start server immediately, load model and register with MCP in background.
    
    FIXED: Now targets 'deepguard_brain' for MCP registration.
    """
    logger.info("🚀 Video Agent (I3D) starting...")
    
    # Start model loading in background thread
    threading.Thread(target=load_model_background, daemon=True).start()
    
    # Start MCP registration in background thread
    threading.Thread(target=register_with_coordinator, daemon=True).start()
    
    logger.info("✅ Server started - model loading & MCP registration in progress...")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)