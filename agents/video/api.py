"""
DeepGuard Video Agent - Temporal Deepfake Detection
===================================================
Detects video deepfakes using ResNet3D temporal analysis.
"""

import os
import cv2
import uuid
import shutil
import time
import logging
import threading
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models.video import R3D_18_Weights
from fastapi import FastAPI, File, UploadFile
import requests

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("DeepGuardVideo")

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="DeepGuard Video Agent",
    version="2.0",
    description="Video deepfake detection using ResNet3D"
)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_PATH = os.path.join("models_i3d", "best_i3d.pth")
NUM_FRAMES = 32  # Match training config
IMG_SIZE = (224, 224) # Match training config (r3d uses 224 usually, check train config)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = None
model_loaded = False

# Verdict thresholds — tunable via env vars for strictness control
# Lower FAKE_THRESHOLD → catches more deepfakes (fewer false negatives, more false positives)
FAKE_THRESHOLD = float(os.getenv("VIDEO_FAKE_THRESHOLD", "40.0"))
SUSPICIOUS_THRESHOLD = float(os.getenv("VIDEO_SUSPICIOUS_THRESHOLD", "25.0"))
logger.info(f"Video thresholds: FAKE>={FAKE_THRESHOLD}%, SUSPICIOUS>={SUSPICIOUS_THRESHOLD}%")

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
            "display_name": "Video Agent",
            "description": "Analyzes video frames for temporal inconsistencies, frame-level artifacts, and motion anomalies using ResNet3D",
            "version": "2.0.0",
            "agent_type": "neural",
            "modality": "video",
            "capabilities": [
                {
                    "name": "temporal_consistency",
                    "description": "Detects temporal anomalies",
                    "confidence_threshold": 0.8
                },
                {
                    "name": "frame_interpolation_detection",
                    "description": "Identifies fake frames",
                    "confidence_threshold": 0.75
                },
                {
                    "name": "motion_analysis",
                    "description": "Analyzes motion patterns",
                    "confidence_threshold": 0.7
                }
            ],
            "input_schema": {
                "modality": "video",
                "formats": ["mp4", "avi", "mov", "mkv", "webm"],
                "max_size_mb": 500
            },
            "output_schema": {
                "verdict_field": "verdict",
                "confidence_field": "fake_probability",
                "details_field": "details"
            },
            "endpoint": "http://deepguard_video:8003/analyze",
            "health_endpoint": "http://deepguard_video:8003/health"
        }
    }
    
    # Use COORDINATOR_URL env var (base URL) + append /mcp/register
    coordinator_base = os.getenv("COORDINATOR_URL", "http://deepguard_brain:8000")
    register_url = f"{coordinator_base.rstrip('/')}/mcp/register"
    max_retries = 15
    retry_delay = 5

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"📝 Attempting to register with Coordinator (Attempt {attempt}/{max_retries})...")

            response = requests.post(
                register_url,
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
# MODEL ARCHITECTURE
# ============================================================================
def get_model_architecture() -> Optional[nn.Module]:
    """
    Build ResNet3D architecture matching training configuration.

    Strategy:
    1. Try to load with Kinetics400 pretrained weights (requires internet or cache)
    2. If download fails, fall back to random-init weights (no internet needed)

    Returns:
        ResNet3D model or None if building fails
    """
    # --- Strategy 1: Try pretrained weights (uses cache if already downloaded) ---
    try:
        import torch.hub
        # Disable download progress bar and set short timeout to detect failure fast
        old_timeout = None
        try:
            import socket
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(10)  # 10s connection timeout
        except Exception:
            pass

        weights = R3D_18_Weights.KINETICS400_V1
        m = models.video.r3d_18(weights=weights)

        try:
            import socket
            if old_timeout is not None:
                socket.setdefaulttimeout(old_timeout)
        except Exception:
            pass

        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 2)
        logger.info("✅ ResNet3D architecture built with Kinetics400 pretrained weights")
        return m

    except Exception as e:
        logger.warning(f"⚠️ Could not load pretrained weights (no internet/cache): {e}")
        logger.info("🔄 Falling back to random-init R3D-18 (no internet required)...")

    # --- Strategy 2: Random-init weights (always works offline) ---
    try:
        m = models.video.r3d_18(weights=None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, 2)
        logger.info("✅ ResNet3D architecture built with random weights (offline mode)")
        logger.warning("⚠️ Model uses random weights — accuracy will be low until custom weights are loaded from disk")
        return m
    except Exception as e:
        logger.error(f"❌ Failed to build ResNet3D architecture even without pretrained weights: {e}")
        return None

# ============================================================================
# FRAME EXTRACTION & PREPROCESSING
# ============================================================================
def extract_frames(video_path: str, num_frames: int = 16, num_clips: int = 3) -> Optional[torch.Tensor]:
    """
    Extract multiple clips of consecutive frames from video for robust temporal analysis.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames per clip (default 16 for ResNet3D)
        num_clips: Number of clips to extract (Start, Middle, End)
        
    Returns:
        Tensor of shape (num_clips, C, T, H, W) or None if extraction fails
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"❌ Failed to open video: {video_path}")
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logger.error("❌ Video has 0 frames")
            return None

        # Determine clip starting positions
        # Ensure we don't go out of bounds (total_frames - num_frames)
        max_start = max(0, total_frames - num_frames)
        start_indices = np.linspace(0, max_start, num_clips, dtype=int)
        
        all_clips = []
        
        for start_idx in start_indices:
            clip_frames = []
            # Reset pointer to start of this clip
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_idx))
            
            for _ in range(num_frames):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    clip_frames.append(frame)
                else:
                    break
            
            # Pad clip if partial (e.g. end of video or read error)
            if not clip_frames:
                continue
                
            while len(clip_frames) < num_frames:
                clip_frames.append(clip_frames[-1])
                
            # Preprocessing transform
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])
            
            # Stack frames: (T, C, H, W) -> (C, T, H, W) for R3D
            tensors = [transform(f) for f in clip_frames]
            clip_tensor = torch.stack(tensors, dim=1)
            all_clips.append(clip_tensor)
        
        cap.release()

        if not all_clips:
            logger.error("❌ No clips extracted from video")
            return None

        # Stack clips into batch: (B, C, T, H, W) where B=num_clips
        batch_tensor = torch.stack(all_clips, dim=0)
        
        logger.info(f"✅ Extracted {len(all_clips)} clips of {num_frames} frames each")
        return batch_tensor
        
    except Exception as e:
        logger.error(f"❌ Frame extraction failed: {e}")
        return None

# ============================================================================
# HEURISTIC ANALYSIS (Fallback)
# ============================================================================
def heuristic_video_analysis(video_path: str, max_frames: int = 30) -> Dict[str, Any]:
    """
    Fallback heuristic analysis based on video metadata.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum frames to check
        
    Returns:
        Dictionary with heuristic metrics
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Cannot open video", "black_frames_ratio": 0.0}
        
        frame_count = 0
        black_frames = 0
        
        # Read metadata BEFORE iterating/releasing
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Check if frame is mostly black
            if np.mean(frame.astype(float)) < 5:
                black_frames += 1
            frame_count += 1

        cap.release()

        return {
            "black_frames_ratio": black_frames / max(1, frame_count),
            "total_frames_checked": frame_count,
            "fps": fps,
            "total_frames": total_frames
        }
        
    except Exception as e:
        logger.error(f"❌ Heuristic analysis failed: {e}")
        return {"error": str(e), "black_frames_ratio": 0.0}

# ============================================================================
# MODEL LOADING (Background Thread)
# ============================================================================
def load_model_background() -> None:
    """
    Load ResNet3D model and weights in background.
    """
    global model, model_loaded
    
    logger.info(f"🚀 Initializing Video Agent on {DEVICE}...")
    
    try:
        # Step 1: Build architecture
        model = get_model_architecture()
        if model is None:
            logger.error("❌ Architecture initialization failed")
            return

        # Step 2: Load custom weights if available
        if os.path.exists(MODEL_PATH):
            logger.info(f"📂 Loading weights from {MODEL_PATH}")
            try:
                checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
                
                # Handle checkpoint dictionary vs raw state_dict
                if isinstance(checkpoint, dict):
                    if "model" in checkpoint:
                        state_dict = checkpoint["model"]
                    elif "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                    
                model.load_state_dict(state_dict, strict=True)
                logger.info("✅ ResNet3D weights loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load weights: {e}")
                logger.warning("⚠️ Using pretrained Kinetics400 weights only")
        else:
            logger.warning(f"⚠️ Weights file not found: {MODEL_PATH}")
            logger.warning("⚠️ Using pretrained Kinetics400 weights")

        # Step 3: Move to device and set eval mode
        model.to(DEVICE)
        model.eval()
        model_loaded = True
        logger.info(f"✅ Video Model ready using I3D (ResNet3D backbone) on {DEVICE}")
        
    except Exception as e:
        logger.error(f"❌ Critical startup error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        model = None
        model_loaded = False

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze video for deepfake detection.
    
    Features:
    - ResNet3D temporal analysis (if model loaded)
    - Heuristic metadata analysis as fallback
    - Bulletproof error handling
    
    Returns:
        Dictionary with verdict, fake_probability, and details
    """
    # REJECT IMAGES - Video agent expects video files only
    mime_type = file.content_type or ""
    fname = file.filename or ""
    if mime_type.startswith("image") or any(fname.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp']):
        logger.warning(f"⚠️ Video agent cannot process images: {fname}")
        return {
            "verdict": "INCONCLUSIVE",
            "fake_probability": 0.0,
            "confidence": 0.0,
            "details": "Video agent only processes videos, not images",
            "error": "Invalid file type for video agent"
        }
    
    temp_path = f"/tmp/{uuid.uuid4()}.mp4"
    
    try:
        # Save uploaded video (read bytes first for async compatibility)
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # 1. AI Model Analysis
        ai_verdict = "Unknown"
        ai_prob = 0.0
        ai_details = "Model not loaded"
        
        if model_loaded and model is not None:
            # Now returns (B, C, T, H, W) where B = num_clips
            # Increased to 10 clips for denser coverage (catching fleeting deepfakes)
            video_tensor = extract_frames(temp_path, num_frames=NUM_FRAMES, num_clips=10)
            
            if video_tensor is not None:
                try:
                    # video_tensor is already (B, C, T, H, W), just move to device
                    video_tensor = video_tensor.to(DEVICE)
                    
                    with torch.no_grad():
                        output = model(video_tensor) # Output: (B, 2)
                        probs = torch.softmax(output, dim=1) # (B, 2)
                        
                        # Get fake probabilities for all clips
                        fake_probs = probs[:, 1].cpu().numpy() * 100
                        
                        # Aggressive "Needle in Haystack" Logic
                        # Deepfakes often appear only in segments. Validating the whole video based on mean is wrong.
                        max_prob = float(np.max(fake_probs))
                        mean_prob = float(np.mean(fake_probs))
                        
                        logger.info(f"🎞️ Clip probabilities: {fake_probs.round(2)}%")
                        
                        if max_prob > 80.0:
                             # If we are >80% confident on any segment, TRUST IT. It's likely a deepfake.
                             ai_prob = max_prob
                             logger.info(f"🚨 High-confidence deepfake segment detected ({max_prob:.2f}%)! Overriding mean.")
                        elif max_prob > 60.0:
                             # If suspicious segments exist, weight them heavily
                             ai_prob = (mean_prob + max_prob * 2) / 3
                        else:
                             # Otherwise, average out noise
                             ai_prob = mean_prob
                        
                    if ai_prob >= FAKE_THRESHOLD:
                        ai_verdict = "Fake"
                    elif ai_prob >= SUSPICIOUS_THRESHOLD:
                        ai_verdict = "Suspicious"
                    else:
                        ai_verdict = "Real"
                    ai_details = f"I3D Score: {ai_prob:.1f}% (Max: {max_prob:.1f}%, Mean: {mean_prob:.1f}%, Threshold: {FAKE_THRESHOLD:.0f}%)"

                    logger.info(f"📊 Final Analysis: {ai_verdict} ({ai_prob:.2f}% >= threshold {FAKE_THRESHOLD:.0f}%)")
                    
                except Exception as model_error:
                    logger.error(f"❌ Model inference failed: {model_error}")
                    ai_details = f"Model inference error: {str(model_error)[:50]}"
            else:
                ai_details = "Frame extraction failed"
        
        # 2. Heuristic Metadata Analysis
        meta = heuristic_video_analysis(temp_path)
        
        # 3. Combine Results
        if model_loaded and ai_verdict != "Unknown":
            # Use AI model result
            final_verdict = ai_verdict
            final_prob = ai_prob
            final_details = f"{ai_details} | Metadata: {meta}"
        else:
            # Fallback to heuristics
            final_verdict = "Real"  # Safe default
            final_prob = 0.0
            final_details = "Heuristic fallback (Model unavailable)"
            
            # Check for suspicious patterns
            if meta.get('black_frames_ratio', 0) > 0.3:
                final_verdict = "Fake"
                final_prob = 90.0
                final_details += f" - High black frame ratio: {meta['black_frames_ratio']:.2f}"
        
        return {
            "verdict": final_verdict,
            "fake_probability": round(final_prob, 2),
            "details": final_details,
            "clip_probabilities": fake_probs.tolist() if 'fake_probs' in locals() else []
        }

    except Exception as e:
        logger.error(f"❌ Video analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "verdict": "Real",
            "fake_probability": 0.0,
            "details": f"Analysis error: {str(e)[:100]}",
            "error": str(e)[:200]
        }
        
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for Docker"""
    return {
        "status": "healthy",
        "agent": "video",
        "model_loaded": model_loaded,
        "device": DEVICE,
        "num_frames": NUM_FRAMES,
        "mode": "neural" if model_loaded else "heuristic"
    }

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with agent info"""
    return {
        "agent": "DeepGuard Video Agent",
        "version": "2.0",
        "model": "ResNet3D (r3d_18)",
        "status": "operational" if model_loaded else "heuristic_mode"
    }

# ============================================================================
# STARTUP EVENT
# ============================================================================
@app.on_event("startup")
async def startup() -> None:
    """
    Start server immediately, load model and register with MCP in background.
    
    FIXED: Now targets 'deepguard_brain' for MCP registration.
    """
    logger.info("🚀 Video Agent starting...")
    
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