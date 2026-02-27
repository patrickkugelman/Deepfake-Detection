"""
DeepGuard Bio Agent - Physiological Signal Analysis
===================================================
Analyzes heart rate (rPPG), micro-movements, and biometric liveness indicators.
"""

import os
import cv2
import uuid
import shutil
import time
import logging
import threading
from typing import Dict, Any, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, UploadFile
import requests

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("DeepGuardBio")

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="DeepGuard Bio Agent",
    version="2.0",
    description="Biometric liveness detection via rPPG and micro-movement analysis"
)

# ============================================================================
# CONFIGURATION
# ============================================================================
HAAR_CASCADE_PATH = os.path.join(
    os.path.dirname(__file__),
    "haarcascade_frontalface_default.xml"
)

# Global face detector
FACE_CASCADE: Optional[cv2.CascadeClassifier] = None

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
            "name": "bio",
            "display_name": "Bio Agent",
            "description": "Verifies liveness through rPPG pulse detection and micro-expression analysis using physiological signals",
            "version": "2.0.0",
            "agent_type": "neural",
            "modality": "video",
            "capabilities": [
                {
                    "name": "pulse_detection",
                    "description": "Detects heart rate via rPPG",
                    "confidence_threshold": 0.85
                },
                {
                    "name": "micro_expressions",
                    "description": "Analyzes facial micro-movements",
                    "confidence_threshold": 0.75
                },
                {
                    "name": "liveness_verification",
                    "description": "Confirms biological presence",
                    "confidence_threshold": 0.8
                }
            ],
            "input_schema": {
                "modality": "video",
                "formats": ["mp4", "avi", "mov"],
                "max_size_mb": 200,
                "requires_preprocessing": True
            },
            "output_schema": {
                "verdict_field": "verdict",
                "confidence_field": "fake_probability",
                "details_field": "details",
                "additional_fields": ["pulse_detected", "heart_rate_bpm"]
            },
            "endpoint": "http://deepguard_bio:8005/analyze",
            "health_endpoint": "http://deepguard_bio:8005/health"
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
# FACE DETECTION & ROI EXTRACTION
# ============================================================================
def get_roi(frame: np.ndarray) -> np.ndarray:
    """
    Detect face and return forehead ROI for rPPG analysis.
    Falls back to center-top crop if face detection fails.
    
    Args:
        frame: BGR image frame
        
    Returns:
        ROI region (numpy array)
    """
    if FACE_CASCADE is None:
        # Fallback: center-top crop (approximate forehead location)
        h, w = frame.shape[:2]
        return frame[h//4:h//2, w//3:2*w//3]

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Pick largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            # Forehead region (top 25% of face, middle 60% width)
            forehead = frame[
                y:y+int(h*0.25),
                x+int(w*0.2):x+int(w*0.8)
            ]
            return forehead
    except Exception as e:
        logger.warning(f"Face detection error: {e}")
    
    # Fallback
    h, w = frame.shape[:2]
    return frame[h//4:h//2, w//3:2*w//3]

# ============================================================================
# rPPG PULSE DETECTION
# ============================================================================
def extract_pulse(video_path: str) -> Dict[str, Any]:
    """
    Extract pulse signal using remote Photoplethysmography (rPPG).
    
    Algorithm:
    1. Extract forehead ROI from each frame
    2. Compute mean green channel value (strongest PPG signal)
    3. Apply FFT to detect periodic signal
    4. Bandpass filter 42-180 BPM (0.7-3.0 Hz)
    5. Calculate SNR to verify signal quality
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with detection results
    """
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps < 1:
            fps = 30.0  # Default fallback
        
        green_means = []
        max_frames = 300  # ~10 seconds at 30fps
        
        count = 0
        while count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            roi = get_roi(frame)
            # Green channel (index 1) has strongest PPG signal
            mean_g = float(np.mean(roi[:, :, 1]))
            green_means.append(mean_g)
            count += 1
        
        cap.release()
        
        if len(green_means) < 30:  # Need at least ~1 second
            return {
                "detected": False,
                "bpm": 0.0,
                "snr": 0.0,
                "msg": "Video too short for pulse analysis"
            }

        # Signal Processing
        signal = np.array(green_means)
        
        # 1. Normalize (remove DC component)
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
        
        # 2. Fast Fourier Transform
        L = len(signal)
        fft = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(L, 1/fps)
        
        # 3. Bandpass filter: 0.7 Hz - 3.0 Hz (42-180 BPM)
        interest_idx = np.where((freqs >= 0.7) & (freqs <= 3.0))[0]
        
        if len(interest_idx) == 0:
            return {
                "detected": False,
                "bpm": 0.0,
                "snr": 0.0,
                "msg": "No frequency components in human heart rate range"
            }
        
        # 4. Find peak frequency in band
        band_fft = np.abs(fft[interest_idx])
        peak_idx = np.argmax(band_fft)
        peak_freq = freqs[interest_idx][peak_idx]
        
        bpm = float(peak_freq * 60.0)
        
        # 5. Signal Quality Check (Signal-to-Noise Ratio)
        peak_power = band_fft[peak_idx]
        avg_power = np.mean(band_fft)
        snr = float(peak_power / avg_power if avg_power > 0 else 0)
        
        # Threshold: SNR > 3.0 indicates VERY strong periodic signal
        is_human = snr > 3.0 and 42 <= bpm <= 180
        
        if is_human:
             msg = f"Pulse: {bpm:.1f} BPM (SNR: {snr:.2f})"
        else:
             msg = f"Pulse unclear (Noise/SNR: {snr:.2f})" # Hide BPM if not confident to avoid confusion

        return {
            "detected": bool(is_human),
            "bpm": round(bpm, 1) if is_human else 0.0,
            "snr": round(snr, 2),
            "msg": msg
        }

    except Exception as e:
        logger.error(f"❌ rPPG extraction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "detected": False,
            "bpm": 0.0,
            "snr": 0.0,
            "msg": f"rPPG error: {str(e)[:100]}"
        }

# ============================================================================
# MICRO-MOVEMENT & LIVENESS DETECTION
# ============================================================================
def detect_liveness(video_path: str) -> Dict[str, Any]:
    """
    Detect liveness through micro-movements and frame differences.
    
    Natural human videos show subtle motion even when still.
    Static images or paused videos show no motion.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with liveness metrics
    """
    try:
        cap = cv2.VideoCapture(video_path)
        ret, prev = cap.read()
        if not ret:
            return {"alive": False, "score": 0.0}
        
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        diffs = []
        max_frames = 60  # Check ~2 seconds at 30fps
        
        count = 0
        while count < max_frames:
            ret, curr = cap.read()
            if not ret:
                break
            
            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            
            # Frame Difference (motion detection)
            diff = cv2.absdiff(prev_gray, curr_gray) # type: ignore
            # Type-safe conversion for Pylance
            diff_arr = np.array(diff)
            score = float(np.mean(diff_arr)) # type: ignore
            diffs.append(score)
            
            prev_gray = curr_gray
            count += 1
        
        cap.release()
        
        if not diffs:
            return {"alive": False, "score": 0.0}
        
        # Type-safe conversion to list for np.mean
        diffs_array = np.array(diffs)
        avg_motion = float(np.mean(diffs_array))
        std_motion = float(np.std(diffs_array))
        
        # Logic:
        # - Too little motion (<0.5) = Static Image/Paused Video
        # - Natural micro-movement: 0.5 - 5.0 range
        # - Excessive motion (>10) = Camera shake or unnatural movement
        alive = 0.5 < avg_motion < 10.0
        
        return {
            "alive": bool(alive),
            "score": round(avg_motion, 2),
            "std": round(std_motion, 2)
        }
        
    except Exception as e:
        logger.error(f"❌ Liveness detection failed: {e}")
        return {"alive": False, "score": 0.0}

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze video for biometric liveness.
    
    Features:
    - rPPG pulse detection (heart rate via green channel)
    - Micro-movement analysis (frame differencing)
    - Liveness verification logic
    
    Returns:
        Dictionary with verdict, pulse data, and liveness metrics
    """
    temp_path = f"/tmp/{uuid.uuid4()}.mp4"
    
    try:
        # Save uploaded video
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        logger.info(f"🫀 Analyzing biometric signals from {file.filename}")
        
        # 1. Pulse Detection (rPPG)
        pulse_data = extract_pulse(temp_path)
        
        # 2. Liveness Detection (Micro-movements)
        motion_data = detect_liveness(temp_path)
        
        # 3. Verdict Logic
        verdict = "INCONCLUSIVE"
        status = "INCONCLUSIVE"
        reason = "Analysis incomplete"
        fake_prob = 0.0
        
        # Helper to check if we actually analyzed a face or just the whole frame (fallback)
        # Note: extract_pulse uses get_roi internally. We need to trust its output or improve it.
        # Ideally, we should check if face was detected *before* pulse extraction.
        # For now, we infer based on pulse_data content.
        
        if pulse_data["detected"]:
            # Strong pulse signal detected -> Likely Real
            fake_prob = 10.0
            verdict = "REAL"
            status = "OK"
            reason = f"Strong pulse detected: {pulse_data['bpm']} BPM"
            logger.info(f"✅ Pulse detected: {pulse_data['bpm']} BPM")
        else:
            # check if face detection failed (requires tracking in extract_pulse, or simplified heuristic here)
            # If msg implies noise/short video, we might still have a face but bad signal.
            
            if FACE_CASCADE is None:
                 # If detector failed to load, we can't be sure
                 status = "ERROR"
                 verdict = "INCONCLUSIVE_TECHNICAL"
                 reason = "Face detector not initialized"
                 
            elif motion_data["alive"]:
                # Moving but no pulse detected
                fake_prob = 60.0
                verdict = "FAKE"
                status = "OK"
                reason = "Motion analysis indicates liveness but no biological pulse found (possible deepfake)"
                logger.warning("⚠️ Motion detected but no pulse - suspicious")
            
            elif not motion_data["alive"] and motion_data["score"] < 0.5:
                 # No pulse, no motion -> Static image
                 fake_prob = 80.0
                 verdict = "FAKE"
                 status = "OK"
                 reason = "Static image or freeze frame detected (no physiological signs)"
            else:
                 # Motion exists but isn't "alive" (too random?) and no pulse.
                 status = "INCONCLUSIVE"
                 verdict = "INCONCLUSIVE_INSUFFICIENT_DATA"
                 reason = "Signal quality too low for determination"

        details = f"{pulse_data['msg']} | Motion: {motion_data['score']}"
        
        logger.info(f"📊 Bio analysis - Verdict: {verdict} ({fake_prob}% fake)")
        
        return {
            "status": status,
            "verdict": verdict,
            "fake_probability": round(fake_prob, 2),
            "confidence": round(100.0 - fake_prob if verdict == "REAL" else fake_prob, 2),
            "pulse_detected": pulse_data["detected"],
            "heart_rate_bpm": pulse_data["bpm"],
            "pulse": pulse_data,
            "liveness": motion_data,
            "details": details,
            "reason": reason
        }
        
    except Exception as e:
        logger.error(f"❌ Bio analysis failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "status": "ERROR",
            "verdict": "INCONCLUSIVE_TECHNICAL",
            "fake_probability": 0.0,
            "confidence": 0.0,
            "pulse_detected": False,
            "heart_rate_bpm": 0.0,
            "details": f"Analysis error: {str(e)[:100]}",
            "reason": "Internal agent error",
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
        "agent": "bio",
        "face_detector_loaded": FACE_CASCADE is not None,
        "mode": "rppg_liveness"
    }

@app.get("/")
async def root() -> Dict[str, Any]:  # Changed from Dict[str, str] to Dict[str, Any]
    """Root endpoint with agent info"""
    return {
        "agent": "DeepGuard Bio Agent",
        "version": "2.0",
        "methods": ["rPPG", "micro-movement analysis"],  # This is a list
        "status": "operational"
    }

# ============================================================================
# STARTUP EVENT
# ============================================================================
@app.on_event("startup")
async def startup_event() -> None:
    """
    Load face detector and register with MCP in background.
    
    FIXED: Now targets 'deepguard_brain' for MCP registration.
    """
    global FACE_CASCADE
    
    logger.info("🚀 Bio Agent starting...")
    
    # Load Haar Cascade face detector
    if os.path.exists(HAAR_CASCADE_PATH):
        try:
            FACE_CASCADE = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
            if FACE_CASCADE.empty():
                logger.warning("⚠️ Face detector loaded but is empty")
                FACE_CASCADE = None
            else:
                logger.info("✅ Face detector loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load face detector: {e}")
            FACE_CASCADE = None
    else:
        logger.warning(f"⚠️ Haar Cascade not found at {HAAR_CASCADE_PATH}")
        logger.warning("⚠️ Will use fallback ROI extraction")
    
    # Start MCP registration in background thread
    threading.Thread(target=register_with_coordinator, daemon=True).start()
    
    logger.info("✅ Bio Agent ready - rPPG & liveness detection operational")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)