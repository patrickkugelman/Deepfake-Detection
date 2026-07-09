"""
DeepGuard Audio Agent - Voice Cloning Detection
================================================
Detects synthetic voices, TTS artifacts, and audio deepfakes using Wav2Vec2.
"""

import os
import shutil
import uuid
import logging
import time
import threading
from typing import Dict, Any

import torch
import librosa
import numpy as np
from fastapi import FastAPI, File, UploadFile
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import requests

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DeepGuardAudio")

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(title="DeepGuard Audio Agent", version="2.0")

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_PATH = os.path.join("models", "best_audio_model.pth")
BASE_MODEL = "facebook/wav2vec2-base"

# Global model variables (initialized to None for safety)
model = None
feature_extractor = None

# ============================================================================
# MCP REGISTRATION (Updated to use deepguard_brain)
# ============================================================================
def register_with_coordinator() -> None:
    """
    Register this agent with the Coordinator via MCP Protocol.
    
    FIXED: Now uses 'deepguard_brain' as hostname (matches docker-compose).
    """
    mcp_payload = {
        "tool": {
            "name": "audio",
            "display_name": "Audio Agent",
            "description": "Analyzes audio files for synthetic voice signatures, artifacts, and spectral anomalies using Wav2Vec2",
            "version": "2.0.0",
            "agent_type": "neural",
            "modality": "audio",
            "capabilities": [
                {
                    "name": "voice_cloning_detection",
                    "description": "Identifies synthetic voices",
                    "confidence_threshold": 0.8
                },
                {
                    "name": "spectral_analysis",
                    "description": "Analyzes frequency patterns",
                    "confidence_threshold": 0.7
                },
                {
                    "name": "artifact_detection",
                    "description": "Detects audio processing artifacts",
                    "confidence_threshold": 0.75
                }
            ],
            "input_schema": {
                "modality": "audio",
                "formats": ["wav", "mp3", "flac", "ogg", "m4a"],
                "max_size_mb": 100
            },
            "output_schema": {
                "verdict_field": "verdict",
                "confidence_field": "fake_probability",
                "details_field": "details"
            },
            "endpoint": "http://deepguard_audio:8004/analyze",
            "health_endpoint": "http://deepguard_audio:8004/health"
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
# MODEL LOADING (Background Thread)
# ============================================================================
def load_model_background() -> None:
    """
    Load Wav2Vec2 model in background without blocking server startup.
    Handles architecture mismatches and provides fallback to heuristics.
    """
    global model, feature_extractor
    
    try:
        logger.info("🔊 Loading Wav2Vec2 Pipeline (background)...")
        
        # Step 1: Load feature extractor
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)
            logger.info("✅ Feature extractor loaded")
        except Exception as e:
            logger.error(f"❌ Feature extractor failed: {e}")
            feature_extractor = None
            return
        
        # Step 2: Load base model
        try:
            model = AutoModelForAudioClassification.from_pretrained(
                BASE_MODEL, 
                num_labels=2
            )
            logger.info("✅ Base Wav2Vec2 model loaded")
        except Exception as e:
            logger.error(f"❌ Base model loading failed: {e}")
            model = None
            return
        
        # Step 3: Try to load custom weights
        if os.path.exists(MODEL_PATH):
            logger.info(f"📂 Weights found: {MODEL_PATH}")
            try:
                state_dict = torch.load(MODEL_PATH, map_location="cpu")
                
                # Check for architecture mismatch
                model_keys = set(model.state_dict().keys())
                weight_keys = set(state_dict.keys())
                
                missing_keys = model_keys - weight_keys
                unexpected_keys = weight_keys - model_keys
                
                if missing_keys or unexpected_keys:
                    logger.warning("⚠️ Architecture mismatch detected!")
                    logger.warning(f"   Missing keys: {len(missing_keys)}")
                    logger.warning(f"   Unexpected keys: {len(unexpected_keys)}")
                    logger.warning("⚠️ Discarding base model — falling back to heuristic mode")
                    # IMPORTANT: without this, `model` stayed set to the base (untrained)
                    # architecture, so /analyze kept running inference through it and
                    # reported "mode": "neural" at /health — the real heuristic fallback
                    # below never actually executed.
                    model = None
                else:
                    # Perfect match - load weights
                    model.load_state_dict(state_dict, strict=True)
                    logger.info("✅ Custom weights loaded successfully")
                    
            except Exception as e:
                logger.error(f"❌ Failed to load weights: {e}")
                logger.warning("⚠️ Discarding base model — falling back to heuristic mode")
                model = None
        else:
            logger.warning(f"⚠️ No weights found at {MODEL_PATH}")
            logger.warning("⚠️ Discarding base model (untrained classification head) — falling back to heuristic mode")
            model = None
        
        # Step 4: Set to eval mode
        if model is not None:
            model.eval()
            logger.info("✅ Audio model ready for inference")
        else:
            logger.info("ℹ️ No trained weights available — /analyze will use heuristic_audio_analysis()")
            
    except Exception as e:
        logger.error(f"❌ Critical model loading error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        model = None
        feature_extractor = None

# ============================================================================
# HEURISTIC ANALYSIS (Fallback)
# ============================================================================
def heuristic_audio_analysis(audio: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Fallback heuristic analysis when model fails.
    Uses spectral analysis, silence detection, and signal statistics.
    
    Args:
        audio: Audio waveform as numpy array
        sr: Sample rate (Hz)
        
    Returns:
        Dictionary with verdict, fake_probability, and details
    """
    try:
        # 1. Digital silence check
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude < 0.005:
            return {
                "verdict": "Fake",
                "fake_probability": 100.0,
                "details": "Heuristic: Absolute Digital Silence Detected"
            }
        
        # 2. Spectral analysis
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        frequency = np.fft.fftfreq(len(audio), 1/sr)
        
        # Only use positive frequencies
        positive_freq_idx = frequency > 0
        magnitude = magnitude[positive_freq_idx]
        frequency = frequency[positive_freq_idx]
        
        if np.sum(magnitude) > 0:
            spectral_centroid = np.sum(frequency * magnitude) / np.sum(magnitude)
        else:
            spectral_centroid = 0
        
        # 3. Zero crossing rate
        zero_crossings = np.sum(np.diff(np.signbit(audio)))
        zcr = zero_crossings / len(audio)
        
        # 4. Energy level
        energy = np.mean(audio ** 2)
        
        # 5. Heuristic scoring
        fake_score = 0.0
        
        # Very low energy = suspicious
        if energy < 0.0001:
            fake_score += 30.0
        
        # Unusual spectral centroid
        if spectral_centroid < 500 or spectral_centroid > 4000:
            fake_score += 20.0
        
        # Unusual zero crossing rate
        if zcr < 0.01 or zcr > 0.3:
            fake_score += 15.0
        
        # Very uniform signal
        std_dev = np.std(audio)
        if std_dev < 0.001:
            fake_score += 25.0
        
        # Check for compression artifacts
        if len(audio) > 1000:
            low_freq_energy = np.mean(np.abs(audio[:len(audio)//4]) ** 2)
            high_freq_energy = np.mean(np.abs(audio[3*len(audio)//4:]) ** 2)
            if low_freq_energy > 0 and high_freq_energy / low_freq_energy > 2.0:
                fake_score += 20.0
        
        fake_prob = min(fake_score, 95.0)  # Cap at 95%
        
        return {
            "verdict": "Fake" if fake_prob > 50 else "Real",
            "fake_probability": fake_prob,
            "details": f"Heuristic Analysis (Energy: {energy:.6f}, ZCR: {zcr:.4f}, Spectral: {spectral_centroid:.1f}Hz)"
        }
        
    except Exception as e:
        logger.error(f"❌ Heuristic analysis failed: {e}")
        return {
            "verdict": "Real",
            "fake_probability": 0.0,
            "details": f"Heuristic error: {str(e)}"
        }

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze audio file for deepfake detection.
    
    Features:
    - Wav2Vec2 model inference (if available)
    - Heuristic fallback for robustness
    - Bulletproof error handling
    
    Returns:
        Dictionary with verdict, fake_probability, and details
    """
    temp_path = f"/tmp/{uuid.uuid4()}.wav"
    
    try:
        # 1. Save uploaded file
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # 2. Load audio with librosa
        try:
            audio, sr_float = librosa.load(temp_path, sr=16000, duration=10.0)
            sr = int(sr_float)  # Type-safe conversion
        except Exception as e:
            logger.error(f"❌ Failed to load audio: {e}")
            return {
                "status": "ERROR",
                "verdict": "INCONCLUSIVE_TECHNICAL",
                "fake_probability": 0.0,
                "confidence": 0.0,
                "details": f"Audio loading error: {str(e)}",
                "error": str(e)[:200]
            }
        
        # 3. Quick heuristic check (digital silence)
        max_amplitude = np.max(np.abs(audio))
        if max_amplitude < 0.005:
            return {
                "verdict": "Fake",
                "fake_probability": 100.0,
                "details": "Heuristic: Absolute Digital Silence"
            }
        
        # 4. Try model inference if available
        if model is not None and feature_extractor is not None:
            try:
                inputs = feature_extractor(
                    audio, 
                    sampling_rate=16000, 
                    return_tensors="pt", 
                    padding=True
                )
                
                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Support for class inversion via env var
                    # When INVERT_AUDIO_CLASSES=true: index 0=FAKE, index 1=REAL
                    if os.getenv("INVERT_AUDIO_CLASSES", "false").lower() == "true":
                        prob_fake = probs[0][0].item() * 100
                        prob_real = probs[0][1].item() * 100
                        logger.info(f"🔄 Class Inversion Active: index0=FAKE({prob_fake:.2f}%), index1=REAL({prob_real:.2f}%)")
                    else:
                        prob_real = probs[0][0].item() * 100
                        prob_fake = probs[0][1].item() * 100

                    logger.info(f"📊 Model output - Real: {prob_real:.2f}%, Fake: {prob_fake:.2f}%")
                    
                    # Uncertainty warning
                    if abs(prob_real - prob_fake) < 5.0:
                        logger.warning(f"⚠️ Model very uncertain (diff: {abs(prob_real - prob_fake):.2f}%)")
                
                return {
                    "verdict": "Fake" if prob_fake > 50 else "Real",
                    "fake_probability": prob_fake,
                    "details": f"AI Model: {prob_fake:.2f}% fake confidence"
                }
                
            except Exception as model_error:
                logger.error(f"❌ Model inference failed: {model_error}")
                logger.info("🔄 Falling back to heuristic analysis")
                # Fall through to heuristics
        else:
            logger.warning("⚠️ Model unavailable, using heuristic fallback")
        
        # 5. Fallback to heuristics
        result = heuristic_audio_analysis(audio, sr)
        if model is None or feature_extractor is None:
            result["details"] += " (Model Unavailable)"
        return result

    except Exception as e:
        logger.error(f"❌ Critical analysis error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "status": "ERROR",
            "verdict": "INCONCLUSIVE_TECHNICAL",
            "fake_probability": 0.0,
            "confidence": 0.0,
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
        "agent": "audio",
        "model_loaded": model is not None,
        "feature_extractor_loaded": feature_extractor is not None,
        "mode": "neural" if model is not None else "heuristic"
    }

# ============================================================================
# STARTUP EVENT
# ============================================================================
@app.on_event("startup")
async def startup_event() -> None:
    """
    Start server immediately, load model and register with MCP in background.
    
    FIXED: Now targets 'deepguard_brain' for MCP registration.
    """
    logger.info("🚀 Audio Agent starting...")
    
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
    uvicorn.run(app, host="0.0.0.0", port=8004)