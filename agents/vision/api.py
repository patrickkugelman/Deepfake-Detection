"""
DeepGuard Vision Agent - Image Deepfake Detection
=================================================
Detects image manipulation, GANs, and face-swaps using Xception CNN.
"""

import os
import io
import time
import logging
import threading
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import timm
import requests

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger("DeepGuardVision")

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="DeepGuard Vision Agent",
    version="2.0",
    description="Image deepfake detection using Xception CNN"
)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_PATH = os.path.join("models", "best_vision_model.pth")
BASE_MODEL = "xception"

# Global model variables
model: Optional[nn.Module] = None
model_loaded = False

# Image preprocessing transform
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Standard Xception input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

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
            "name": "vision",
            "display_name": "Vision Agent",
            "description": "Analyzes images for manipulation artifacts, facial inconsistencies, and GAN signatures using Xception CNN",
            "version": "2.0.0",
            "agent_type": "neural",
            "modality": "image",
            "capabilities": [
                {
                    "name": "face_detection",
                    "description": "Detects faces in images",
                    "confidence_threshold": 0.8
                },
                {
                    "name": "gan_artifact_detection",
                    "description": "Identifies GAN-generated artifacts",
                    "confidence_threshold": 0.75
                },
                {
                    "name": "metadata_forensics",
                    "description": "Analyzes EXIF and metadata",
                    "confidence_threshold": 0.6
                }
            ],
            "input_schema": {
                "modality": "image",
                "formats": ["jpg", "jpeg", "png", "bmp", "webp"],
                "max_size_mb": 50
            },
            "output_schema": {
                "verdict_field": "verdict",
                "confidence_field": "fake_probability",
                "details_field": "details"
            },
            "endpoint": "http://deepguard_vision:8002/analyze",
            "health_endpoint": "http://deepguard_vision:8002/health"
        }
    }
    
    coordinator_base = os.getenv("COORDINATOR_URL", "http://deepguard_brain:8000")
    register_url = f"{coordinator_base.rstrip('/')}/mcp/register"
    max_retries = 15
    retry_delay = 5

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"📝 Attempting to register with Coordinator at {register_url} (Attempt {attempt}/{max_retries})...")

            response = requests.post(
                register_url,
                json=mcp_payload,
                timeout=10
            )

            if response.status_code in [200, 201]:
                logger.info(f"✅ Successfully registered with Coordinator!")
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

    logger.error(f"❌ Failed to register with Coordinator after {max_retries} retries")
    logger.info("ℹ️  Agent will continue running, but won't appear in MCP registry")

# ============================================================================
# MODEL LOADING
# ============================================================================
def load_xception() -> Optional[nn.Module]:
    """
    Load Xception model architecture using timm library.
    
    Returns:
        Xception model or None if loading fails
    """
    try:
        # Use timm to create standard Xception architecture
        m = timm.create_model(BASE_MODEL, pretrained=False, num_classes=2)
        logger.info("✅ Xception architecture initialized (via timm)")
        return m
    except Exception as e:
        logger.error(f"❌ Failed to create Xception model via timm: {e}")
        return None

def load_model_background() -> None:
    """
    Load Xception model and weights in background.
    Handles architecture mismatches with partial loading.
    """
    global model, model_loaded
    
    logger.info("🚀 Initializing Vision Agent...")
    
    try:
        # Step 1: Create Xception architecture
        model = load_xception()
        if model is None:
            logger.error("❌ Failed to create model architecture")
            model_loaded = False
            return
        
        # Step 2: Try to load weights
        if os.path.exists(MODEL_PATH):
            logger.info(f"📂 Found weights file: {MODEL_PATH}")
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
                    
                    # Try partial loading (only compatible layers)
                    try:
                        model_state = model.state_dict()
                        compatible_state = {}
                        incompatible_keys = []
                        
                        for key, value in state_dict.items():
                            if key in model_state:
                                # Check if shapes match
                                if model_state[key].shape == value.shape:
                                    compatible_state[key] = value
                                else:
                                    incompatible_keys.append(f"{key}: {value.shape} vs {model_state[key].shape}")
                        
                        if incompatible_keys:
                            logger.warning(f"   ⚠️ {len(incompatible_keys)} incompatible layer shapes (will skip)")
                        
                        if compatible_state:
                            model.load_state_dict(compatible_state, strict=False)
                            loaded_count = len(compatible_state)
                            total_count = len(model_state)
                            logger.warning(f"⚠️ Loaded {loaded_count}/{total_count} compatible layers")
                            logger.warning("⚠️ Model using partial weights + random init for mismatched layers")
                        else:
                            logger.error("❌ No compatible layers found - cannot load weights")
                            model = None
                            model_loaded = False
                            return
                            
                    except Exception as e2:
                        logger.error(f"❌ Partial load failed: {e2}")
                        import traceback
                        logger.error(traceback.format_exc())
                        model = None
                        model_loaded = False
                        return
                else:
                    # Perfect match - load normally
                    model.load_state_dict(state_dict, strict=True)
                    logger.info("✅ Weights loaded successfully")
                
            except Exception as e:
                logger.error(f"❌ Failed to load weights: {e}")
                model = None
                model_loaded = False
                return
        else:
            logger.warning(f"⚠️ Weights file not found: {MODEL_PATH}")
            logger.warning("⚠️ Running in untrained mode (heuristic fallback will be used)")
            model = None
            model_loaded = False
            return
        
        # Step 3: Set to eval mode
        if model is not None:
            model.eval()
            model_loaded = True
            logger.info("✅ Vision model ready for inference")
        else:
            model_loaded = False
            
    except Exception as e:
        logger.error(f"❌ Critical startup error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        model = None
        model_loaded = False

# ============================================================================
# HEURISTIC ANALYSIS (Fallback)
# ============================================================================
def heuristic_analysis(image: Image.Image) -> Dict[str, Any]:
    """
    Fallback heuristic analysis when model fails.
    Uses image metadata, noise patterns, and basic statistics.
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with verdict, fake_probability, and details
    """
    try:
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Basic statistics
        mean_brightness = float(np.mean(img_array))
        std_dev = float(np.std(img_array))
        
        # Check for suspicious patterns
        # 1. Very uniform images (potential compression artifacts)
        if std_dev < 10:
            fake_prob = 70.0
            reason = "Heuristic: Extremely uniform image (potential compression artifact)"
        # 2. Very dark or very bright images
        elif mean_brightness < 20 or mean_brightness > 235:
            fake_prob = 60.0
            reason = "Heuristic: Extreme brightness levels detected"
        # 3. Check for metadata anomalies
        elif hasattr(image, 'getexif'):
            try:
                exif = image.getexif()
                if exif is None or len(exif) == 0:
                    fake_prob = 55.0
                    reason = "Heuristic: Missing EXIF metadata"
                else:
                    fake_prob = 45.0
                    reason = "Heuristic: Basic analysis (model unavailable)"
            except Exception:
                fake_prob = 45.0
                reason = "Heuristic: Basic analysis (model unavailable)"
        else:
            # Default: slightly suspicious but not definitive
            fake_prob = 45.0
            reason = "Heuristic: Basic analysis (model unavailable)"
        
        return {
            "verdict": "Fake" if fake_prob > 50 else "Real",
            "fake_probability": fake_prob,
            "details": reason
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
    Analyze image for deepfake detection.
    
    Features:
    - Xception CNN inference (if model loaded)
    - Heuristic fallback for robustness
    - Bulletproof error handling
    
    Vision agent ONLY processes images. Videos are rejected (PIL cannot open them).
    
    Returns:
        Dictionary with verdict, fake_probability, and details
    """
    try:
        content = await file.read()
        
        # REJECT VIDEOS IMMEDIATELY - Vision agent uses PIL which cannot open video files
        mime_type = file.content_type or ""
        fname = file.filename or ""
        if mime_type.startswith("video") or any(fname.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv']):
            logger.warning(f"⚠️ Vision agent cannot process videos: {fname}")
            return {
                "verdict": "INCONCLUSIVE",
                "fake_probability": 0.0,
                "confidence": 0.0,
                "details": "Vision agent only processes images, not videos",
                "error": "Invalid file type for vision agent"
            }
        
        # Process as image
        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            logger.error(f"❌ Not a valid image: {e}")
            return {
                "verdict": "INCONCLUSIVE",
                "fake_probability": 0.0,
                "confidence": 0.0,
                "details": f"Failed to open as image: {str(e)}",
                "error": "Invalid image file"
            }
        
        # If model is not loaded, use heuristic fallback
        if not model_loaded or model is None:
            logger.warning("⚠️ Model unavailable, using heuristic fallback")
            return heuristic_analysis(image)
        
        # Try model inference
        try:
            tensor = transform(image).unsqueeze(0) # type: ignore
            tensor.requires_grad_() # Enable gradients for saliency map
            
            # Model inference (with gradients enabled for saliency)
            output = model(tensor)
            
            # Handle different Xception output formats
            if isinstance(output, tuple):
                logits = output[0]  # Xception returns (logits, features)
            else:
                logits = output
            
            probs = torch.softmax(logits, dim=1)
            fake_prob = float(probs[0][1].item() * 100)

            # --- Grad-CAM Generation (Gradient-weighted Class Activation Mapping) ---
            try:
                import cv2
                import hashlib
                import base64

                # ── Grad-CAM hooks ──────────────────────────────────────────
                # Găsim ultimul layer convoluțional din modelul timm Xception.
                # timm expune conv_head (1×1 conv) sau blocks[-1] (ultimul block).
                target_layer = None
                if hasattr(model, 'conv_head'):
                    target_layer = model.conv_head
                elif hasattr(model, 'blocks') and len(model.blocks) > 0:
                    target_layer = model.blocks[-1]

                activations_store = {}
                gradients_store = {}

                def fwd_hook(module, inp, out):
                    activations_store['value'] = out  # no detach — needed for backward()

                def bwd_hook(module, grad_in, grad_out):
                    gradients_store['value'] = grad_out[0].detach()

                handles = []
                if target_layer is not None:
                    handles.append(target_layer.register_forward_hook(fwd_hook))
                    handles.append(target_layer.register_full_backward_hook(bwd_hook))

                # Re-run forward cu gradienți activați
                tensor_grad = transform(image).unsqueeze(0)
                tensor_grad.requires_grad_(True)
                out2 = model(tensor_grad)
                logits2 = out2[0] if isinstance(out2, tuple) else out2
                score = logits2[0][1]  # Scorul clasei FAKE
                model.zero_grad()
                score.backward()

                # Eliberăm hookurile
                for h in handles:
                    h.remove()

                # ── Calculăm CAM ────────────────────────────────────────────
                grad_cam_generated = False
                if 'value' in activations_store and 'value' in gradients_store:
                    acts = activations_store['value']   # [1, C, H, W]
                    grads = gradients_store['value']    # [1, C, H, W]

                    # Global Average Pooling pe gradienți → ponderi per canal
                    weights = grads.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

                    # Combinare ponderată a activation maps
                    cam = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, H, W]
                    cam = torch.nn.functional.relu(cam)

                    # Normalize la [0, 1]
                    cam_min, cam_max = cam.min(), cam.max()
                    if (cam_max - cam_min).item() > 1e-8:
                        cam = (cam - cam_min) / (cam_max - cam_min)

                    cam_np = cam.squeeze().cpu().numpy()  # [H, W]
                    grad_cam_generated = True
                else:
                    # Fallback la gradient pe input dacă hookurile nu au funcționat
                    if tensor_grad.grad is not None:
                        gradients = tensor_grad.grad.data.abs()[0]
                        cam_np, _ = torch.max(gradients, dim=0)
                        cam_np = cam_np.cpu().numpy()
                        cam_min, cam_max = cam_np.min(), cam_np.max()
                        if cam_max - cam_min > 1e-8:
                            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
                        grad_cam_generated = True
                    else:
                        cam_np = np.zeros((7, 7), dtype=np.float32)

                # ── Generăm imaginea overlay ─────────────────────────────────
                file_hash = hashlib.sha256(content).hexdigest()
                heatmap_filename = f"{file_hash}_gradcam.jpg"
                evidence_dir = "/app/data/evidence"
                os.makedirs(evidence_dir, exist_ok=True)
                heatmap_path = os.path.join(evidence_dir, heatmap_filename)

                orig_w, orig_h = image.size
                cam_uint8 = (255 * cam_np).astype(np.uint8)
                cam_resized = cv2.resize(cam_uint8, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                heatmap_color = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)

                # Overlay pe imaginea originală (BGR)
                image_bgr = np.array(image.convert("RGB"))[:, :, ::-1].copy()
                overlay = cv2.addWeighted(image_bgr, 0.6, heatmap_color, 0.4, 0)
                cv2.imwrite(heatmap_path, overlay)

                # Encode base64 pentru răspuns direct în JSON
                with open(heatmap_path, "rb") as f:
                    heatmap_b64 = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")

                logger.info(f"🎨 Grad-CAM saved: {heatmap_path} (layer_found={grad_cam_generated})")

                evidence = [{
                    "type": "grad_cam",
                    "title": "Grad-CAM — Hartă de Activare (Explainable AI)",
                    "description": (
                        "Zonele roșii/galbene indică regiunile care au influențat cel mai mult "
                        "decizia rețelei neuronale pentru clasa FAKE. "
                        "Grad-CAM (Gradient-weighted Class Activation Mapping) vizualizează "
                        "atenția modelului Xception CNN."
                    ),
                    "url": f"/evidence/{heatmap_filename}",
                    "base64_image": heatmap_b64,
                    "confidence": fake_prob,
                    "method": "Grad-CAM" if grad_cam_generated and 'value' in activations_store else "Gradient-Input Fallback"
                }]
            except Exception as viz_e:
                logger.warning(f"Failed to generate Grad-CAM: {viz_e}")
                evidence = []
            # ----------------------------------------------------------------
            
            logger.info(f"📊 Vision analysis - Fake probability: {fake_prob:.2f}%")
            
            return {
                "verdict": "Fake" if fake_prob > 50 else "Real",
                "fake_probability": round(fake_prob, 2),
                "details": f"Xception CNN Confidence: {fake_prob:.2f}%",
                "evidence": evidence
            }
            
        except Exception as model_error:
            logger.error(f"❌ Model inference failed: {model_error}")
            # Fallback to heuristics
            logger.info("🔄 Falling back to heuristic analysis")
            result = heuristic_analysis(image)
            result["details"] = f"Heuristic (Model Error: {str(model_error)[:50]})"
            return result
            
    except Exception as e:
        logger.error(f"❌ Critical analysis error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            "verdict": "Real",
            "fake_probability": 0.0,
            "details": f"Analysis error: {str(e)[:100]}",
            "error": str(e)[:200]
        }

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for Docker"""
    return {
        "status": "healthy",
        "agent": "vision",
        "model_loaded": model_loaded,
        "model_type": BASE_MODEL,
        "mode": "neural" if model_loaded else "heuristic"
    }

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with agent info"""
    return {
        "agent": "DeepGuard Vision Agent",
        "version": "2.0",
        "model": BASE_MODEL,
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
    logger.info("🚀 Vision Agent starting...")
    
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
    uvicorn.run(app, host="0.0.0.0", port=8002)