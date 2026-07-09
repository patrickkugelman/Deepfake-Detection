"""
DeepGuard Bio Agent - Physiological & Biological Plausibility Analysis
=======================================================================
Video: rPPG heart-rate detection (CHROM signal + Butterworth bandpass +
sliding-window majority vote) plus micro-movement liveness.
Image: no temporal signal exists in a single frame, so pulse detection is
physically impossible. Instead, static images get a biological-plausibility
check (skin-tone consistency, micro-texture, facial symmetry) as a
complementary signal to the vision agent - NOT a pulse measurement.
"""

import os
import cv2
import uuid
import shutil
import time
import logging
import threading
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from scipy.signal import butter, filtfilt, detrend
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
    version="3.0",
    description="rPPG pulse detection (video) + biological-plausibility analysis (image)"
)

# ============================================================================
# CONFIGURATION
# ============================================================================
AGENT_DIR = os.path.dirname(__file__)
HAAR_CASCADE_PATH = os.path.join(AGENT_DIR, "haarcascade_frontalface_default.xml")
DNN_PROTO_PATH = os.path.join(AGENT_DIR, "deploy.prototxt")
DNN_MODEL_PATH = os.path.join(AGENT_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

BBox = Tuple[int, int, int, int]  # x, y, w, h

# Global face detectors
FACE_CASCADE: Optional[cv2.CascadeClassifier] = None
FACE_NET: Optional[cv2.dnn.Net] = None

# ============================================================================
# MCP REGISTRATION
# ============================================================================
def register_with_coordinator() -> None:
    """Register this agent with the Coordinator via MCP Protocol."""
    mcp_payload = {
        "tool": {
            "name": "bio",
            "display_name": "Bio Agent",
            "description": (
                "Verifies biological plausibility: rPPG pulse + micro-movement liveness for "
                "video, skin-tone/texture/symmetry plausibility analysis for still images"
            ),
            "version": "3.0.0",
            "agent_type": "neural",
            "modality": "video,image",
            "capabilities": [
                {
                    "name": "pulse_detection",
                    "description": "Detects heart rate via rPPG (video only)",
                    "confidence_threshold": 0.85
                },
                {
                    "name": "micro_expressions",
                    "description": "Analyzes facial micro-movements (video only)",
                    "confidence_threshold": 0.75
                },
                {
                    "name": "liveness_verification",
                    "description": "Confirms biological presence in video",
                    "confidence_threshold": 0.8
                },
                {
                    "name": "image_biological_plausibility",
                    "description": "Skin-tone/texture/symmetry plausibility check for a single image (no pulse - no temporal signal exists in a still frame)",
                    "confidence_threshold": 0.6
                }
            ],
            "input_schema": {
                "modality": "video,image",
                "formats": ["mp4", "avi", "mov", "jpg", "jpeg", "png", "webp", "bmp"],
                "max_size_mb": 200,
                "requires_preprocessing": True
            },
            "output_schema": {
                "verdict_field": "verdict",
                "confidence_field": "fake_probability",
                "details_field": "details",
                "additional_fields": ["analysis_type", "pulse_detected", "heart_rate_bpm"]
            },
            "endpoint": "http://deepguard_bio:8005/analyze",
            "health_endpoint": "http://deepguard_bio:8005/health"
        }
    }

    coordinator_base = os.getenv("COORDINATOR_URL", "http://deepguard_brain:8000")
    register_url = f"{coordinator_base.rstrip('/')}/mcp/register"
    max_retries = 15
    retry_delay = 5

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"📝 Attempting to register with Coordinator (Attempt {attempt}/{max_retries})...")

            response = requests.post(register_url, json=mcp_payload, timeout=10)

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
# FACE DETECTION (DNN primary, Haar Cascade fallback)
# ============================================================================
def _detect_face_dnn(frame: np.ndarray, conf_threshold: float = 0.6) -> Optional[BBox]:
    """SSD ResNet10 face detector (2017) - far more robust to angle/lighting/
    partial occlusion than Haar Cascade (2001), which is the primary reason
    the old pipeline silently fell back to a center-crop so often."""
    if FACE_NET is None:
        return None
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    FACE_NET.setInput(blob)
    detections = FACE_NET.forward()

    best_bbox = None
    best_conf = conf_threshold
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence <= best_conf:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            best_bbox = (x1, y1, x2 - x1, y2 - y1)
            best_conf = confidence
    return best_bbox


def _detect_face_haar(frame: np.ndarray) -> Optional[BBox]:
    if FACE_CASCADE is None:
        return None
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            return (int(x), int(y), int(w), int(h))
    except Exception as e:
        logger.warning(f"Haar face detection error: {e}")
    return None


def detect_face(frame: np.ndarray) -> Optional[BBox]:
    """DNN first (more accurate), Haar Cascade as a fast fallback if the DNN
    model failed to load or found nothing."""
    bbox = _detect_face_dnn(frame)
    if bbox is not None:
        return bbox
    return _detect_face_haar(frame)


def _forehead_roi(frame: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
    """Forehead region (top 20%, middle 60% of the face box) - the classic
    rPPG sampling region: mostly bare skin, minimal occlusion by hair/glasses/
    facial hair compared to cheeks or chin."""
    x, y, w, h = bbox
    y1 = max(0, y + int(h * 0.05))
    y2 = min(frame.shape[0], y + int(h * 0.25))
    x1 = max(0, x + int(w * 0.2))
    x2 = min(frame.shape[1], x + int(w * 0.8))
    if y2 <= y1 or x2 <= x1:
        return None
    roi = frame[y1:y2, x1:x2]
    return roi if roi.size > 0 else None


def _track_bbox(frame: np.ndarray, last_bbox: Optional[BBox], frame_idx: int,
                 redetect_interval: int = 5, smoothing: float = 0.6) -> Optional[BBox]:
    """Re-detects every `redetect_interval` frames and exponentially smooths
    the bbox in between, instead of running independent Haar detection on
    every single frame. Independent per-frame detection is what caused most
    of the jitter/noise in the old rPPG signal: a face box that moves 5-10px
    frame-to-frame due to detector noise (not actual head movement) shifts
    the sampled skin pixels and injects spurious high-frequency energy that
    swamps the tiny (~1% brightness modulation) real pulse signal."""
    need_redetect = (last_bbox is None) or (frame_idx % redetect_interval == 0)
    if not need_redetect:
        return last_bbox

    new_bbox = detect_face(frame)
    if new_bbox is None:
        return last_bbox  # keep previous box; better than snapping to a fallback crop mid-stream
    if last_bbox is None:
        return new_bbox

    return tuple(int(smoothing * l + (1 - smoothing) * n) for l, n in zip(last_bbox, new_bbox))  # type: ignore

# ============================================================================
# rPPG SIGNAL EXTRACTION (CHROM method)
# ============================================================================
def _extract_rgb_trace(video_path: str, max_seconds: float = 30.0) -> Tuple[np.ndarray, float, int]:
    """Samples mean R/G/B of the tracked forehead ROI per frame.
    Returns (rgb_trace[N,3], fps, frames_with_face)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps < 1:
        fps = 30.0

    max_frames = int(fps * max_seconds)
    rgb_trace: List[Tuple[float, float, float]] = []
    frames_with_face = 0
    last_bbox: Optional[BBox] = None
    frame_idx = 0

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        bbox = _track_bbox(frame, last_bbox, frame_idx)
        last_bbox = bbox

        roi = _forehead_roi(frame, bbox) if bbox is not None else None
        if roi is None:
            h, w = frame.shape[:2]
            roi = frame[h // 4:h // 2, w // 3:2 * w // 3]
        else:
            frames_with_face += 1

        b_mean = float(np.mean(roi[:, :, 0]))
        g_mean = float(np.mean(roi[:, :, 1]))
        r_mean = float(np.mean(roi[:, :, 2]))
        rgb_trace.append((r_mean, g_mean, b_mean))
        frame_idx += 1

    cap.release()
    return np.array(rgb_trace), fps, frames_with_face


def _chrom_signal(rgb_trace: np.ndarray) -> np.ndarray:
    """CHROM method (de Haan & Jeanne, 2013): combines all 3 channels via a
    chrominance projection that cancels specular reflection/lighting changes
    common to R, G and B, instead of relying on the raw green channel alone
    (which is far noisier under any lighting variation or compression)."""
    r, g, b = rgb_trace[:, 0], rgb_trace[:, 1], rgb_trace[:, 2]
    rn = r / (np.mean(r) + 1e-6)
    gn = g / (np.mean(g) + 1e-6)
    bn = b / (np.mean(b) + 1e-6)

    x = 3 * rn - 2 * gn
    y = 1.5 * rn + gn - 1.5 * bn

    std_x = np.std(x)
    std_y = np.std(y)
    alpha = std_x / std_y if std_y > 1e-6 else 1.0

    return x - alpha * y


def _bandpass_filter(signal: np.ndarray, fps: float, low: float = 0.7, high: float = 3.0, order: int = 3) -> np.ndarray:
    """Butterworth bandpass (42-180 BPM) with detrending, replacing the old
    approach of taking a raw FFT of the unfiltered signal and just looking at
    which bins fall in-band - a proper filter suppresses out-of-band noise
    energy before spectral analysis instead of just masking it post-hoc."""
    signal = detrend(signal)
    nyq = fps / 2.0
    low_n = max(low / nyq, 1e-4)
    high_n = min(high / nyq, 0.999)
    if low_n >= high_n:
        return signal
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, signal)


def _analyze_window(segment: np.ndarray, fps: float) -> Optional[Dict[str, float]]:
    try:
        filtered = _bandpass_filter(segment, fps)
    except Exception:
        return None

    L = len(filtered)
    if L < 8:
        return None

    windowed = filtered * np.hanning(L)
    fft = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(L, 1 / fps)

    band_idx = np.where((freqs >= 0.7) & (freqs <= 3.0))[0]
    if len(band_idx) == 0:
        return None

    band_power = np.abs(fft[band_idx])
    peak_i = int(np.argmax(band_power))
    peak_freq = float(freqs[band_idx][peak_i])
    bpm = peak_freq * 60.0

    peak_power = float(band_power[peak_i])
    avg_power = float(np.mean(band_power))
    snr = peak_power / avg_power if avg_power > 0 else 0.0

    return {"bpm": bpm, "snr": snr}


def _sigmoid(x: float, midpoint: float, steepness: float) -> float:
    return float(1.0 / (1.0 + np.exp(-steepness * (x - midpoint))))


def extract_pulse(video_path: str) -> Dict[str, Any]:
    """rPPG pulse extraction via sliding-window CHROM analysis.

    Instead of one fixed ~10s window and a hard SNR>3.0 cutoff, this splits
    the trace into overlapping windows, computes BPM+SNR per window, and
    aggregates via median + consistency ratio. This is both more robust
    (a single noisy window no longer decides the whole verdict) and gives a
    continuous confidence value instead of a step function at SNR==3.0.
    """
    try:
        rgb_trace, fps, frames_with_face = _extract_rgb_trace(video_path)

        if len(rgb_trace) < 30:
            return {
                "detected": False, "bpm": 0.0, "snr": 0.0, "confidence": 0.0,
                "face_coverage": 0.0,
                "msg": "Video too short for pulse analysis"
            }

        face_coverage = frames_with_face / len(rgb_trace)
        chrom = _chrom_signal(rgb_trace)

        window_len = min(int(fps * 8), len(chrom))
        step = max(int(fps * 2), 1)

        window_results = []
        start = 0
        while start + window_len <= len(chrom) or (start == 0 and window_len == len(chrom)):
            seg = chrom[start:start + window_len]
            res = _analyze_window(seg, fps)
            if res is not None:
                window_results.append(res)
            if start + window_len >= len(chrom):
                break
            start += step

        if not window_results:
            return {
                "detected": False, "bpm": 0.0, "snr": 0.0, "confidence": 0.0,
                "face_coverage": round(face_coverage, 2),
                "msg": "No frequency components in human heart rate range"
            }

        bpms = np.array([w["bpm"] for w in window_results])
        snrs = np.array([w["snr"] for w in window_results])
        median_bpm = float(np.median(bpms))
        median_snr = float(np.median(snrs))

        consistent = np.abs(bpms - median_bpm) <= 8.0
        consistency_ratio = float(np.mean(consistent))

        # Continuous confidence: sigmoid on SNR (replaces the hard SNR>3.0 cutoff),
        # scaled down when windows disagree on BPM (majority-vote robustness) and
        # when the face tracker actually had a face for only part of the clip.
        raw_confidence = _sigmoid(median_snr, midpoint=2.2, steepness=1.5)
        confidence = raw_confidence * (0.4 + 0.6 * consistency_ratio) * (0.5 + 0.5 * face_coverage)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        bpm_plausible = 42 <= median_bpm <= 180
        is_human = bool(confidence > 0.55 and bpm_plausible)

        if is_human:
            msg = f"Pulse: {median_bpm:.1f} BPM (SNR: {median_snr:.2f}, {len(window_results)} windows, {consistency_ratio*100:.0f}% agree)"
        else:
            msg = f"Pulse unclear (SNR: {median_snr:.2f}, confidence: {confidence*100:.0f}%)"

        return {
            "detected": is_human,
            "bpm": round(median_bpm, 1) if is_human else 0.0,
            "snr": round(median_snr, 2),
            "confidence": round(confidence * 100, 1),
            "face_coverage": round(face_coverage, 2),
            "windows_analyzed": len(window_results),
            "windows_agreeing": round(consistency_ratio * 100, 1),
            "msg": msg
        }

    except Exception as e:
        logger.error(f"❌ rPPG extraction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "detected": False, "bpm": 0.0, "snr": 0.0, "confidence": 0.0,
            "msg": f"rPPG error: {str(e)[:100]}"
        }

# ============================================================================
# MICRO-MOVEMENT & LIVENESS DETECTION
# ============================================================================
def detect_liveness(video_path: str) -> Dict[str, Any]:
    """Detect liveness through micro-movements and frame differences."""
    try:
        cap = cv2.VideoCapture(video_path)
        ret, prev = cap.read()
        if not ret:
            return {"alive": False, "score": 0.0}

        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        diffs = []
        max_frames = 60

        count = 0
        while count < max_frames:
            ret, curr = cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, curr_gray)  # type: ignore
            diff_arr = np.array(diff)
            score = float(np.mean(diff_arr))  # type: ignore
            diffs.append(score)

            prev_gray = curr_gray
            count += 1

        cap.release()

        if not diffs:
            return {"alive": False, "score": 0.0}

        diffs_array = np.array(diffs)
        avg_motion = float(np.mean(diffs_array))
        std_motion = float(np.std(diffs_array))

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
# IMAGE-MODE: BIOLOGICAL PLAUSIBILITY (no temporal signal -> no pulse)
# ============================================================================
def _bell_score(value: float, ideal_center: float, ideal_width: float) -> float:
    """Gaussian-shaped score in [0,1], peaking at ideal_center. Used because
    both "too little" and "too much" of each signal below are suspicious
    (e.g. near-zero skin-tone variance looks over-smoothed/generated, but
    very high variance looks like a blending seam) - a plain threshold can't
    express that, a bell curve can."""
    return float(np.exp(-((value - ideal_center) ** 2) / (2 * ideal_width ** 2)))


def analyze_image_biology(image_path: str) -> Dict[str, Any]:
    """Static-image biological-plausibility check.

    A single frame carries no temporal information, so genuine rPPG pulse
    detection is physically impossible here - this is a deliberately
    different, complementary signal to the vision agent, not a pulse
    measurement: skin-tone consistency, micro-texture consistency and facial
    symmetry across face regions, each scored against the range typical of a
    naturally photographed face (GAN/diffusion generation and face-swap
    blending tend to push these either to unnaturally uniform or unnaturally
    inconsistent extremes).
    """
    img = cv2.imread(image_path)
    if img is None:
        return {
            "status": "ERROR", "analysis_type": "image_plausibility",
            "verdict": "INCONCLUSIVE_TECHNICAL", "fake_probability": 0.0, "confidence": 0.0,
            "pulse_detected": None, "heart_rate_bpm": 0.0,
            "details": "Could not decode image", "reason": "Unreadable image file"
        }

    bbox = detect_face(img)
    if bbox is None:
        return {
            "status": "INCONCLUSIVE", "analysis_type": "image_plausibility",
            "verdict": "INCONCLUSIVE_INSUFFICIENT_DATA", "fake_probability": 0.0, "confidence": 0.0,
            "pulse_detected": None, "heart_rate_bpm": 0.0,
            "details": "No face detected in image",
            "reason": "Bio agent requires a visible face for biological-plausibility analysis"
        }

    x, y, w, h = bbox
    face = img[y:y + h, x:x + w]
    if face.size == 0 or w < 40 or h < 40:
        return {
            "status": "INCONCLUSIVE", "analysis_type": "image_plausibility",
            "verdict": "INCONCLUSIVE_INSUFFICIENT_DATA", "fake_probability": 0.0, "confidence": 0.0,
            "pulse_detected": None, "heart_rate_bpm": 0.0,
            "details": "Detected face region too small for reliable analysis",
            "reason": "Face crop below minimum usable resolution"
        }

    regions = {
        "forehead": face[int(h * 0.05):int(h * 0.25), int(w * 0.25):int(w * 0.75)],
        "left_cheek": face[int(h * 0.40):int(h * 0.65), int(w * 0.05):int(w * 0.35)],
        "right_cheek": face[int(h * 0.40):int(h * 0.65), int(w * 0.65):int(w * 0.95)],
        "chin": face[int(h * 0.75):int(h * 0.95), int(w * 0.30):int(w * 0.70)],
    }
    regions = {name: roi for name, roi in regions.items() if roi.size > 0}

    if len(regions) < 3:
        return {
            "status": "INCONCLUSIVE", "analysis_type": "image_plausibility",
            "verdict": "INCONCLUSIVE_INSUFFICIENT_DATA", "fake_probability": 0.0, "confidence": 0.0,
            "pulse_detected": None, "heart_rate_bpm": 0.0,
            "details": "Not enough face regions extractable (occlusion or extreme pose)",
            "reason": "Insufficient facial regions for plausibility analysis"
        }

    # 1. Skin-tone consistency (Cr/Cb chroma channels across regions)
    tone_samples = []
    for roi in regions.values():
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        tone_samples.append((float(np.mean(ycrcb[:, :, 1])), float(np.mean(ycrcb[:, :, 2]))))
    tone_arr = np.array(tone_samples)
    tone_std = float(np.mean(np.std(tone_arr, axis=0)))
    tone_score = _bell_score(tone_std, ideal_center=3.0, ideal_width=4.0)

    # 2. Micro-texture consistency (Laplacian variance = high-frequency roughness)
    texture_samples = []
    for roi in regions.values():
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        texture_samples.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    texture_arr = np.array(texture_samples)
    texture_mean = float(np.mean(texture_arr))
    texture_cv = float(np.std(texture_arr) / (texture_mean + 1e-6))
    # Center calibrated against real photographed faces (CelebA-HQ sample), which
    # typically show texture_cv ~0.5-0.9 across forehead/cheek/chin (natural pore/
    # lighting variation) - centering at 0.3 as originally guessed scored real faces
    # too low. Both near-zero (over-smoothed/generated) and very high (injected noise)
    # texture_cv remain penalized.
    texture_score = _bell_score(texture_cv, ideal_center=0.7, ideal_width=0.55)

    # 3. Facial symmetry (mirror-flip difference)
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.resize(gray_face, (256, 256))
    left_half = gray_face[:, :128]
    right_half = cv2.flip(gray_face[:, 128:], 1)
    asymmetry = float(np.mean(cv2.absdiff(left_half, right_half))) / 255.0
    # Center calibrated against real photographed faces (CelebA-HQ sample), which
    # typically land ~0.07-0.15 due to natural pose/lighting asymmetry.
    symmetry_score = _bell_score(asymmetry, ideal_center=0.11, ideal_width=0.07)

    plausibility = float(np.clip(
        tone_score * 0.35 + texture_score * 0.35 + symmetry_score * 0.30, 0.0, 1.0
    ))
    fake_prob = round((1.0 - plausibility) * 100, 2)
    confidence = round(abs(plausibility - 0.5) * 200, 2)

    details = (
        f"skin_tone_std={tone_std:.2f} texture_cv={texture_cv:.2f} "
        f"asymmetry={asymmetry:.3f} plausibility={plausibility:.2f}"
    )

    if plausibility >= 0.6:
        verdict, status = "REAL", "OK"
        reason = "Skin tone, texture and symmetry are consistent with a naturally photographed face"
    elif plausibility <= 0.4:
        verdict, status = "FAKE", "OK"
        reason = "Skin tone/texture/symmetry patterns are atypical of a naturally photographed face (possible AI-generated or blended content)"
    else:
        verdict, status = "INCONCLUSIVE", "INCONCLUSIVE"
        reason = "Biological-plausibility signal is ambiguous"

    return {
        "status": status,
        "analysis_type": "image_plausibility",
        "verdict": verdict,
        "fake_probability": fake_prob,
        "confidence": confidence,
        "pulse_detected": None,  # not applicable - no temporal signal in a still image
        "heart_rate_bpm": 0.0,
        "details": details,
        "reason": reason,
        "plausibility_breakdown": {
            "skin_tone_consistency": round(tone_score, 3),
            "texture_consistency": round(texture_score, 3),
            "facial_symmetry": round(symmetry_score, 3),
        }
    }

# ============================================================================
# API ENDPOINTS
# ============================================================================
def _suffix_for(filename: Optional[str]) -> str:
    if not filename:
        return ".mp4"
    ext = os.path.splitext(filename)[1].lower()
    return ext if ext else ".mp4"


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze media for biological plausibility.

    - Video: rPPG pulse detection (CHROM + Butterworth + sliding-window) + micro-movement liveness.
    - Image: skin-tone/texture/symmetry biological-plausibility check (no pulse - physically
      impossible from a single frame).
    """
    suffix = _suffix_for(file.filename)
    is_image = suffix in IMAGE_EXTENSIONS
    temp_path = f"/tmp/{uuid.uuid4()}{suffix}"

    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if is_image:
            logger.info(f"🖼️ Analyzing image biological plausibility from {file.filename}")
            result = analyze_image_biology(temp_path)
            logger.info(f"📊 Bio (image) analysis - Verdict: {result['verdict']} ({result['fake_probability']}% fake)")
            return result

        logger.info(f"🫀 Analyzing biometric signals (video) from {file.filename}")

        pulse_data = extract_pulse(temp_path)
        motion_data = detect_liveness(temp_path)

        verdict = "INCONCLUSIVE"
        status = "INCONCLUSIVE"
        reason = "Analysis incomplete"
        fake_prob = 0.0

        if pulse_data["detected"]:
            fake_prob = 10.0
            verdict = "REAL"
            status = "OK"
            reason = f"Strong pulse detected: {pulse_data['bpm']} BPM"
            logger.info(f"✅ Pulse detected: {pulse_data['bpm']} BPM")
        else:
            if FACE_NET is None and FACE_CASCADE is None:
                status = "ERROR"
                verdict = "INCONCLUSIVE_TECHNICAL"
                reason = "Face detector not initialized"
            elif motion_data["alive"]:
                fake_prob = 60.0
                verdict = "FAKE"
                status = "OK"
                reason = "Motion analysis indicates liveness but no biological pulse found (possible deepfake)"
                logger.warning("⚠️ Motion detected but no pulse - suspicious")
            elif not motion_data["alive"] and motion_data["score"] < 0.5:
                fake_prob = 80.0
                verdict = "FAKE"
                status = "OK"
                reason = "Static image or freeze frame detected (no physiological signs)"
            else:
                status = "INCONCLUSIVE"
                verdict = "INCONCLUSIVE_INSUFFICIENT_DATA"
                reason = "Signal quality too low for determination"

        details = f"{pulse_data['msg']} | Motion: {motion_data['score']}"

        logger.info(f"📊 Bio analysis - Verdict: {verdict} ({fake_prob}% fake)")

        return {
            "status": status,
            "analysis_type": "video_rppg",
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
            "analysis_type": "image_plausibility" if is_image else "video_rppg",
            "verdict": "INCONCLUSIVE_TECHNICAL",
            "fake_probability": 0.0,
            "confidence": 0.0,
            "pulse_detected": None,
            "heart_rate_bpm": 0.0,
            "details": f"Analysis error: {str(e)[:100]}",
            "reason": "Internal agent error",
            "error": str(e)[:200]
        }

    finally:
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
        "dnn_face_detector_loaded": FACE_NET is not None,
        "haar_face_detector_loaded": FACE_CASCADE is not None,
        "mode": "rppg_liveness_and_image_plausibility"
    }


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with agent info"""
    return {
        "agent": "DeepGuard Bio Agent",
        "version": "3.0",
        "methods": ["rPPG (CHROM)", "micro-movement analysis", "image biological plausibility"],
        "status": "operational"
    }

# ============================================================================
# STARTUP EVENT
# ============================================================================
@app.on_event("startup")
async def startup_event() -> None:
    """Load face detectors and register with MCP in background."""
    global FACE_CASCADE, FACE_NET

    logger.info("🚀 Bio Agent starting...")

    if os.path.exists(DNN_PROTO_PATH) and os.path.exists(DNN_MODEL_PATH):
        try:
            FACE_NET = cv2.dnn.readNetFromCaffe(DNN_PROTO_PATH, DNN_MODEL_PATH)
            logger.info("✅ DNN face detector (SSD ResNet10) loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load DNN face detector: {e}")
            FACE_NET = None
    else:
        logger.warning(f"⚠️ DNN face detector files not found at {AGENT_DIR} - will rely on Haar Cascade fallback")

    if os.path.exists(HAAR_CASCADE_PATH):
        try:
            FACE_CASCADE = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
            if FACE_CASCADE.empty():
                logger.warning("⚠️ Haar Cascade loaded but is empty")
                FACE_CASCADE = None
            else:
                logger.info("✅ Haar Cascade fallback detector loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load Haar Cascade: {e}")
            FACE_CASCADE = None
    else:
        logger.warning(f"⚠️ Haar Cascade not found at {HAAR_CASCADE_PATH}")

    if FACE_NET is None and FACE_CASCADE is None:
        logger.error("❌ No face detector available - bio agent will run in degraded fallback-crop mode")

    threading.Thread(target=register_with_coordinator, daemon=True).start()

    logger.info("✅ Bio Agent ready - rPPG, liveness & image-plausibility detection operational")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
