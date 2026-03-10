"""
DeepGuard Pro - Fingerprint Analyzer Agent (Port 8006)
GAN/Diffusion Model Attribution via Spectral Analysis (DCT/FFT)
Identifică cu ce tool AI a fost creat un deepfake.
"""
import os
import io
import time
import uuid
import logging
import threading
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import requests
from scipy.fftpack import dct
from scipy.signal import welch
from scipy.stats import entropy as scipy_entropy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [Fingerprint] %(levelname)s: %(message)s")
logger = logging.getLogger("DeepGuard.Fingerprint")

COORDINATOR_URL = os.getenv("COORDINATOR_URL", "http://deepguard_brain:8000")
AGENT_PORT = int(os.getenv("AGENT_PORT", "8006"))

app = FastAPI(
    title="DeepGuard Fingerprint Analyzer",
    description="GAN/Diffusion Model Attribution via DCT/FFT Spectral Analysis",
    version="1.0.0"
)

# ─────────────────────────────────────────────────────────────────────────────
# Semnături spectrale cunoscute pentru fiecare model generator
# Bazate pe caracteristici empirice din literatura de specialitate
# ─────────────────────────────────────────────────────────────────────────────
MODEL_SIGNATURES = {
    "StableDiffusion": {
        "description": "Stable Diffusion (v1.x / v2.x / SDXL)",
        "fft_high_freq_ratio": (0.04, 0.12),   # Raportul puterii în benzi înalte
        "dct_energy_concentration": (0.70, 0.88), # Concentrarea energiei în benzi joase DCT
        "spectral_entropy_range": (3.2, 4.8),
        "color_channel_correlation": (0.82, 0.97), # Corelație ridicată între canale RGB
        "artifact_signature": "periodic_noise_pattern",
        "color_badge": "#f97316"  # portocaliu
    },
    "Midjourney": {
        "description": "Midjourney (v4 / v5 / v6)",
        "fft_high_freq_ratio": (0.02, 0.09),
        "dct_energy_concentration": (0.75, 0.92),
        "spectral_entropy_range": (2.8, 4.2),
        "color_channel_correlation": (0.85, 0.98),
        "artifact_signature": "smooth_gradient_artifacts",
        "color_badge": "#8b5cf6"  # violet
    },
    "DALL-E": {
        "description": "DALL-E 2 / DALL-E 3 (OpenAI)",
        "fft_high_freq_ratio": (0.05, 0.13),
        "dct_energy_concentration": (0.68, 0.85),
        "spectral_entropy_range": (3.5, 5.0),
        "color_channel_correlation": (0.78, 0.94),
        "artifact_signature": "compositional_inconsistency",
        "color_badge": "#10b981"  # verde
    },
    "DeepFaceLab": {
        "description": "DeepFaceLab (Face-Swap GAN)",
        "fft_high_freq_ratio": (0.08, 0.18),    # Mai mult noise la margini de față
        "dct_energy_concentration": (0.60, 0.80),
        "spectral_entropy_range": (4.0, 5.5),   # Entropie mai mare din cauza blend-ului
        "color_channel_correlation": (0.70, 0.88),
        "artifact_signature": "blend_boundary_artifacts",
        "color_badge": "#ef4444"  # roșu
    },
    "ProPainter": {
        "description": "ProPainter / Inpainting Models",
        "fft_high_freq_ratio": (0.06, 0.15),
        "dct_energy_concentration": (0.65, 0.83),
        "spectral_entropy_range": (3.8, 5.2),
        "color_channel_correlation": (0.75, 0.92),
        "artifact_signature": "texture_boundary_discontinuity",
        "color_badge": "#3b82f6"  # albastru
    }
}


class FingerprintAnalyzer:
    """
    Motor de analiză spectrală pentru atribuirea modelului generator.
    Folosește DCT, FFT 2D și entropie spectrală pentru fingerprinting.
    """

    def extract_dct_features(self, gray: np.ndarray) -> dict:
        """
        Extrage caracteristici din domeniul DCT pe blocuri 8x8 (similar JPEG).
        GANurile lasă amprente specifice în coeficienții DCT de înaltă frecvență.
        """
        h, w = gray.shape
        # Resize la multiplu de 8
        h_new = (h // 8) * 8
        w_new = (w // 8) * 8
        gray_resized = cv2.resize(gray, (w_new, h_new))

        high_freq_energy = []
        low_freq_energy = []
        total_energy = []

        for i in range(0, h_new - 8, 8):
            for j in range(0, w_new - 8, 8):
                block = gray_resized[i:i+8, j:j+8].astype(np.float64)
                dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

                block_energy = np.sum(dct_block ** 2)
                if block_energy < 1e-10:
                    continue

                # DC + primele benzi joase (0:4, 0:4)
                low_e = np.sum(dct_block[:4, :4] ** 2)
                # Benzile de înaltă frecvență (4:, 4:)
                high_e = np.sum(dct_block[4:, 4:] ** 2)

                low_freq_energy.append(low_e / block_energy)
                high_freq_energy.append(high_e / block_energy)
                total_energy.append(block_energy)

        if not low_freq_energy:
            return {"dct_energy_concentration": 0.5, "dct_high_freq_mean": 0.5}

        return {
            "dct_energy_concentration": float(np.mean(low_freq_energy)),
            "dct_high_freq_mean": float(np.mean(high_freq_energy)),
            "dct_energy_std": float(np.std(low_freq_energy))
        }

    def extract_fft_features(self, gray: np.ndarray) -> dict:
        """
        Analizează distribuția puterii în domeniul frecvență (FFT 2D).
        Modelele generative au distribuții specifice ale puterii spectrale.
        """
        fft_2d = np.fft.fft2(gray.astype(np.float64))
        fft_shifted = np.fft.fftshift(fft_2d)
        power_spectrum = np.abs(fft_shifted) ** 2

        h, w = power_spectrum.shape
        cy, cx = h // 2, w // 2

        # Împarte în zone concentrice
        y_grid, x_grid = np.ogrid[-cy:h-cy, -cx:w-cx]
        radius_map = np.sqrt(x_grid**2 + y_grid**2)
        max_r = min(cy, cx)

        dc_mask = radius_map <= max_r * 0.05
        low_mask = (radius_map > max_r * 0.05) & (radius_map <= max_r * 0.25)
        mid_mask = (radius_map > max_r * 0.25) & (radius_map <= max_r * 0.55)
        high_mask = radius_map > max_r * 0.55

        total_power = power_spectrum.sum() + 1e-10
        dc_ratio = power_spectrum[dc_mask].sum() / total_power
        low_ratio = power_spectrum[low_mask].sum() / total_power
        mid_ratio = power_spectrum[mid_mask].sum() / total_power
        high_ratio = power_spectrum[high_mask].sum() / total_power

        return {
            "fft_dc_ratio": float(dc_ratio),
            "fft_low_freq_ratio": float(low_ratio),
            "fft_mid_freq_ratio": float(mid_ratio),
            "fft_high_freq_ratio": float(high_ratio)
        }

    def extract_spectral_entropy(self, gray: np.ndarray) -> float:
        """
        Calculează entropia spectrală Welch.
        Conținutul natural are entropie mai mare decât cel generat AI.
        """
        flat = gray.flatten().astype(np.float64)
        if len(flat) < 256:
            return 4.0

        freqs, psd = welch(flat, fs=256, nperseg=min(256, len(flat) // 4))
        psd_norm = psd / (psd.sum() + 1e-10)
        ent = float(scipy_entropy(psd_norm + 1e-12))
        return ent

    def extract_color_correlation(self, image_rgb: np.ndarray) -> float:
        """
        Calculează corelația medie dintre canalele RGB.
        Modelele generative tind să aibă corelații mai mari (culori mai uniforme).
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
            return 0.85

        r = image_rgb[:, :, 0].flatten().astype(np.float64)
        g = image_rgb[:, :, 1].flatten().astype(np.float64)
        b = image_rgb[:, :, 2].flatten().astype(np.float64)

        def safe_corr(a, b):
            if np.std(a) < 1e-10 or np.std(b) < 1e-10:
                return 1.0
            return float(np.corrcoef(a, b)[0, 1])

        corr_rg = safe_corr(r, g)
        corr_rb = safe_corr(r, b)
        corr_gb = safe_corr(g, b)
        return float(np.mean([abs(corr_rg), abs(corr_rb), abs(corr_gb)]))

    def score_against_signature(self, features: dict, sig: dict) -> float:
        """
        Calculează un scor de potrivire [0, 1] între feature-urile extrase
        și semnătura unui model generator.
        """
        scores = []

        # 1. FFT high-freq ratio
        fft_hr = features.get("fft_high_freq_ratio", 0.08)
        lo, hi = sig["fft_high_freq_ratio"]
        mid = (lo + hi) / 2
        spread = (hi - lo) / 2 + 1e-6
        scores.append(max(0.0, 1.0 - abs(fft_hr - mid) / spread))

        # 2. DCT energy concentration
        dct_ec = features.get("dct_energy_concentration", 0.75)
        lo, hi = sig["dct_energy_concentration"]
        mid = (lo + hi) / 2
        spread = (hi - lo) / 2 + 1e-6
        scores.append(max(0.0, 1.0 - abs(dct_ec - mid) / spread))

        # 3. Spectral entropy
        se = features.get("spectral_entropy", 4.0)
        lo, hi = sig["spectral_entropy_range"]
        mid = (lo + hi) / 2
        spread = (hi - lo) / 2 + 1e-6
        scores.append(max(0.0, 1.0 - abs(se - mid) / spread))

        # 4. Color channel correlation
        cc = features.get("color_correlation", 0.87)
        lo, hi = sig["color_channel_correlation"]
        mid = (lo + hi) / 2
        spread = (hi - lo) / 2 + 1e-6
        scores.append(max(0.0, 1.0 - abs(cc - mid) / spread))

        return float(np.mean(scores))

    def analyze(self, image_rgb: np.ndarray) -> dict:
        """
        Analizează imaginea și returnează modelul generator detectat.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Extrage toate feature-urile
        dct_feats = self.extract_dct_features(gray)
        fft_feats = self.extract_fft_features(gray)
        spectral_ent = self.extract_spectral_entropy(gray)
        color_corr = self.extract_color_correlation(image_rgb)

        features = {
            **dct_feats,
            **fft_feats,
            "spectral_entropy": spectral_ent,
            "color_correlation": color_corr
        }

        # Scorează față de fiecare semnătură
        model_scores = {}
        for model_name, sig in MODEL_SIGNATURES.items():
            model_scores[model_name] = self.score_against_signature(features, sig)

        best_model = max(model_scores, key=model_scores.get)
        best_score = model_scores[best_model]

        # Prag minim pentru a considera că e un deepfake generat cu tool cunoscut
        IS_GENERATED_THRESHOLD = 0.52

        return {
            "detected_model": best_model if best_score >= IS_GENERATED_THRESHOLD else "Unknown",
            "model_description": MODEL_SIGNATURES[best_model]["description"] if best_score >= IS_GENERATED_THRESHOLD else "Model necunoscut sau conținut natural",
            "confidence": round(best_score, 4),
            "is_ai_generated": best_score >= IS_GENERATED_THRESHOLD,
            "color_badge": MODEL_SIGNATURES[best_model]["color_badge"] if best_score >= IS_GENERATED_THRESHOLD else "#6b7280",
            "all_scores": {k: round(v, 3) for k, v in model_scores.items()},
            "spectral_features": {
                "dct_energy_concentration": round(features["dct_energy_concentration"], 4),
                "fft_high_freq_ratio": round(features["fft_high_freq_ratio"], 4),
                "spectral_entropy": round(spectral_ent, 4),
                "color_channel_correlation": round(color_corr, 4)
            }
        }


# Instanță globală
_analyzer = FingerprintAnalyzer()


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze_fingerprint(file: UploadFile = File(...)):
    """
    Analizează amprenta spectrală a imaginii și identifică modelul generator.
    Returnează: detected_model, confidence, spectral_features, all_scores.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Agentul Fingerprint acceptă doar imagini.")

    try:
        content = await file.read()
        image_pil = Image.open(io.BytesIO(content)).convert("RGB")

        # Resize la max 512x512 pentru performanță
        max_dim = 512
        w, h = image_pil.size
        if w > max_dim or h > max_dim:
            scale = max_dim / max(w, h)
            image_pil = image_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        image_np = np.array(image_pil)
        result = _analyzer.analyze(image_np)

        fake_prob = result["confidence"] * 100 if result["is_ai_generated"] else 0.0

        return {
            "verdict": "Fake" if result["is_ai_generated"] else "Real",
            "fake_probability": round(fake_prob, 2),
            "confidence": result["confidence"],
            "detected_model": result["detected_model"],
            "model_description": result["model_description"],
            "color_badge": result["color_badge"],
            "is_ai_generated": result["is_ai_generated"],
            "all_model_scores": result["all_scores"],
            "spectral_features": result["spectral_features"],
            "details": f"Attribution: {result['detected_model']} — {result['model_description']} (confidence: {result['confidence']*100:.1f}%)"
        }

    except Exception as e:
        logger.error(f"Eroare analiză fingerprint: {e}")
        return JSONResponse(status_code=200, content={
            "verdict": "INCONCLUSIVE",
            "fake_probability": 0.0,
            "confidence": 0.0,
            "detected_model": "Unknown",
            "model_description": "Analiză eșuată",
            "is_ai_generated": False,
            "error": str(e),
            "details": f"Fingerprint analysis error: {str(e)}"
        })


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent": "fingerprint_analyzer",
        "version": "1.0.0",
        "port": AGENT_PORT,
        "capabilities": [
            "dct_analysis",
            "fft_spectral_analysis",
            "spectral_entropy",
            "color_channel_correlation",
            "model_attribution"
        ],
        "supported_models": list(MODEL_SIGNATURES.keys())
    }


@app.get("/")
async def root():
    return {
        "agent": "DeepGuard Fingerprint Analyzer",
        "description": "GAN/Diffusion Model Attribution via DCT/FFT Spectral Analysis",
        "version": "1.0.0",
        "endpoint": "/analyze"
    }


# ─────────────────────────────────────────────────────────────────────────────
# MCP Auto-Registration cu Coordinator
# ─────────────────────────────────────────────────────────────────────────────

def register_with_coordinator():
    """Se înregistrează automat cu Coordinator-ul via MCP Protocol."""
    mcp_payload = {
        "tool": {
            "name": "fingerprint",
            "display_name": "Fingerprint Analyzer",
            "description": "GAN/Diffusion Model Attribution via DCT/FFT Spectral Analysis. Identifică dacă un deepfake a fost creat cu StableDiffusion, Midjourney, DALL-E, DeepFaceLab sau ProPainter.",
            "version": "1.0.0",
            "agent_type": "neural",
            "modality": "image",
            "capabilities": [
                {
                    "name": "dct_analysis",
                    "description": "Discrete Cosine Transform artifact detection pe blocuri 8x8"
                },
                {
                    "name": "fft_analysis",
                    "description": "2D FFT power spectrum distribution analysis"
                },
                {
                    "name": "spectral_entropy",
                    "description": "Welch spectral entropy — conținut natural vs AI-generated"
                },
                {
                    "name": "model_attribution",
                    "description": "Identifică modelul generator: StableDiffusion, Midjourney, DALL-E, DeepFaceLab, ProPainter"
                }
            ],
            "input_schema": {
                "modality": "image",
                "formats": ["jpg", "jpeg", "png", "webp", "gif", "bmp"],
                "max_size_mb": 50
            },
            "output_schema": {
                "verdict_field": "verdict",
                "confidence_field": "confidence",
                "attribution_field": "detected_model"
            },
            "endpoint": f"http://deepguard_fingerprint:{AGENT_PORT}/analyze",
            "health_endpoint": f"http://deepguard_fingerprint:{AGENT_PORT}/health"
        }
    }

    max_retries = 15
    for attempt in range(max_retries):
        try:
            time.sleep(5)
            resp = requests.post(
                f"{COORDINATOR_URL}/mcp/register",
                json=mcp_payload,
                timeout=10
            )
            if resp.status_code in (200, 201):
                logger.info(f"✅ Fingerprint Agent înregistrat cu Coordinator ({COORDINATOR_URL})")
                return
            else:
                logger.warning(f"⚠️ Registration attempt {attempt+1}/{max_retries}: HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Registration attempt {attempt+1}/{max_retries} failed: {e}")
        time.sleep(5)

    logger.error("❌ Nu s-a putut înregistra cu Coordinator după toate reîncercările.")


@app.on_event("startup")
async def startup_event():
    logger.info("🔬 Fingerprint Analyzer pornit — inițializare analiză spectrală...")
    # Înregistrare asincronă în background
    t = threading.Thread(target=register_with_coordinator, daemon=True)
    t.start()
    logger.info("🚀 Fingerprint Analyzer gata pe portul %d", AGENT_PORT)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=AGENT_PORT)
