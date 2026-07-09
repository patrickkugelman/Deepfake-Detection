import os
import io
import time
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

MODEL_SIGNATURES = {
    "BigGAN": {
        "description": "BigGAN (GAN condiționat pe clasă, stil ImageNet)",
        "fft_high_freq_ratio": (0.00023, 0.00157),
        "dct_energy_concentration": (0.99678, 0.99970),
        "spectral_falloff_slope": (-0.28402, -0.24380),
        "spectral_entropy_range": (2.119, 3.080),
        "color_channel_correlation": (0.8154, 0.9878),
        "color_badge": "#f97316"
    },
    "CIPS": {
        "description": "CIPS (Conditionally Independent Pixel Synthesis)",
        "fft_high_freq_ratio": (0.00056, 0.00403),
        "dct_energy_concentration": (0.99155, 0.99908),
        "spectral_falloff_slope": (-0.26169, -0.17515),
        "spectral_entropy_range": (2.101, 3.201),
        "color_channel_correlation": (0.8441, 0.9794),
        "color_badge": "#8b5cf6"
    },
    "CycleGAN": {
        "description": "CycleGAN (traducere imagine-la-imagine nepereche)",
        "fft_high_freq_ratio": (0.00150, 0.01237),
        "dct_energy_concentration": (0.97529, 0.99706),
        "spectral_falloff_slope": (-0.21915, -0.14495),
        "spectral_entropy_range": (2.577, 3.914),
        "color_channel_correlation": (0.7749, 0.9811),
        "color_badge": "#14b8a6"
    },
    "DDPM": {
        "description": "DDPM (Denoising Diffusion Probabilistic Model)",
        "fft_high_freq_ratio": (0.00065, 0.00385),
        "dct_energy_concentration": (0.99210, 0.99898),
        "spectral_falloff_slope": (-0.23973, -0.18004),
        "spectral_entropy_range": (2.237, 2.933),
        "color_channel_correlation": (0.8101, 0.9806),
        "color_badge": "#3b82f6"
    },
    "Denoising Diffusion GAN": {
        "description": "Denoising Diffusion GAN (hibrid difuzie-GAN)",
        "fft_high_freq_ratio": (0.00094, 0.00287),
        "dct_energy_concentration": (0.98889, 0.99826),
        "spectral_falloff_slope": (-0.23657, -0.19389),
        "spectral_entropy_range": (1.869, 2.527),
        "color_channel_correlation": (0.9054, 0.9828),
        "color_badge": "#ec4899"
    },
    "Diffusion-StyleGAN2": {
        "description": "Diffusion-StyleGAN2 (StyleGAN2 regularizat prin difuzie)",
        "fft_high_freq_ratio": (0.00125, 0.00730),
        "dct_energy_concentration": (0.98600, 0.99784),
        "spectral_falloff_slope": (-0.21650, -0.15801),
        "spectral_entropy_range": (2.174, 3.150),
        "color_channel_correlation": (0.8518, 0.9796),
        "color_badge": "#a855f7"
    },
    "Face Synthetics (rendered)": {
        "description": "Microsoft Face Synthetics (fețe randate 3D sintetic)",
        "fft_high_freq_ratio": (0.00104, 0.00413),
        "dct_energy_concentration": (0.99542, 0.99874),
        "spectral_falloff_slope": (-0.22851, -0.18941),
        "spectral_entropy_range": (1.987, 2.826),
        "color_channel_correlation": (0.8115, 0.9817),
        "color_badge": "#f59e0b"
    },
    "GANsformer": {
        "description": "GANsformer (GAN pe bază de transformer)",
        "fft_high_freq_ratio": (0.00078, 0.00238),
        "dct_energy_concentration": (0.99422, 0.99882),
        "spectral_falloff_slope": (-0.23511, -0.19733),
        "spectral_entropy_range": (2.023, 2.812),
        "color_channel_correlation": (0.8807, 0.9794),
        "color_badge": "#06b6d4"
    },
    "GauGAN": {
        "description": "GauGAN (sinteză semantică de imagini, NVIDIA)",
        "fft_high_freq_ratio": (0.00124, 0.00733),
        "dct_energy_concentration": (0.98436, 0.99812),
        "spectral_falloff_slope": (-0.22252, -0.16848),
        "spectral_entropy_range": (2.359, 3.314),
        "color_channel_correlation": (0.8564, 0.9823),
        "color_badge": "#eab308"
    },
    "Generative Inpainting": {
        "description": "Gated Convolution Inpainting",
        "fft_high_freq_ratio": (0.00186, 0.01415),
        "dct_energy_concentration": (0.97243, 0.99650),
        "spectral_falloff_slope": (-0.20079, -0.14434),
        "spectral_entropy_range": (2.479, 3.501),
        "color_channel_correlation": (0.7828, 0.9756),
        "color_badge": "#ef4444"
    },
    "GLIDE": {
        "description": "GLIDE (difuzie ghidată prin text, OpenAI)",
        "fft_high_freq_ratio": (0.00051, 0.00642),
        "dct_energy_concentration": (0.98288, 0.99937),
        "spectral_falloff_slope": (-0.28284, -0.18503),
        "spectral_entropy_range": (2.227, 3.215),
        "color_channel_correlation": (0.7327, 0.9830),
        "color_badge": "#6366f1"
    },
    "LaMa (inpainting)": {
        "description": "LaMa (inpainting cu măști mari)",
        "fft_high_freq_ratio": (0.00085, 0.00647),
        "dct_energy_concentration": (0.98629, 0.99851),
        "spectral_falloff_slope": (-0.24434, -0.17166),
        "spectral_entropy_range": (1.996, 3.155),
        "color_channel_correlation": (0.8480, 0.9833),
        "color_badge": "#d946ef"
    },
    "Latent Diffusion": {
        "description": "Latent Diffusion Model (LDM)",
        "fft_high_freq_ratio": (0.00077, 0.00813),
        "dct_energy_concentration": (0.98270, 0.99883),
        "spectral_falloff_slope": (-0.24364, -0.16146),
        "spectral_entropy_range": (2.188, 3.202),
        "color_channel_correlation": (0.7370, 0.9863),
        "color_badge": "#0ea5e9"
    },
    "Natural": {
        "description": "Caracteristici spectrale consistente cu o fotografie naturală",
        "fft_high_freq_ratio": (0.00145, 0.01027),
        "dct_energy_concentration": (0.97522, 0.99713),
        "spectral_falloff_slope": (-0.20795, -0.14855),
        "spectral_entropy_range": (2.214, 3.171),
        "color_channel_correlation": (0.8520, 0.9863),
        "color_badge": "#22c55e"
    }
}

_NATURAL_FALLOFF_MID = sum(MODEL_SIGNATURES["Natural"]["spectral_falloff_slope"]) / 2
_GENERATOR_FALLOFF_MID = sum(
    sum(sig["spectral_falloff_slope"]) / 2
    for name, sig in MODEL_SIGNATURES.items() if name != "Natural"
) / (len(MODEL_SIGNATURES) - 1)
FALLOFF_DECISION_BOUNDARY = (_NATURAL_FALLOFF_MID + _GENERATOR_FALLOFF_MID) / 2
FALLOFF_SIGMOID_SCALE = abs(_GENERATOR_FALLOFF_MID - _NATURAL_FALLOFF_MID) / 2

class FingerprintAnalyzer:
    def extract_dct_features(self, gray: np.ndarray) -> dict:
        h, w = gray.shape
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

                low_e = np.sum(dct_block[:4, :4] ** 2)
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
        fft_2d = np.fft.fft2(gray.astype(np.float64))
        fft_shifted = np.fft.fftshift(fft_2d)
        power_spectrum = np.abs(fft_shifted) ** 2

        h, w = power_spectrum.shape
        cy, cx = h // 2, w // 2

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

    def extract_spectral_falloff(self, gray: np.ndarray, n_bins: int = 40) -> dict:
        fft = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
        power = np.log(np.abs(fft) ** 2 + 1e-8)

        h, w = power.shape
        cy, cx = h // 2, w // 2
        y, x = np.indices((h, w))
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r_max = min(cy, cx)

        r_bin = np.clip((r / max(r_max, 1) * n_bins).astype(int), 0, n_bins - 1)
        sums = np.bincount(r_bin.ravel(), weights=power.ravel(), minlength=n_bins)
        counts = np.bincount(r_bin.ravel(), minlength=n_bins)
        profile = sums / np.maximum(counts, 1)

        lo, hi = 2, n_bins - 2
        bins_idx = np.arange(lo, hi)
        values = profile[lo:hi]
        valid = np.isfinite(values)
        if valid.sum() < 5:
            return {"spectral_falloff_slope": -0.15}

        slope = float(np.polyfit(bins_idx[valid], values[valid], 1)[0])
        return {"spectral_falloff_slope": slope}

    def extract_spectral_entropy(self, gray: np.ndarray) -> float:
        flat = gray.flatten().astype(np.float64)
        if len(flat) < 256:
            return 4.0

        freqs, psd = welch(flat, fs=256, nperseg=min(256, len(flat) // 4))
        psd_norm = psd / (psd.sum() + 1e-10)
        ent = float(scipy_entropy(psd_norm + 1e-12))
        return ent

    def extract_color_correlation(self, image_rgb: np.ndarray) -> float:
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
        def band_score(value, lo, hi):
            mid = (lo + hi) / 2
            spread = (hi - lo) / 2 + 1e-6
            return max(0.0, 1.0 - abs(value - mid) / spread)

        weighted = [
            (5.0, band_score(features.get("spectral_falloff_slope", -0.15), *sig["spectral_falloff_slope"])),
            (1.0, band_score(features.get("fft_high_freq_ratio", 0.08), *sig["fft_high_freq_ratio"])),
            (1.0, band_score(features.get("dct_energy_concentration", 0.75), *sig["dct_energy_concentration"])),
            (1.0, band_score(features.get("spectral_entropy", 4.0), *sig["spectral_entropy_range"])),
            (1.0, band_score(features.get("color_correlation", 0.87), *sig["color_channel_correlation"])),
        ]
        total_weight = sum(w for w, _ in weighted)
        return sum(w * s for w, s in weighted) / total_weight

    def analyze(self, image_rgb: np.ndarray) -> dict:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        dct_feats = self.extract_dct_features(gray)
        fft_feats = self.extract_fft_features(gray)
        falloff_feats = self.extract_spectral_falloff(gray)
        spectral_ent = self.extract_spectral_entropy(gray)
        color_corr = self.extract_color_correlation(image_rgb)

        features = {
            **dct_feats,
            **fft_feats,
            **falloff_feats,
            "spectral_entropy": spectral_ent,
            "color_correlation": color_corr
        }

        slope = features["spectral_falloff_slope"]
        z = (FALLOFF_DECISION_BOUNDARY - slope) / (FALLOFF_SIGMOID_SCALE + 1e-6)
        gen_probability = float(1.0 / (1.0 + np.exp(-z)))
        
        DETECTION_THRESHOLD = 0.65
        is_generated = gen_probability >= DETECTION_THRESHOLD

        detected_model = "Unknown"
        model_scores = {}
        
        if is_generated:
            for model_name, sig in MODEL_SIGNATURES.items():
                if model_name == "Natural":
                    continue
                model_scores[model_name] = self.score_against_signature(features, sig)
            if model_scores:
                detected_model = max(model_scores.keys(), key=lambda k: model_scores[k])

        confidence = gen_probability if is_generated else (1.0 - gen_probability)

        return {
            "detected_model": detected_model if is_generated else "Unknown",
            "model_description": MODEL_SIGNATURES[detected_model]["description"] if is_generated else "Caracteristici spectrale consistente cu o fotografie naturală",
            "confidence": round(confidence, 4),
            "is_ai_generated": is_generated,
            "color_badge": MODEL_SIGNATURES[detected_model]["color_badge"] if is_generated else "#22c55e",
            "all_scores": {k: round(v, 3) for k, v in model_scores.items()} if is_generated else {},
            "spectral_features": {
                "dct_energy_concentration": round(features["dct_energy_concentration"], 4),
                "fft_high_freq_ratio": round(features["fft_high_freq_ratio"], 4),
                "spectral_falloff_slope": round(features["spectral_falloff_slope"], 4),
                "spectral_entropy": round(spectral_ent, 4),
                "color_channel_correlation": round(color_corr, 4)
            }
        }

_analyzer = FingerprintAnalyzer()

@app.post("/analyze")
async def analyze_fingerprint(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Agentul Fingerprint acceptă doar imagini.")

    try:
        content = await file.read()
        image_pil = Image.open(io.BytesIO(content)).convert("RGB")

        max_dim = 512
        w, h = image_pil.size
        if w > max_dim or h > max_dim:
            scale = max_dim / max(w, h)
            image_pil = image_pil.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

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

def register_with_coordinator():
    mcp_payload = {
        "tool": {
            "name": "fingerprint",
            "display_name": "Fingerprint Analyzer",
            "description": "GAN/Diffusion Model Attribution via DCT/FFT Spectral Analysis.",
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
                    "description": "Welch spectral entropy"
                },
                {
                    "name": "model_attribution",
                    "description": "Identifică modelul generator"
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
    logger.info("🔬 Fingerprint Analyzer pornit")
    t = threading.Thread(target=register_with_coordinator, daemon=True)
    t.start()
    logger.info("🚀 Fingerprint Analyzer gata pe portul %d", AGENT_PORT)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=AGENT_PORT)