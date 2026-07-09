"""
Recalibrates MODEL_SIGNATURES in api.py using REAL sample images instead of the
hand-typed "empirical" ranges that shipped originally (never actually measured against
real generator output).

Measures the five spectral features FingerprintAnalyzer extracts at inference time (DCT
energy concentration, FFT high-freq ratio, spectral falloff slope, spectral entropy,
color channel correlation) directly from real images of each of the 13 generators, plus a
pooled "Natural" signature from 7 real-photo categories. Importing FingerprintAnalyzer
from api.py (rather than reimplementing the math) guarantees the calibration uses the
exact same feature code that runs in production.

Source images: the original archive/ folder (from agents/vision/dataset/archive/) has
since been deleted after its contents were merged into agents/vision/dataset/Train and
Test - this reads the same images back out of there instead, using the "archive_<cat>_"
filename prefix that prepare_archive_dataset.py gave them, so the source data doesn't
need to be re-downloaded.

spectral_falloff_slope carries most of the discriminating power (see the comment in
api.py's extract_spectral_falloff) - the other four all landed in nearly-identical
ranges across every generator AND real photos during the first calibration pass.

This also changes what "is_ai_generated" means: previously an image whose features
matched NONE of the 5 old signatures well still got weakly attributed to whichever was
"least wrong", gated by an arbitrary IS_GENERATED_THRESHOLD=0.52 magic number. With
"Natural" now competing directly, api.py's analyze() picks whichever signature -
including Natural - fits best, so a genuinely natural image can win outright instead of
being forced into a low-confidence generator guess.

Usage:
    python calibrate_signatures.py --samples_per_category 300
Writes signatures.json (a MODEL_SIGNATURES-shaped dict, ranges as [lo, hi] percentile bands).
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from api import FingerprintAnalyzer  # reuse the exact feature-extraction code used at inference

# archive category name -> display name shown in the UI
FAKE_CATEGORIES = {
    "big_gan": "BigGAN",
    "cips": "CIPS",
    "cycle_gan": "CycleGAN",
    "ddpm": "DDPM",
    "denoising_diffusion_gan": "Denoising Diffusion GAN",
    "diffusion_gan": "Diffusion-StyleGAN2",
    "face_synthetics": "Face Synthetics (rendered)",
    "gansformer": "GANsformer",
    "gau_gan": "GauGAN",
    "generative_inpainting": "Generative Inpainting",
    "glide": "GLIDE",
    "lama": "LaMa (inpainting)",
    "latent_diffusion": "Latent Diffusion",
}
REAL_CATEGORIES = ["afhq", "celebahq", "coco", "ffhq", "imagenet", "landscape", "lsun"]
IMG_EXTS = (".jpg", ".jpeg", ".png")
MAX_DIM = 512  # matches api.py's inference-time resize, so calibration sees the same conditions

FEATURE_KEYS = ("dct_energy_concentration", "fft_high_freq_ratio", "spectral_falloff_slope",
                "spectral_entropy", "color_channel_correlation")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vision_dataset_dir", type=str,
                    default=str(Path(__file__).parent.parent / "vision" / "dataset"),
                    help="Reads archive_<category>_* files back out of Train/{Real,Fake} "
                         "(and Test/{Real,Fake} as a fallback) since dataset/archive/ was deleted")
    p.add_argument("--samples_per_category", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default=str(Path(__file__).parent / "signatures.json"))
    p.add_argument("--lo_pct", type=float, default=15.0)
    p.add_argument("--hi_pct", type=float, default=85.0)
    return p.parse_args()


def sample_images(dataset_dir: Path, label: str, cat_dir_name: str, n: int, rng: random.Random) -> list:
    pattern = f"archive_{cat_dir_name}_*"
    imgs = [f for f in (dataset_dir / "Train" / label).glob(pattern) if f.suffix.lower() in IMG_EXTS]
    imgs += [f for f in (dataset_dir / "Test" / label).glob(pattern) if f.suffix.lower() in IMG_EXTS]
    rng.shuffle(imgs)
    return imgs[:n]


def extract_features(path: Path, analyzer: FingerprintAnalyzer):
    try:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if w > MAX_DIM or h > MAX_DIM:
            scale = MAX_DIM / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        dct_feats = analyzer.extract_dct_features(gray)
        fft_feats = analyzer.extract_fft_features(gray)
        falloff_feats = analyzer.extract_spectral_falloff(gray)
        ent = analyzer.extract_spectral_entropy(gray)
        color_corr = analyzer.extract_color_correlation(arr)
        return {
            "dct_energy_concentration": dct_feats["dct_energy_concentration"],
            "fft_high_freq_ratio": fft_feats["fft_high_freq_ratio"],
            "spectral_falloff_slope": falloff_feats["spectral_falloff_slope"],
            "spectral_entropy": ent,
            "color_channel_correlation": color_corr,
        }
    except Exception:
        return None


def collect_stats(image_paths: list, analyzer: FingerprintAnalyzer) -> dict:
    feats = {k: [] for k in FEATURE_KEYS}
    for path in image_paths:
        f = extract_features(path, analyzer)
        if f is None:
            continue
        for k in FEATURE_KEYS:
            feats[k].append(f[k])
    return feats


def to_signature(feats: dict, lo_pct: float, hi_pct: float) -> dict:
    def rng(key):
        arr = np.array(feats[key])
        return [float(np.percentile(arr, lo_pct)), float(np.percentile(arr, hi_pct))]

    return {
        "fft_high_freq_ratio": rng("fft_high_freq_ratio"),
        "dct_energy_concentration": rng("dct_energy_concentration"),
        "spectral_falloff_slope": rng("spectral_falloff_slope"),
        "spectral_entropy_range": rng("spectral_entropy"),
        "color_channel_correlation": rng("color_channel_correlation"),
        "n_samples": len(feats["dct_energy_concentration"]),
    }


def main():
    args = parse_args()
    dataset_dir = Path(args.vision_dataset_dir)
    analyzer = FingerprintAnalyzer()
    rng = random.Random(args.seed)

    if not dataset_dir.exists():
        print(f"Vision dataset folder not found: {dataset_dir}")
        return

    signatures = {}

    print(f"=== Calibrating {len(FAKE_CATEGORIES)} FAKE generator signatures ({args.samples_per_category} samples each) ===")
    for cat_dir_name, display_name in FAKE_CATEGORIES.items():
        imgs = sample_images(dataset_dir, "Fake", cat_dir_name, args.samples_per_category, rng)
        feats = collect_stats(imgs, analyzer)
        if not feats["dct_energy_concentration"]:
            print(f"  ! {cat_dir_name}: no valid images found, skipping")
            continue
        sig = to_signature(feats, args.lo_pct, args.hi_pct)
        signatures[display_name] = sig
        print(f"  {display_name}: n={sig['n_samples']} falloff={sig['spectral_falloff_slope']} "
              f"dct={sig['dct_energy_concentration']}")

    print(f"\n=== Calibrating pooled Natural (real-photo) signature from {len(REAL_CATEGORIES)} categories ===")
    per_cat_n = max(1, args.samples_per_category // len(REAL_CATEGORIES))
    natural_paths = []
    for cat_dir_name in REAL_CATEGORIES:
        natural_paths.extend(sample_images(dataset_dir, "Real", cat_dir_name, per_cat_n, rng))
    feats = collect_stats(natural_paths, analyzer)
    if not feats["dct_energy_concentration"]:
        print("  ! no valid Natural images found")
    else:
        sig = to_signature(feats, args.lo_pct, args.hi_pct)
        signatures["Natural"] = sig
        print(f"  Natural: n={sig['n_samples']} falloff={sig['spectral_falloff_slope']} "
              f"dct={sig['dct_energy_concentration']}")

    with open(args.output, "w") as f:
        json.dump(signatures, f, indent=2)
    print(f"\nWritten {args.output}")


if __name__ == "__main__":
    main()
