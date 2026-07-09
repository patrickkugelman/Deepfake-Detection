import argparse
import random
import shutil
from pathlib import Path

REAL_CATEGORIES = ["afhq", "celebahq", "coco", "ffhq", "imagenet", "landscape", "lsun"]
FAKE_CATEGORIES = [
    "big_gan", "cips", "cycle_gan", "ddpm", "denoising_diffusion_gan", "diffusion_gan",
    "face_synthetics", "gansformer", "gau_gan", "generative_inpainting", "glide",
    "lama", "latent_diffusion",
]
IMG_EXTS = (".jpg", ".jpeg", ".png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--archive_dir", type=str,
                    default=str(Path(__file__).parent / "dataset" / "archive"))
    p.add_argument("--output_dir", type=str,
                    default=str(Path(__file__).parent / "dataset"),
                    help="agents/vision/dataset by default - same root train_vision.py reads from")
    p.add_argument("--max_per_category", type=int, default=8000,
                    help="Cap images taken per top-level archive folder before balancing "
                         "(some, like coco/lsun, have 150k-240k)")
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def collect_images(category_dir: Path, cap: int, rng: random.Random) -> list:
    images = [f for f in category_dir.rglob("*")
              if f.suffix.lower() in IMG_EXTS and f.stat().st_size > 0]
    rng.shuffle(images)
    return images[:cap]


def gather_label(categories: list, archive: Path, cap: int, rng: random.Random) -> list:
    """Collect capped images from every category folder for one label (Real or Fake)."""
    pooled = []
    for cat in categories:
        cat_dir = archive / cat
        if not cat_dir.exists():
            print(f"  ! {cat} not found in archive, skipping")
            continue
        images = collect_images(cat_dir, cap, rng)
        if not images:
            print(f"  ! {cat}: no images found, skipping")
            continue
        pooled.extend((f, cat) for f in images)
        print(f"  {cat}: {len(images)} images")
    return pooled


def split_and_copy(pooled: list, label: str, out: Path,
                    val_ratio: float, test_ratio: float, rng: random.Random) -> dict:
    rng.shuffle(pooled)
    n = len(pooled)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    splits = {
        "Test": pooled[:n_test],
        "Validation": pooled[n_test:n_test + n_val],
        "Train": pooled[n_test + n_val:],
    }
    counts = {}
    for split_name, entries in splits.items():
        dest_dir = out / split_name / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for f, cat in entries:
            try:
                shutil.copy2(f, dest_dir / f"archive_{cat}_{f.name}")
                copied += 1
            except OSError as e:
                # Transient disk-contention errors shouldn't abort the whole run.
                print(f"  ! Skipping {f.name}: {e}")
        counts[split_name] = copied
    return counts


def main():
    args = parse_args()
    archive = Path(args.archive_dir)
    out = Path(args.output_dir)
    rng = random.Random(args.seed)

    if not archive.exists():
        print(f"Archive folder not found: {archive}")
        return

    print(f"=== Collecting REAL categories (cap {args.max_per_category}/category) ===")
    real_pool = gather_label(REAL_CATEGORIES, archive, args.max_per_category, rng)

    print(f"\n=== Collecting FAKE categories (cap {args.max_per_category}/category) ===")
    fake_pool = gather_label(FAKE_CATEGORIES, archive, args.max_per_category, rng)

    print(f"\nCollected before balancing: {len(real_pool)} real, {len(fake_pool)} fake")
    n = min(len(real_pool), len(fake_pool))
    rng.shuffle(real_pool)
    rng.shuffle(fake_pool)
    real_pool, fake_pool = real_pool[:n], fake_pool[:n]
    print(f"Balanced to {n} each (matches the smaller pool, mirrors prepare_genimage_dataset.py)")

    real_counts = split_and_copy(real_pool, "Real", out, args.val_ratio, args.test_ratio, rng)
    fake_counts = split_and_copy(fake_pool, "Fake", out, args.val_ratio, args.test_ratio, rng)

    print(f"\nAdded Real: {real_counts}")
    print(f"Added Fake: {fake_counts}")
    print(f"\nMerged into {out} - combined with whatever was already in Train/Validation/Test.")
    print("train_vision.py's DeepfakeDataset globs recursively, so no code changes were needed "
          "for it to pick these up - just point --data_dir at this same dataset root.")


if __name__ == "__main__":
    main()
