"""
Builds a small, FIXED regression/benchmark set for the vision agent - a stable pool of
images that never gets included in future training runs, so successive model retrains can
be compared on the exact same evidence instead of each getting evaluated against a
different, reshuffled Test split.

Why this is needed: prepare_archive_dataset.py (and prepare_genimage_dataset.py before it)
ADD new images into Train/Validation/Test every time they run, and Test itself is what
train_vision.py's final evaluation uses - so "Test accuracy" from two different training
runs isn't really comparable, since Test's contents themselves keep growing/changing.
This script copies a one-time, versioned sample OUT of Test into a separate folder that
nothing else writes to, specifically from Test/ (never Train/), so the current
best_vision_model.pth (already trained on Train) hasn't seen any of it - it started as a
genuine held-out check, and stays fixed going forward regardless of what happens to Test/.

Usage:
    python build_regression_set.py --samples_per_category 20
"""
import argparse
import random
import shutil
import re
from pathlib import Path
from collections import defaultdict

IMG_EXTS = (".jpg", ".jpeg", ".png")
ARCHIVE_PREFIX_RE = re.compile(r"^archive_([a-z_0-9]+)_img")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source_dir", type=str,
                    default=str(Path(__file__).parent / "dataset" / "Test"),
                    help="Only ever sources from Test/, never Train/ - so the current "
                         "trained model hasn't seen these images")
    p.add_argument("--output_dir", type=str,
                    default=str(Path(__file__).parent / "regression_test_set"))
    p.add_argument("--samples_per_category", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def group_by_category(label_dir: Path) -> dict:
    groups = defaultdict(list)
    for f in label_dir.iterdir():
        if f.suffix.lower() not in IMG_EXTS:
            continue
        m = ARCHIVE_PREFIX_RE.match(f.name)
        category = m.group(1) if m else "_original_dataset"
        groups[category].append(f)
    return groups


def main():
    args = parse_args()
    source = Path(args.source_dir)
    out = Path(args.output_dir)
    rng = random.Random(args.seed)

    if out.exists():
        print(f"{out} already exists - delete it first if you want to rebuild from scratch "
              f"(not doing that automatically, since this set is meant to stay FIXED once built).")
        return

    summary = {}
    for label in ("Real", "Fake"):
        label_dir = source / label
        if not label_dir.exists():
            print(f"  ! {label_dir} not found, skipping")
            continue
        groups = group_by_category(label_dir)
        dest_dir = out / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        for category, files in sorted(groups.items()):
            rng.shuffle(files)
            picked = files[:args.samples_per_category]
            for f in picked:
                shutil.copy2(f, dest_dir / f.name)
            summary[f"{label}/{category}"] = len(picked)
            print(f"  {label}/{category}: {len(picked)} images")

    total = sum(summary.values())
    print(f"\nBuilt regression_test_set with {total} images across {len(summary)} categories at {out}")
    print("This set is now fixed - future prepare_archive_dataset.py / prepare_genimage_dataset.py "
          "runs won't touch it (different folder), so evaluate_regression.py results stay comparable "
          "across model retrains.")


if __name__ == "__main__":
    main()
