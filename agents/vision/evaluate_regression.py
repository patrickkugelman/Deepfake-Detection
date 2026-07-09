"""
Evaluates a trained vision model against the FIXED regression_test_set built by
build_regression_set.py - the same images every time, regardless of how many times
Train/Test get regrown by prepare_archive_dataset.py / prepare_genimage_dataset.py, so
results across different training runs are actually comparable.

Reports overall accuracy plus a per-category breakdown (grouped by the archive_<category>_
filename prefix) - useful for spotting "this retrain got better overall but got worse on
GLIDE specifically", which a single aggregate accuracy number would hide.

Usage:
    python evaluate_regression.py --checkpoint models/best_vision_model.pth
Writes regression_test_set/results_<timestamp>.json so successive runs can be diffed.
"""
import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = (".jpg", ".jpeg", ".png")
ARCHIVE_PREFIX_RE = re.compile(r"^archive_([a-z_0-9]+)_img")
IMG_SIZE = 299  # matches train_vision.py's Xception input size


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--regression_dir", type=str,
                    default=str(Path(__file__).parent / "regression_test_set"))
    p.add_argument("--checkpoint", type=str,
                    default=str(Path(__file__).parent / "models" / "best_vision_model.pth"))
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


class RegressionDataset(Dataset):
    def __init__(self, root_dir: Path, transform):
        self.transform = transform
        self.samples = []  # (path, label, category)
        for label_name, label_idx in [("Real", 0), ("Fake", 1)]:
            label_dir = root_dir / label_name
            if not label_dir.exists():
                continue
            for f in label_dir.iterdir():
                if f.suffix.lower() not in IMG_EXTS:
                    continue
                m = ARCHIVE_PREFIX_RE.match(f.name)
                category = m.group(1) if m else "_original_dataset"
                self.samples.append((f, label_idx, f"{label_name}/{category}"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, category = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
        except Exception:
            img = torch.zeros((3, IMG_SIZE, IMG_SIZE))
        return img, label, category


def load_model(checkpoint_path: str, device):
    model = timm.create_model("xception", pretrained=False, num_classes=2, drop_rate=0.5)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    regression_dir = Path(args.regression_dir)
    if not regression_dir.exists():
        print(f"{regression_dir} not found - run build_regression_set.py first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = RegressionDataset(regression_dir, transform)
    if len(dataset) == 0:
        print("Regression set is empty.")
        return
    print(f"Loaded {len(dataset)} regression images")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    per_category = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_correct = 0
    overall_total = 0

    with torch.no_grad():
        for imgs, labels, categories in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct = preds.eq(labels)
            for i, cat in enumerate(categories):
                per_category[cat]["total"] += 1
                per_category[cat]["correct"] += int(correct[i].item())
            overall_correct += int(correct.sum().item())
            overall_total += labels.size(0)

    overall_acc = 100.0 * overall_correct / overall_total
    print(f"\n=== Overall accuracy: {overall_acc:.2f}% ({overall_correct}/{overall_total}) ===")
    print("\nPer-category breakdown:")
    category_results = {}
    for cat in sorted(per_category.keys()):
        stats = per_category[cat]
        acc = 100.0 * stats["correct"] / stats["total"]
        category_results[cat] = {"accuracy": round(acc, 2), "correct": stats["correct"], "total": stats["total"]}
        print(f"  {cat:35s}: {acc:6.2f}% ({stats['correct']}/{stats['total']})")

    results = {
        "checkpoint": args.checkpoint,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall_accuracy": round(overall_acc, 2),
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "per_category": category_results,
    }
    out_path = regression_dir / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten {out_path}")


if __name__ == "__main__":
    main()
