"""
Validates every GenVideo-Real/GenVideo-Fake clip and repairs corrupt ones by
re-extracting them from the original .tar shards.

Why: the first extraction ran concurrently with two large HuggingFace downloads,
and the resulting disk contention left some mp4s truncated ("moov atom not found"
during training — the mp4's metadata atom lives at the end of the file, so a
truncated copy is unreadable even though the file exists and has plausible size).

The filenames encode their origin (dataset_shard_000012_000345_SEINE_678.mp4 →
member "000345_SEINE_678.mp4" inside dataset_shard_000012.tar), so each corrupt
file can be restored from its exact source.

Usage:
    python repair_genvideo_dataset.py --shards_dir D:/deepfake-detector-structure/datasets_augmentation/video_genvideo
"""
import argparse
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import cv2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shards_dir", type=str, required=True)
    p.add_argument("--dataset_dir", type=str, default=str(Path(__file__).parent / "dataset"))
    p.add_argument("--delete_unrepairable", action="store_true",
                   help="Delete files that are corrupt and can't be re-extracted (default: leave and report)")
    return p.parse_args()


def is_valid_video(path: Path) -> bool:
    """A clip is valid if OpenCV can open it AND decode its first frame."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        return bool(ret and frame is not None)
    finally:
        cap.release()


SHARD_RE = re.compile(r"^(dataset_shard_\d+)_(.+)$")


def main():
    args = parse_args()
    shards_dir = Path(args.shards_dir)
    dataset_dir = Path(args.dataset_dir)

    corrupt_by_shard: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    checked = 0
    corrupt = 0

    for folder in ("GenVideo-Real", "GenVideo-Fake"):
        for clip in sorted((dataset_dir / folder).glob("*.mp4")):
            checked += 1
            if checked % 1000 == 0:
                print(f"  scanned {checked} clips ({corrupt} corrupt so far)...")
            if is_valid_video(clip):
                continue
            corrupt += 1
            m = SHARD_RE.match(clip.name)
            if m:
                corrupt_by_shard[m.group(1) + ".tar"].append((clip, m.group(2)))
            else:
                print(f"  ! corrupt but unparseable name (can't locate source): {clip.name}")

    print(f"\nScan done: {checked} clips checked, {corrupt} corrupt")

    repaired = 0
    failed = []
    for shard_name, entries in sorted(corrupt_by_shard.items()):
        shard_path = shards_dir / shard_name
        if not shard_path.exists():
            print(f"  ! missing shard {shard_name} — cannot repair {len(entries)} clips")
            failed.extend(e[0] for e in entries)
            continue
        print(f"Re-extracting {len(entries)} clips from {shard_name}...")
        with tarfile.open(shard_path, "r") as tar:
            members = {m.name: m for m in tar.getmembers()}
            for dest, member_name in entries:
                member = members.get(member_name)
                if member is None:
                    print(f"  ! {member_name} not found in {shard_name}")
                    failed.append(dest)
                    continue
                with tar.extractfile(member) as src, open(dest, "wb") as out:
                    out.write(src.read())
                if is_valid_video(dest):
                    repaired += 1
                else:
                    print(f"  ! still corrupt after re-extraction (bad source?): {dest.name}")
                    failed.append(dest)

    print(f"\nRepaired: {repaired}/{corrupt}")
    if failed:
        print(f"Unrepairable: {len(failed)}")
        if args.delete_unrepairable:
            for f in failed:
                f.unlink(missing_ok=True)
            print("Deleted unrepairable files so training skips nothing mid-epoch.")
        else:
            for f in failed[:10]:
                print(f"  - {f}")
    print("\nDone. Dataset is now clean for train_i3d.py --dataset_type augmented.")


if __name__ == "__main__":
    main()
