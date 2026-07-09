import os
import sys
import shutil
import multiprocessing as mp
import queue as queue_module
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Windows' console defaults to the cp1252 codepage, which can't encode the emoji used in
# the print() messages below (raises UnicodeEncodeError otherwise). sys.stdout is typed as
# TextIO (no reconfigure) even though real stdout is a TextIOWrapper that has it - getattr
# sidesteps the static type-checker error while keeping the same runtime behavior.
_reconfigure = getattr(sys.stdout, "reconfigure", None)
if _reconfigure is not None:
    _reconfigure(encoding="utf-8", errors="replace")

# Must match train_i3d.py's --num_frames default (Config.NUM_FRAMES) - this checks exactly
# the frames extract_frames() will actually request, not just the first/last.
NUM_FRAMES = 16

# Max seconds to wait for one file's check before treating it as hung. decord can hang
# (not raise) on some malformed videos - a plain try/except in the same process can't
# catch a hang, only a separate process that gets killed from outside can.
PER_FILE_TIMEOUT = 20.0


def _worker_loop(task_queue, result_queue, num_frames):
    """Runs in its own process so it can be killed from outside if it hangs. Persistent
    (checks many files, not spawned fresh per file) to avoid paying process-spawn/import
    overhead for every one of 18k+ files - only actually respawned when a file hangs."""
    import decord
    import numpy as np
    decord.bridge.set_bridge('torch')

    while True:
        path_str = task_queue.get()
        if path_str is None:
            break
        vr = None
        try:
            vr = decord.VideoReader(path_str, ctx=decord.cpu())
            total_frames = len(vr)
            if total_frames == 0:
                result_queue.put((path_str, False))
            else:
                indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
                _ = vr.get_batch(list(indices))
                result_queue.put((path_str, True))
        except Exception:
            result_queue.put((path_str, False))
        finally:
            if vr is not None:
                del vr


def _spawn_worker(num_frames):
    ctx = mp.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    proc = ctx.Process(target=_worker_loop, args=(task_q, result_q, num_frames), daemon=True)
    proc.start()
    return proc, task_q, result_q


def safe_clean_dataset(data_dir="dataset", quarantine_dir="unsupported_videos"):
    dataset_path = Path(data_dir)
    quarantine_path = Path(quarantine_dir)

    quarantine_path.mkdir(parents=True, exist_ok=True)

    print(f"🔎 Căutăm videoclipuri nesuportate/corupte în: {dataset_path.absolute()}")

    all_videos = list(dataset_path.rglob("*.mp4"))
    print(f"Total fișiere găsite: {len(all_videos)}")

    corrupted_count = 0
    hung_count = 0

    worker, task_q, result_q = _spawn_worker(NUM_FRAMES)

    try:
        for video_path in tqdm(all_videos, desc="Verificare și mutare"):
            task_q.put(str(video_path))
            try:
                _, is_ok = result_q.get(timeout=PER_FILE_TIMEOUT)
                is_corrupted = not is_ok
            except queue_module.Empty:
                # Worker hasn't responded in time - it's stuck on this file. Kill it and
                # start a fresh one so the scan can keep going instead of hanging forever.
                hung_count += 1
                tqdm.write(f"  ⏱️ Timeout ({PER_FILE_TIMEOUT:.0f}s) - decord s-a blocat pe: {video_path.name}")
                worker.terminate()
                worker.join(timeout=5)
                worker, task_q, result_q = _spawn_worker(NUM_FRAMES)
                is_corrupted = True

            if is_corrupted:
                try:
                    destination = quarantine_path / video_path.name
                    shutil.move(str(video_path), str(destination))
                    corrupted_count += 1
                except Exception as e:
                    tqdm.write(f"  ⚠️ Nu am putut muta {video_path}: {e}")
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)

    print("\n✅ Verificare terminată în siguranță!")
    print(f"📦 Au fost mutate {corrupted_count} videoclipuri în folderul '{quarantine_dir}' "
          f"({hung_count} dintre ele au blocat decord, restul au dat eroare curată).")


if __name__ == "__main__":
    safe_clean_dataset()
