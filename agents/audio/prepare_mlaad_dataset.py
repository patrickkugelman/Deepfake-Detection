"""
!!! LICENSE WARNING — READ BEFORE USING !!!
MLAAD (mueller91/MLAAD) is published under CC-BY-NC-4.0: "strictly for
non-commercial academic research use... Commercial use is not permitted."
DeepGuard Pro has a paid SaaS billing layer (see saas_layer memory / Stripe
integration) — training a production model on MLAAD and shipping it in a paid
product would violate this license. Use this only for non-commercial
research/evaluation, or replace with a permissively-licensed alternative
(e.g. ASVspoof, which has more permissive research-use terms — check its
license too) before using the resulting weights commercially.

Converts MLAAD's `fake/<language>/<model>/audio_*.wav` files into
`Train_MLAAD_<language>_<model>/` folders that train_audio.py's CustomAudioDataset
already knows how to consume (any folder under dataset/ starting with "Train" is
treated as a fake-audio source, capped at max_fake_per_folder each — see
train_audio.py's CustomAudioDataset).

MLAAD ships fake audio only (no paired real audio) — it's built by re-synthesizing
M-AILABS text with ~50 different TTS models. The existing real corpus already used
here (dataset/Real/wavs, LJSpeech) stays as-is; MLAAD only adds new FAKE examples
from much more modern TTS systems (2023-2024 era) than the original training data.

Source: https://huggingface.co/datasets/mueller91/MLAAD (gated — request access first)

Usage:
    python prepare_mlaad_dataset.py --source_dir D:/deepfake-detector-structure/datasets_augmentation/audio_mlaad
"""
import argparse
import shutil
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source_dir", type=str, required=True,
                   help="Folder containing MLAAD's fake/<language>/<model>/ subfolders as downloaded from HF")
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).parent / "dataset"),
                   help="agents/audio/dataset by default — same root train_audio.py reads from")
    p.add_argument("--max_per_model", type=int, default=1200,
                   help="Cap files copied per language/model combo (matches train_audio.py's max_fake_per_folder default)")
    p.add_argument("--languages", type=str, default="",
                   help="Comma-separated language codes to include (e.g. 'en,de,fr'). Empty = all available.")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("REMINDER: MLAAD is CC-BY-NC-4.0 — non-commercial use only.")
    print("Do not ship weights trained on this data in a paid product.")
    print("=" * 70)

    source_dir = Path(args.source_dir) / "fake"
    if not source_dir.exists():
        print(f"Expected {source_dir} — download MLAAD's fake/ folder first.")
        return

    out_root = Path(args.output_dir)
    wanted_langs = set(args.languages.split(",")) if args.languages else None

    total_copied = 0
    for lang_dir in sorted(source_dir.iterdir()):
        if not lang_dir.is_dir():
            continue
        if wanted_langs and lang_dir.name not in wanted_langs:
            continue

        for model_dir in sorted(lang_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            wavs = list(model_dir.glob("*.wav"))[: args.max_per_model]
            if not wavs:
                continue

            dest = out_root / f"Train_MLAAD_{lang_dir.name}_{model_dir.name}"
            dest.mkdir(parents=True, exist_ok=True)
            for w in wavs:
                shutil.copy2(w, dest / w.name)

            total_copied += len(wavs)
            print(f"{lang_dir.name}/{model_dir.name}: {len(wavs)} files -> {dest.name}")

    print(f"\nTotal fake audio files added: {total_copied}")
    print(f"Written under {out_root} as Train_MLAAD_* folders — train_audio.py will "
          "pick these up automatically (any 'Train*' folder counts as a fake source).")


if __name__ == "__main__":
    main()
