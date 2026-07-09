"""
Converts the HC3 (Human-ChatGPT Comparison Corpus) dataset into the
True.csv / Fake.csv format expected by train_text.py.

Replaces the previous dataset (Kaggle/ISOT fake-news, backed up under
dataset/_old_fake_news_dataset_backup/) which trained the agent to detect
misinformation-style news writing instead of human-vs-AI text authorship —
a completely different task from what this agent is meant to do.

HC3 source: https://huggingface.co/datasets/Hello-SimpleAI/HC3
"""
import json
import csv
import random
from pathlib import Path
from huggingface_hub import hf_hub_download

SEED = 42
MAX_HUMAN_ANSWERS_PER_QUESTION = 1  # cap so REAL doesn't outnumber FAKE (usually 1 chatgpt answer/question)

def main():
    random.seed(SEED)
    dataset_dir = Path(__file__).parent / "dataset"
    dataset_dir.mkdir(exist_ok=True)

    print("Downloading HC3 (all.jsonl) from HuggingFace...")
    path = hf_hub_download(repo_id="Hello-SimpleAI/HC3", filename="all.jsonl", repo_type="dataset")

    human_texts = []
    ai_texts = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            humans = row.get("human_answers") or []
            ais = row.get("chatgpt_answers") or []

            # Cap human answers per question to avoid a class imbalance
            # (many Reddit ELI5 questions have several human answers but only one ChatGPT answer).
            for h in humans[:MAX_HUMAN_ANSWERS_PER_QUESTION]:
                h = (h or "").strip()
                if len(h) >= 20:
                    human_texts.append(h)

            for a in ais:
                a = (a or "").strip()
                if len(a) >= 20:
                    ai_texts.append(a)

    print(f"Collected {len(human_texts)} human samples, {len(ai_texts)} ChatGPT samples")

    # Balance classes to the smaller of the two, shuffled.
    random.shuffle(human_texts)
    random.shuffle(ai_texts)
    n = min(len(human_texts), len(ai_texts))
    human_texts = human_texts[:n]
    ai_texts = ai_texts[:n]
    print(f"Balanced to {n} samples per class ({2*n} total)")

    true_csv = dataset_dir / "True.csv"
    fake_csv = dataset_dir / "Fake.csv"

    with open(true_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text"])
        for t in human_texts:
            writer.writerow([t])

    with open(fake_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text"])
        for t in ai_texts:
            writer.writerow([t])

    print(f"Wrote {true_csv} ({n} rows) and {fake_csv} ({n} rows)")

if __name__ == "__main__":
    main()
