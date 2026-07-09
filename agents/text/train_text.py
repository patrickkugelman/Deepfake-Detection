import os
import argparse
import sys
import logging
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake Text Training")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save models and logs")
    return parser.parse_args()

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "train.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_local_data(data_dir):
    texts = []
    labels = []
    
    logging.info(f"Scanning data in {data_dir}...")
    
    # 1. Try CSV loading (Priority)
    try:
        import pandas as pd
        true_csv = os.path.join(data_dir, "True.csv")
        fake_csv = os.path.join(data_dir, "Fake.csv")
        
        if os.path.exists(true_csv) and os.path.exists(fake_csv):
            logging.info("Found True.csv and Fake.csv. Loading...")
            
            # Read True.csv (Real) -> Label 0
            df_true = pd.read_csv(true_csv)
            if 'text' in df_true.columns:
                t_texts = df_true['text'].dropna().astype(str).tolist()
                texts.extend(t_texts)
                labels.extend([0] * len(t_texts))
                logging.info(f"Loaded {len(t_texts)} real samples from CSV")
            
            # Read Fake.csv (Fake) -> Label 1
            df_fake = pd.read_csv(fake_csv)
            if 'text' in df_fake.columns:
                f_texts = df_fake['text'].dropna().astype(str).tolist()
                texts.extend(f_texts)
                labels.extend([1] * len(f_texts))
                logging.info(f"Loaded {len(f_texts)} fake samples from CSV")
                
    except ImportError:
        logging.warning("Pandas not installed. Skipping CSV loading.")
    except Exception as e:
        logging.error(f"Error loading CSVs: {e}")

    # 2. Try Folder loading (Recursive)
    # Looking for folders like "Real", "Human", "True" vs "Fake", "AI"
    
    possible_real = ["Real", "Human", "True"]
    possible_fake = ["Fake", "AI", "False"]
    
    found_folders = False
    
    # helper for recursive loading
    def scan_folder(folder_path, label):
        count = 0
        if not os.path.exists(folder_path): return 0
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.txt'):
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read().strip()
                            if len(content) > 10: # Min length check
                                texts.append(content)
                                labels.append(label)
                                count += 1
                    except Exception: pass
        return count

    # Scan root subdirectories
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for real_name in possible_real:
        if real_name in subdirs:
            # Case exact match or inside dataset/train/Real
            c = scan_folder(os.path.join(data_dir, real_name), 0)
            if c > 0: logging.info(f"Loaded {c} real samples from folder {real_name}"); found_folders = True
            
    for fake_name in possible_fake:
         if fake_name in subdirs:
            c = scan_folder(os.path.join(data_dir, fake_name), 1)
            if c > 0: logging.info(f"Loaded {c} fake samples from folder {fake_name}"); found_folders = True

    # 3. Check for specific 'train' folder structure if nothing else found
    if not found_folders and len(texts) == 0:
        train_dir = os.path.join(data_dir, "train")
        if os.path.exists(train_dir):
            logging.info("Checking 'train' subdirectory...")
            # Recursively check inside train for Real/Fake subfolders
            for real_name in possible_real:
                c = scan_folder(os.path.join(train_dir, real_name), 0)
                if c > 0: logging.info(f"Loaded {c} real samples from train/{real_name}")
            
            for fake_name in possible_fake:
                c = scan_folder(os.path.join(train_dir, fake_name), 1)
                if c > 0: logging.info(f"Loaded {c} fake samples from train/{fake_name}")

    return texts, labels

def train_text_agent():
    args = parse_args()
    setup_logging(args.output_dir)
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    
    # Load Data
    texts, labels = load_local_data(args.data_dir)
    
    if len(texts) == 0:
        logging.error("No data found. Exiting.")
        sys.exit(1)
        
    logging.info(f"Loaded {len(texts)} samples.")
    
    # Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=args.seed)
    
    # Tokenizer
    logging.info("Loading Tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    train_ds = TextDataset(train_texts, train_labels, tokenizer)
    val_ds = TextDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)
    
    # Model
    logging.info("Loading Model...")
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    model = model.to(device) # type: ignore
    
    optimizer = AdamW(model.parameters(), lr=args.lr)  # torch.optim.AdamW has no correct_bias kwarg (that was transformers-specific)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    best_acc = 0.0
    logging.info("STARTING TEXT TRAINING")
    sys.stdout.flush()
    
    try:
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            correct = 0; total = 0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            
            for batch in loop:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                model.zero_grad()
                
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                _, preds = torch.max(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                loop.set_postfix(acc=f"{100*correct/total:.1f}%", loss=f"{loss.item():.4f}")
                
            # Validation
            model.eval()
            val_correct = 0; val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids, attention_mask=attention_mask)
                    _, preds = torch.max(outputs.logits, dim=1)
                    
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = 100 * val_correct / val_total
            logging.info(f"Epoch {epoch+1}: Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_text_model.pth"))
                
                tokenizer_path = os.path.join(args.output_dir, "tokenizer_model")
                if not os.path.exists(tokenizer_path):
                    os.makedirs(tokenizer_path)
                tokenizer.save_pretrained(tokenizer_path)
                logging.info(f"New Best Model Saved ({best_acc:.2f}%)")
            
            torch.save(model.state_dict(), os.path.join(args.output_dir, "last_text_model.pth"))

        metrics = {"best_val_acc": best_acc}
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        
        logging.info("Training complete.")
        sys.stdout.flush()
    except KeyboardInterrupt:
        logging.warning("Training interrupted by user")
        sys.stdout.flush()
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.stdout.flush()
        sys.exit(1)

if __name__ == "__main__":
    train_text_agent()
