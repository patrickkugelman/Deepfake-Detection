import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, get_linear_schedule_with_warmup
import librosa
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import warnings
import random
import argparse
import logging
import json

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake Audio Training")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Config:
    MODEL_NAME = "facebook/wav2vec2-base" 
    SAMPLING_RATE = 16000     
    MAX_DURATION = 3.0        
    ACCUMULATION_STEPS = 8    
    WEIGHT_DECAY = 0.01       
    NUM_WORKERS = 0  

class CustomAudioDataset(Dataset):
    def __init__(self, root_dir, feature_extractor, max_fake_per_folder=1200):
        self.root = root_dir
        self.feature_extractor = feature_extractor
        self.samples = []
        
        logging.info(f"Scanning data in {root_dir}...")
        
        # 1. REAL (dataset/Real/wavs or dataset/Real)
        real_path = os.path.join(root_dir, "Real", "wavs")
        if not os.path.exists(real_path):
             real_path = os.path.join(root_dir, "Real")
        
        if os.path.exists(real_path):
            real_files = []
            for r, _, f in os.walk(real_path):
                for file in f:
                     if file.lower().endswith('.wav'): real_files.append(os.path.join(r, file))
            
            for f in real_files:
                self.samples.append((f, 0)) # 0 = REAL
            logging.info(f"   Real Files: {len(real_files)}")
        else:
            logging.error(f"CRITICAL: Missing folder '{real_path}'!")
        
        # 2. FAKE (dataset/Train1 ... Train10)
        total_fake = 0
        train_folders = [d for d in os.listdir(root_dir) if d.startswith("Train")]
        
        for folder_name in train_folders:
            fake_path = os.path.join(root_dir, folder_name)
            if os.path.exists(fake_path):
                files = []
                for r, _, f in os.walk(fake_path):
                    for file in f:
                        if file.lower().endswith('.wav'): files.append(os.path.join(r, file))
                
                if len(files) > max_fake_per_folder:
                    random.shuffle(files)
                    files = files[:max_fake_per_folder]
                
                for f in files: self.samples.append((f, 1)) # 1 = FAKE
                total_fake += len(files)
                logging.info(f"   {folder_name}: {len(files)} files")

        random.shuffle(self.samples)
        
        r_c = sum(1 for s in self.samples if s[1]==0)
        f_c = sum(1 for s in self.samples if s[1]==1)
        logging.info(f"FINAL BALANCE -> Real: {r_c} | Fake: {f_c}")
        
        if len(self.samples) == 0:
            logging.error("ERROR: Dataset is empty!")
            sys.exit(1)

    def get_speaker_id(self, filename: str) -> str:
        """
        Extract speaker/chapter ID from filename.
        Strategies:
        1. LJSpeech format: 'LJ001-0001.wav' -> 'LJ001'
        2. Generic format: 'gen_1.wav' -> Use Parent Folder Name (e.g., 'Train1') as identity
           This prevents splitting files from the same generation batch.
        """
        basename = os.path.basename(filename)
        
        # Strategy 1: LJSpeech Prefix
        if basename.startswith("LJ"):
            # Ex: LJ001-0001.wav -> LJ001
            parts = basename.split('-')
            if len(parts) > 1:
                return parts[0]
        
        # Strategy 2: Fallback to Parent Directory Name
        # This groups all "gen_x" files from "Train1" as a single "speaker/source"
        return os.path.basename(os.path.dirname(filename))

    def split_by_speaker(self, val_ratio=0.15, seed=42):
        """Split dataset ensuring speakers don't overlap between train and val"""
        random.seed(seed)
        
        # Group samples by speaker
        speaker_map = {}
        for path, label in self.samples:
            spk_id = self.get_speaker_id(path)
            if spk_id not in speaker_map:
                speaker_map[spk_id] = []
            speaker_map[spk_id].append((path, label))
            
        speakers = list(speaker_map.keys())
        random.shuffle(speakers)
        
        split_idx = int(len(speakers) * (1 - val_ratio))
        train_speakers = set(speakers[:split_idx])
        val_speakers = set(speakers[split_idx:])
        
        train_samples = []
        val_samples = []
        
        for spk, samples in speaker_map.items():
            if spk in train_speakers:
                train_samples.extend(samples)
            else:
                val_samples.extend(samples)
                
        logging.info(f"Speaker-Aware Split: {len(train_speakers)} speakers in Train, {len(val_speakers)} in Val")
        logging.info(f"   Train samples: {len(train_samples)} | Val samples: {len(val_samples)}")
        
        return train_samples, val_samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        target_len = int(Config.MAX_DURATION * Config.SAMPLING_RATE)
        
        try:
            audio, sr = librosa.load(path, sr=Config.SAMPLING_RATE, duration=Config.MAX_DURATION)
            
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
            else:
                audio = audio[:target_len]
            
            inputs = self.feature_extractor(
                audio, 
                sampling_rate=Config.SAMPLING_RATE, 
                return_tensors="pt", 
                padding="max_length", 
                max_length=target_len,
                truncation=True
            )
            
            return inputs.input_values.squeeze(0), torch.tensor(label)
            
        except Exception as e:
            logging.warning(f"Error loading {path}: {e}")
            return torch.zeros((target_len)), torch.tensor(label)

def train_engine():
    args = parse_args()
    setup_logging(args.output_dir)
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    logging.info("Initializing Feature Extractor...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(Config.MODEL_NAME)
    
    full_dataset = CustomAudioDataset(args.data_dir, feature_extractor=feature_extractor)
    
    # Speaker-Aware Split
    logging.info("Performing rigorous Speaker-Aware Split...")
    train_samples, val_samples = full_dataset.split_by_speaker(val_ratio=0.15, seed=args.seed)
    
    # Create Subsets manually to keep transforms working
    # We cheat slightly by creating new Dataset instances with filtered samples
    # to avoid complex Subset logic with custom methods
    
    train_ds = CustomAudioDataset(args.data_dir, feature_extractor=feature_extractor)
    train_ds.samples = train_samples # Override samples
    
    val_ds = CustomAudioDataset(args.data_dir, feature_extractor=feature_extractor)
    val_ds.samples = val_samples # Override samples
    
    # Windows compatibility: num_workers=0 on Windows to avoid multiprocessing issues
    import platform
    num_workers = 0 if platform.system() == 'Windows' else (Config.NUM_WORKERS if hasattr(Config, 'NUM_WORKERS') else 4)
    pin_memory = True if torch.cuda.is_available() else False
    
    logging.info(f"DataLoader config: num_workers={num_workers}, pin_memory={pin_memory}")
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    logging.info("Loading Model Architecture...")
    sys.stdout.flush()
    
    try:
        logging.info(f"Downloading/loading {Config.MODEL_NAME} from HuggingFace (this may take a moment)...")
        logging.info("Note: First-time download may take several minutes depending on internet speed")
        sys.stdout.flush()
        
        model = AutoModelForAudioClassification.from_pretrained(
            Config.MODEL_NAME, 
            num_labels=2
        )
        logging.info("✅ Model architecture loaded successfully")
        sys.stdout.flush()
        
        model.freeze_feature_extractor()
        logging.info("✅ Feature extractor frozen")
        sys.stdout.flush()
        
        model.to(device)
        logging.info(f"✅ Model moved to device: {device}")
        sys.stdout.flush()
    except KeyboardInterrupt:
        logging.error("Model initialization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Failed to load model: {e}")
        logging.error(f"Error type: {type(e).__name__}")
        import traceback
        logging.error(traceback.format_exc())
        sys.stdout.flush()
        sys.exit(1)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=Config.WEIGHT_DECAY)
    
    total_steps = (len(train_loader) // Config.ACCUMULATION_STEPS) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )
    
    scaler = GradScaler()
    
    best_acc = 0.0
    logging.info("STARTING TRAINING")
    sys.stdout.flush()
    
    try:
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            correct = 0; total = 0
            optimizer.zero_grad()
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            
            for i, (inputs, labels) in enumerate(loop):
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                with autocast():
                    outputs = model(inputs, labels=labels)
                    loss = outputs.loss / Config.ACCUMULATION_STEPS 
                
                scaler.scale(loss).backward()
                
                if (i + 1) % Config.ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                
                train_loss += loss.item() * Config.ACCUMULATION_STEPS
                
                with torch.no_grad():
                    preds = torch.argmax(outputs.logits, dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                
                current_lr = scheduler.get_last_lr()[0]
                loop.set_postfix(acc=f"{100*correct/total:.1f}%", loss=f"{loss.item()*Config.ACCUMULATION_STEPS:.4f}", lr=f"{current_lr:.2e}")
                
            final_train_acc = 100 * correct / total
            final_train_loss = train_loss / len(train_loader)

            # Validation
            model.eval()
            val_correct = 0; val_total = 0
            val_loss = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = model(inputs, labels=labels)
                    
                    loss = outputs.loss
                    val_loss += loss.item()
                    
                    preds = torch.argmax(outputs.logits, dim=-1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = 100 * val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            
            logging.info(f"Epoch {epoch+1}: Train Acc: {final_train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_audio_model.pth"))
                logging.info(f"New Best Model Saved ({best_acc:.2f}%)")
            
            torch.save(model.state_dict(), os.path.join(args.output_dir, "last_audio_model.pth"))

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
    torch.multiprocessing.freeze_support()
    train_engine()