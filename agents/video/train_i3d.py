"""
DeepGuard Pro - I3D Video Deepfake Training (Optimized with Decord & V2)
Features:
- Dedicated I3D (Inflated 3D ConvNet) training
- Mixup & CutMix augmentation
- Focal Loss (Default) & Label Smoothing
- Automatic Mixed Precision (AMP)
- WeightedRandomSampler (Class Imbalance Fix)
- Progress Bars (tqdm)
- Decord Video Loading (Fast & Deadlock-Free)
- Vectorized v2 Transforms
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import v2  # Folosim v2 pentru transformari pe video
import numpy as np
import random
import time
from tqdm import tqdm # type: ignore
from torch.cuda.amp import autocast, GradScaler
import argparse
import sys
import logging
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score

# Inlocuim OpenCV cu Decord pentru citire nativa pe tensor mult mai rapida
try:
    import decord
    decord.bridge.set_bridge('torch')
except ImportError:
    print("⚠️ decord not installed. Run: pip install decord")
    sys.exit(1)

# Optional: Weights & Biases
try:
    import wandb # type: ignore
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ wandb not installed. Run: pip install wandb")

try:
    from models.i3d import I3D
except ImportError as e:
    print(f"❌ Error importing model: {e}")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="DeepGuard I3D Training")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="dataset", help="Dataset root")
    parser.add_argument("--dataset_type", type=str, default="augmented",
                        choices=["faceforensics", "celebdf", "dfdc", "custom", "hybrid", "augmented"],
                        help="Dataset structure type. 'augmented' (default) combines Celeb-real/"
                             "YouTube-real/Celeb-synthesis (the original Celeb-DF data) together "
                             "with GenVideo-Real/GenVideo-Fake in one unified training run, so "
                             "training from scratch here doesn't lose the Celeb-DF signal - it's "
                             "included, just relearned together with GenVideo rather than resumed "
                             "from an old checkpoint's weights.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Num workers for DataLoader (0 for debugging). GPU sat at 0%% "
                             "utilization with the old default of 4 - Decord video decoding is "
                             "heavier per-sample than image loading, so it needs more parallel "
                             "workers to keep the GPU fed. Raise further if nvidia-smi still shows "
                             "0%% right after starting.")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (lower for I3D)")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Frames per clip (16 is the standard for video deepfake detection)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Advanced
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="Mixup alpha")
    parser.add_argument("--no_focal_loss", action="store_true", help="Disable Focal Loss (Default: Enabled)")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="models_i3d", help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--model", type=str, default="r3d_18", help="Model name (default: r3d_18)")
    
    return parser.parse_args()

def setup_logging(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Path(output_dir) / "train.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Lasam benchmark pe True ulterior pentru modele fixate ca dimensiune
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Config:
    NUM_FRAMES = 16
    IMG_SIZE = (224, 224)
    ACCUMULATION_STEPS = 4
    LABEL_SMOOTHING = 0.1

# Max seconds to wait for a single batch from a DataLoader worker before treating it as
# hung (see the timeout= usage below) - decord can hang (not raise) on some malformed
# videos, and normal batches take ~1-2s, so anything past this is a stuck worker, not slowness.
DATALOADER_TIMEOUT = 180

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    def forward(self, x, target):
        logprobs = nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        return (self.confidence * nll_loss + self.smoothing * smooth_loss).mean()

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_video_transforms(img_size, augment=False):
    """Transformari v2 care opereaza direct pe batch-ul de cadre (T, C, H, W)"""
    transform_list = [
        v2.Resize(img_size, antialias=True),
        v2.ToDtype(torch.float32, scale=True), # Converteste [0, 255] in [0.0, 1.0]
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    
    if augment:
        aug_list = [
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2)
        ]
        transform_list = aug_list + transform_list
        
    return v2.Compose(transform_list)

def extract_frames(video_path, num_frames=16, img_size=(224, 224), augment=False):
    """Extract frames using Decord, mult mai rapid si sigur decat cv2"""
    try:
        # num_threads=1: decord defaults to num_threads=0 (auto/all-cores) *per VideoReader
        # instance*. Under a multi-worker DataLoader, every worker process opens its own
        # VideoReader, so with the default each of N worker processes independently tries to
        # grab every CPU core for internal decode threads - N-way oversubscription that shows
        # up as 100% CPU with WORSE throughput than fewer workers (observed: 8 workers ran
        # slower than 4). Pinning each reader to 1 internal thread lets the DataLoader's
        # worker-process-level parallelism be the actual source of parallelism instead.
        vr = decord.VideoReader(str(video_path), ctx=decord.cpu(), num_threads=1)
        total_frames = len(vr)
        
        if total_frames == 0:
            print(f"\n❌ Video gol: {video_path}")
            return None
            
        if augment and total_frames >= num_frames:
            start_idx = random.randint(0, max(0, total_frames - num_frames - 1))
            indices = np.linspace(start_idx, start_idx + num_frames - 1, num_frames, dtype=int)
        else:
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            
        # Get batch direct. Forma: (T, H, W, C)
        frames = vr.get_batch(indices)
        
        # Transformare pentru torchvision v2: (T, C, H, W)
        frames = frames.permute(0, 3, 1, 2)
        
        # Aplicam transformarile (vectorizat pe intreg videoclipul)
        transform = get_video_transforms(img_size, augment)
        frames = transform(frames)
        
        # Modelele 3D asteapta input in format (C, T, H, W)
        frames = frames.permute(1, 0, 2, 3)
        
        return frames

    except Exception as e:
        # Prinde orice eroare (codec lipsa, video corupt etc.)
        print(f"\n❌ Eroare citire video {video_path} - Fallback activat. Eroare: {e}")
        return None

class DeepFakeDataset(Dataset):
    def __init__(self, root_dir, augment=False, dataset_type="hybrid"):
        self.samples = []
        self.augment = augment
        root_path = Path(root_dir)
        
        if dataset_type == "hybrid":
            for real_dir in ["Celeb-real", "YouTube-real"]:
                path = root_path / real_dir
                if path.exists():
                     for v in path.glob("*.mp4"): self.samples.append((v, 0))
            
            fake_path = root_path / "Celeb-synthesis"
            if fake_path.exists():
                 for v in fake_path.glob("*.mp4"): self.samples.append((v, 1))
        
        elif dataset_type == "custom":
             for lbl, cls in enumerate(["Real", "Fake"]):
                 p = root_path / cls
                 if p.exists():
                     for v in p.glob("*.mp4"): self.samples.append((v, lbl))

        elif dataset_type == "augmented":
            for real_dir in ["Celeb-real", "YouTube-real", "GenVideo-Real"]:
                path = root_path / real_dir
                if path.exists():
                    for v in path.glob("*.mp4"): self.samples.append((v, 0))

            for fake_dir in ["Celeb-synthesis", "GenVideo-Fake"]:
                path = root_path / fake_dir
                if path.exists():
                    for v in path.glob("*.mp4"): self.samples.append((v, 1))

        logging.info(f"Loaded {len(self.samples)} samples. Type: {dataset_type}")

    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = extract_frames(path, Config.NUM_FRAMES, Config.IMG_SIZE, self.augment)
        
        # FALLBACK LOOP: Daca videoclipul este corupt, tragem la sorti alt index din setul de date
        # Astfel prevenim invatarea cu tensori goi si blocarea antrenamentului.
        attempts = 0
        while frames is None and attempts < 10:
            idx = random.randint(0, len(self.samples) - 1)
            path, label = self.samples[idx]
            frames = extract_frames(path, Config.NUM_FRAMES, Config.IMG_SIZE, self.augment)
            attempts += 1
            
        # Daca dupa 10 incercari tot pica (putin probabil, doar daca ai datele extrem de corupte), punem 0
        if frames is None:
            frames = torch.zeros((3, Config.NUM_FRAMES, *Config.IMG_SIZE), dtype=torch.float32)
            
        return frames, label
    
    def get_labels(self):
        return [s[1] for s in self.samples]

def get_model(model_name="r3d_18", num_classes=2):
    logging.info(f"Building Model: {model_name}...")
    
    if model_name == "custom_i3d" or model_name == "i3d":
        model = I3D(num_classes=num_classes)
        
    elif model_name == "r3d_18":
        from torchvision.models.video import r3d_18, R3D_18_Weights
        weights = R3D_18_Weights.KINETICS400_V1
        model = r3d_18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        logging.info("Loaded ResNet3D-18 pretrained weights")
        
    elif model_name == "mc3_18":
        from torchvision.models.video import mc3_18, MC3_18_Weights
        weights = MC3_18_Weights.KINETICS400_V1
        model = mc3_18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        logging.info("Loaded MC3-18 pretrained weights")
        
    elif model_name == "r2plus1d_18":
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        weights = R2Plus1D_18_Weights.KINETICS400_V1
        model = r2plus1d_18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        logging.info("Loaded R2+1D-18 pretrained weights")
        
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model

def train_epoch(model, loader, criterion, optimizer, scaler, device, args):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Training")
    
    for i, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        if args.mixup_alpha > 0:
            inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, args.mixup_alpha)
            with autocast(enabled=args.use_amp):
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            with autocast(enabled=args.use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
        
        loss = loss / Config.ACCUMULATION_STEPS
        scaler.scale(loss).backward()
        
        if (i + 1) % Config.ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * Config.ACCUMULATION_STEPS
        _, preds = outputs.max(1)
        total += labels.size(0)
        
        if args.mixup_alpha > 0:
            correct += (lam * preds.eq(labels_a).sum().item() + (1 - lam) * preds.eq(labels_b).sum().item())
        else:
            correct += preds.eq(labels).sum().item()
            
        pbar.set_postfix({'loss': f"{loss.item()*Config.ACCUMULATION_STEPS:.4f}"})
            
    return total_loss / len(loader), 100.0 * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_labels, all_probs = [], []
    pbar = tqdm(loader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
    acc = 100.0 * correct / total
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return total_loss / len(loader), acc, auc

def main():
    args = parse_args()
    setup_logging(args.output_dir)
    set_seed(args.seed)

    Config.NUM_FRAMES = args.num_frames
    
    # Optimizare CuDNN pt modele cu input fix 
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    logging.info(f"Frames per clip: {Config.NUM_FRAMES} | cuDNN benchmark: ON")

    args.use_amp = torch.cuda.is_available()

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project="deepguard-i3d", config=vars(args))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    full_ds = DeepFakeDataset(args.data_dir, augment=False, dataset_type=args.dataset_type)
    if len(full_ds) == 0:
        logging.error("Dataset empty!")
        sys.exit(1)

    train_len = int(0.8 * len(full_ds))
    val_len = len(full_ds) - train_len
    generator = torch.Generator().manual_seed(args.seed)
    train_ds_subset, val_ds_subset = torch.utils.data.random_split(full_ds, [train_len, val_len], generator=generator)
    
    import copy
    train_ds = copy.deepcopy(full_ds)
    val_ds = copy.deepcopy(full_ds)
    
    train_ds.augment = True
    val_ds.augment = False
    
    train_ds = torch.utils.data.Subset(train_ds, train_ds_subset.indices)
    val_ds = torch.utils.data.Subset(val_ds, val_ds_subset.indices)
    
    logging.info("Calculating class weights for balanced sampling...")
    train_indices = train_ds.indices
    all_labels = np.array(full_ds.get_labels())
    train_labels = all_labels[train_indices]
    
    classes, counts = np.unique(train_labels, return_counts=True)
    logging.info(f"Distribution: {dict(zip(classes, counts))}")
    
    weight_per_class = 1.0 / counts
    sample_weights = np.array([weight_per_class[label] for label in train_labels])
    sample_weights = torch.from_numpy(sample_weights).double()
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights)) # type: ignore

    # Acum luam num_workers din argparse (--workers).
    # Seteaza la 0 in terminal (--workers 0) daca vrei sa faci debugging curat.
    # timeout=DATALOADER_TIMEOUT: daca un fisier scapa nedetectat de clean.dataset.py si decord
    # se blocheaza la citire (nu arunca exceptie, pur si simplu nu mai raspunde), fara timeout
    # DataLoader-ul astepta la infinit - exact ce s-a intamplat la Validation. Cu workers
    # persistenti (persistent_workers=True) acelasi worker ramane blocat si la epocile
    # urmatoare, deci fara asta antrenarea s-ar bloca din nou in acelasi loc de fiecare data.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers > 0),
                              timeout=DATALOADER_TIMEOUT if args.workers > 0 else 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers > 0),
                            timeout=DATALOADER_TIMEOUT if args.workers > 0 else 0)
    
    model = get_model(model_name=args.model, num_classes=2).to(device)
    
    use_focal = not args.no_focal_loss
    logging.info(f"Loss Function: {'Focal Loss' if use_focal else 'Label Smoothing'}")
    criterion = FocalLoss() if use_focal else LabelSmoothingCrossEntropy(args.label_smoothing)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler(enabled=args.use_amp)
    
    best_auc = 0.0
    start_epoch = 0
    
    if args.checkpoint and Path(args.checkpoint).exists():
        logging.info(f"Loading checkpoint {args.checkpoint}")
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_auc = ckpt.get('best_auc', 0.0)

    logging.info("Starting I3D Training...")
    
    for epoch in range(start_epoch, args.epochs):
        try:
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, args)
            val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)
        except RuntimeError as e:
            msg = str(e)
            if "DataLoader worker" in msg or "timed out" in msg.lower():
                # A worker didn't return a batch within DATALOADER_TIMEOUT - almost certainly a
                # video decord hangs on outright (rather than a clean, catchable corruption error,
                # which clean.dataset.py would already have quarantined). Save what was already
                # trained this epoch before exiting so a re-run doesn't lose ~an epoch's work.
                emergency_path = Path(args.output_dir) / f"emergency_epoch{epoch+1}_{args.model}.pth"
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_auc': best_auc
                }, emergency_path)
                logging.error(
                    f"A DataLoader worker didn't return a batch within {DATALOADER_TIMEOUT}s during "
                    f"epoch {epoch+1} - a specific video is hanging decord, not raising a catchable "
                    f"error (clean.dataset.py would already have quarantined a clean-erroring file). "
                    f"Saved progress to {emergency_path}. Re-run clean.dataset.py (it now also checks "
                    f"the last frame of every clip, not just the first) to find and quarantine it, "
                    f"then resume with --checkpoint {emergency_path}"
                )
                sys.exit(1)
            raise

        scheduler.step()
        
        logging.info(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% AUC: {val_auc:.4f}")
        
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({"train_loss": train_loss, "val_auc": val_auc})
            
        if val_auc > best_auc:
            best_auc = val_auc
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_auc': best_auc
            }
            torch.save(checkpoint, Path(args.output_dir) / f"best_{args.model}.pth")
            if args.model == "r3d_18":
                # agents/video/api.py hardcodes MODEL_PATH="models_i3d/best_i3d.pth" and always
                # builds an r3d_18 architecture to load it into - without this, a freshly
                # retrained r3d_18 checkpoint would sit in best_r3d_18.pth and never actually
                # get served, silently leaving the old checkpoint in production.
                torch.save(checkpoint, Path(args.output_dir) / "best_i3d.pth")
            logging.info(f"Saved Best Model (AUC: {best_auc:.4f} | Acc: {val_acc:.2f}%)")

if __name__ == "__main__":
    main()