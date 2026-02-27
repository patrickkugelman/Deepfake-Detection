"""
DeepGuard Pro - I3D Video Deepfake Training
Features:
- Dedicated I3D (Inflated 3D ConvNet) training
- Mixup & CutMix augmentation
- Focal Loss (Default) & Label Smoothing
- Automatic Mixed Precision (AMP)
- WeightedRandomSampler (Class Imbalance Fix)
- Progress Bars (tqdm)
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
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
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
    parser.add_argument("--dataset_type", type=str, default="hybrid", 
                       choices=["faceforensics", "celebdf", "dfdc", "custom", "hybrid"],
                       help="Dataset structure type")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (lower for I3D)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Advanced
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="Mixup alpha")
    # Focal loss default True
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Config:
    NUM_FRAMES = 32
    IMG_SIZE = (224, 224)
    ACCUMULATION_STEPS = 4
    LABEL_SMOOTHING = 0.1

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

def extract_frames(video_path, num_frames=64, img_size=(224, 224), augment=False):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None
    
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0: cap.release(); return None
    
    if augment and total >= num_frames:
        start_idx = random.randint(0, max(0, total - num_frames - 1))
        indices = list(range(start_idx, min(start_idx + num_frames, total)))
        if len(indices) < num_frames:
             indices = np.linspace(0, total - 1, num_frames, dtype=int)
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    
    if not frames: return None
    while len(frames) < num_frames: frames.append(frames[-1])
    
    if augment:
        if random.random() > 0.5: frames = [cv2.flip(f, 1) for f in frames]
        if random.random() > 0.7:
             alpha = random.uniform(0.8, 1.2)
             beta = random.randint(-10, 10)
             frames = [cv2.convertScaleAbs(f, alpha=alpha, beta=beta) for f in frames]

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])
    
    try:
        return torch.stack([transform(f) for f in frames], dim=1)
    except: return None

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

        logging.info(f"Loaded {len(self.samples)} samples. Type: {dataset_type}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = extract_frames(path, Config.NUM_FRAMES, Config.IMG_SIZE, self.augment)
        if frames is None: frames = torch.zeros((3, Config.NUM_FRAMES, *Config.IMG_SIZE), dtype=torch.float32)
        return frames, label
    
    def get_labels(self):
        return [s[1] for s in self.samples]

def make_weights_for_balanced_classes(dataset):
    labels = np.array(dataset.get_labels())
    classes, counts = np.unique(labels, return_counts=True)
    weight_per_class = 1.0 / counts
    weights = np.array([weight_per_class[label] for label in labels])
    return torch.from_numpy(weights).double()

def get_model(model_name="r3d_18", num_classes=2):
    logging.info(f"Building Model: {model_name}...")
    
    if model_name == "custom_i3d" or model_name == "i3d":
        logging.warning("Using I3D model (custom or default) without pretrained weights! Training may be slow or fail on small datasets.")
        model = I3D(num_classes=num_classes)
        
    elif model_name == "r3d_18":
        from torchvision.models.video import r3d_18, R3D_18_Weights
        weights = R3D_18_Weights.KINETICS400_V1
        model = r3d_18(weights=weights)
        # Replace head
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        logging.info("Loaded ResNet3D-18 with Kinetics-400 pretrained weights")
        
    elif model_name == "mc3_18":
        from torchvision.models.video import mc3_18, MC3_18_Weights
        weights = MC3_18_Weights.KINETICS400_V1
        model = mc3_18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        logging.info("Loaded MC3-18 with Kinetics-400 pretrained weights")
        
    elif model_name == "r2plus1d_18":
        from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
        weights = R2Plus1D_18_Weights.KINETICS400_V1
        model = r2plus1d_18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        logging.info("Loaded R2+1D-18 with Kinetics-400 pretrained weights")
        
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
    
    args.use_amp = torch.cuda.is_available()

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project="deepguard-i3d", config=vars(args))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    full_ds = DeepFakeDataset(args.data_dir, augment=False, dataset_type=args.dataset_type)
    if len(full_ds) == 0:
        logging.error("Dataset empty!")
        sys.exit(1)

    # Split indices first
    train_len = int(0.8 * len(full_ds))
    val_len = len(full_ds) - train_len
    generator = torch.Generator().manual_seed(args.seed)
    train_ds_subset, val_ds_subset = torch.utils.data.random_split(full_ds, [train_len, val_len], generator=generator)
    
    # Create separate datasets to allow different augmentation
    import copy
    train_ds = copy.deepcopy(full_ds)
    val_ds = copy.deepcopy(full_ds)
    
    train_ds.augment = True
    val_ds.augment = False
    
    # Re-wrap in subset logic using the INDICES from random_split
    # We need to access samples by index from the split
    # Better way: Subsets hold indices. We just point them to the new datasets.
    train_ds = torch.utils.data.Subset(train_ds, train_ds_subset.indices)
    val_ds = torch.utils.data.Subset(val_ds, val_ds_subset.indices)
    
    # --- WEIGHTED SAMPLER LOGIC ---
    logging.info("Calculating class weights for balanced sampling...")
    # Extract training indices
    train_indices = train_ds.indices
    # Get all labels from the main dataset
    all_labels = np.array(full_ds.get_labels())
    # Filter labels for only training data
    train_labels = all_labels[train_indices]
    
    # Calculate weights
    classes, counts = np.unique(train_labels, return_counts=True)
    logging.info(f"Distribution: {dict(zip(classes, counts))}")
    
    weight_per_class = 1.0 / counts
    sample_weights = np.array([weight_per_class[label] for label in train_labels])
    sample_weights = torch.from_numpy(sample_weights).double()
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights)) # type: ignore
    logging.info("WeightedRandomSampler detached")
    
    # Windows supports workers if protected by if __name__ == "__main__"
    num_workers = 4 
    
    # Sampler replaces shuffle=True
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    
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
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device, args)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        logging.info(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% AUC: {val_auc:.4f}")
        
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({"train_loss": train_loss, "val_auc": val_auc})
            
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_auc': best_auc
            }, Path(args.output_dir) / "best_i3d.pth")
            logging.info(f"Saved Best I3D (AUC: {best_auc:.4f} | Acc: {val_acc:.2f}%)")

if __name__ == "__main__":
    main()