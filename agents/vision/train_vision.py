import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import timm
from sklearn.metrics import classification_report
import argparse
import json
import logging

# Rezolvă eroarea "image file is truncated" pentru dataset-uri mari
ImageFile.LOAD_TRUNCATED_IMAGES = True

def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake Vision Training")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset root (must contain Train/Validation/Test)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, split='Train', transform=None):
        self.root = Path(root_dir) / split
        self.transform = transform
        self.samples = []
        
        self.classes = ['Real', 'Fake'] 
        self.class_to_idx = {'Real': 0, 'Fake': 1}
        
        if not self.root.exists():
            logging.error(f"CRITICAL ERROR: Folder {self.root} does not exist!")
            sys.exit(1)

        logging.info(f"Scanning {split} set in {self.root}...")
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                logging.warning(f"Warning: Class {cls_name} not found in {split}")
                continue
                
            files = list(cls_dir.glob('**/*.jpg')) + list(cls_dir.glob('**/*.png')) + list(cls_dir.glob('**/*.jpeg'))
            for f in files:
                self.samples.append((str(f), self.class_to_idx[cls_name]))
                
        real_n = sum(1 for s in self.samples if s[1] == 0)
        fake_n = sum(1 for s in self.samples if s[1] == 1)
        logging.info(f"Loaded {len(self.samples)} samples (Real: {real_n}, Fake: {fake_n})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            logging.warning(f"Corrupt image skipped: {path}")
            return torch.zeros((3, 299, 299)), label

def get_transforms(is_train=True, img_size=299):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Optional
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def main():
    args = parse_args()
    setup_logging(args.output_dir)
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    logging.info(f"Args: {args}")

    # Initialize Datasets and Loaders
    train_ds = DeepfakeDataset(args.data_dir, split='Train', transform=get_transforms(is_train=True))
    val_ds = DeepfakeDataset(args.data_dir, split='Validation', transform=get_transforms(is_train=False))
    
    # Check if dataloaders are empty
    if len(train_ds) == 0 or len(val_ds) == 0:
        logging.error("Datasets are empty. Exiting.")
        sys.exit(1)

    # Windows compatibility: num_workers=0 on Windows to avoid multiprocessing issues
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 4
    pin_memory = True if torch.cuda.is_available() else False
    
    logging.info(f"DataLoader config: num_workers={num_workers}, pin_memory={pin_memory}")
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=pin_memory)

    # Model
    logging.info(f"Initializing Xception model...")
    sys.stdout.flush()  # Ensure output is flushed immediately
    
    try:
        logging.info("Loading Xception from timm (downloading pretrained weights if needed)...")
        logging.info("Note: First-time download may take several minutes depending on internet speed")
        sys.stdout.flush()
        
        # Create model with error handling
        model = timm.create_model('xception', pretrained=True, num_classes=2, drop_rate=0.5)
        logging.info("✅ Xception model architecture created")
        sys.stdout.flush()
        
        model = model.to(device)
        logging.info(f"✅ Model moved to device: {device}")
        sys.stdout.flush()
        
        # Test forward pass to ensure model works
        logging.info("Testing model forward pass...")
        test_input = torch.randn(1, 3, 299, 299).to(device)
        with torch.no_grad():
            test_output = model(test_input)
        logging.info(f"✅ Model forward pass successful. Output shape: {test_output.shape}")
        sys.stdout.flush()
        
    except KeyboardInterrupt:
        logging.error("Model initialization interrupted by user")
        sys.stdout.flush()
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Failed to initialize Xception model: {e}")
        logging.error(f"Error type: {type(e).__name__}")
        import traceback
        logging.error(traceback.format_exc())
        sys.stdout.flush()
        sys.exit(1)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
    criterion = LabelSmoothingLoss(classes=2, smoothing=0.1)
    scaler = GradScaler()

    best_acc = 0.0
    
    logging.info("Starting training...")
    sys.stdout.flush()
    
    try:
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            correct = 0; total = 0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
            
            for imgs, labels in loop:
                imgs, labels = imgs.to(device), labels.to(device)
                
                with autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()
                
                loop.set_postfix(loss=loss.item(), acc=100*correct/total)
                
            final_train_acc = 100 * correct / total
            final_train_loss = train_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_correct = 0; val_total = 0
            val_loss_acc = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    # Compute val loss for logging
                    # Note: using same criterion (LabelSmoothing) or simple CrossEntropy
                    loss = criterion(outputs, labels) 
                    val_loss_acc += loss.item()

                    _, preds = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += preds.eq(labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            avg_val_loss = val_loss_acc / len(val_loader) if len(val_loader) > 0 else 0
            
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            logging.info(f"Epoch {epoch+1}: Train Loss: {final_train_loss:.4f} Acc: {final_train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.2f}% | LR: {current_lr:.2e}")
            
            # Save Best
            if val_acc > best_acc:
                best_acc = val_acc
                best_path = os.path.join(args.output_dir, "best_vision_model.pth")
                torch.save(model.state_dict(), best_path)
                logging.info(f"New Best Model Saved: {best_acc:.2f}%")
            
            # Save Last
            last_path = os.path.join(args.output_dir, "last_vision_model.pth")
            torch.save(model.state_dict(), last_path)
        
        # Final Metrics
        metrics = {
            "best_val_acc": best_acc,
            "final_train_acc": final_train_acc,
            "final_train_loss": final_train_loss
        }
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
    main()
