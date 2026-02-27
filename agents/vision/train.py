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
import timm  # Library-ul magic pentru modele SOTA
from sklearn.metrics import classification_report, confusion_matrix

# Rezolvă eroarea "image file is truncated" pentru dataset-uri mari
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# --- 1. CONFIGURAȚIE DE TOP (RTX 4060) ---
# ==========================================
class Config:
    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4               # Pentru citire rapidă de pe SSD
    PIN_MEMORY = True
    
    # Model
    MODEL_NAME = 'xception'       # 'xception' este regele deepfake-urilor pe imagini. 
                                  # Alte optiuni bune: 'efficientnet_b4', 'convnext_tiny'
    PRETRAINED = True
    NUM_CLASSES = 2
    DROPOUT = 0.5
    
    # Antrenament
    NUM_EPOCHS = 30
    BATCH_SIZE = 32               # Ajustează dacă ai OOM (Out of Memory)
    LEARNING_RATE = 1e-4          # Start standard pentru fine-tuning
    WEIGHT_DECAY = 1e-4           # Regularizare L2
    LABEL_SMOOTHING = 0.1         # Truc anti-overfitting
    
    # Paths (Bazat pe structura ta din poză)
    DATA_DIR = "dataset"          # Folderul rădăcină
    SAVE_PATH = "best_vision_model.pth"

print(f"🚀 NASA-Grade Vision System Initialized")
print(f"   Device: {Config.DEVICE} | Model: {Config.MODEL_NAME}")

# ==========================================
# --- 2. DATASET ROBUST (Cu verificare erori) ---
# ==========================================
class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, split='Train', transform=None):
        self.root = Path(root_dir) / split
        self.transform = transform
        self.samples = []
        
        # Mapping: Real=0, Fake=1 (Sau invers, verificăm folderul)
        # Structura ta: dataset/Train/Fake și dataset/Train/Real
        self.classes = ['Real', 'Fake'] 
        self.class_to_idx = {'Real': 0, 'Fake': 1}
        
        if not self.root.exists():
            print(f"❌ CRITICAL ERROR: Folderul {self.root} nu există!"); sys.exit(1)

        print(f"📂 Scanning {split} set...")
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                print(f"   ⚠️ Warning: Clasa {cls_name} nu există în {split}")
                continue
                
            # Caută recursiv imagini
            files = list(cls_dir.glob('**/*.jpg')) + list(cls_dir.glob('**/*.png')) + list(cls_dir.glob('**/*.jpeg'))
            for f in files:
                self.samples.append((str(f), self.class_to_idx[cls_name]))
                
        # Statistici
        real_n = sum(1 for s in self.samples if s[1] == 0)
        fake_n = sum(1 for s in self.samples if s[1] == 1)
        print(f"   ✅ Loaded: {len(self.samples)} (Real: {real_n}, Fake: {fake_n})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            # Încărcare imagine
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"⚠️ Corrupt image skipped: {path}")
            # Returnăm un tensor random și labelul corect ca să nu crape batch-ul
            return torch.zeros((3, 299, 299)), label

# ==========================================
# --- 3. TRANSFORMS (Augmentare Pro) ---
# ==========================================
def get_transforms(is_train=True):
    # Xception cere input 299x299
    IMG_SIZE = 299 
    
    if is_train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Critic pentru deepfakes
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalizare Xception
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

# ==========================================
# --- 4. ENGINE: MODEL & LOSS ---
# ==========================================
def get_model():
    print(f"🔄 Downloading/Loading {Config.MODEL_NAME} from TIMM...")
    model = timm.create_model(Config.MODEL_NAME, pretrained=Config.PRETRAINED, num_classes=Config.NUM_CLASSES, drop_rate=Config.DROPOUT)
    return model

# Funcție Loss Custom pentru a preveni Overfitting
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

# ==========================================
# --- 5. MAIN TRAINING LOOP ---
# ==========================================
if __name__ == "__main__":
    
    # 1. Dataset Setup
    train_ds = DeepfakeDataset(Config.DATA_DIR, split='Train', transform=get_transforms(is_train=True))
    val_ds = DeepfakeDataset(Config.DATA_DIR, split='Validation', transform=get_transforms(is_train=False))
    test_ds = DeepfakeDataset(Config.DATA_DIR, split='Test', transform=get_transforms(is_train=False))
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, 
                              num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, 
                            num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                             num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)

    # 2. Model Init
    model = get_model().to(Config.DEVICE)
    
    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
    
    # 4. Loss & Scaler
    criterion = LabelSmoothingLoss(classes=Config.NUM_CLASSES, smoothing=Config.LABEL_SMOOTHING)
    scaler = GradScaler()

    best_acc = 0.0
    
    print("\n🔥 STARTING IMAGE DEEPFAKE TRAINING 🔥\n")

    for epoch in range(Config.NUM_EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        correct = 0; total = 0
        
        loop = tqdm(train_loader, desc=f"Epoca {epoch+1}/{Config.NUM_EPOCHS}", leave=False)
        
        for imgs, labels in loop:
            imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            # Mixed Precision Forward
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Stats
            train_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100*correct/total)
            
        final_train_acc = 100 * correct / total
        final_train_loss = train_loss / len(train_loader)
        
        # --- VALIDATION ---
        model.eval()
        val_correct = 0; val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                val_total += labels.size(0)
                val_correct += preds.eq(labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        # --- UPDATE & SAVE ---
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step() # Update LR pt epoch urmatoare
        
        print(f"📊 EPOCA {epoch+1}: Train Acc: {final_train_acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {current_lr:.2e}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Config.SAVE_PATH)
            print(f"   ✅ NEW RECORD! Model Saved ({best_acc:.2f}%)")
            
    # ==========================================
    # --- 6. TESTARE FINALĂ (MOMENTUL ADEVĂRULUI) ---
    # ==========================================
    print("\n🏁 TRAINING FINISHED. RUNNING FINAL TEST ON UNSEEN DATA...")
    model.load_state_dict(torch.load(Config.SAVE_PATH)) # Încărcăm cel mai bun model
    model.eval()
    
    test_correct = 0; test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            
            test_total += labels.size(0)
            test_correct += preds.eq(labels).sum().item()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    final_test_acc = 100 * test_correct / test_total
    print(f"\n🏆 FINAL TEST ACCURACY: {final_test_acc:.2f}%")
    
    # Raport Detaliat
    print("\n📊 DETAILED REPORT:")
    print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))