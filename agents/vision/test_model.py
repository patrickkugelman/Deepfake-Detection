import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import timm
from tqdm import tqdm

# --- CONFIGURĂRI (Trebuie să fie la fel ca la antrenare) ---
IMG_SIZE = (299, 299)
BATCH_SIZE = 16
MODEL_PATH = "model_final.pth" # Fișierul generat de scriptul anterior

# Clasa Dataset (Trebuie definită și aici ca să putem încărca datele)
class DeepfakeImageDataset(Dataset):
    def __init__(self, root_dir, phase, transform=None):
        self.base_path = Path(root_dir) / phase
        self.transform = transform
        self.samples = []
        for label, class_name in enumerate(["Real", "Fake"]):
            class_dir = self.base_path / class_name
            if class_dir.exists():
                for ext in ["*.jpg", "*.png", "*.jpeg", "*.webp"]:
                    for img in class_dir.glob(ext):
                        self.samples.append((str(img), label))
        print(f"[{phase}] Încărcate: {len(self.samples)} imagini.")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform: img = self.transform(img)
            return img, label
        except: return torch.zeros(3, *IMG_SIZE), label

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍  Rulez testul pe: {device}")

    # Verificăm dacă există modelul antrenat
    if not Path(MODEL_PATH).exists():
        print("❌ Eroare: Nu găsesc fișierul 'model_final.pth'. Rulează întâi train.py!")
        return

    # Transformări simple (fără augmentare, doar resize)
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Încărcăm datele de VALIDATION
    print("\n--- Încărcare Date de Validare ---")
    val_dataset = DeepfakeImageDataset("dataset", "Validation", transform=transform)
    if len(val_dataset) == 0:
        print("❌ Folderul dataset/Validation este gol!")
        return
        
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Încărcăm Modelul și Greutățile salvate
    print("🤖  Încarc modelul antrenat...")
    model = timm.create_model('xception', num_classes=2) # Nu mai punem pretrained=True, că încărcăm noi greutățile
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval() # Punem modelul în mod de examinare (fără învățare)

    # Variabile pentru statistici detaliate
    correct = 0
    total = 0
    
    # Statistici pe clase
    correct_real = 0
    total_real = 0
    correct_fake = 0
    total_fake = 0

    print("\n📝 Start Evaluare...")
    with torch.no_grad(): # Nu calculăm gradienți (economisește memorie și timp)
        for images, labels in tqdm(val_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Calculăm statistici detaliate (Real vs Fake)
            # Label 0 = Real, Label 1 = Fake
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                
                if label == 0: # E poză Reală
                    total_real += 1
                    if pred == 0: correct_real += 1
                else: # E poză Fake
                    total_fake += 1
                    if pred == 1: correct_fake += 1

    # Raport Final
    acc = 100 * correct / total
    acc_real = 100 * correct_real / total_real if total_real > 0 else 0
    acc_fake = 100 * correct_fake / total_fake if total_fake > 0 else 0

    print("\n" + "="*30)
    print("      RAPORT FINAL VALIDARE")
    print("="*30)
    print(f"✅ Acuratețe Totală: {acc:.2f}%")
    print("-" * 30)
    print(f"📷 REAL: A detectat corect {correct_real} din {total_real} ({acc_real:.2f}%)")
    print(f"🤖 FAKE: A detectat corect {correct_fake} din {total_fake} ({acc_fake:.2f}%)")
    print("="*30)

if __name__ == "__main__":
    evaluate()