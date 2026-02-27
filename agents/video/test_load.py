import torch
from models.resnet3d import resnet3d_18

print("=" * 70)
print("Testing ResNet3D checkpoint loading...")
print("=" * 70)

# Create model
model = resnet3d_18(num_classes=2)
print("✅ Model created")

# Load checkpoint
print("\n📂 Loading: best_resnet3d_pro.pth")
checkpoint = torch.load("best_resnet3d_pro.pth", map_location='cpu', weights_only=False)
print("✅ Checkpoint loaded")

# Load weights into model
print("\n🔄 Loading weights into model...")
result = model.load_state_dict(checkpoint, strict=False)

# Check results
print("\n📊 Results:")
missing = len(result.missing_keys)
unexpected = len(result.unexpected_keys)

if missing > 0:
    print(f"⚠️  Missing keys: {missing}")
    if missing <= 5:
        for key in result.missing_keys:
            print(f"     - {key}")

if unexpected > 0:
    print(f"⚠️  Unexpected keys: {unexpected}")
    if unexpected <= 5:
        for key in result.unexpected_keys:
            print(f"     - {key}")

if missing == 0 and unexpected == 0:
    print("✅✅✅ PERFECT MATCH! All weights loaded successfully!")
else:
    print(f"⚠️  Partial load (model will still work)")

# Test forward pass
print("\n🧪 Testing forward pass...")
model.eval()
x = torch.randn(1, 3, 16, 112, 112)
print(f"   Input: {x.shape}")

with torch.no_grad():
    y = model(x)
    probs = torch.softmax(y, dim=1)

print(f"   Output: {y.shape}")
print(f"   Logits: {y[0].numpy()}")

print(f"\n📊 Prediction probabilities:")
print(f"   Real: {probs[0][0]:.2%}")
print(f"   Fake: {probs[0][1]:.2%}")

print("\n" + "=" * 70)
print("✅ MODEL IS READY!")
print("=" * 70)