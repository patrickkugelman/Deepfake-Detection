import torch
from models.resnet3d import resnet3d_18

print("DIAGNOSTIC: Checkpoint vs Model Structure")
print("=" * 70)

# Load checkpoint
checkpoint = torch.load("best_resnet3d_pro.pth", map_location='cpu', weights_only=False)
ckpt_keys = list(checkpoint.keys())

# Load model
model = resnet3d_18(num_classes=2)
model_keys = list(model.state_dict().keys())

print(f"\n📊 Summary:")
print(f"   Checkpoint keys: {len(ckpt_keys)}")
print(f"   Model keys: {len(model_keys)}")

print(f"\n📋 First 20 checkpoint keys:")
for i, key in enumerate(ckpt_keys[:20]):
    print(f"   {i+1:2d}. {key}")

print(f"\n📋 First 20 model keys:")
for i, key in enumerate(model_keys[:20]):
    print(f"   {i+1:2d}. {key}")

# Find differences
print(f"\n🔍 Sample missing keys (in model but not in checkpoint):")
result = model.load_state_dict(checkpoint, strict=False)
for i, key in enumerate(result.missing_keys[:10]):
    print(f"   {i+1}. {key}")

print(f"\n🔍 Sample unexpected keys (in checkpoint but not in model):")
for i, key in enumerate(result.unexpected_keys[:10]):
    print(f"   {i+1}. {key}")

print("=" * 70)