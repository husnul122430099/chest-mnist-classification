# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from model import EfficientChestNet
from datareader import get_data_loaders  # pastikan file ini sudah ada
from sklearn.metrics import accuracy_score

# =========================
# 🔧 Hyperparameters
# =========================
EPOCHS = 100 
WARMUP_EPOCHS = 5
BATCH_SIZE = 32
LR_MAX = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*70)
print("EfficientChestNet - Binary Classification (Chest X-Ray)")
print("="*70)
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ WARNING: Training on CPU - akan lebih lambat!")
print("="*70)

# =========================
# 🔹 Load Dataset
# =========================
train_loader, val_loader, num_classes, in_channels = get_data_loaders(batch_size=BATCH_SIZE)

# =========================
# 🔹 Initialize Model, Loss, Optimizer, Scheduler
# =========================
model = EfficientChestNet(in_channels=in_channels, num_classes=num_classes).to(DEVICE)

# Hitung total parameter
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal model parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}\n")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)
scheduler = OneCycleLR(
    optimizer,
    max_lr=LR_MAX,
    steps_per_epoch=len(train_loader),
    epochs=EPOCHS,
    pct_start=WARMUP_EPOCHS / EPOCHS,
    div_factor=10,
    final_div_factor=100
)

# =========================
# 🔹 Training Function
# =========================
def train_one_epoch(epoch):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
    for images, labels in loop:
        images = images.to(DEVICE)
        labels = labels.float().to(DEVICE)
        
        # Pastikan labels bentuknya (N,) untuk BCEWithLogitsLoss
        if labels.dim() > 1:
            labels = labels.squeeze()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()

        preds = (outputs > 0).float()
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_samples
    acc = 100.0 * total_correct / total_samples
    return avg_loss, acc


# =========================
# 🔹 Validation Function
# =========================
def validate_one_epoch():
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE)
            
            # Pastikan labels bentuknya (N,) untuk BCEWithLogitsLoss
            if labels.dim() > 1:
                labels = labels.squeeze()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            preds = (outputs > 0).float()
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    acc = 100.0 * total_correct / total_samples
    return avg_loss, acc


# =========================
# 🔹 Main Training Loop
# =========================
best_val_acc = 0.0
patience = 15
patience_counter = 0

print("="*70)
print("TRAINING CONFIGURATION")
print("="*70)
print(f"- Epochs: {EPOCHS}")
print(f"- Batch Size: {BATCH_SIZE}")
print(f"- Initial Learning Rate: {LR_MAX}")
print(f"- Weight Decay: {WEIGHT_DECAY}")
print(f"- Warmup Epochs: {WARMUP_EPOCHS}")
print(f"- Early Stopping Patience: {patience}")
print("="*70)

print("\n🚀 Starting EfficientChestNet training...\n")

for epoch in range(EPOCHS):
    avg_train_loss, train_acc = train_one_epoch(epoch)
    avg_val_loss, val_acc = validate_one_epoch()

    current_lr = optimizer.param_groups[0]['lr']
    is_best = val_acc > best_val_acc
    warmup_marker = " [WARMUP]" if epoch < WARMUP_EPOCHS else ""
    status_marker = " ✓✓ NEW BEST!" if is_best else ""

    # Print hasil epoch
    print(f"Epoch [{epoch+1}/{EPOCHS}]{warmup_marker} | "
          f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
          f"LR: {current_lr:.6f}{status_marker}")

    # Simpan model terbaik
    if is_best:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'val_loss': avg_val_loss,
            'train_acc': train_acc,
            'train_loss': avg_train_loss,
        }, "best_efficientchestnet.pth")
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= patience:
        print(f"\n⚠️ Early stopping triggered at epoch {epoch+1}")
        print(f"No improvement for {patience} consecutive epochs")
        break

print("\n" + "="*70)
print("TRAINING COMPLETED")
print("="*70)
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

if best_val_acc >= 90.0:
    print("🎉 TARGET ACHIEVED: Val Acc >= 90%!")
else:
    print(f"⚠️ Target not fully reached. Best Val Acc: {best_val_acc:.2f}%")
print("="*70)
