# train_mobilenet.py
"""
Training script khusus untuk MobileNet models
Optimized untuk mencapai val accuracy > 92% dengan training cepat
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from datareader import get_data_loaders
from mobilenet_model import SuperMobileNet, EnhancedMobileNet, FastChestNet
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions
import time
import os

# ==================== CONFIGURATION ====================
# Pilih model: 'super', 'enhanced', atau 'fast'
MODEL_CHOICE = 'super'  # RECOMMENDED untuk 92%+ accuracy

# Hyperparameters
EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 0.002
WEIGHT_DECAY = 4e-4

# Training settings
TARGET_ACCURACY = 92.0
GRADIENT_CLIP = 0.5
USE_WARMUP = True
WARMUP_EPOCHS = 5

# File paths
SAVE_DIR = 'checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)

# ==================== DATA AUGMENTATION ====================

class DataAugmentation:
    """Data augmentation untuk meningkatkan generalization"""
    def __init__(self):
        self.noise_level = 0.06
        
    def __call__(self, images):
        # Random noise
        if torch.rand(1).item() > 0.5:
            noise = torch.randn_like(images) * self.noise_level
            images = images + noise
            images = torch.clamp(images, 0, 1)
        return images

# ==================== MAIN TRAINING ====================

def train_mobilenet():
    """Main training function"""
    
    print(f"\n{'='*80}")
    print(f"Model: {MODEL_CHOICE.upper()}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Batch Size: {BATCH_SIZE} | LR: {LEARNING_RATE} | Epochs: {EPOCHS}")
    print(f"{'='*80}\n")
    
    # 1. Load data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # 2. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 3. Initialize model
    if MODEL_CHOICE == 'super':
        model = SuperMobileNet(in_channels=in_channels, num_classes=num_classes).to(device)
    elif MODEL_CHOICE == 'enhanced':
        model = EnhancedMobileNet(in_channels=in_channels, num_classes=num_classes).to(device)
    elif MODEL_CHOICE == 'fast':
        model = FastChestNet(in_channels=in_channels, num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Invalid model choice: {MODEL_CHOICE}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}\n")
    
    # 4. Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # 5. Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # 6. Data augmentation
    augmentation = DataAugmentation()
    
    # 7. History
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []
    
    best_val_acc = 0.0
    best_epoch = 0
    patience = 0
    max_patience = 15
    
    print("--- Memulai Training ---\n")
    
    # 8. Training loop
    for epoch in range(EPOCHS):
        # TRAINING
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Warmup LR
        if USE_WARMUP and epoch < WARMUP_EPOCHS:
            lr = LEARNING_RATE * (epoch + 1) / WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            
            # Augmentation
            if epoch >= 5:
                images = augmentation(images)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            
            running_loss += loss.item()
            with torch.no_grad():
                predicted = (outputs > 0).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
        
        if not USE_WARMUP or epoch >= WARMUP_EPOCHS:
            scheduler.step()
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # VALIDATION
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device)
                
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item()
                
                predicted = (outputs > 0).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)
        
        # Save best
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch + 1
            patience = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_accuracy,
            }, os.path.join(SAVE_DIR, f'best_{MODEL_CHOICE}_model.pth'))
            print(f"✓ New best model saved! Val Acc: {val_accuracy:.2f}%")
        else:
            patience += 1
        
        # Print
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | "
              f"LR: {current_lr:.6f}")
        
        # Target reached
        if val_accuracy >= TARGET_ACCURACY and epoch >= 30:
            print(f"\n🎯 Target {TARGET_ACCURACY}% reached!")
            break
        
        # Early stopping
        if patience >= max_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Summary
    print("\n--- Training Selesai ---")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    # Load best
    checkpoint = torch.load(os.path.join(SAVE_DIR, f'best_{MODEL_CHOICE}_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Visualize
    plot_training_history(train_losses_history, val_losses_history,
                         train_accs_history, val_accs_history)
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)
    
    return model, best_val_acc


if __name__ == '__main__':
    model, best_acc = train_mobilenet()
    
    if best_acc >= TARGET_ACCURACY:
        print(f"\n✅ SUCCESS! Achieved {best_acc:.2f}%!")
    else:
        print(f"\n⚠️  Best: {best_acc:.2f}% (Target: {TARGET_ACCURACY}%)")
    
    print(f"\n✓ Training completed!\n")
