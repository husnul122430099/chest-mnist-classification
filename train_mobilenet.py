# train_mobilenet.py
"""
Training script khusus untuk MobileNet models (FastChestNet & UltraFastChestNet)
Optimized untuk mencapai val accuracy > 92% dengan training cepat
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from datareader import get_data_loaders
from mobilenet_model import FastChestNet, UltraFastChestNet
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions
import time
import os

# ==================== CONFIGURATION ====================
# Pilih model: 'fast' atau 'ultrafast'
MODEL_CHOICE = 'ultrafast'  # Ubah ini untuk mencoba model lain

# Hyperparameters
EPOCHS = 35
BATCH_SIZE = 64
LEARNING_RATE = 0.004
WEIGHT_DECAY = 2e-4

# Training settings
TARGET_ACCURACY = 92.0
GRADIENT_CLIP = 1.0

# File paths
SAVE_DIR = 'checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)

# ==================== MAIN TRAINING ====================

def train_mobilenet():
    """Main training function"""
    
    print(f"\n{'='*80}")
    print(f"{'MOBILENET CHEST-MNIST TRAINING':^80}")
    print(f"{'='*80}")
    print(f"Model: {MODEL_CHOICE.upper()}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Batch Size: {BATCH_SIZE} | LR: {LEARNING_RATE} | Epochs: {EPOCHS}")
    print(f"{'='*80}\n")
    
    # 1. Load data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # 2. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 3. Initialize model
    if MODEL_CHOICE == 'fast':
        model = FastChestNet(in_channels=in_channels, num_classes=num_classes).to(device)
    elif MODEL_CHOICE == 'ultrafast':
        model = UltraFastChestNet(in_channels=in_channels, num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Invalid model choice: {MODEL_CHOICE}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}\n")
    
    # 4. Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # 5. Learning rate scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )
    
    # 6. Training history
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []
    
    best_val_acc = 0.0
    
    print("--- Memulai Training ---\n")
    
    # 7. Training loop
    for epoch in range(EPOCHS):
        # === TRAINING PHASE ===
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP)
            
            optimizer.step()
            scheduler.step()
            
            # Statistics
            running_loss += loss.item()
            predicted = (outputs > 0).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # === VALIDATION PHASE ===
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
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_accuracy,
                'val_loss': avg_val_loss,
            }
            torch.save(checkpoint, os.path.join(SAVE_DIR, f'best_{MODEL_CHOICE}_model.pth'))
            print(f"✓ New best model saved! Val Acc: {val_accuracy:.2f}%")
        
        # Print epoch results
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | "
              f"LR: {current_lr:.6f}")
        
        # Early stop if target reached
        if val_accuracy >= TARGET_ACCURACY and epoch >= 20:
            print(f"\n🎯 Target accuracy {TARGET_ACCURACY}% reached at epoch {epoch+1}!")
            break
    
    # 8. Training summary
    print("\n--- Training Selesai ---")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # 9. Load best model
    best_checkpoint = torch.load(os.path.join(SAVE_DIR, f'best_{MODEL_CHOICE}_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # 10. Visualizations
    plot_training_history(
        train_losses_history,
        val_losses_history,
        train_accs_history,
        val_accs_history
    )
    
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)
    
    return model, best_val_acc


if __name__ == '__main__':
    model, best_acc = train_mobilenet()
    print(f"\n✓ Training completed! Best accuracy: {best_acc:.2f}%\n")
