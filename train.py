# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import EfficientNetB0ChestMNIST
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions

# --- Hyperparameter untuk EfficientNet-B0 ---
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001  # Lower LR untuk pretrained model
WEIGHT_DECAY = 1e-4

def train():
    # 1. Memuat Data
    print("="*70)
    print("EfficientNet-B0 CHEST MNIST CLASSIFICATION - Target: Val Acc > 90%")
    print("="*70)
    
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # 2. Inisialisasi Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load EfficientNet-B0 dengan pretrained weights
    model = EfficientNetB0ChestMNIST(
        in_channels=in_channels, 
        num_classes=num_classes, 
        pretrained=True
    ).to(device)
    
    # Hitung total parameter
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"\nModel: EfficientNet-B0 (Pretrained on ImageNet)")
    print()
    
    # 3. Loss Function dan Optimizer
    criterion = nn.BCEWithLogitsLoss()
    
    # AdamW optimizer - lower LR untuk pretrained model
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )
    
    # History tracking
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []
    
    best_val_acc = 0.0
    best_epoch = 0
    
    print("="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"Configuration:")
    print(f"- Epochs: {EPOCHS}")
    print(f"- Batch Size: {BATCH_SIZE}")
    print(f"- Learning Rate: {LEARNING_RATE}")
    print(f"- Optimizer: AdamW")
    print(f"- Scheduler: OneCycleLR")
    print(f"- Weight Decay: {WEIGHT_DECAY}")
    print("="*70)
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
            predicted = (outputs > 0).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # --- Validation Phase ---
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
        
        # Simpan history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_accuracy,
                'val_loss': avg_val_loss,
            }, 'best_efficientnet_b0.pth')
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | "
                  f"LR: {current_lr:.6f} ✓ NEW BEST")
        else:
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | "
                  f"LR: {current_lr:.6f}")

    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    if best_val_acc >= 90.0:
        print(f"✓ TARGET ACHIEVED: Val Acc > 90%")
        print(f"  Exceeded by: {best_val_acc - 90.0:.2f}%")
    else:
        print(f"✗ Target not reached. Gap: {90.0 - best_val_acc:.2f}%")
    
    print("="*70)
    
    # Load best model
    checkpoint = torch.load('best_efficientnet_b0.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Plot hasil
    print("\nGenerating training history plot...")
    plot_training_history(train_losses_history, val_losses_history, 
                         train_accs_history, val_accs_history)
    
    print("Generating validation predictions visualization...")
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)
    
    print("\n" + "="*70)
    print("ALL TASKS COMPLETED!")
    print("="*70)
    print("Generated files:")
    print("- best_efficientnet_b0.pth (model checkpoint)")
    print("- training_history.png")
    print("- val_predictions.png")
    print("="*70)

if __name__ == '__main__':
    train()
