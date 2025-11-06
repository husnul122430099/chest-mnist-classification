# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import EfficientNetB0ChestMNIST
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions
import numpy as np

# --- Hyperparameter Optimal untuk Val Acc > 92% ---
EPOCHS = 100 
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MIN_LR = 1e-6

def train():
    # 1. Memuat Data
    print("="*70)
    print("EfficientNet-B0 CHEST MNIST - Target: Val Acc > 92%")
    print("="*70)
    
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # 2. Inisialisasi Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = EfficientNetB0ChestMNIST(
        in_channels=in_channels, 
        num_classes=num_classes, 
        pretrained=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    print(f"Model: EfficientNet-B0 (Pretrained on ImageNet)")
    print()
    
    # 3. Loss Function
    criterion = nn.BCEWithLogitsLoss()
    
    # 4. Optimizer - Unfreeze ALL layers dari awal
    # Gunakan differential learning rates
    feature_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
        else:
            feature_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': feature_params, 'lr': LEARNING_RATE * 0.1},
        {'params': classifier_params, 'lr': LEARNING_RATE}
    ], weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    
    # OneCycleLR - proven scheduler untuk fast convergence
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[LEARNING_RATE * 0.1, LEARNING_RATE],
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )
    
    # Backup: ReduceLROnPlateau jika stuck
    scheduler_plateau = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10,
        min_lr=MIN_LR
    )
    
    # History tracking
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []
    learning_rates = []
    
    best_val_acc = 0.0
    best_epoch = 0
    patience = 25
    patience_counter = 0
    
    print("="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"- Epochs: {EPOCHS}")
    print(f"- Batch Size: {BATCH_SIZE}")
    print(f"- Learning Rate (Classifier): {LEARNING_RATE}")
    print(f"- Learning Rate (Features): {LEARNING_RATE * 0.1}")
    print(f"- Optimizer: AdamW (differential LR)")
    print(f"- Scheduler: OneCycleLR + ReduceLROnPlateau (backup)")
    print(f"- Weight Decay: {WEIGHT_DECAY}")
    print(f"- All layers UNFROZEN from start")
    print(f"- Early Stopping Patience: {patience}")
    print("="*70)
    
    print("\n🚀 STARTING TRAINING")
    print("="*70)
    
    # Training Loop
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.float().to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()  # Step per batch untuk OneCycleLR
            
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
        
        all_preds = []
        all_labels = []
        
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
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        # Update plateau scheduler (backup)
        old_lr = optimizer.param_groups[0]['lr']
        scheduler_plateau.step(val_accuracy)
        new_lr = optimizer.param_groups[0]['lr']
        
        if old_lr != new_lr:
            print(f"\n📉 Plateau Scheduler: LR reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        # Save history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]
        
        # Per-class accuracy
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        class_0_acc = 100 * np.sum((all_preds == 0) & (all_labels == 0)) / max(np.sum(all_labels == 0), 1)
        class_1_acc = 100 * np.sum((all_preds == 1) & (all_labels == 1)) / max(np.sum(all_labels == 1), 1)
        
        # Save best model
        is_best = False
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch + 1
            patience_counter = 0
            is_best = True
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_accuracy,
                'val_loss': avg_val_loss,
                'train_acc': train_accuracy,
                'train_loss': avg_train_loss,
                'class_0_acc': class_0_acc,
                'class_1_acc': class_1_acc,
            }, 'best_efficientnet_b0.pth')
        else:
            patience_counter += 1
        
        # Print progress dengan format yang diminta
        status_marker = " ✓✓ NEW BEST!" if is_best else ""
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | "
              f"LR: {current_lr:.6f}{status_marker}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            break
        
        # Adaptive boost jika progress lambat
        if epoch == 40 and best_val_acc < 88.0:
            print("\n⚡ Progress too slow. Boosting learning rate...")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 2.0

    # Training Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    
    checkpoint = torch.load('best_efficientnet_b0.pth', weights_only=False)  # ← TAMBAHKAN weights_only=False
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Best Results (Epoch {best_epoch}):")
    print(f"- Validation Accuracy: {best_val_acc:.2f}%")
    print(f"- Validation Loss: {checkpoint['val_loss']:.4f}")
    print(f"- Training Accuracy: {checkpoint['train_acc']:.2f}%")
    print(f"- Class 0 (Cardiomegaly) Acc: {checkpoint['class_0_acc']:.2f}%")
    print(f"- Class 1 (Pneumothorax) Acc: {checkpoint['class_1_acc']:.2f}%")
    
    if best_val_acc >= 92.0:
        print(f"\n🎉 TARGET ACHIEVED: Val Acc > 92%")
        print(f"   Exceeded by: {best_val_acc - 92.0:.2f}%")
    elif best_val_acc >= 90.0:
        print(f"\n✓ Good Result: Val Acc > 90%")
        print(f"  Gap to 92%: {92.0 - best_val_acc:.2f}%")
    else:
        print(f"\n⚠️  Target not reached")
        print(f"   Gap to 92%: {92.0 - best_val_acc:.2f}%")
    
    print("="*70)
    
    # Visualizations
    print("\nGenerating visualizations...")
    plot_training_history(train_losses_history, val_losses_history, 
                         train_accs_history, val_accs_history)
    
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)
    
    # Plot learning rate
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule (OneCycleLR)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('learning_rate_schedule.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*70)
    print("ALL TASKS COMPLETED!")
    print("="*70)
    print("Generated files:")
    print("- best_efficientnet_b0.pth")
    print("- training_history.png")
    print("- val_predictions.png")
    print("- learning_rate_schedule.png")
    print("="*70)

if __name__ == '__main__':
    train()
