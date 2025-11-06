import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from datareader_improved import get_data_loaders, NEW_CLASS_NAMES
from model_improved import DenseNet121
from utils import plot_training_history, visualize_random_val_predictions
import numpy as np
import matplotlib.pyplot as plt

# --- Hyperparameter Optimal untuk Val Acc > 91% ---
EPOCHS = 100  # Lebih banyak epochs
BATCH_SIZE = 32  # Batch size lebih kecil untuk update lebih sering
LEARNING_RATE = 0.001  # Lower LR
WEIGHT_DECAY = 1e-4
MIN_LR = 1e-7

def train():
    # 1. Memuat Data
    print("="*70)
    print("DenseNet121 CHEST MNIST - Target: Val Acc > 91%")
    print("="*70)
    
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # 2. Inisialisasi Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = DenseNet121(
        in_channels=in_channels, 
        num_classes=num_classes,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        compression=0.5,
        drop_rate=0.25  # Increase dropout untuk regularisasi lebih kuat
    ).to(device)
    
    # Hitung total parameter
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total model parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"\nModel: DenseNet121")
    print(f"- Growth rate: 32")
    print(f"- Block config: [6, 12, 24, 16]")
    print(f"- Compression: 0.5")
    print(f"- Dropout rate: 0.25")
    print()
    
    # 3. Loss Function
    criterion = nn.BCEWithLogitsLoss()
    
    # 4. Optimizer dengan differential learning rates
    # Separate parameters untuk different layers
    dense_blocks = []
    transition_blocks = []
    classifier_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'denseblock' in name:
            dense_blocks.append(param)
        elif 'transition' in name:
            transition_blocks.append(param)
        elif 'classifier' in name:
            classifier_params.append(param)
        else:
            other_params.append(param)
    
    # AdamW optimizer dengan differential LR
    optimizer = optim.AdamW([
        {'params': other_params, 'lr': LEARNING_RATE * 0.1},
        {'params': dense_blocks, 'lr': LEARNING_RATE * 0.5},
        {'params': transition_blocks, 'lr': LEARNING_RATE * 0.5},
        {'params': classifier_params, 'lr': LEARNING_RATE}
    ], weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    
    # Dual scheduler: CosineAnnealing + ReduceLROnPlateau
    scheduler_cosine = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=15,  # Restart setiap 15 epochs
        T_mult=2,
        eta_min=MIN_LR
    )
    
    # FIX: Hapus parameter 'verbose' yang tidak ada
    scheduler_plateau = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=8,
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
    patience = 20
    patience_counter = 0
    
    print("="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"- Epochs: {EPOCHS}")
    print(f"- Batch Size: {BATCH_SIZE}")
    print(f"- Learning Rate (Classifier): {LEARNING_RATE}")
    print(f"- Learning Rate (Dense/Transition): {LEARNING_RATE * 0.5}")
    print(f"- Learning Rate (Others): {LEARNING_RATE * 0.1}")
    print(f"- Optimizer: AdamW (differential LR)")
    print(f"- Scheduler: CosineAnnealingWarmRestarts + ReduceLROnPlateau")
    print(f"- Weight Decay: {WEIGHT_DECAY}")
    print(f"- Gradient Clipping: max_norm=1.0")
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
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping untuk stabilitas
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
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
        
        # Store predictions untuk analysis
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
        
        # Update schedulers
        scheduler_cosine.step()
        
        # Manual print untuk plateau scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler_plateau.step(val_accuracy)
        new_lr = optimizer.param_groups[0]['lr']
        
        if old_lr != new_lr:
            print(f"\n📉 ReduceLROnPlateau: Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        # Save history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate per-class accuracy
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
                'scheduler_cosine_state_dict': scheduler_cosine.state_dict(),
                'scheduler_plateau_state_dict': scheduler_plateau.state_dict(),
                'val_acc': val_accuracy,
                'val_loss': avg_val_loss,
                'train_acc': train_accuracy,
                'train_loss': avg_train_loss,
                'class_0_acc': class_0_acc,
                'class_1_acc': class_1_acc,
            }, 'best_densenet121.pth')
        else:
            patience_counter += 1
        
        # Print progress dengan format yang diminta (SIMPLIFIED)
        status_marker = " ✓✓ NEW BEST!" if is_best else ""
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | "
              f"LR: {current_lr:.6f}{status_marker}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            break
        
        # Adaptive adjustment jika stuck
        if epoch > 30 and val_accuracy < 88.0:
            print("\n⚠️  Validation accuracy is low. Adjusting learning rate...")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5

    # Training Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    
    # Load best model untuk final evaluation
    checkpoint = torch.load('best_densenet121.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Best Results (Epoch {best_epoch}):")
    print(f"- Validation Accuracy: {best_val_acc:.2f}%")
    print(f"- Validation Loss: {checkpoint['val_loss']:.4f}")
    print(f"- Training Accuracy: {checkpoint['train_acc']:.2f}%")
    print(f"- Class 0 (Cardiomegaly) Acc: {checkpoint['class_0_acc']:.2f}%")
    print(f"- Class 1 (Pneumothorax) Acc: {checkpoint['class_1_acc']:.2f}%")
    
    if best_val_acc >= 91.0:
        print(f"\n🎉 TARGET ACHIEVED: Val Acc > 91%")
        print(f"   Exceeded by: {best_val_acc - 91.0:.2f}%")
    else:
        print(f"\n⚠️  Target not reached")
        print(f"   Gap to 91%: {91.0 - best_val_acc:.2f}%")
    
    print("="*70)
    
    # Plot hasil
    print("\nGenerating visualizations...")
    plot_training_history(train_losses_history, val_losses_history, 
                         train_accs_history, val_accs_history)
    
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)
    
    # Plot learning rate schedule
    plt.figure(figsize=(10, 5))
    plt.plot(learning_rates)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('learning_rate_schedule.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*70)
    print("ALL TASKS COMPLETED!")
    print("="*70)
    print("Generated files:")
    print("- best_densenet121.pth (model checkpoint)")
    print("- training_history.png")
    print("- val_predictions.png")
    print("- learning_rate_schedule.png")
    print("="*70)

if __name__ == '__main__':
    train()