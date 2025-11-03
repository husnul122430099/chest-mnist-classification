import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from datareader_improved import get_data_loaders, NEW_CLASS_NAMES
from model_improved import DenseNet121
from utils import plot_training_history, visualize_random_val_predictions

# --- Hyperparameter Optimal untuk DenseNet121 ---
EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 0.002  # Slightly lower for DenseNet
WEIGHT_DECAY = 1e-4

def train():
    # 1. Memuat Data
    print("="*70)
    print("DenseNet121 CHEST MNIST CLASSIFICATION - Target: Val Acc > 92%")
    print("="*70)
    
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # 2. Inisialisasi Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = DenseNet121(in_channels=in_channels, num_classes=num_classes).to(device)
    
    # Hitung total parameter
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    print(f"Model: DenseNet121 (growth_rate=32, blocks=[6,12,24,16])")
    print()
    
    # 3. Loss Function dan Optimizer
    criterion = nn.BCEWithLogitsLoss()
    
    # AdamW optimizer
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
    print(f"Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | Initial LR: {LEARNING_RATE}")
    print(f"Optimizer: AdamW | Scheduler: OneCycleLR | Weight Decay: {WEIGHT_DECAY}")
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
        
        # Save best model
        current_lr = optimizer.param_groups[0]['lr']
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_densenet121.pth')
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | "
                  f"LR: {current_lr:.6f} ✓ BEST")
        else:
            print(f"Epoch [{epoch+1:3d}/{EPOCHS}] | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | "
                  f"LR: {current_lr:.6f}")

    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    
    if best_val_acc >= 92.0:
        print("✓ TARGET ACHIEVED: Val Acc > 92%")
    else:
        print(f"✗ Target not reached. Gap: {92.0 - best_val_acc:.2f}%")
    
    print("="*70)
    
    # Load best model
    model.load_state_dict(torch.load('best_densenet121.pth'))
    
    # Plot hasil
    print("\nGenerating training history plot...")
    plot_training_history(train_losses_history, val_losses_history, 
                         train_accs_history, val_accs_history)
    
    print("Generating validation predictions visualization...")
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)
    
    print("\nAll visualizations saved successfully!")
    print(f"Model saved as: best_densenet121.pth")

if __name__ == '__main__':
    train()