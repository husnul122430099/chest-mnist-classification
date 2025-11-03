# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import EfficientChestNet
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions

# --- Hyperparameter yang dioptimalkan ---
EPOCHS = 40
BATCH_SIZE = 32  # Increased for better gradient estimates
LEARNING_RATE = 0.001  # Higher initial LR with scheduler
WEIGHT_DECAY = 1e-4  # L2 regularization

def train():
    # 1. Memuat Data
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # 2. Inisialisasi Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = EfficientChestNet(in_channels=in_channels, num_classes=num_classes).to(device)
    print(model)
    
    # 3. Mendefinisikan Loss Function dan Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning Rate Scheduler untuk adaptive learning (hapus verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Inisialisasi list untuk menyimpan history
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []
    
    best_val_acc = 0.0
    
    print("\n--- Memulai Training ---")
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
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
            
            # Gradient clipping untuk stabilitas
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            # Hitung training accuracy
            predicted = (outputs > 0).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # --- Fase Validasi ---
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
        
        # Update learning rate based on validation loss
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Manual print jika LR berubah
        if old_lr != new_lr:
            print(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
        
        # Simpan history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"✓ New best model saved! Val Acc: {val_accuracy:.2f}%")
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | "
              f"LR: {new_lr:.6f}")

    print("--- Training Selesai ---")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    # Load best model untuk visualisasi
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Tampilkan plot
    plot_training_history(train_losses_history, val_losses_history, 
                         train_accs_history, val_accs_history)

    # Visualisasi prediksi pada 10 gambar random dari validation set
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)

if __name__ == '__main__':
    train()
