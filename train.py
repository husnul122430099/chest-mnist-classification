# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datareader import get_data_loaders, NEW_CLASS_NAMES
from model import EfficientChestNet
import matplotlib.pyplot as plt
from utils import plot_training_history, visualize_random_val_predictions

# --- Hyperparameter ---
EPOCHS = 30
BATCH_SIZE = 64  # Lebih besar karena menggunakan BatchNorm
LEARNING_RATE = 0.001  # Lebih besar karena BatchNorm memungkinkan learning rate yang lebih agresif
WEIGHT_DECAY = 1e-4  # L2 regularization

#Menampilkan plot riwayat training dan validasi setelah training selesai.

def train():
    # 1. Memuat Data
    print("="*70)
    print("EfficientChestNet - Binary Classification")
    print("="*70)
    
    train_loader, val_loader, num_classes, in_channels = get_data_loaders(BATCH_SIZE)
    
    # 2. Inisialisasi Model
    model = EfficientChestNet(in_channels=in_channels, num_classes=num_classes)
    
    # Pindahkan model ke GPU jika tersedia
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\nUsing device: {device}")
    
    # Hitung total parameter
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")
    print()
    
    # 3. Mendefinisikan Loss Function dan Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Inisialisasi list untuk menyimpan history
    train_losses_history = []
    val_losses_history = []
    train_accs_history = []
    val_accs_history = []
    
    best_val_acc = 0.0
    best_epoch = 0
    
    print("="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    print(f"- Epochs: {EPOCHS}")
    print(f"- Batch Size: {BATCH_SIZE}")
    print(f"- Learning Rate: {LEARNING_RATE}")
    print(f"- Weight Decay: {WEIGHT_DECAY}")
    print(f"- Optimizer: AdamW")
    print(f"- Scheduler: ReduceLROnPlateau")
    print("="*70)
    
    print("\n🚀 STARTING TRAINING")
    print("="*70)
    
    # 4. Training Loop
    for epoch in range(EPOCHS):
        # Training phase
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
            
            # Update metrics
            running_loss += loss.item()
            predicted = (outputs > 0).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
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
        
        # Update learning rate berdasarkan validation loss
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Simpan model terbaik
        is_best = False
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch + 1
            is_best = True
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_accuracy,
                'val_loss': avg_val_loss,
                'train_acc': train_accuracy,
                'train_loss': avg_train_loss,
            }, 'best_model.pth')
        
        # Simpan history
        train_losses_history.append(avg_train_loss)
        val_losses_history.append(avg_val_loss)
        train_accs_history.append(train_accuracy)
        val_accs_history.append(val_accuracy)
        
        # Print progress dengan format yang diminta
        status_marker = " ✓✓ NEW BEST!" if is_best else ""
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}% | "
              f"LR: {current_lr:.6f}{status_marker}")

    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    
    # Load model terbaik untuk evaluasi
    checkpoint = torch.load('best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Best Results (Epoch {best_epoch}):")
    print(f"- Validation Accuracy: {best_val_acc:.2f}%")
    print(f"- Validation Loss: {checkpoint['val_loss']:.4f}")
    print(f"- Training Accuracy: {checkpoint['train_acc']:.2f}%")
    print("="*70)
    
    # Tampilkan plot
    print("\nGenerating visualizations...")
    plot_training_history(train_losses_history, val_losses_history, 
                         train_accs_history, val_accs_history)

    # Visualisasi prediksi pada validation set
    visualize_random_val_predictions(model, val_loader, num_classes, count=10)
    
    print("\n" + "="*70)
    print("ALL TASKS COMPLETED!")
    print("="*70)
    print("Generated files:")
    print("- best_model.pth")
    print("- training_history.png")
    print("- val_predictions.png")
    print("="*70)

if __name__ == '__main__':
    train()