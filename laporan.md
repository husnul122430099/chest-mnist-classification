# Laporan Eksperimen: Klasifikasi Biner ChestMNIST

**Nama:** Husnul Fatimah  
**NIM:** 122430099  
**Program Studi:** Teknik Biomedis  
**Mata Kuliah:** Kecerdasan Buatan

---

## 📋 Ringkasan Proyek

**Tujuan:** Membangun model klasifikasi biner untuk membedakan antara dua kondisi medis pada dataset ChestMNIST:
- **Class 0**: Cardiomegaly (Pembesaran Jantung)
- **Class 1**: Pneumothorax (Kolaps Paru-paru)

**Target Akurasi:** ≥ 90% validation accuracy

---

## 🔬 Eksperimen 1: EfficientChestNet dengan SE Blocks (train.py)

### 1.1 Arsitektur Model

**File:** `model.py`

```python
class EfficientChestNet:
    Base Architecture:
    - MobileNetV2-style Bottleneck Blocks
    - Squeeze-and-Excitation (SE) Blocks
    - Residual Connections
    
    Components:
    1. Stem: Conv2d(1→32) + BN + GELU
    2. Stage 1: 2 Bottleneck blocks (32→64 channels)
    3. Stage 2: 2 Bottleneck blocks (64→128 channels)  
    4. Stage 3: 3 Bottleneck blocks + SE (128→256 channels)
    5. Head: AdaptiveAvgPool → Dropout(0.4) → Linear(256→128) 
             → GELU → Dropout(0.2) → Linear(128→1)
    
    Bottleneck Block:
    - Expansion factor: 4x
    - Depthwise separable convolutions
    - SE attention (reduction=4)
    - Residual skip connection
```

**Total Parameters:** ~1,283,073 (1.28M)

**Key Features:**
- ✅ Lightweight architecture (1.28M params)
- ✅ Channel attention mechanism (SE blocks)
- ✅ Efficient depthwise separable convolutions
- ✅ Strong regularization (Dropout 0.4, 0.2)

### 1.2 Dataset Configuration

**File:** `datareader.py`

**Pemilihan Data:**
```python
CLASS_A_IDX = 1  # Cardiomegaly
CLASS_B_IDX = 7  # Pneumothorax

Filtering Strategy:
- Hanya single-label samples
- Multi-label samples diabaikan untuk menghindari ambiguitas
```

**Distribusi Dataset:**
| Split | Cardiomegaly | Pneumothorax | Total | Imbalance Ratio |
|-------|--------------|--------------|-------|-----------------|
| Train | 754 samples  | 1,552 samples | 2,306 | 1:2.06 |
| Test  | 243 samples  | 439 samples   | 682   | 1:1.81 |

**Transformasi (Basic):**
```python
Training & Validation:
- ToTensor()
- Normalize(mean=[0.5], std=[0.5])

Augmentation: Minimal (hanya normalisasi)
```

### 1.3 Training Strategy

**File:** `train.py`

**Hyperparameters:**
```python
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
```

**Optimizer:**
```python
AdamW:
- lr: 0.001
- weight_decay: 1e-4
- betas: (0.9, 0.999)
```

**Scheduler:**
```python
OneCycleLR:
- max_lr: 0.001
- pct_start: 0.05 (5% warmup)
- div_factor: 10
- final_div_factor: 100
- anneal_strategy: linear
```

**Regularization:**
- Gradient Clipping: max_norm=1.0
- Dropout: 0.4, 0.2
- Weight Decay: 1e-4

**Early Stopping:**
- Patience: 15 epochs

### 1.4 Hasil Eksperimen 1

**Berdasarkan grafik training_history.png:**

| Metric | Value |
|--------|-------|
| **Best Val Accuracy** | **~72-73%** ⚠️ |
| Final Train Accuracy | ~85-86% |
| Best Val Loss | ~0.56 |
| Final Train Loss | ~0.36 |
| Training Time | ~8-10 min (CPU) |
| Best Epoch | ~8-9 |
| Convergence | Stabil setelah epoch 8 |

**Analisis Grafik:**
- 📈 **Training Loss & Accuracy:**
  - Loss menurun smooth dari 0.68 → 0.36
  - Accuracy naik stabil dari 60% → 86%
  - Konvergensi baik tanpa oscillation

- 📊 **Validation Loss & Accuracy:**
  - Val loss stabil di ~0.56-0.58 setelah epoch 5
  - Val accuracy plateau di ~72-73%
  - **Gap besar antara train (86%) dan val (73%)** ⚠️

**Masalah yang Teridentifikasi:**
1. **Overfitting Moderate:**
   - Train acc: 86% vs Val acc: 73% (gap 13%)
   - Model terlalu fit ke training data

2. **Underfitting ke Validation:**
   - Val accuracy stuck di 72-73%
   - Tidak mencapai target 90%

3. **Class Imbalance Impact:**
   - Ratio 1:2.06 kemungkinan menyebabkan bias
   - Model cenderung prediksi kelas mayoritas

4. **Augmentasi Minimal:**
   - Hanya normalisasi, tidak ada geometric augmentation
   - Dataset kecil (2,306 samples) butuh augmentasi kuat

---

## 🔬 Eksperimen 2: EfficientChestNet Residual (train_improved.py)

### 2.1 Arsitektur Model

**File:** `model_improved.py`

```python
class EfficientChestNet (Residual Version):
    Base Architecture:
    - ResNet-style Residual Blocks
    - Batch Normalization setelah setiap conv
    - Progressive channel expansion
    
    Architecture:
    1. Conv1: Conv2d(1→64) + BN + ReLU
    2. Layer1: 2x ResidualBlock (64→64)
    3. Layer2: 2x ResidualBlock (64→128, stride=2)  # 28→14
    4. Layer3: 2x ResidualBlock (128→256, stride=2) # 14→7
    5. Conv_Final: Conv2d(256→512) + BN + ReLU + AdaptiveAvgPool
    6. Classifier: 
       - Dropout(0.5) → Linear(512→256) → BN → ReLU
       - Dropout(0.4) → Linear(256→128) → BN → ReLU
       - Dropout(0.3) → Linear(128→1)
    
    Residual Block:
    - Conv(3×3) → BN → ReLU → Conv(3×3) → BN
    - Skip connection (1×1 conv jika stride≠1 atau channels berbeda)
    - Final ReLU setelah addition
```

**Total Parameters:** ~7,895,553 (7.9M)

**Key Features:**
- ✅ Deep residual architecture (skip connections)
- ✅ Progressive dropout (0.5 → 0.4 → 0.3)
- ✅ Batch normalization di setiap layer
- ✅ Stronger regularization

### 2.2 Dataset Configuration (Improved)

**File:** `datareader_improved.py`

**Enhanced Data Augmentation:**
```python
Training Transform:
- ToTensor()
- RandomRotation(±15°)
- RandomAffine:
  - translate: (0.1, 0.1)  # 10% shift
  - scale: (0.9, 1.1)       # 10% zoom
- RandomHorizontalFlip(p=0.5)
- Normalize(mean=[0.5], std=[0.5])

Validation Transform:
- ToTensor()
- Normalize(mean=[0.5], std=[0.5])
```

**DataLoader Optimization:**
```python
num_workers=2
pin_memory=True
shuffle=True (train only)
```

### 2.3 Training Strategy (Improved)

**File:** `train_improved.py`

**Hyperparameters:**
```python
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.0005  # Lebih kecil
WEIGHT_DECAY = 5e-4     # Lebih kuat
```

**Optimizer:**
```python
AdamW:
- lr: 0.0005 (lebih konservatif)
- weight_decay: 5e-4 (regularisasi lebih kuat)
```

**Scheduler:**
```python
ReduceLROnPlateau:
- mode: 'min' (monitor val_loss)
- factor: 0.5
- patience: 5 epochs
- Adaptive learning rate adjustment
```

**Regularization:**
- **Stronger Dropout:** 0.5 → 0.4 → 0.3 (progressive)
- **Data Augmentation:** Rotation, Affine, Flip
- **Weight Decay:** 5e-4 (5× lebih kuat)
- **Gradient Clipping:** max_norm=1.0
- **Batch Normalization:** Di semua conv dan linear layers

**Early Stopping:**
- Patience: 15 epochs

### 2.4 Hasil Eksperimen 2 (Expected)

| Metric | Expected Value |
|--------|----------------|
| Best Val Accuracy | **~75-78%** (estimasi) |
| Final Train Accuracy | ~88-90% |
| Best Val Loss | ~0.52-0.54 |
| Training Time | ~12-15 min (CPU) |
| Best Epoch | ~12-15 |
| Model Size | 7.9M parameters |

**Keuntungan:**
- ✅ Residual connections untuk gradient flow
- ✅ Augmentasi kuat mengurangi overfitting
- ✅ Progressive dropout untuk regularisasi
- ✅ ReduceLROnPlateau adaptif

**Trade-offs:**
- ⚠️ Model lebih besar (7.9M vs 1.28M params)
- ⚠️ Training lebih lambat
- ⚠️ Augmentasi memperlambat data loading

---

## 📊 Perbandingan Eksperimen

### 3.1 Arsitektur Comparison

| Aspek | EfficientChestNet (SE) | EfficientChestNet (Residual) |
|-------|------------------------|------------------------------|
| **Base Architecture** | MobileNetV2 Bottleneck | ResNet-style Residual |
| **Parameters** | 1.28M | 7.9M (6× lebih besar) |
| **Attention Mechanism** | ✅ SE Blocks | ❌ None |
| **Skip Connections** | ✅ In bottlenecks | ✅ In residual blocks |
| **Batch Norm** | ✅ In bottlenecks | ✅ Everywhere |
| **Dropout Strategy** | 0.4, 0.2 | 0.5, 0.4, 0.3 (progressive) |
| **Model Complexity** | Lightweight | Heavy |

### 3.2 Training Strategy Comparison

| Aspek | Eksperimen 1 | Eksperimen 2 |
|-------|--------------|--------------|
| **Data Augmentation** | Minimal (normalisasi) | Kuat (Rotation, Affine, Flip) |
| **Learning Rate** | 1e-3 | 5e-4 (lebih konservatif) |
| **Weight Decay** | 1e-4 | 5e-4 (5× lebih kuat) |
| **Scheduler** | OneCycleLR | ReduceLROnPlateau |
| **Warmup** | 5 epochs | None |
| **LR Adjustment** | Cyclic (predetermined) | Adaptive (plateau-based) |
| **Epochs** | 100 | 30 |
| **Early Stop Patience** | 15 | 15 |

### 3.3 Hasil Perbandingan

| Metric | Eksperimen 1 | Eksperimen 2 (Est.) | Winner |
|--------|--------------|---------------------|--------|
| Val Accuracy | 72-73% | 75-78% (est.) | Exp 2 🏆 |
| Train Accuracy | 85-86% | 88-90% (est.) | Exp 2 🏆 |
| Overfitting Gap | 13% | 10-12% (est.) | Exp 2 🏆 |
| Training Time | 8-10 min | 12-15 min | Exp 1 🏆 |
| Model Size | 1.28M | 7.9M | Exp 1 🏆 |
| Inference Speed | Faster | Slower | Exp 1 🏆 |

**Kesimpulan:**
- **Eksperimen 1** lebih cocok untuk deployment (lightweight, fast)
- **Eksperimen 2** lebih cocok untuk maximum accuracy (heavy, slow)
- **Keduanya belum mencapai target 90%** ⚠️

---

## 🔍 Analisis Masalah & Solusi

### 4.1 Class Imbalance ⚠️

**Problem:**
```
Training Set:
- Cardiomegaly: 754 samples (32.7%)
- Pneumothorax: 1,552 samples (67.3%)
Ratio: 1:2.06

Test Set:
- Cardiomegaly: 243 samples (35.6%)  
- Pneumothorax: 439 samples (64.4%)
Ratio: 1:1.81
```

**Impact:**
- Model bias ke kelas mayoritas (Pneumothorax)
- Cardiomegaly under-represented
- Akurasi tinggi di kelas 1, rendah di kelas 0

**Solutions (BELUM DIIMPLEMENTASI):**
```python
1. Weighted Loss:
   pos_weight = torch.tensor([2.06], device=device)
   criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

2. Class Weights in Sampler:
   from torch.utils.data import WeightedRandomSampler
   class_weights = [2.06, 1.0]
   sample_weights = [class_weights[label] for label in labels]
   sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

3. Focal Loss:
   class FocalLoss(nn.Module):
       def __init__(self, alpha=0.25, gamma=2):
           # Focus on hard examples

4. SMOTE Oversampling:
   from imblearn.over_sampling import SMOTE
   # Balance dataset synthetically
```

### 4.2 Dataset Size Limitation

**Problem:**
```
Total Training Samples: 2,306
Total Validation Samples: 682

Sangat kecil untuk deep learning (ideal: 10K+)
```

**Solutions:**
```python
1. External Data Augmentation:
   - Mixup / CutMix
   - AutoAugment
   - RandAugment

2. Transfer Learning:
   from torchvision.models import efficientnet_b0
   model = efficientnet_b0(pretrained=True)
   # Fine-tune on ChestMNIST

3. Self-Supervised Pre-training:
   - SimCLR / MoCo
   - Pre-train on unlabeled chest X-rays

4. Ensemble Methods:
   - Train multiple models dengan different seeds
   - Average predictions
```

### 4.3 Overfitting (Moderate)

**Observation dari Grafik:**
```
Epoch ~8-100:
- Train loss terus turun: 0.45 → 0.36
- Val loss plateau: ~0.56-0.58
- Gap: ~0.20 (moderate overfitting)
```

**Solutions (PARTIALLY IMPLEMENTED):**
```python
✅ DONE:
- Dropout (0.4, 0.2)
- Weight Decay (1e-4 / 5e-4)
- Gradient Clipping
- Data Augmentation (Exp 2)

❌ TODO:
- Stronger augmentation (CutOut, MixUp)
- Label Smoothing
- Stochastic Depth
- Test-Time Augmentation (TTA)
```

### 4.4 Learning Rate Tuning

**Observation:**
```
Eksperimen 1:
- OneCycleLR: smooth convergence
- Val acc plateau setelah epoch 8
- LR mungkin terlalu tinggi (1e-3)

Eksperimen 2:
- ReduceLROnPlateau: adaptive
- LR: 5e-4 (lebih konservatif)
- Expected: slower but more stable
```

**Recommendation:**
```python
# Learning Rate Finder
from torch_lr_finder import LRFinder
lr_finder = LRFinder(model, optimizer, criterion)
lr_finder.range_test(train_loader, end_lr=1, num_iter=100)
lr_finder.plot()  # Cari valley
optimal_lr = 3e-4  # Contoh hasil

# Warmup + Cosine Annealing
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=10,     # Restart setiap 10 epochs
    T_mult=2,   # Double period setiap restart
    eta_min=1e-6
)
```

### 4.5 Model Architecture Limitations

**EfficientChestNet (SE):**
- ✅ Lightweight & fast
- ❌ Mungkin undercapacity untuk medical images
- ❌ SE blocks butuh lebih banyak channels untuk efektif

**EfficientChestNet (Residual):**
- ✅ Deeper architecture
- ✅ Better gradient flow
- ❌ Masih belum cukup untuk 90% acc

**Recommendation:**
```python
# Try Pre-trained Models
1. EfficientNet-B3/B4 (pretrained ImageNet)
2. ResNet50 / ResNet101
3. DenseNet121 / DenseNet169
4. Vision Transformer (ViT-Base)
5. Swin Transformer

# Medical-Specific Models
1. CheXNet (pretrained on ChestX-ray14)
2. DenseNet121-ChestXRay
3. ResNet50-RadImageNet
```

---

## 📈 Output & Visualisasi

### 5.1 Generated Files

**Eksperimen 1:**
1. `best_efficientchestnet.pth` (1.28M params)
2. `training_history.png` (Loss & Accuracy curves)

**Eksperimen 2:**
1. `best_model_improved.pth` (7.9M params)
2. `training_history_improved.png` (expected)

### 5.2 Analisis Grafik (training_history.png)

**Training Loss (Blue):**
- Start: 0.68
- End: 0.36
- Trend: Smooth descent, no oscillation
- Convergence: Good

**Validation Loss (Red):**
- Start: 0.68
- Plateau: 0.56-0.58 (setelah epoch 5)
- Trend: Stabil, slight upward fluctuation
- Issue: Tidak turun setelah epoch 8

**Training Accuracy (Blue):**
- Start: 60%
- End: 85-86%
- Trend: Steady increase
- Convergence: Good

**Validation Accuracy (Red):**
- Start: 60%
- Plateau: 72-73% (setelah epoch 8)
- Trend: Flat dengan minor fluctuation
- Issue: Stuck, tidak improve

**Key Insights:**
1. ⚠️ **Overfitting Moderate:** Train >> Val (gap 13%)
2. ⚠️ **Early Plateau:** Val metrics stuck setelah epoch 8
3. ⚠️ **Underperformance:** Val acc 72% << Target 90%
4. ✅ **Stable Training:** Tidak ada collapse atau oscillation

---

## 🎯 Rekomendasi Improvement

### 6.1 Immediate Actions (Priority 1)

**1. Address Class Imbalance:**
```python
# Implementasi Weighted Loss
class_counts = [754, 1552]
class_weights = [1552/754, 1.0]  # [2.06, 1.0]
pos_weight = torch.tensor([2.06], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**2. Stronger Data Augmentation:**
```python
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(20),  # Increase dari 15
    transforms.RandomAffine(
        degrees=0,
        translate=(0.15, 0.15),  # Increase dari 0.1
        scale=(0.85, 1.15)       # Increase range
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),  # NEW
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # NEW
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

**3. Learning Rate Adjustment:**
```python
# Reduce LR further
LEARNING_RATE = 3e-4  # Dari 5e-4 / 1e-3

# Longer warmup
WARMUP_EPOCHS = 10  # Dari 5

# Add Cosine Annealing
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(
    optimizer, T_0=15, T_mult=2, eta_min=1e-6
)
```

### 6.2 Advanced Techniques (Priority 2)

**1. Transfer Learning:**
```python
from torchvision.models import efficientnet_b3
model = efficientnet_b3(pretrained=True)

# Modify first conv for grayscale
model.features[0][0] = nn.Conv2d(
    1, 40, kernel_size=3, stride=2, padding=1, bias=False
)

# Freeze early layers
for param in model.features[:5].parameters():
    param.requires_grad = False

# Replace classifier
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(1536, 1)
)
```

**2. Ensemble Methods:**
```python
# Train 5 models dengan random seeds
models = []
for seed in [42, 123, 456, 789, 1011]:
    torch.manual_seed(seed)
    model = EfficientChestNet()
    train_model(model)
    models.append(model)

# Ensemble prediction
def ensemble_predict(models, x):
    preds = [torch.sigmoid(m(x)) for m in models]
    return torch.mean(torch.stack(preds), dim=0)
```

**3. Test-Time Augmentation (TTA):**
```python
def tta_predict(model, image, n_augmentations=10):
    preds = []
    for _ in range(n_augmentations):
        aug_image = augment(image)  # Random augmentation
        pred = torch.sigmoid(model(aug_image))
        preds.append(pred)
    return torch.mean(torch.stack(preds))
```

**4. Focal Loss:**
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

criterion = FocalLoss(alpha=0.25, gamma=2)
```

**5. K-Fold Cross-Validation:**
```python
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    model = EfficientChestNet()
    train_fold(model, train_idx, val_idx)
    # Save best model per fold
```

### 6.3 Long-Term Improvements (Priority 3)

**1. Architecture Search:**
```python
# Try different architectures
candidates = [
    'efficientnet_b4',
    'resnet50',
    'densenet121',
    'vit_base_patch16_224',
    'swin_tiny_patch4_window7_224'
]

# Evaluate each on validation set
```

**2. Hyperparameter Tuning:**
```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    wd = trial.suggest_float('wd', 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.2, 0.6)
    
    model = EfficientChestNet(dropout=dropout)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    val_acc = train_and_evaluate(model, optimizer)
    return val_acc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

**3. Mixed Precision Training:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, labels in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 📝 Kesimpulan

### 7.1 Key Findings

**Eksperimen 1 (EfficientChestNet SE):**
- ✅ Lightweight (1.28M params) & fast inference
- ✅ Stable training dengan smooth convergence
- ✅ SE attention mechanism efektif untuk channel attention
- ❌ Val accuracy stuck di 72-73% (jauh dari target 90%)
- ❌ Moderate overfitting (gap 13%)
- ❌ Augmentasi minimal tidak cukup

**Eksperimen 2 (EfficientChestNet Residual):**
- ✅ Deeper architecture dengan residual connections
- ✅ Stronger regularization (progressive dropout)
- ✅ Kuat augmentasi data (rotation, affine, flip)
- ✅ Adaptive LR dengan ReduceLROnPlateau
- ❌ Model lebih besar (7.9M params)
- ⏳ Expected val acc: 75-78% (masih < 90%)

**Root Causes untuk Underperformance:**
1. **Class Imbalance (1:2.06)** - Model bias ke kelas mayoritas
2. **Small Dataset (2,306 samples)** - Insufficient untuk deep learning
3. **Limited Augmentation (Exp 1)** - Tidak cukup variasi data
4. **Model Capacity** - Mungkin undercapacity untuk medical images
5. **No Transfer Learning** - Mulai dari scratch tanpa pretrained weights

### 7.2 Best Practices yang Telah Diterapkan

✅ **Regularization:**
- Dropout (0.4-0.5, 0.2-0.3)
- Weight Decay (1e-4, 5e-4)
- Gradient Clipping (max_norm=1.0)
- Batch Normalization

✅ **Training Strategy:**
- OneCycleLR / ReduceLROnPlateau
- Early Stopping (patience=15)
- Warmup (5 epochs)
- Model checkpointing

✅ **Data Augmentation (Exp 2):**
- RandomRotation (±15°)
- RandomAffine (translate, scale)
- RandomHorizontalFlip

✅ **Architecture Design:**
- Residual connections
- SE attention mechanism
- Progressive dropout
- Depthwise separable convolutions

### 7.3 Rekomendasi Final

**Untuk Mencapai 90% Val Accuracy:**