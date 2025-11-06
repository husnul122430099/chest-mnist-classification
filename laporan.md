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

---

## 🔬 Eksperimen 1: EfficientNet-B0 (train.py)

### 1.1 Arsitektur Model

**File:** `model.py`

```python
class EfficientNetB0ChestMNIST:
    - Base: EfficientNet-B0 pretrained (ImageNet)
    - Input: 28x28 grayscale (1 channel)
    - Output: Binary classification (1 neuron)
    
    Modifikasi:
    1. First layer: Conv2d(3→1 channel)
       - Weight averaging: RGB → Grayscale
    2. Classifier:
       - Dropout(0.3) → Linear(1280→512) → ReLU
       - Dropout(0.2) → Linear(512→1)
```

**Total Parameters:** ~5.3M (pretrained dari ImageNet)

### 1.2 Dataset Configuration

**File:** `datareader.py`

**Pemilihan Data:**
```python
CLASS_A_IDX = 1  # Cardiomegaly
CLASS_B_IDX = 7  # Pneumothorax

Filtering:
- Hanya single-label samples
- Multi-label samples diabaikan
```

**Distribusi Dataset:**
| Split | Cardiomegaly | Pneumothorax | Total |
|-------|--------------|--------------|-------|
| Train | 350 samples  | 750 samples  | 1,100 |
| Test  | 90 samples   | 200 samples  | 290   |

**Transformasi:**
```python
Training & Validation:
- ToTensor()
- Normalize(mean=[0.5], std=[0.5])
```

### 1.3 Training Strategy

**File:** `train.py`

**Hyperparameters:**
```python
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
```

**Optimizer:**
```python
AdamW with Differential Learning Rates:
- Features: LR × 0.1 (0.0001)
- Classifier: LR × 1.0 (0.001)
```

**Scheduler:**
```python
Primary: OneCycleLR
- max_lr: [0.0001, 0.001]
- pct_start: 0.3 (30% warmup)
- anneal_strategy: cosine

Backup: ReduceLROnPlateau
- factor: 0.5
- patience: 10 epochs
```

**Regularization:**
- Weight Decay: 1e-4
- Gradient Clipping: max_norm=1.0
- Dropout: 0.3, 0.2

**Early Stopping:**
- Patience: 25 epochs

### 1.4 Hasil Eksperimen 1

| Metric | Value |
|--------|-------|
| Best Val Accuracy | **TBD** % |
| Best Val Loss | **TBD** |
| Training Time | ~10-15 min (CPU) |
| Best Epoch | **TBD** |
| Class 0 Accuracy | **TBD** % |
| Class 1 Accuracy | **TBD** % |

**Analisis:**
- Pretrained weights membantu konvergensi cepat
- OneCycleLR memberikan smooth learning curve
- Differential LR mencegah catastrophic forgetting

---

## 🔬 Eksperimen 2: DenseNet121 (train_improved.py)

### 2.1 Arsitektur Model

**File:** `model_improved.py`

```python
class DenseNet121:
    - Growth rate: 32
    - Block config: [6, 12, 24, 16]
    - Compression: 0.5
    - Dropout: 0.25
    
    Architecture:
    1. Conv2d(1→64) + BN + ReLU + MaxPool
    2. DenseBlock1 (6 layers)  → Transition1
    3. DenseBlock2 (12 layers) → Transition2
    4. DenseBlock3 (24 layers) → Transition3
    5. DenseBlock4 (16 layers)
    6. Global Average Pooling
    7. Linear(num_features → 1)
```

**Total Parameters:** ~7.0M (trained from scratch)

### 2.2 Dataset Configuration (Improved)

**File:** `datareader_improved.py`

**Enhanced Data Augmentation:**
```python
Training Transform:
- RandomRotation(±15°)
- RandomAffine:
  - translate: (0.1, 0.1)
  - scale: (0.9, 1.1)
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
```

### 2.3 Training Strategy (Improved)

**File:** `train_improved.py`

**Hyperparameters:**
```python
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
DROPOUT = 0.25  # Increased
```

**Optimizer dengan Differential LR:**
```python
AdamW:
- Other params: LR × 0.1 (0.0001)
- Dense blocks: LR × 0.5 (0.0005)
- Transition blocks: LR × 0.5 (0.0005)
- Classifier: LR × 1.0 (0.001)
```

**Dual Scheduler Strategy:**
```python
1. CosineAnnealingWarmRestarts:
   - T_0: 15 epochs
   - T_mult: 2
   - eta_min: 1e-7

2. ReduceLROnPlateau:
   - mode: max
   - factor: 0.5
   - patience: 8 epochs
```

**Regularization:**
- Stronger Dropout: 0.25
- Data Augmentation: Rotation, Affine, Flip
- Weight Decay: 1e-4
- Gradient Clipping: max_norm=1.0

**Early Stopping:**
- Patience: 20 epochs

**Adaptive Learning:**
```python
if epoch > 30 and val_accuracy < 88%:
    LR *= 0.5  # Reduce if stuck
```

### 2.4 Hasil Eksperimen 2

| Metric | Value |
|--------|-------|
| Best Val Accuracy | **TBD** % |
| Best Val Loss | **TBD** |
| Training Time | ~15-20 min (CPU) |
| Best Epoch | **TBD** |
| Class 0 Accuracy | **TBD** % |
| Class 1 Accuracy | **TBD** % |

**Analisis:**
- DenseNet121 memiliki feature reuse yang efisien
- Augmentasi kuat mencegah overfitting
- Dual scheduler memberikan fleksibilitas learning

---

## 📊 Perbandingan Eksperimen

### 3.1 Perbedaan Utama

| Aspek | EfficientNet-B0 | DenseNet121 |
|-------|-----------------|-------------|
| **Pretrained** | ✅ ImageNet | ❌ From scratch |
| **Parameters** | ~5.3M | ~7.0M |
| **Data Augmentation** | Basic | Strong (Rotation, Affine, Flip) |
| **Dropout** | 0.3, 0.2 | 0.25 |
| **Scheduler** | OneCycleLR + Plateau | CosineWarmRestart + Plateau |
| **Differential LR Groups** | 2 groups | 4 groups |
| **Target Acc** | > 92% | > 91% |
| **Training Time** | Faster | Slower |

### 3.2 Strategi Training

**EfficientNet-B0:**
- ✅ Transfer learning dari ImageNet
- ✅ Konvergensi lebih cepat
- ✅ Model lebih ringan
- ❌ Augmentasi minimal

**DenseNet121:**
- ✅ Feature reuse yang efisien
- ✅ Augmentasi kuat
- ✅ Regularisasi lebih agresif
- ❌ Training dari nol (lebih lambat)
- ❌ Model lebih besar

### 3.3 Hyperparameter Tuning

**Learning Rate Strategy:**
```
EfficientNet-B0:
├── Features: 0.0001
└── Classifier: 0.001

DenseNet121:
├── Other: 0.0001
├── Dense Blocks: 0.0005
├── Transitions: 0.0005
└── Classifier: 0.001
```

**Scheduler Comparison:**

| Feature | OneCycleLR | CosineWarmRestart |
|---------|------------|-------------------|
| Warmup | 30% | Per cycle |
| Restart | ❌ | ✅ Every 15 epochs |
| Annealing | Smooth | Periodic |
| Best For | Fast convergence | Escaping local minima |

---

## 🔍 Analisis Masalah yang Ditemukan

### 4.1 Class Imbalance

**Problem:**
```
Training Set:
- Cardiomegaly: 350 samples (31.8%)
- Pneumothorax: 750 samples (68.2%)

Ratio: 1:2.14
```

**Impact:**
```
Epoch [7/100] | Val: 63.20%
- C0: 9.9%  ← Model bias ke kelas 1
- C1: 92.7% ← Hampir selalu prediksi kelas 1
```

**Potential Solutions (Belum Diimplementasi):**
```python
1. Weighted Loss:
   pos_weight = torch.tensor([2.14])
   criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

2. Focal Loss:
   Focus on hard examples

3. SMOTE/Oversampling:
   Balance dataset
```

### 4.2 PyTorch 2.6 Compatibility

**Problem:**
```python
_pickle.UnpicklingError: Weights only load failed
```

**Root Cause:**
- PyTorch 2.6 changed default `weights_only=True`
- Checkpoint berisi numpy arrays yang tidak diizinkan

**Solution:**
```python
# FIXED in both files:
checkpoint = torch.load('model.pth', weights_only=False)
```

### 4.3 Learning Rate Tuning

**Observation:**
```
Epoch [7/100] | LR: 0.000016

Problem: LR terlalu kecil terlalu cepat
```

**Recommendation:**
```python
# Increase warmup phase
scheduler = OneCycleLR(
    optimizer,
    pct_start=0.4  # 40% warmup (dari 30%)
)
```

---

## 📈 Output & Visualisasi

### 5.1 Generated Files

**Setiap eksperimen menghasilkan:**

1. **Model Checkpoint:**
   - `best_efficientnet_b0.pth`
   - `best_densenet121.pth`

2. **Training Visualizations:**
   - `training_history.png` (Loss & Accuracy curves)
   - `val_predictions.png` (Sample predictions)
   - `learning_rate_schedule.png` (LR over epochs)

### 5.2 Checkpoint Contents

```python
checkpoint = {
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'scheduler_state_dict': OrderedDict,  # train_improved.py only
    'val_acc': float,
    'val_loss': float,
    'train_acc': float,
    'train_loss': float,
    'class_0_acc': float,
    'class_1_acc': float,
}
```

---

## 🎯 Rekomendasi Improvement

### 6.1 Immediate Actions

1. **Address Class Imbalance:**
   ```python
   # Tambahkan weighted loss
   pos_weight = torch.tensor([2.14], device=device)
   criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
   ```

2. **Increase Training Time:**
   ```python
   # Both experiments
   EPOCHS = 150  # Dari 100
   ```

3. **Ensemble Method:**
   ```python
   # Average predictions from both models
   pred = (efficientnet_pred + densenet_pred) / 2
   ```

### 6.2 Advanced Techniques

1. **Test-Time Augmentation (TTA):**
   ```python
   # Predict pada multiple augmented versions
   # Average predictions
   ```

2. **Learning Rate Finder:**
   ```python
   # Find optimal LR automatically
   from torch_lr_finder import LRFinder
   ```

3. **Mixed Precision Training:**
   ```python
   # Faster training dengan AMP
   scaler = torch.cuda.amp.GradScaler()
   ```

4. **K-Fold Cross-Validation:**
   ```python
   # More robust evaluation
   kf = KFold(n_splits=5)
   ```

---

## 📝 Kesimpulan

### 7.1 Key Findings

1. **EfficientNet-B0** lebih cocok untuk:
   - Fast prototyping
   - Limited computational resources
   - Transfer learning scenarios

2. **DenseNet121** lebih cocok untuk:
   - Maximum accuracy
   - Strong regularization needs
   - Training from scratch

3. **Common Challenges:**
   - Class imbalance (~1:2 ratio)
   - Small dataset (1,100 train, 290 val)
   - Grayscale medical images (28×28)

### 7.2 Best Practices Applied

✅ **Transfer Learning** (EfficientNet)  
✅ **Data Augmentation** (DenseNet)  
✅ **Differential Learning Rates**  
✅ **Multiple Schedulers**  
✅ **Gradient Clipping**  
✅ **Early Stopping**  
✅ **Comprehensive Logging**  
✅ **Visualization Tools**  

### 7.3 Next Steps

1. ⏳ **Run full training** (100 epochs)
2. 📊 **Analyze confusion matrix**
3. 🔧 **Implement class balancing**
4. 🎯 **Fine-tune hyperparameters**
5. 🤝 **Ensemble both models**
6. 📈 **Evaluate on additional metrics** (F1, ROC-AUC)

---

## 📚 References

- **EfficientNet:** [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- **DenseNet:** [Huang et al., 2017](https://arxiv.org/abs/1608.06993)
- **MedMNIST:** [Yang et al., 2021](https://arxiv.org/abs/2010.14925)
- **OneCycleLR:** [Smith, 2018](https://arxiv.org/abs/1803.09820)

---

**Disusun oleh:**  
Husnul Fatimah (122430099)  
Teknik Biomedis  
Mata Kuliah Kecerdasan Buatan  

**Tanggal:** 6 November 2025  
**Environment:** PyTorch 2.6, Python 3.13