import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from medmnist import ChestMNIST

# --- Konfigurasi Kelas Biner ---
CLASS_A_IDX = 1  # 'Cardiomegaly'
CLASS_B_IDX = 7  # 'Pneumothorax'

NEW_CLASS_NAMES = {0: 'Cardiomegaly', 1: 'Pneumothorax'}

class FilteredBinaryDataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform
        
        full_dataset = ChestMNIST(split=split, transform=None, download=True)
        original_labels = full_dataset.labels

        indices_a = np.where((original_labels[:, CLASS_A_IDX] == 1) & (original_labels.sum(axis=1) == 1))[0]
        indices_b = np.where((original_labels[:, CLASS_B_IDX] == 1) & (original_labels.sum(axis=1) == 1))[0]

        self.images = []
        self.labels = []

        for idx in indices_a:
            self.images.append(full_dataset[idx][0])
            self.labels.append(0)

        for idx in indices_b:
            self.images.append(full_dataset[idx][0])
            self.labels.append(1)
        
        print(f"Split: {split}")
        print(f"Jumlah Cardiomegaly (label 0): {len(indices_a)}")
        print(f"Jumlah Pneumothorax (label 1): {len(indices_b)}")
        print()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor([label])


def get_data_loaders(batch_size):
    """
    Data loaders dengan augmentasi yang lebih kuat untuk training
    """
    # Training transform dengan augmentasi agresif
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(15),  # Rotasi random ±15 derajat
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # Translasi random
            scale=(0.9, 1.1)       # Scaling random
        ),
        transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontal
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Validation transform tanpa augmentasi
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = FilteredBinaryDataset('train', train_transform)
    val_dataset = FilteredBinaryDataset('test', val_transform)
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    n_classes = 2
    n_channels = 1
    
    print("Dataset ChestMNIST berhasil difilter untuk klasifikasi biner!")
    print(f"Kelas yang digunakan: {NEW_CLASS_NAMES[0]} (Label 0) dan {NEW_CLASS_NAMES[1]} (Label 1)")
    print(f"Jumlah data training: {len(train_dataset)}")
    print(f"Jumlah data validasi: {len(val_dataset)}")
    
    return train_loader, val_loader, n_classes, n_channels