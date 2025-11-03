# model.py

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNetB0ChestMNIST(nn.Module):
    """
    EfficientNet-B0 dengan pretrained ImageNet weights
    Dimodifikasi untuk ChestMNIST (28x28 grayscale, 2 classes)
    Target: Val Acc > 90%
    """
    def __init__(self, in_channels=1, num_classes=2, pretrained=True):
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.efficientnet = efficientnet_b0(weights=weights)
        else:
            self.efficientnet = efficientnet_b0(weights=None)
        
        # Modifikasi layer pertama untuk grayscale (1 channel)
        # Original: Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        original_first_layer = self.efficientnet.features[0][0]
        self.efficientnet.features[0][0] = nn.Conv2d(
            in_channels,  # 1 channel untuk grayscale
            original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=False
        )
        
        # Jika pretrained, copy weights dari 3 channel ke 1 channel
        if pretrained:
            with torch.no_grad():
                # Average RGB weights menjadi grayscale
                self.efficientnet.features[0][0].weight = nn.Parameter(
                    original_first_layer.weight.mean(dim=1, keepdim=True)
                )
        
        # Modifikasi classifier untuk binary classification
        # Original classifier: Linear(in_features=1280, out_features=1000)
        in_features = self.efficientnet.classifier[1].in_features
        
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 1 if num_classes == 2 else num_classes)
        )
    
    def forward(self, x):
        return self.efficientnet(x)


class EfficientChestNet(nn.Module):
    """
    Model CNN yang lebih dalam dengan Batch Normalization dan Dropout
    (Model alternatif jika tidak ingin menggunakan pretrained)
    """
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        
        # Block 1: 28x28 → 14x14
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1)
        )
        
        # Block 2: 14x14 → 7x7
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        
        # Block 3: 7x7 → 3x3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2)
        )
        
        # Global Average Pooling + Fully Connected
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 1 if num_classes == 2 else num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x


class SimpleCNN(nn.Module):
    """Legacy SimpleCNN - kept for backward compatibility"""
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2)
        self.pool = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1 if num_classes == 2 else num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    
    print("="*70)
    print("TESTING EfficientNet-B0 for ChestMNIST")
    print("="*70)
    
    # Test EfficientNet-B0
    model = EfficientNetB0ChestMNIST(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, pretrained=True)
    print("\nModel Architecture (EfficientNet-B0):")
    print(model)
    
    dummy_input = torch.randn(8, IN_CHANNELS, 28, 28)
    print(f"\nInput shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*70}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"{'='*70}")
    
    print("\n✓ EfficientNet-B0 model test successful!")
    print("\nKey Features:")
    print("- Pretrained on ImageNet")
    print("- Modified first layer for grayscale input (1 channel)")
    print("- Custom classifier for binary classification")
    print("- Target: Val Acc > 90%")