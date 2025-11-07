# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class EfficientChestNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Residual blocks with increasing channels
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)  # 28x28 -> 14x14
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)  # 14x14 -> 7x7
        
        # Additional convolution block
        self.conv_final = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        )
        
        # Classifier with batch normalization
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1 if num_classes == 2 else num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
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
        # Initial convolution
        x = self.conv1(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Final convolution and pooling
        x = self.conv_final(x)
        
        # Flatten and classify
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# --- Bagian untuk pengujian ---
if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    BATCH_SIZE = 64
    
    print("--- Menguji Model 'EfficientChestNet' (Improved) ---")
    
    # Buat model
    model = EfficientChestNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    print("\nArsitektur Model:")
    print(model)
    
    # Test forward pass
    dummy_input = torch.randn(BATCH_SIZE, IN_CHANNELS, 28, 28)
    output = model(dummy_input)
    
    # Hitung jumlah parameter
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nUkuran input: {dummy_input.shape}")
    print(f"Ukuran output: {output.shape}")
    print(f"Total parameter: {total_params:,}")
    print(f"Parameter yang dapat dilatih: {trainable_params:,}")
    print("\nPengujian model 'EfficientChestNet' (Improved) berhasil.")