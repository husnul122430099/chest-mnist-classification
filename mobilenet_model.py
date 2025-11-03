# mobilenet_model.py

import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution untuk efisiensi komputasi
    Mengurangi parameter hingga 8-9x dibanding conv biasa
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=3, stride=stride, 
            padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InvertedResidual(nn.Module):
    """
    Inverted Residual Block dari MobileNetV2
    Expand -> Depthwise -> Project dengan skip connection
    """
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(in_channels * expand_ratio)
        self.use_residual = self.stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            # Expansion
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise + Pointwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class FastChestNet(nn.Module):
    """
    MobileNetV2-inspired CNN untuk ChestMNIST
    - Cepat: Menggunakan depthwise separable convolutions
    - Akurat: Inverted residual blocks dengan skip connections
    - Target: Val acc > 90% dengan training cepat
    """
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        
        # Initial convolution: 28x28 -> 28x28
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # Inverted Residual Blocks
        # Format: (expand_ratio, out_channels, num_blocks, stride)
        self.blocks = nn.Sequential(
            # 28x28 -> 14x14
            InvertedResidual(32, 16, stride=1, expand_ratio=1),
            InvertedResidual(16, 24, stride=2, expand_ratio=6),
            InvertedResidual(24, 24, stride=1, expand_ratio=6),
            
            # 14x14 -> 7x7
            InvertedResidual(24, 32, stride=2, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            
            # 7x7 -> 3x3
            InvertedResidual(32, 64, stride=2, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
        )
        
        # Final convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True)
        )
        
        # Global Average Pooling + Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 1 if num_classes == 2 else num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.conv1(x)      # (N, 32, 28, 28)
        x = self.blocks(x)     # (N, 64, 3, 3)
        x = self.conv2(x)      # (N, 128, 3, 3)
        x = self.avgpool(x)    # (N, 128, 1, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x) # (N, 1)
        return x


class UltraFastChestNet(nn.Module):
    """
    Versi lebih cepat lagi dengan parameter lebih sedikit
    Target: Training super cepat dengan val acc > 90%
    """
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1: 28x28 -> 14x14
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 64, stride=1),
            
            # Block 2: 14x14 -> 7x7
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            
            # Block 3: 7x7 -> 3x3
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 1 if num_classes == 2 else num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# --- Bagian untuk pengujian ---
if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    
    print("=" * 60)
    print("--- Menguji Model 'FastChestNet' (MobileNetV2-inspired) ---")
    print("=" * 60)
    
    model1 = FastChestNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    print("\nArsitektur FastChestNet:")
    print(model1)
    
    dummy_input = torch.randn(32, IN_CHANNELS, 28, 28)
    output1 = model1(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output1.shape}")
    
    total_params1 = sum(p.numel() for p in model1.parameters())
    trainable_params1 = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params1:,}")
    print(f"Trainable parameters: {trainable_params1:,}")
    
    print("\n" + "=" * 60)
    print("--- Menguji Model 'UltraFastChestNet' ---")
    print("=" * 60)
    
    model2 = UltraFastChestNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    print("\nArsitektur UltraFastChestNet:")
    print(model2)
    
    output2 = model2(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output2.shape}")
    
    total_params2 = sum(p.numel() for p in model2.parameters())
    trainable_params2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params2:,}")
    print(f"Trainable parameters: {trainable_params2:,}")
    
    print("\n" + "=" * 60)
    print("PERBANDINGAN:")
    print(f"FastChestNet parameters: {total_params1:,}")
    print(f"UltraFastChestNet parameters: {total_params2:,}")
    print(f"Reduction: {100 - (total_params2/total_params1*100):.1f}%")
    print("=" * 60)
    
    print("\n✓ Pengujian kedua model berhasil!")
    print("\nREKOMENDASI:")
    print("- Gunakan UltraFastChestNet untuk training tercepat")
    print("- Gunakan FastChestNet untuk akurasi lebih tinggi")