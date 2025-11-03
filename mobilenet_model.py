# mobilenet_model.py
"""
Advanced MobileNet models untuk ChestMNIST Classification
Optimized untuk mencapai validation accuracy > 92% dengan training cepat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution - Efisien komputasi"""
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


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block - Channel Attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module - Spatial + Channel Attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa_input)
        x = x * sa
        
        return x


class InvertedResidual(nn.Module):
    """Inverted Residual Block dengan SE/CBAM"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio, use_cbam=False):
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
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])
        
        self.conv = nn.Sequential(*layers)
        
        # Attention module
        if use_cbam:
            self.attention = CBAM(hidden_dim, reduction=4)
        else:
            self.attention = SEBlock(hidden_dim, reduction=4)
        
        # Pointwise
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        out = self.conv(x)
        out = self.attention(out)
        out = self.project(out)
        
        if self.use_residual:
            return x + out
        else:
            return out


class SuperMobileNet(nn.Module):
    """
    Super MobileNet - Model terbaik untuk 92%+ accuracy
    Features:
    - Very Deep architecture (15+ blocks)
    - CBAM attention mechanism
    - Progressive channel expansion
    - Strong regularization
    """
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        
        # Stem: 28x28 -> 28x28
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.Conv2d(32, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # Stage 1: 28x28 -> 28x28
        self.stage1 = nn.Sequential(
            InvertedResidual(32, 24, 1, expand_ratio=1, use_cbam=False),
            InvertedResidual(24, 24, 1, expand_ratio=6, use_cbam=True),
        )
        
        # Stage 2: 28x28 -> 14x14
        self.stage2 = nn.Sequential(
            InvertedResidual(24, 32, 2, expand_ratio=6, use_cbam=True),
            InvertedResidual(32, 32, 1, expand_ratio=6, use_cbam=True),
            InvertedResidual(32, 32, 1, expand_ratio=6, use_cbam=True),
            InvertedResidual(32, 32, 1, expand_ratio=6, use_cbam=True),
        )
        
        # Stage 3: 14x14 -> 7x7
        self.stage3 = nn.Sequential(
            InvertedResidual(32, 64, 2, expand_ratio=6, use_cbam=True),
            InvertedResidual(64, 64, 1, expand_ratio=6, use_cbam=True),
            InvertedResidual(64, 64, 1, expand_ratio=6, use_cbam=True),
            InvertedResidual(64, 64, 1, expand_ratio=6, use_cbam=True),
            InvertedResidual(64, 64, 1, expand_ratio=6, use_cbam=True),
        )
        
        # Stage 4: 7x7 -> 4x4
        self.stage4 = nn.Sequential(
            InvertedResidual(64, 96, 2, expand_ratio=6, use_cbam=True),
            InvertedResidual(96, 96, 1, expand_ratio=6, use_cbam=True),
            InvertedResidual(96, 96, 1, expand_ratio=6, use_cbam=True),
            InvertedResidual(96, 96, 1, expand_ratio=6, use_cbam=True),
        )
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(96, 320, 1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU6(inplace=True)
        )
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(320, 160),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(160, 80),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(80, 1 if num_classes == 2 else num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class EnhancedMobileNet(nn.Module):
    """
    Enhanced MobileNet - Balanced model
    Target: 91-94% accuracy
    """
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        
        # Initial: 28x28 -> 28x28
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # Blocks with attention
        self.blocks = nn.Sequential(
            # Stage 1: 28x28
            InvertedResidual(32, 16, 1, expand_ratio=1, use_cbam=False),
            
            # Stage 2: 28x28 -> 14x14
            InvertedResidual(16, 24, 2, expand_ratio=6, use_cbam=True),
            InvertedResidual(24, 24, 1, expand_ratio=6, use_cbam=True),
            InvertedResidual(24, 24, 1, expand_ratio=6, use_cbam=True),
            
            # Stage 3: 14x14 -> 7x7
            InvertedResidual(24, 40, 2, expand_ratio=6, use_cbam=True),
            InvertedResidual(40, 40, 1, expand_ratio=6, use_cbam=True),
            InvertedResidual(40, 40, 1, expand_ratio=6, use_cbam=True),
            InvertedResidual(40, 40, 1, expand_ratio=6, use_cbam=True),
            
            # Stage 4: 7x7 -> 3x3
            InvertedResidual(40, 80, 2, expand_ratio=6, use_cbam=True),
            InvertedResidual(80, 80, 1, expand_ratio=6, use_cbam=True),
            InvertedResidual(80, 80, 1, expand_ratio=6, use_cbam=True),
        )
        
        # Final
        self.conv2 = nn.Sequential(
            nn.Conv2d(80, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True)
        )
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 1 if num_classes == 2 else num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class FastChestNet(nn.Module):
    """
    Fast ChestNet - Quick training
    Target: 89-92% accuracy
    """
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        self.blocks = nn.Sequential(
            InvertedResidual(32, 16, 1, 1, use_cbam=False),
            InvertedResidual(16, 24, 2, 6, use_cbam=True),
            InvertedResidual(24, 24, 1, 6, use_cbam=True),
            InvertedResidual(24, 40, 2, 6, use_cbam=True),
            InvertedResidual(40, 40, 1, 6, use_cbam=True),
            InvertedResidual(40, 40, 1, 6, use_cbam=True),
            InvertedResidual(40, 64, 2, 6, use_cbam=True),
            InvertedResidual(64, 64, 1, 6, use_cbam=True),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU6(inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(192, 96),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(96, 1 if num_classes == 2 else num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# Test module
if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    BATCH_SIZE = 32
    
    print("=" * 80)
    print("TESTING NEW MOBILENET MODELS")
    print("=" * 80)
    
    # Test SuperMobileNet
    print("\n1. SuperMobileNet (BEST - Target 92-95%)")
    print("-" * 80)
    model1 = SuperMobileNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    dummy_input = torch.randn(BATCH_SIZE, IN_CHANNELS, 28, 28)
    output1 = model1(dummy_input)
    params1 = sum(p.numel() for p in model1.parameters())
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output1.shape}")
    print(f"Parameters:   {params1:,}")
    
    # Test EnhancedMobileNet
    print("\n2. EnhancedMobileNet (BALANCED - Target 91-94%)")
    print("-" * 80)
    model2 = EnhancedMobileNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    output2 = model2(dummy_input)
    params2 = sum(p.numel() for p in model2.parameters())
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output2.shape}")
    print(f"Parameters:   {params2:,}")
    
    # Test FastChestNet
    print("\n3. FastChestNet (FAST - Target 89-92%)")
    print("-" * 80)
    model3 = FastChestNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    output3 = model3(dummy_input)
    params3 = sum(p.numel() for p in model3.parameters())
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output3.shape}")
    print(f"Parameters:   {params3:,}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Model':<25} {'Parameters':>15} {'Target Accuracy':>20} {'Speed':>15}")
    print("-" * 80)
    print(f"{'SuperMobileNet':<25} {params1:>15,} {'92-95%':>20} {'Medium':>15}")
    print(f"{'EnhancedMobileNet':<25} {params2:>15,} {'91-94%':>20} {'Fast':>15}")
    print(f"{'FastChestNet':<25} {params3:>15,} {'89-92%':>20} {'Very Fast':>15}")
    print("=" * 80)
    
    print("\n✓ All models tested successfully!")
    print("\nRECOMMENDATION:")
    print("→ Use SuperMobileNet for BEST accuracy (92%+)")
    print("→ Use EnhancedMobileNet for BALANCE")
    print("→ Use FastChestNet for SPEED")