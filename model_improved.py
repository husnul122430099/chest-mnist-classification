import torch
import torch.nn as nn
import torch.nn.functional as F

class _DenseLayer(nn.Module):
    """Dense Layer untuk DenseNet"""
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                              kernel_size=1, stride=1, bias=False)
        
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                              kernel_size=3, stride=1, padding=1, bias=False)
        
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Module):
    """Dense Block yang berisi beberapa Dense Layers"""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class _Transition(nn.Module):
    """Transition layer antara Dense Blocks"""
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                             kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        x = self.pool(x)
        return x


class DenseNet121(nn.Module):
    """
    DenseNet121 yang dimodifikasi untuk ChestMNIST (28x28 grayscale)
    Target: Val Acc > 92%
    
    Arsitektur:
    - Growth rate: 32
    - Block config: [6, 12, 24, 16] (DenseNet-121)
    - Compression: 0.5
    """
    def __init__(self, in_channels=1, num_classes=2, growth_rate=32, 
                 block_config=(6, 12, 24, 16), compression=0.5, drop_rate=0.2):
        super().__init__()
        
        # Initial convolution (28x28 -> 14x14)
        num_init_features = 64
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, num_init_features, kernel_size=3, 
                     stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Dense Blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=4,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            # Transition layer (kecuali setelah dense block terakhir)
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=int(num_features * compression)
                )
                self.features.add_module(f'transition{i+1}', trans)
                num_features = int(num_features * compression)
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        
        # Global Average Pooling + Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, 1 if num_classes == 2 else num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inisialisasi weights dengan Kaiming/Xavier"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class ImprovedChestNet(nn.Module):
    """Alias untuk backward compatibility"""
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.model = DenseNet121(in_channels=in_channels, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    NUM_CLASSES = 2
    IN_CHANNELS = 1
    
    print("="*70)
    print("TESTING DenseNet121 for ChestMNIST")
    print("="*70)
    
    model = DenseNet121(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)
    print("\nModel Architecture:")
    print(model)
    
    # Test forward pass
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
    
    print("\n✓ DenseNet121 model test successful!")
    print("\nKey Features:")
    print("- Growth rate: 32")
    print("- Block config: [6, 12, 24, 16] (DenseNet-121)")
    print("- Compression: 0.5")
    print("- Dropout: 0.2")
    print("- Target: Val Acc > 92%")