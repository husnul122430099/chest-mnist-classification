# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# =======================
# 🔸 Squeeze-and-Excitation Block
# =======================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_dim = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x)

# =======================
# 🔸 Bottleneck Block (MobileNetV2 style)
# =======================
class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, expansion=4, stride=1, use_se=True):
        super().__init__()
        hidden_ch = in_ch * expansion
        self.use_res = (in_ch == out_ch and stride == 1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, stride, 1, groups=hidden_ch, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.GELU(),
            SEBlock(hidden_ch) if use_se else nn.Identity(),
            nn.Conv2d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
    def forward(self, x):
        y = self.conv(x)
        return x + y if self.use_res else y

# =======================
# 🔸 EfficientChestNet (nama diubah untuk kompatibilitas)
# =======================
class EfficientChestNet(nn.Module):
    """Improved model with SE attention and residual bottlenecks."""
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        dims = [32, 64, 128, 256]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims[0]),
            nn.GELU()
        )

        self.stage1 = nn.Sequential(
            Bottleneck(dims[0], dims[0]),
            Bottleneck(dims[0], dims[1], stride=2)
        )
        self.stage2 = nn.Sequential(
            Bottleneck(dims[1], dims[1]),
            Bottleneck(dims[1], dims[2], stride=2)
        )
        self.stage3 = nn.Sequential(
            Bottleneck(dims[2], dims[2]),
            Bottleneck(dims[2], dims[3], stride=2),
            Bottleneck(dims[3], dims[3]),
            SEBlock(dims[3])
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(dims[3], 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1 if num_classes == 2 else num_classes)
        )

        self._init_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x.squeeze(1) if x.shape[1] == 1 else x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# =======================
# 🔸 Pengujian Cepat
# =======================
if __name__ == "__main__":
    print("="*70)
    print("Testing EfficientChestNet Model")
    print("="*70)
    
    model = EfficientChestNet(in_channels=1, num_classes=2)
    print("\nModel Architecture:")
    print(model)
    
    # Test forward pass
    x = torch.randn(16, 1, 28, 28)
    y = model(x)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print("\n" + "="*70)
    print("Model test completed successfully!")
    print("="*70)
