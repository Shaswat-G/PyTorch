import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convblock(x)
        return x


class ModernCNN(nn.Module):
    def __init__(self, num_classes: int, channel_dimension: int = 3):
        super().__init__()

        # Extractor
        self.extractor = nn.Sequential(
            ConvBlock(channel_dimension, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2, 2),
        )

        # Pool - Global AVg pooling to summarize HxW to make it agnostic to input image size and flatten to B x C.
        self.pool = nn.AdaptiveAvgPool2d(1)

        # MLP Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extractor(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x


# Simple Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # Reduce channels to 1 for attention map
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),  # Channel reduction
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, 1),  # To single channel
            nn.Sigmoid(),  # Attention weights [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_map = self.attention(x)  # Shape: [B, 1, H, W]
        return x * attention_map  # Element-wise multiplication


class ModernCNNWithAttention(nn.Module):
    def __init__(self, num_classes: int, channel_dimension: int = 3):
        super().__init__()

        # Extractor (same as your original)
        self.extractor = nn.Sequential(
            ConvBlock(channel_dimension, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2, 2),
        )

        # Add attention after feature extraction
        self.attention = SpatialAttention(256)

        # Pool and classifier (same as your original)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features
        x = self.extractor(x)

        # Apply attention
        x = self.attention(x)

        # Pool and classify
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

    def get_attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """Get the attention map for visualization"""
        features = self.extractor(x)
        attention_map = self.attention.attention(features)
        return attention_map

    # Training paradigm methods
    def freeze_extractor(self):
        """Freeze the feature extractor for transfer learning"""
        for param in self.extractor.parameters():
            param.requires_grad = False

    def unfreeze_extractor(self):
        """Unfreeze the feature extractor for fine-tuning"""
        for param in self.extractor.parameters():
            param.requires_grad = True

    def freeze_classifier(self):
        """Freeze the classifier"""
        for param in self.classifier.parameters():
            param.requires_grad = False

    def unfreeze_classifier(self):
        """Unfreeze the classifier"""
        for param in self.classifier.parameters():
            param.requires_grad = True

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the extractor (before pooling)"""
        return self.extractor(x)

    def get_pooled_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled features (latent representation)"""
        features = self.extractor(x)
        pooled = self.pool(features)
        return pooled.flatten(start_dim=1)

    def extract_pretrained_extractor(self):
        """Surgically extract the extractor for use in other models"""
        return self.extractor


# Test the models
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create models
    model = ModernCNN(num_classes=10).to(device)
    attention_model = ModernCNNWithAttention(num_classes=10).to(device)

    # Test input
    x = torch.randn(4, 3, 224, 224, device=device)

    print("🧪 TESTING FUNCTIONALITY")
    print("-" * 40)

    # Test forward pass
    with torch.no_grad():
        output = model(x)
        attention_output = attention_model(x)
        print(f"Original model output: {output.shape}")
        print(f"Attention model output: {attention_output.shape}")

    # Test training paradigms
    print("\n🎯 TRAINING PARADIGMS")
    print("-" * 40)

    # 1. Feature extraction (frozen extractor)
    model.freeze_extractor()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen extractor - Trainable params: {trainable_params:,}")

    # 2. Fine-tuning (unfreeze everything)
    model.unfreeze_extractor()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Full fine-tuning - Trainable params: {trainable_params:,}")

    # Test feature extraction
    with torch.no_grad():
        features = model.get_features(x)
        pooled = model.get_pooled_features(x)
        print(f"Feature maps: {features.shape}")
        print(f"Pooled features: {pooled.shape}")

    # Test attention visualization
    with torch.no_grad():
        attention_map = attention_model.get_attention_map(x)
        print(f"Attention map: {attention_map.shape}")

    print("\n✅ All tests passed!")
