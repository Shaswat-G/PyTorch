import torch
import torch.nn as nn
import torch.nn.functional as F


# Basic CNN

class BasicCNN(nn.Module):
    def __init__(self, num_classes : int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128*4*4, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool(x)

        x = torch.relu(self.conv2(x))
        x = self.pool(x)

        x = torch.relu(self.conv3(x))
        x = self.pool(x)

        # flatten
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# =============================================================================
# UNDERSTANDING CONVOLUTION PARAMETERS
# =============================================================================


def analyze_conv_parameters():
    """Understand how convolution parameters affect output size"""

    print("=== CONVOLUTION PARAMETER ANALYSIS ===")

    # Input tensor: (batch=1, channels=3, height=32, width=32)
    input_tensor = torch.randn(1, 3, 32, 32)
    print(f"Input shape: {input_tensor.shape}")

    # Different convolution configurations
    configs = [
        {"kernel_size": 3, "padding": 0, "stride": 1, "name": "3x3, no padding"},
        {"kernel_size": 3, "padding": 1, "stride": 1, "name": "3x3, padding=1"},
        {"kernel_size": 5, "padding": 2, "stride": 1, "name": "5x5, padding=2"},
        {"kernel_size": 3, "padding": 1, "stride": 2, "name": "3x3, stride=2"},
    ]

    for config in configs:
        conv = nn.Conv2d(3, 16, **{k: v for k, v in config.items() if k != "name"})
        output = conv(input_tensor)
        print(f"{config['name']:20} → {output.shape}")

    print(f"\nFormula: out_size = (in_size + 2*padding - kernel_size) / stride + 1")


# =============================================================================
# POOLING OPERATIONS
# =============================================================================


class PoolingComparison(nn.Module):
    """Compare different pooling operations"""

    def __init__(self):
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.adaptivepool = nn.AdaptiveAvgPool2d((1, 1))  # Always outputs 1x1

    def forward(self, x):
        print(f"Input shape: {x.shape}")

        max_out = self.maxpool(x)
        print(f"MaxPool2d output: {max_out.shape}")

        avg_out = self.avgpool(x)
        print(f"AvgPool2d output: {avg_out.shape}")

        adaptive_out = self.adaptivepool(x)
        print(f"AdaptiveAvgPool2d output: {adaptive_out.shape}")

        return max_out, avg_out, adaptive_out


## Modern cnn
# Plan: 3 pairs of 2 convolution blocks -> each block having conv2d, bn, relu. Each pair of conv blocks will be followed by maxpool2d, and one global pool, classifier fcn.

class ConvBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)

class ModernCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Features
        self.feature_extractor = nn.Sequential(
            ConvBlock(3, 64, 3, 1, 1),
            ConvBlock(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128, 3, 1, 1),
            ConvBlock(128, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256, 3, 1, 1),
            ConvBlock(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
        )

        # Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classify
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.global_pool(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x


# =============================================================================
# DEMONSTRATION
# =============================================================================


def demonstrate_cnn_basics():
    """Demonstrate CNN concepts with examples"""

    print("=== CNN BASIC DEMONSTRATION ===")

    # Create sample input (CIFAR-10 like)
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 32, 32)

    # Basic CNN
    basic_cnn = BasicCNN(num_classes=10)
    output = basic_cnn(input_tensor)

    print(f"Basic CNN:")
    print(f"  Input: {input_tensor.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in basic_cnn.parameters()):,}")

    # Modern CNN
    modern_cnn = ModernCNN(num_classes=10)
    output2 = modern_cnn(input_tensor)

    print(f"\nModern CNN:")
    print(f"  Input: {input_tensor.shape}")
    print(f"  Output: {output2.shape}")
    print(f"  Parameters: {sum(p.numel() for p in modern_cnn.parameters()):,}")

    # Analyze convolution parameters
    print("\n")
    analyze_conv_parameters()

    # Pooling comparison
    print(f"\n=== POOLING COMPARISON ===")
    pooling_demo = PoolingComparison()
    test_input = torch.randn(1, 64, 8, 8)
    pooling_demo(test_input)


if __name__ == "__main__":
    demonstrate_cnn_basics()
