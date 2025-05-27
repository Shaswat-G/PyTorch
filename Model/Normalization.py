import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# BATCH NORMALIZATION - DETAILED IMPLEMENTATION AND ANALYSIS
# =============================================================================


class CustomBatchNorm1d(nn.Module):
    """Custom BatchNorm to understand the internals"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()

        # Learnable parameters (same as nn.BatchNorm1d)
        self.weight = nn.Parameter(torch.ones(num_features))  # gamma (scale)
        self.bias = nn.Parameter(torch.zeros(num_features))  # beta (shift)

        # Running statistics (not learnable, updated during training)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features

    def forward(self, x):
        # x shape: (batch_size, num_features) for 1D case

        if self.training:
            # Training mode: use batch statistics
            batch_mean = x.mean(dim=0, keepdim=True)  # Mean across batch dimension
            batch_var = x.var(
                dim=0, keepdim=True, unbiased=False
            )  # Variance across batch

            # Update running statistics with exponential moving average
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * batch_mean.squeeze()
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * batch_var.squeeze()

            # Normalize using batch statistics
            x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            # Evaluation mode: use running statistics
            x_normalized = (x - self.running_mean) / torch.sqrt(
                self.running_var + self.eps
            )

        # Scale and shift
        return x_normalized * self.weight + self.bias


class CustomLayerNorm(nn.Module):
    """Custom LayerNorm to understand the internals"""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(normalized_shape))  # gamma
        self.bias = nn.Parameter(torch.zeros(normalized_shape))  # beta

        self.eps = eps
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # x shape: (batch_size, ..., normalized_shape)
        # Normalize across the last dimension(s)

        mean = x.mean(dim=-1, keepdim=True)  # Mean across features
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # Variance across features

        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        return x_normalized * self.weight + self.bias


# =============================================================================
# COMPARATIVE ANALYSIS MODELS
# =============================================================================


class MLPWithBatchNorm(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(
                        hidden_size
                    ),  # BatchNorm after linear, before activation
                    nn.ReLU(),
                ]
            )
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLPWithLayerNorm(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.LayerNorm(
                        hidden_size
                    ),  # LayerNorm after linear, before activation
                    nn.ReLU(),
                ]
            )
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MLPWithoutNorm(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(prev_size, hidden_size), nn.ReLU()])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# =============================================================================
# NORMALIZATION ANALYSIS AND DEMONSTRATION
# =============================================================================


def analyze_normalization_effects():
    # Create test data with different batch sizes
    batch_sizes = [32, 8, 1]  # Different batch sizes to show BatchNorm sensitivity
    input_size, hidden_size, output_size = 100, 64, 10

    print("=== NORMALIZATION ANALYSIS ===")

    for batch_size in batch_sizes:
        print(f"\n--- Batch Size: {batch_size} ---")

        # Create models
        bn_model = MLPWithBatchNorm(input_size, [hidden_size], output_size)
        ln_model = MLPWithLayerNorm(input_size, [hidden_size], output_size)
        no_norm_model = MLPWithoutNorm(input_size, [hidden_size], output_size)

        # Create input with some distribution shift
        x = torch.randn(batch_size, input_size) * 2 + 1  # Mean=1, Std=2

        print(f"Input stats - Mean: {x.mean():.3f}, Std: {x.std():.3f}")

        # Training mode analysis
        for name, model in [
            ("No Norm", no_norm_model),
            ("BatchNorm", bn_model),
            ("LayerNorm", ln_model),
        ]:
            model.train()

            # Get intermediate activations (after first linear layer)
            if name == "No Norm":
                intermediate = model.network[0](x)  # First linear layer
            else:
                intermediate = model.network[1](
                    model.network[0](x)
                )  # After normalization

            print(
                f"{name:10} - Mean: {intermediate.mean().item():.3f}, Std: {intermediate.std().item():.3f}"
            )


def demonstrate_batchnorm_vs_layernorm():
    """Show the key difference in normalization dimensions"""
    batch_size, seq_len, features = 4, 6, 8

    # Create data: (batch_size, seq_len, features) - like NLP sequences
    x = torch.randn(batch_size, seq_len, features) * 3 + 2

    print("=== BATCH NORM vs LAYER NORM DIMENSIONS ===")
    print(f"Input shape: {x.shape}")
    print(f"Input mean: {x.mean():.3f}, std: {x.std():.3f}")

    # Reshape for BatchNorm1d (it expects 2D: batch_size, features)
    x_2d = x.view(-1, features)  # (batch_size * seq_len, features)

    # Apply BatchNorm1d
    bn = nn.BatchNorm1d(features)
    x_bn = bn(x_2d).view(batch_size, seq_len, features)

    # Apply LayerNorm
    ln = nn.LayerNorm(features)
    x_ln = ln(x)

    print(f"\nAfter BatchNorm - Mean: {x_bn.mean():.3f}, Std: {x_bn.std():.3f}")
    print(f"After LayerNorm - Mean: {x_ln.mean():.3f}, Std: {x_ln.std():.3f}")

    # Show per-sample statistics
    print(f"\nPer-sample analysis (first 2 samples):")
    for i in range(2):
        print(f"Sample {i+1}:")
        print(f"  Original    - Mean: {x[i].mean():.3f}, Std: {x[i].std():.3f}")
        print(f"  BatchNorm   - Mean: {x_bn[i].mean():.3f}, Std: {x_bn[i].std():.3f}")
        print(f"  LayerNorm   - Mean: {x_ln[i].mean():.3f}, Std: {x_ln[i].std():.3f}")


def analyze_parameters():
    """Analyze learnable parameters in normalization layers"""
    features = 128

    bn = nn.BatchNorm1d(features)
    ln = nn.LayerNorm(features)

    print("=== NORMALIZATION PARAMETERS ===")
    print(f"Feature dimension: {features}")

    print(f"\nBatchNorm1d parameters:")
    for name, param in bn.named_parameters():
        print(f"  {name}: {param.shape} - {param.numel()} parameters")

    print(f"\nBatchNorm1d buffers (non-learnable):")
    for name, buffer in bn.named_buffers():
        print(f"  {name}: {buffer.shape}")

    print(f"\nLayerNorm parameters:")
    for name, param in ln.named_parameters():
        print(f"  {name}: {param.shape} - {param.numel()} parameters")

    print(f"\nLayerNorm buffers:")
    print(
        f"  {len(list(ln.named_buffers()))} buffers (LayerNorm has no running statistics)"
    )


# Run demonstrations
if __name__ == "__main__":
    analyze_normalization_effects()
    print("\n" + "=" * 60 + "\n")
    demonstrate_batchnorm_vs_layernorm()
    print("\n" + "=" * 60 + "\n")
    analyze_parameters()
