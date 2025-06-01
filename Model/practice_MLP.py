import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# --- Model Class


class FlexMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout_p: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super().__init__()  # Critical: Must call parent constructor

        layers = []
        prev_size = input_size

        # Build hidden layers
        for hidden_size in hidden_sizes:
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))

            # Optional batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))  # Must specify num_features

            # Activation and dropout
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_p))

            prev_size = hidden_size

        # Output layer (no activation - let loss function handle it)
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_num_parameters(self) -> int:
        """Utility method to count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --- Alternative Implementation using nn.ModuleList ---


class FlexMLPModuleList(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout_p: float = 0.2,
        use_batch_norm: bool = True,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()

        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        self.activation = activation

        # Store all layer dimensions
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        # Create linear layers using ModuleList
        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(layer_sizes[i], layer_sizes[i + 1])
                for i in range(len(layer_sizes) - 1)
            ]
        )

        # Create batch norm layers (only for hidden layers)
        if use_batch_norm and hidden_sizes:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(size) for size in hidden_sizes]
            )
        else:
            self.batch_norm_layers = None

        # Create dropout layers
        self.dropout_layers = nn.ModuleList(
            [nn.Dropout(p=dropout_p) for _ in range(len(hidden_sizes))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through hidden layers
        for i, linear_layer in enumerate(
            self.linear_layers[:-1]
        ):  # All except output layer
            x = linear_layer(x)

            # Apply batch norm if enabled
            if self.batch_norm_layers is not None:
                x = self.batch_norm_layers[i](x)

            # Apply activation
            x = self.activation(x)

            # Apply dropout
            x = self.dropout_layers[i](x)

        # Output layer (no activation, batch norm, or dropout)
        x = self.linear_layers[-1](x)
        return x

    def get_layer_info(self) -> str:
        """Utility method to inspect layer architecture"""
        info = []
        for i, layer in enumerate(self.linear_layers):
            layer_type = "Hidden" if i < len(self.linear_layers) - 1 else "Output"
            info.append(f"{layer_type} Layer {i}: {layer.in_features} → {layer.out_features}")
        return "\n".join(info)

    def get_num_parameters(self) -> int:
        """Utility method to count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_layer(self, layer_index: int):
        """Freeze a specific linear layer"""
        if 0 <= layer_index < len(self.linear_layers):
            for param in self.linear_layers[layer_index].parameters():
                param.requires_grad = False

    def unfreeze_layer(self, layer_index: int):
        """Unfreeze a specific linear layer"""
        if 0 <= layer_index < len(self.linear_layers):
            for param in self.linear_layers[layer_index].parameters():
                param.requires_grad = True


# Test both implementations
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("=" * 60)

    # Test 1: Compare Sequential vs ModuleList implementations
    print("COMPARISON: Sequential vs ModuleList")
    print("-" * 40)

    # Create both models with same architecture
    hidden_sizes = [512, 256, 128]
    model_seq = FlexMLP(input_size=784, hidden_sizes=hidden_sizes, output_size=10).to(device)
    model_list = FlexMLPModuleList(input_size=784, hidden_sizes=hidden_sizes, output_size=10).to(device)

    print(f"Sequential Model Parameters: {model_seq.get_num_parameters():,}")
    print(f"ModuleList Model Parameters: {model_list.get_num_parameters():,}")

    # Test with same input
    x = torch.randn(32, 784, device=device)

    # Compare outputs (should be different due to random initialization)
    with torch.no_grad():
        out_seq = model_seq(x)
        out_list = model_list(x)

    print(f"Sequential output shape: {out_seq.shape}")
    print(f"ModuleList output shape: {out_list.shape}")
    print(f"Output shapes match: {out_seq.shape == out_list.shape}")

    print("\n" + "=" * 60)

    # Test 2: ModuleList-specific features
    print("MODULELIST-SPECIFIC FEATURES")
    print("-" * 40)

    print("Layer Architecture:")
    print(model_list.get_layer_info())

    # Test layer freezing
    print(f"\nBefore freezing - Trainable params: {model_list.get_num_parameters():,}")
    model_list.freeze_layer(0)  # Freeze first layer
    trainable_after_freeze = sum(p.numel() for p in model_list.parameters() if p.requires_grad)
    print(f"After freezing layer 0 - Trainable params: {trainable_after_freeze:,}")

    # Test layer access
    print(f"\nFirst linear layer: {model_list.linear_layers[0]}")
    print(f"Number of linear layers: {len(model_list.linear_layers)}")
    print(f"Number of dropout layers: {len(model_list.dropout_layers)}")

    print("\n" + "=" * 60)

    # Test 3: Edge cases
    print("EDGE CASE TESTING")
    print("-" * 40)

    # No hidden layers
    simple_model = FlexMLPModuleList(input_size=10, hidden_sizes=[], output_size=2).to(device)
    simple_input = torch.randn(5, 10, device=device)
    simple_output = simple_model(simple_input)
    print(f"No hidden layers - Input: {simple_input.shape}, Output: {simple_output.shape}")

    # Single hidden layer
    single_model = FlexMLPModuleList(input_size=784, hidden_sizes=[128], output_size=10).to(device)
    single_output = single_model(x)
    print(f"Single hidden layer - Output: {single_output.shape}")

    # Custom activation
    custom_model = FlexMLPModuleList(input_size=784, hidden_sizes=[256], output_size=10, activation=nn.LeakyReLU(0.1)).to(device)
    custom_output = custom_model(x)
    print(f"Custom activation (LeakyReLU) - Output: {custom_output.shape}")

    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
