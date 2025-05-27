import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple


# Basic MLP


class BasicMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return x


# Sequential MLP
class SequentialMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# Flexible MLP
class FlexibleMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation="relu",
        dropout_prob=0.0,
    ) -> None:
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))

            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            elif activation.lower() == "tanh":
                layers.append(nn.Tanh())
            elif activation.lower() == "sigmoid":
                layers.append(nn.Sigmoid())

            if dropout_prob > 0:
                layers.append(nn.Dropout(p=dropout_prob))

            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class AdvancedMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation="relu",
        weight_init="xavier",
        dropout_prob=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation.lower()
        self.weight_init = weight_init.lower()
        self.dropout_prob = dropout_prob

        prev_size = input_size

        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        self.output_layer = nn.Linear(prev_size, output_size)

        if self.dropout_prob > 0:
            self.dropout_layer = nn.Dropout(p=self.dropout_prob)
        else:
            self.dropout_layer = None

        self._init_weight()

    def _init_weight(self):
        for layer in self.layers:
            if self.weight_init == "xavier":
                nn.init.xavier_uniform_(layer.weight)
            elif self.weight_init == "he":
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            elif self.weight_init == "normal":
                nn.init.normal_(layer.weight, mean=0, std=0.01)

            nn.init.zeros_(layer.bias)

        if self.weight_init == "xavier":
            nn.init.xavier_uniform_(self.output_layer.weight)
        elif self.weight_init == "he":
            nn.init.kaiming_uniform_(self.output_layer.weight, nonlinearity="relu")
        elif self.weight_init == "normal":
            nn.init.normal_(self.output_layer.weight, mean=0, std=0.01)

        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # Apply activation
            if self.activation == "relu":
                x = torch.relu(x)
            elif self.activation == "tanh":
                x = torch.tanh(x)
            elif self.activation == "sigmoid":
                x = torch.sigmoid(x)

            if self.dropout_layer:
                x = self.dropout_layer(x)

        x = self.output_layer(x)
        return x


def compare_models():
    input_size, output_size = 784, 10  # MNIST-like dimensions
    hidden_size = 128
    hidden_sizes = [128, 64, 32]

    # Create different models
    basic = BasicMLP(input_size, hidden_size, output_size)
    sequential = SequentialMLP(input_size, hidden_size, output_size)
    flexible = FlexibleMLP(input_size, hidden_sizes, output_size, dropout_prob=0.2)
    advanced = AdvancedMLP(input_size, hidden_sizes, output_size, weight_init="xavier")

    # Test with dummy data
    x = torch.randn(32, input_size)  # batch_size=32

    print("Model Comparison:")
    print(f"Input shape: {x.shape}")

    for name, model in [
        ("Basic", basic),
        ("Sequential", sequential),
        ("Flexible", flexible),
        ("Advanced", advanced),
    ]:
        output = model(x)
        num_params = sum(p.numel() for p in model.parameters())
        layer_1_params = model.get_first_layer_params()
        print(
            f"{name:10} - Output: {output.shape}, Parameters: {num_params:,}, Layer 1 Params: {layer_1_params}"
        )


if __name__ == "__main__":
    compare_models()
