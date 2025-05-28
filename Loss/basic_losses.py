import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 1. Manual MSE Implementation
def manual_mse_loss(predictions, targets):
    """
    Implement MSE loss from scratch
    MSE = (1/n) * Σ(y_pred - y_true)²
    """
    # Calculate squared differences
    squared_diff = (predictions - targets) ** 2

    # Take mean across all samples
    mse = torch.mean(squared_diff)

    return mse


# 2. Compare with PyTorch's built-in MSE
def compare_mse_implementations():
    # Create sample data
    predictions = torch.tensor([2.5, 0.0, 2.1, 7.8], requires_grad=True)
    targets = torch.tensor([3.0, -0.5, 2.0, 8.0])

    # Our implementation
    manual_loss = manual_mse_loss(predictions, targets)

    # PyTorch's implementation
    pytorch_mse = nn.MSELoss()
    pytorch_loss = pytorch_mse(predictions, targets)

    print(f"Manual MSE Loss: {manual_loss.item():.6f}")
    print(f"PyTorch MSE Loss: {pytorch_loss.item():.6f}")
    print(f"Difference: {abs(manual_loss.item() - pytorch_loss.item()):.10f}")

    return manual_loss, pytorch_loss


# 3. Understanding Gradients
def explore_gradients():
    """
    See how loss connects to optimization through gradients
    """
    # Simple linear model: y = w*x + b
    w = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)

    # Sample data
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_true = torch.tensor([3.0, 5.0, 7.0, 9.0])  # Perfect line: y = 2x + 1

    # Forward pass
    y_pred = w * x + b

    # Calculate loss
    loss = manual_mse_loss(y_pred, y_true)

    print(f"Initial loss: {loss.item():.6f}")
    print(f"Initial w: {w.item():.3f}, b: {b.item():.3f}")

    # Backward pass
    loss.backward()

    print(f"Gradient w.r.t w: {w.grad.item():.6f}")
    print(f"Gradient w.r.t b: {b.grad.item():.6f}")

    return w, b, loss


# 4. Visualizing Loss Landscape
def visualize_loss_landscape():
    """
    Plot how loss changes with different parameter values
    """
    # Fixed data
    x = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([2.0, 4.0, 6.0])  # y = 2x

    # Range of weight values to test
    w_values = np.linspace(0, 4, 100)
    losses = []

    for w in w_values:
        w_tensor = torch.tensor(w)
        y_pred = w_tensor * x
        loss = manual_mse_loss(y_pred, y_true)
        losses.append(loss.item())

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(w_values, losses, "b-", linewidth=2)
    plt.axvline(x=2.0, color="r", linestyle="--", label="True weight (w=2)")
    plt.xlabel("Weight Value (w)")
    plt.ylabel("MSE Loss")
    plt.title("Loss Landscape: How Loss Changes with Weight")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    return w_values, losses


if __name__ == "__main__":
    print("=== Comparing MSE Implementations ===")
    compare_mse_implementations()

    print("\n=== Exploring Gradients ===")
    explore_gradients()

    print("\n=== Visualizing Loss Landscape ===")
    visualize_loss_landscape()
