import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 1. Binary Classification: Why not MSE?
def demonstrate_mse_vs_crossentropy():
    """
    Show why MSE fails for classification while Cross-Entropy succeeds
    """
    # Binary classification setup
    # True labels: 0 or 1
    true_labels = torch.tensor([0, 1, 1, 0, 1], dtype=torch.float32)

    # Model outputs (logits before sigmoid)
    logits = torch.tensor([-2.0, 3.0, 1.0, -1.0, 2.0], requires_grad=True)

    # Convert logits to probabilities
    probabilities = torch.sigmoid(logits)

    print("=== Model Outputs ===")
    print(f"Logits: {logits.detach().numpy()}")
    print(f"Probabilities: {probabilities.detach().numpy()}")
    print(f"True Labels: {true_labels.numpy()}")

    # MSE Loss (treating classification as regression)
    mse_loss = nn.MSELoss()(probabilities, true_labels)

    # Cross-Entropy Loss (proper classification loss)
    ce_loss = nn.BCEWithLogitsLoss()(logits, true_labels)

    print(f"\n=== Loss Comparison ===")
    print(f"MSE Loss: {mse_loss.item():.4f}")
    print(f"Cross-Entropy Loss: {ce_loss.item():.4f}")

    # Compute gradients
    mse_loss.backward(retain_graph=True)
    mse_grads = logits.grad.clone()

    logits.grad.zero_()  # Reset gradients
    ce_loss.backward()
    ce_grads = logits.grad.clone()

    print(f"\n=== Gradient Analysis ===")
    print(f"MSE Gradients: {mse_grads.numpy()}")
    print(f"CE Gradients: {ce_grads.numpy()}")

    return mse_loss, ce_loss, mse_grads, ce_grads


# 2. Multi-class Classification
def multiclass_crossentropy_deep_dive():
    """
    Implement and understand multi-class cross-entropy
    """
    # 3-class classification example
    batch_size, num_classes = 4, 3

    # Raw logits from model
    logits = torch.tensor(
        [
            [2.0, 1.0, 0.1],  # Confident about class 0
            [0.5, 2.5, 0.3],  # Confident about class 1
            [0.1, 0.2, 3.0],  # Confident about class 2
            [1.0, 1.1, 1.2],  # Uncertain between all classes
        ],
        requires_grad=True,
    )

    # True class indices
    targets = torch.tensor([0, 1, 2, 2])

    print("=== Multi-class Setup ===")
    print(f"Logits shape: {logits.shape}")
    print(f"Logits:\n{logits.detach().numpy()}")
    print(f"True classes: {targets.numpy()}")

    # Manual cross-entropy implementation
    def manual_crossentropy(logits, targets):
        # Step 1: Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)

        # Step 2: Select probabilities for true classes
        true_class_probs = probs[torch.arange(len(targets)), targets]

        # Step 3: Take negative log and average
        loss = -torch.mean(torch.log(true_class_probs))

        return loss, probs

    # Compare implementations
    manual_loss, probs = manual_crossentropy(logits, targets)
    pytorch_loss = nn.CrossEntropyLoss()(logits, targets)

    print(f"\n=== Probability Distribution ===")
    print(f"Softmax probabilities:\n{probs.detach().numpy()}")
    print(f"Manual CE Loss: {manual_loss.item():.4f}")
    print(f"PyTorch CE Loss: {pytorch_loss.item():.4f}")

    return logits, targets, manual_loss, pytorch_loss


# 3. Understanding Loss Behavior: Confidence vs Uncertainty
def analyze_loss_behavior():
    """
    See how different confidence levels affect loss
    """
    # Binary classification scenarios
    scenarios = [
        ("Very Confident Correct", torch.tensor([5.0]), torch.tensor([1.0])),
        ("Slightly Confident Correct", torch.tensor([0.5]), torch.tensor([1.0])),
        ("Uncertain", torch.tensor([0.0]), torch.tensor([1.0])),
        ("Slightly Confident Wrong", torch.tensor([-0.5]), torch.tensor([1.0])),
        ("Very Confident Wrong", torch.tensor([-5.0]), torch.tensor([1.0])),
    ]

    print("=== Loss Behavior Analysis ===")
    for name, logit, target in scenarios:
        prob = torch.sigmoid(logit)
        ce_loss = nn.BCEWithLogitsLoss()(logit, target)

        print(
            f"{name:25} | Logit: {logit.item():+5.1f} | "
            f"Prob: {prob.item():.3f} | Loss: {ce_loss.item():.3f}"
        )


# 4. The Non-Convex Reality: Neural Network Loss Landscapes
def neural_network_loss_landscape():
    """
    Show how neural networks create non-convex loss landscapes
    """

    # Simple 2-layer network
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(1, 2, bias=False)
            self.layer2 = nn.Linear(2, 1, bias=False)

        def forward(self, x):
            return self.layer2(torch.relu(self.layer1(x)))

    # Fixed dataset
    x = torch.tensor([[1.0], [2.0], [3.0]])
    y = torch.tensor([[2.0], [4.0], [6.0]])  # y = 2x

    # Create network
    net = TinyNet()

    # Manually set weights to create loss landscape
    w1_range = np.linspace(-3, 3, 50)
    w2_range = np.linspace(-3, 3, 50)

    loss_surface = np.zeros((len(w1_range), len(w2_range)))

    for i, w1 in enumerate(w1_range):
        for j, w2 in enumerate(w2_range):
            # Set weights
            with torch.no_grad():
                net.layer1.weight[0, 0] = w1
                net.layer1.weight[1, 0] = w1
                net.layer2.weight[0, 0] = w2
                net.layer2.weight[0, 1] = w2

            # Forward pass and loss
            pred = net(x)
            loss = nn.MSELoss()(pred, y)
            loss_surface[i, j] = loss.item()

    # Plot the non-convex landscape
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.contour(w1_range, w2_range, loss_surface.T, levels=20)
    plt.colorbar(label="Loss")
    plt.xlabel("Weight 1")
    plt.ylabel("Weight 2")
    plt.title("Neural Network Loss Landscape (Non-Convex)")

    plt.subplot(1, 2, 2)
    plt.imshow(loss_surface.T, extent=[-3, 3, -3, 3], origin="lower", cmap="viridis")
    plt.colorbar(label="Loss")
    plt.xlabel("Weight 1")
    plt.ylabel("Weight 2")
    plt.title("Loss Heatmap")

    plt.tight_layout()
    plt.show()

    return loss_surface


if __name__ == "__main__":
    print("1. MSE vs Cross-Entropy for Classification")
    demonstrate_mse_vs_crossentropy()

    print("\n" + "=" * 60 + "\n")

    print("2. Multi-class Cross-Entropy Deep Dive")
    multiclass_crossentropy_deep_dive()

    print("\n" + "=" * 60 + "\n")

    print("3. Loss Behavior Analysis")
    analyze_loss_behavior()

    print("\n" + "=" * 60 + "\n")

    print("4. Neural Network Loss Landscape")
    neural_network_loss_landscape()
