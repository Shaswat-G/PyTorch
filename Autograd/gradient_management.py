import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict

print("=== OPTIMIZERS AND GRADIENT FLOW ===")


# Create a simple 2-parameter optimization problem
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor(-2.0))
        self.w2 = nn.Parameter(torch.tensor(3.0))

    def forward(self):
        # Loss surface: (w1-1)^2 + (w2+0.5)^2
        # Minimum at w1=1, w2=-0.5
        return (self.w1 - 1) ** 2 + (self.w2 + 0.5) ** 2


model = SimpleModel()
print(f"Initial parameters: w1={model.w1.item():.3f}, w2={model.w2.item():.3f}")
print(f"Initial loss: {model().item():.3f}")

print("\n=== COMPARING DIFFERENT OPTIMIZERS ===")


def optimize_with(optimizer_class, lr, num_steps=50, **kwargs):
    """Test different optimizers on the same problem"""
    model_copy = SimpleModel()
    with torch.no_grad():
        model_copy.w1.copy_(model.w1)
        model_copy.w2.copy_(model.w2)

    optimizer = optimizer_class(model_copy.parameters(), lr=lr, **kwargs)

    losses = []
    w1_history = []
    w2_history = []

    for step in range(num_steps):
        optimizer.zero_grad()  # CRITICAL: Clear accumulated gradients

        loss = model_copy()
        loss.backward()

        # Store values before optimizer step
        losses.append(loss.item())
        w1_history.append(model_copy.w1.item())
        w2_history.append(model_copy.w2.item())

        optimizer.step()

    return losses, w1_history, w2_history


# Test different optimizers
optimizers_to_test = [
    (optim.SGD, {"lr": 0.1}, "SGD"),
    (optim.Adam, {"lr": 0.1}, "Adam"),
    (optim.RMSprop, {"lr": 0.1}, "RMSprop"),
    (optim.SGD, {"lr": 0.1, "momentum": 0.9}, "SGD+Momentum"),
]

results = {}
for opt_class, params, name in optimizers_to_test:
    losses, w1_hist, w2_hist = optimize_with(opt_class, **params)
    results[name] = {"losses": losses, "w1": w1_hist, "w2": w2_hist}
    print(
        f"{name}: Final loss = {losses[-1]:.6f}, Final w1 = {w1_hist[-1]:.3f}, Final w2 = {w2_hist[-1]:.3f}"
    )

print("\n=== GRADIENT ACCUMULATION FOR LARGE EFFECTIVE BATCH SIZE ===")


class DataGenerator:
    def __init__(self, batch_size=8, feature_dim=20):
        self.batch_size = batch_size
        self.feature_dim = feature_dim

    def __iter__(self):
        while True:
            X = torch.randn(self.batch_size, self.feature_dim)
            y = torch.randint(0, 2, (self.batch_size,))
            yield X, y


# Create a simple classifier
classifier = nn.Sequential(nn.Linear(20, 10), nn.ReLU(), nn.Linear(10, 2))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

data_gen = DataGenerator()
data_iter = iter(data_gen)

print("Strategy 1: Normal training (batch_size=8)")


def normal_training_step():
    X, y = next(data_iter)

    optimizer.zero_grad()
    outputs = classifier(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    return loss.item()


print("Strategy 2: Gradient accumulation (effective_batch_size=32)")


def gradient_accumulation_step(accumulation_steps=4):
    optimizer.zero_grad()  # Clear gradients once

    total_loss = 0
    for i in range(accumulation_steps):
        X, y = next(data_iter)

        outputs = classifier(X)
        loss = criterion(outputs, y) / accumulation_steps  # Scale loss
        loss.backward()  # Accumulate gradients

        total_loss += loss.item()

    optimizer.step()  # Update once with accumulated gradients
    return total_loss


# Compare the approaches
normal_loss = normal_training_step()
accum_loss = gradient_accumulation_step()

print(f"Normal training loss: {normal_loss:.4f}")
print(f"Gradient accumulation loss: {accum_loss:.4f}")

print("\n=== LEARNING RATE SCHEDULING ===")

# Reset model
model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Different scheduling strategies
schedulers = {
    "StepLR": optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5),
    "ExponentialLR": optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20),
}

scheduler = schedulers["StepLR"]
lr_history = []

for epoch in range(30):
    optimizer.zero_grad()
    loss = model()
    loss.backward()
    optimizer.step()
    scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]
    lr_history.append(current_lr)

    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}, LR = {current_lr:.6f}")

print("\n=== GRADIENT CLIPPING ===")


class ExplodingGradientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
        # Initialize with large weights to cause gradient explosion
        nn.init.constant_(self.linear.weight, 10.0)

    def forward(self, x):
        return self.linear(x)


exploding_model = ExplodingGradientModel()
x_large = torch.randn(1, 10) * 100  # Large input
y_target = torch.tensor([0.0])

optimizer_clip = optim.SGD(exploding_model.parameters(), lr=0.01)

print("Without gradient clipping:")
optimizer_clip.zero_grad()
output = exploding_model(x_large)
loss = nn.MSELoss()(output, y_target)
loss.backward()

print(
    f"Gradient norm before clipping: {torch.nn.utils.clip_grad_norm_(exploding_model.parameters(), float('inf')):.2f}"
)

print("With gradient clipping:")
optimizer_clip.zero_grad()
output = exploding_model(x_large)
loss = nn.MSELoss()(output, y_target)
loss.backward()

# Clip gradients to max norm of 1.0
torch.nn.utils.clip_grad_norm_(exploding_model.parameters(), max_norm=1.0)
print(
    f"Gradient norm after clipping: {torch.nn.utils.clip_grad_norm_(exploding_model.parameters(), float('inf')):.2f}"
)

print("\n=== COMPLETE TRAINING LOOP TEMPLATE ===")


def train_model(model, train_loader, val_loader, epochs=10):
    """Production-ready training loop template"""

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()

        # Record losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

    return train_losses, val_losses


print("Key components of the training loop:")
print("1. optimizer.zero_grad() - Clear accumulated gradients")
print("2. loss.backward() - Compute gradients via backprop")
print("3. optimizer.step() - Update parameters using gradients")
print("4. model.train()/model.eval() - Set appropriate mode")
print("5. torch.no_grad() during validation - Save memory")
print("6. Gradient clipping - Prevent exploding gradients")
print("7. Learning rate scheduling - Adaptive learning rates")

print("\n=== DEBUGGING GRADIENTS ===")


def check_gradients(model, max_norm_threshold=10.0):
    """Utility function to debug gradient issues"""
    total_norm = 0
    param_count = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1

            if param_norm > max_norm_threshold:
                print(f"WARNING: Large gradient in {name}: {param_norm:.4f}")
        else:
            print(f"WARNING: No gradient for {name}")

    total_norm = total_norm ** (1.0 / 2)
    print(f"Total gradient norm: {total_norm:.4f}")
    return total_norm


# Example usage
simple_model = nn.Linear(5, 1)
x = torch.randn(3, 5)
y = torch.randn(3, 1)
loss = nn.MSELoss()(simple_model(x), y)
loss.backward()

check_gradients(simple_model)
