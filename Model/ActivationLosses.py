import torch
import torch.nn as nn
import torch.nn.functional as F


# Demonstration of why we separate model output from activation
class ModelWithSeparateActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 3)  # NO activation - raw logits
        )

    def forward(self, x):
        return self.network(x)  # Returns raw logits


# Create model and dummy data
model = ModelWithSeparateActivation()
x = torch.randn(2, 10)
raw_logits = model(x)

print("=== OUTPUT LAYER TECHNICAL ANALYSIS ===")
print(f"Raw logits: {raw_logits}")
print(f"Logit ranges: [{raw_logits.min():.3f}, {raw_logits.max():.3f}]")

# Different loss functions expect different things:
target_classification = torch.tensor([0, 2])  # Class indices
target_regression = torch.randn(2, 3)

print("\n=== LOSS FUNCTION COMPATIBILITY ===")

# 1. CrossEntropyLoss expects RAW LOGITS (applies softmax internally)
ce_loss = nn.CrossEntropyLoss()
loss_ce = ce_loss(raw_logits, target_classification)
print(f"CrossEntropyLoss with raw logits: {loss_ce:.4f}")

# If you applied softmax first, you'd get wrong results:
softmax_probs = F.softmax(raw_logits, dim=1)
# Don't do this! CrossEntropyLoss will apply softmax again = double softmax
print(f"Softmax probabilities: {softmax_probs}")
print("CrossEntropyLoss expects logits, not probabilities!")

# 2. BCELoss expects probabilities (0-1 range)
bce_loss = nn.BCELoss()
sigmoid_probs = torch.sigmoid(raw_logits)
binary_targets = torch.randint(0, 2, (2, 3)).float()
loss_bce = bce_loss(sigmoid_probs, binary_targets)
print(f"BCELoss with sigmoid probabilities: {loss_bce:.4f}")

# 3. BCEWithLogitsLoss expects RAW LOGITS (applies sigmoid internally)
bce_logits_loss = nn.BCEWithLogitsLoss()
loss_bce_logits = bce_logits_loss(raw_logits, binary_targets)
print(f"BCEWithLogitsLoss with raw logits: {loss_bce_logits:.4f}")

# 4. MSELoss for regression expects raw values
mse_loss = nn.MSELoss()
loss_mse = mse_loss(raw_logits, target_regression)
print(f"MSELoss with raw outputs: {loss_mse:.4f}")

print("\n=== NUMERICAL STABILITY BENEFIT ===")
# Raw logits provide better numerical stability
large_logits = torch.tensor([[10.0, 0.0, -10.0]])
print(f"Large logits: {large_logits}")

# Stable: log-sum-exp trick used internally by CrossEntropyLoss
stable_loss = F.cross_entropy(large_logits, torch.tensor([0]))
print(f"Stable CrossEntropyLoss: {stable_loss:.4f}")

# Unstable: manual softmax then log
manual_softmax = F.softmax(large_logits, dim=1)
manual_log_softmax = torch.log(manual_softmax)
print(f"Manual softmax probabilities: {manual_softmax}")
print("Notice potential numerical issues with extreme values!")
