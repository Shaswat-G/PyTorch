import torch
import torch.nn as nn
import torch.nn.functional as F

"""
QUICK REFERENCE: WHEN TO USE WHICH LOSS

REGRESSION:
- MSE: Standard regression, outliers not a big issue
- MAE: Robust to outliers, want median-like behavior
- Huber: Compromise between MSE and MAE
- Custom: Business-specific penalties (asymmetric costs)

CLASSIFICATION:
- CrossEntropy: Standard multi-class classification
- BCEWithLogits: Binary classification
- Focal: Severe class imbalance - way to weigh minor classes more highly automatically
- Weighted CrossEntropy: Moderate class imbalance
- Label Smoothing: Prevent overconfidence, regularization - traiing technique to smoothen out targets.

SPECIAL CASES:
- Hinge: SVM-style margin-based classification
- Triplet: Similarity learning, face recognition
- Contrastive: Siamese networks
"""

# ============================================================================
# 1. ESSENTIAL LOSS FUNCTIONS - COPY-PASTE READY
# ============================================================================


class LossFunctions:
    @staticmethod
    def get_regression_loss(loss_type="mse"):
        """Standard regression losses"""
        if loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == "mae":
            return nn.L1Loss()
        elif loss_type == "huber":
            return nn.HuberLoss(delta=1.0)  # delta controls transition point

    @staticmethod
    def get_classification_loss(loss_type="ce", num_classes=None, class_weights=None):
        """Standard classification losses"""
        if loss_type == "ce":
            return nn.CrossEntropyLoss(weight=class_weights)
        elif loss_type == "binary":
            return nn.BCEWithLogitsLoss(pos_weight=class_weights)
        elif loss_type == "focal":
            return FocalLoss(alpha=1.0, gamma=2.0)


# ============================================================================
# 2. CUSTOM LOSS FUNCTIONS - TEMPLATES TO MODIFY
# ============================================================================


class AsymmetricLoss(nn.Module):
    """Template for asymmetric penalties (e.g., underestimate vs overestimate)"""

    def __init__(self, under_penalty=2.0, over_penalty=1.0):
        super().__init__()
        self.under_penalty = under_penalty
        self.over_penalty = over_penalty

    def forward(self, pred, target):
        error = pred - target
        loss = torch.where(
            error < 0,
            self.under_penalty * error**2,  # Underestimate penalty
            self.over_penalty * error**2,
        )  # Overestimate penalty
        return loss.mean()


class FocalLoss(nn.Module):
    """For severe class imbalance"""

    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()


class WeightedMSE(nn.Module):
    """Sample-specific weights for regression"""

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weights=None):
        mse = (pred - target) ** 2
        if weights is not None:
            mse = mse * weights
        return mse.mean()


# ============================================================================
# 3. OUTPUT COMPATIBILITY GUIDE
# ============================================================================


def ensure_compatibility():
    """
    CRITICAL: Match your model output to loss function expectations
    """

    # REGRESSION LOSSES
    print("=== REGRESSION OUTPUT REQUIREMENTS ===")
    pred_regression = torch.randn(32, 1)  # [batch_size, 1] or [batch_size]
    target_regression = torch.randn(32, 1)

    mse_loss = nn.MSELoss()
    loss = mse_loss(pred_regression, target_regression)
    print(
        f"Regression - Pred shape: {pred_regression.shape}, Target shape: {target_regression.shape}"
    )
    print(f"MSE Loss: {loss.item():.4f}")

    # BINARY CLASSIFICATION
    print("\n=== BINARY CLASSIFICATION OUTPUT REQUIREMENTS ===")
    logits_binary = torch.randn(32, 1)  # Raw logits, NOT probabilities
    target_binary = torch.randint(0, 2, (32, 1)).float()

    bce_loss = nn.BCEWithLogitsLoss()
    loss = bce_loss(logits_binary, target_binary)
    print(
        f"Binary - Logits shape: {logits_binary.shape}, Target shape: {target_binary.shape}"
    )
    print(f"BCE Loss: {loss.item():.4f}")

    # MULTI-CLASS CLASSIFICATION
    print("\n=== MULTI-CLASS OUTPUT REQUIREMENTS ===")
    logits_multi = torch.randn(32, 5)  # [batch_size, num_classes] - Raw logits
    target_multi = torch.randint(
        0, 5, (32,)
    )  # [batch_size] - Class indices (NOT one-hot)

    ce_loss = nn.CrossEntropyLoss()
    loss = ce_loss(logits_multi, target_multi)
    print(
        f"Multi-class - Logits shape: {logits_multi.shape}, Target shape: {target_multi.shape}"
    )
    print(f"CrossEntropy Loss: {loss.item():.4f}")


# ============================================================================
# 4. QUICK CUSTOMIZATION TEMPLATES
# ============================================================================


def create_custom_loss(loss_type="business_specific"):
    """Templates for common customizations"""

    if loss_type == "business_specific":
        # Example: E-commerce price prediction where underestimating is worse
        class BusinessLoss(nn.Module):
            def forward(self, pred, target):
                error = pred - target
                # Underestimate penalty = 3x, overestimate penalty = 1x
                loss = torch.where(error < 0, 3.0 * error**2, error**2)
                return loss.mean()

        return BusinessLoss()

    elif loss_type == "robust_regression":
        # Huber loss with custom delta
        return nn.HuberLoss(delta=0.5)

    elif loss_type == "weighted_classes":
        # For imbalanced classification
        class_weights = torch.tensor([1.0, 2.0, 5.0])  # Weight rare classes higher
        return nn.CrossEntropyLoss(weight=class_weights)


# ============================================================================
# 5. COMMON MISTAKES AND FIXES
# ============================================================================


def common_mistakes_demo():
    """Common errors and how to fix them"""

    print("=== COMMON MISTAKES ===")

    # MISTAKE 1: Wrong target format for CrossEntropy
    print("1. CrossEntropy target format:")
    logits = torch.randn(4, 3)

    # WRONG: One-hot encoded targets
    try:
        wrong_targets = F.one_hot(torch.tensor([0, 1, 2, 1]), num_classes=3).float()
        loss = nn.CrossEntropyLoss()(logits, wrong_targets)
    except Exception as e:
        print(f"   WRONG (one-hot): Error - {type(e).__name__}")

    # CORRECT: Class indices
    correct_targets = torch.tensor([0, 1, 2, 1])
    loss = nn.CrossEntropyLoss()(logits, correct_targets)
    print(f"   CORRECT (indices): Loss = {loss.item():.4f}")

    # MISTAKE 2: Using probabilities instead of logits
    print("\n2. BCEWithLogits expects logits, not probabilities:")
    probs = torch.sigmoid(torch.randn(4, 1))  # Already probabilities
    logits = torch.randn(4, 1)  # Raw logits
    targets = torch.randint(0, 2, (4, 1)).float()

    # WRONG: Passing probabilities to BCEWithLogits
    try:
        loss = nn.BCEWithLogitsLoss()(probs, targets)
        print(f"   Probabilities to BCEWithLogits: {loss.item():.4f} (incorrect)")
    except:
        pass

    # CORRECT: Use logits
    loss = nn.BCEWithLogitsLoss()(logits, targets)
    print(f"   Logits to BCEWithLogits: {loss.item():.4f} (correct)")


# ============================================================================
# 6. USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("LOSS FUNCTION COMPATIBILITY GUIDE")
    print("=" * 50)

    # Test compatibility
    ensure_compatibility()

    print("\n" + "=" * 50)

    # Show common mistakes
    common_mistakes_demo()

    print("\n" + "=" * 50)
    print("QUICK LOSS SELECTION:")
    print("- Regression with outliers: nn.HuberLoss()")
    print("- Binary classification: nn.BCEWithLogitsLoss()")
    print("- Multi-class: nn.CrossEntropyLoss()")
    print("- Class imbalance: FocalLoss() or weighted CrossEntropyLoss")
    print("- Custom business logic: Inherit from nn.Module, implement forward()")
