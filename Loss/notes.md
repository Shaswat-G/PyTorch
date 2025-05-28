# PyTorch Loss Functions - Complete Design Guide

## Quick Decision Tree

### Regression Problems
- **Standard case**: MSE (`nn.MSELoss`)
- **Outliers present**: Huber (`nn.HuberLoss`) or MAE (`nn.L1Loss`)
- **Asymmetric costs**: Custom loss (penalize over/under-prediction differently)
- **Robust to extreme outliers**: MAE (`nn.L1Loss`)

### Classification Problems
- **Standard multi-class**: CrossEntropy (`nn.CrossEntropyLoss`)
- **Binary classification**: BCE with logits (`nn.BCEWithLogitsLoss`)
- **Severe class imbalance**: Focal Loss or Weighted CrossEntropy
- **Moderate imbalance**: Weighted CrossEntropy
- **Need calibrated confidence**: Label Smoothing + Temperature Scaling
- **Margin-based learning**: Hinge Loss

### Similarity/Embedding Learning
- **Face recognition/verification**: Triplet Loss
- **Siamese networks**: Contrastive Loss
- **Ranking problems**: Margin Ranking Loss

---

## Detailed Use Cases & Implementation

### 1. Regression Loss Functions

#### Mean Squared Error (MSE)
**When to use:**
- Standard regression problems
- When you want to heavily penalize large errors
- Gaussian noise assumptions

**Pros:**
- Differentiable everywhere
- Convex for linear models
- Well-understood properties

**Cons:**
- Sensitive to outliers
- Can dominate training if outliers exist

```python
loss = nn.MSELoss()
# Output: any shape, target: same shape
```

#### Mean Absolute Error (MAE/L1)
**When to use:**
- Robust regression with outliers
- When you want median-like behavior
- Equal penalty for all error magnitudes

**Pros:**
- Robust to outliers
- More stable gradients for large errors

**Cons:**
- Non-differentiable at zero
- Can be slow to converge

```python
loss = nn.L1Loss()
```

#### Huber Loss (Smooth L1)
**When to use:**
- Best of both worlds: MSE + MAE
- Object detection (bounding box regression)
- When you have some outliers but want MSE behavior for small errors

**Key parameter:** `delta` - transition point between L1 and L2 behavior

```python
loss = nn.HuberLoss(delta=1.0)  # delta=1.0 is standard
```

#### Custom Asymmetric Loss
**When to use:**
- Business-specific penalty asymmetries
- Medical diagnosis (false negatives vs false positives have different costs)
- Financial predictions (underestimate vs overestimate costs)

**Design pattern:**
```python
class AsymmetricMSE(nn.Module):
    def __init__(self, under_penalty=2.0, over_penalty=1.0):
        super().__init__()
        self.under_penalty = under_penalty
        self.over_penalty = over_penalty
    
    def forward(self, pred, target):
        error = pred - target
        loss = torch.where(error < 0,
                          self.under_penalty * error**2,
                          self.over_penalty * error**2)
        return loss.mean()
```

### 2. Classification Loss Functions

#### CrossEntropy Loss
**When to use:**
- Standard multi-class classification
- Mutually exclusive classes
- When you want probabilistic outputs

**Critical requirements:**
- Input: Raw logits `[batch_size, num_classes]`
- Target: Class indices `[batch_size]` (NOT one-hot)

```python
loss = nn.CrossEntropyLoss()
# For class weights:
loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 5.0]))
```

#### Binary CrossEntropy with Logits
**When to use:**
- Binary classification
- Multi-label classification (multiple non-exclusive labels)

**Why with logits:**
- Numerically more stable than separate sigmoid + BCE
- Prevents gradient saturation

```python
loss = nn.BCEWithLogitsLoss()
# For positive class weighting:
loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))
```

#### Focal Loss (Custom Implementation Required)
**When to use:**
- Severe class imbalance (1:100+ ratios)
- When easy examples dominate training
- Object detection with many background pixels

**Key parameters:**
- `alpha`: Class weighting (like weighted CE)
- `gamma`: Focus parameter (2.0 is standard)

**How it works:**
- Down-weights easy examples
- Focuses learning on hard examples
- `(1-p_t)^gamma` term does the magic

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()
```

#### Label Smoothing
**When to use:**
- Prevent overconfident predictions
- Regularization technique
- When you need calibrated probabilities

**How it works:**
- Soft targets: `(1-ε) * hard_target + ε/K`
- Typical ε values: 0.1 for vision, 0.01-0.3 for NLP

```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        soft_targets = torch.zeros_like(logits)
        soft_targets.fill_(self.smoothing / (num_classes - 1))
        soft_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        log_probs = F.log_softmax(logits, dim=1)
        return -torch.sum(soft_targets * log_probs, dim=1).mean()
```

### 3. Class Imbalance Strategies

#### Mild Imbalance (1:2 to 1:10)
**Strategy:** Weighted CrossEntropy
```python
# If class 0 has 1000 samples, class 1 has 100 samples
class_weights = torch.tensor([1.0, 10.0])  # Inverse frequency
loss = nn.CrossEntropyLoss(weight=class_weights)
```

#### Moderate Imbalance (1:10 to 1:100)
**Strategy:** Weighted CrossEntropy + Oversampling/Undersampling
```python
# Calculate weights automatically
def calculate_class_weights(targets):
    class_counts = torch.bincount(targets)
    total_samples = len(targets)
    weights = total_samples / (len(class_counts) * class_counts.float())
    return weights
```

#### Severe Imbalance (1:100+)
**Strategy:** Focal Loss + Sampling strategies
- Use Focal Loss with gamma=2.0, alpha tuned to dataset
- Combine with balanced sampling
- Consider evaluation metrics beyond accuracy

### 4. Similarity Learning Losses

#### Triplet Loss
**When to use:**
- Face recognition/verification
- Image similarity search
- Learning embeddings where relative distances matter

**Setup:** Need triplets (anchor, positive, negative)
```python
loss = nn.TripletMarginLoss(margin=1.0, p=2.0)
# margin: how much closer positive should be than negative
# p: norm type (2 for Euclidean distance)
```

#### Contrastive Loss
**When to use:**
- Siamese networks
- Binary similarity tasks
- When you have pairs rather than triplets

```python
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, output1, output2, target):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((1-target) * torch.pow(euclidean_distance, 2) +
                         target * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss
```

---

## Loss Function Selection Checklist

### Before Choosing:
1. **Problem type?** (Regression/Classification/Similarity)
2. **Data characteristics?** (Balanced/Imbalanced/Outliers)
3. **Business constraints?** (Asymmetric costs/Calibration needs)
4. **Model output format?** (Logits/Probabilities/Embeddings)

### Common Gotchas:
- **Wrong target format:** CrossEntropy needs class indices, not one-hot
- **Forgetting logits:** Use BCEWithLogits, not BCE with sigmoid outputs
- **Ignoring imbalance:** Don't use standard CE on 1:100 imbalanced data
- **Scale mismatch:** Ensure loss scale is appropriate for your problem

### Debugging Loss Functions:
1. **Check shapes:** Print input/target shapes
2. **Verify ranges:** Logits should be unbounded, probabilities in [0,1]
3. **Monitor loss values:** Sudden spikes often indicate shape/format issues
4. **Gradient flow:** Use `torch.autograd.grad` to check if gradients flow properly

---

## Advanced Considerations

### Multi-task Learning
When training multiple tasks simultaneously:
```python
total_loss = task1_weight * task1_loss + task2_weight * task2_loss
# Balance task losses by their typical scales
```

### Dynamic Loss Weighting
For changing problem dynamics during training:
```python
class AdaptiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, pred, target):
        base_loss = F.mse_loss(pred, target)
        return self.weight * base_loss  # Weight learned during training
```

### Evaluation Considerations
- **Different metrics for evaluation:** Loss for training ≠ metric for evaluation
- **Calibration:** Check if predicted probabilities match actual accuracy
- **Class-specific performance:** Don't just look at overall accuracy

Remember: The best loss function depends on your specific problem, data characteristics, and business requirements. Start with standard choices and customize based on observed behavior.

# PyTorch Optimizers - Complete Design Guide

## Quick Decision Tree

### Default Recommendations
- **General purpose**: Adam/AdamW (lr=1e-3, weight_decay=1e-4)
- **Computer vision**: SGD with momentum (lr=1e-1, momentum=0.9, weight_decay=1e-4)
- **NLP/Transformers**: AdamW (lr=1e-4 to 5e-4, weight_decay=1e-2)
- **Fine-tuning pretrained**: Adam with lower lr (1e-5 to 1e-4)
- **RNNs**: RMSprop or Adam (lr=1e-3)

### When Standard Approaches Fail
- **Slow convergence**: Increase lr or switch to adaptive optimizer
- **Oscillating loss**: Decrease lr or add momentum
- **Overfitting**: Increase weight_decay or use dropout
- **Underfitting**: Decrease weight_decay, increase model capacity

---

## Optimizer Deep Dive

### 1. SGD (Stochastic Gradient Descent)

#### Basic SGD
**When to use:**
- Simple problems
- When you want full control over learning
- Educational purposes

**Pros:**
- Simple and interpretable
- Often finds better minima (wider basins)
- Less prone to overfitting

**Cons:**
- Requires careful tuning
- Slow convergence
- Struggles with different parameter scales

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

#### SGD with Momentum
**When to use:**
- Computer vision tasks (especially CNNs)
- When you have time to tune hyperparameters
- Training from scratch on large datasets

**How momentum helps:**
- Smooths out oscillations
- Accelerates convergence in consistent directions
- Helps escape small local minima

**Key parameters:**
- `lr`: 0.1 to 0.01 (much higher than Adam)
- `momentum`: 0.9 (standard), 0.95 (for very large datasets)

```python
optimizer = torch.optim.SGD(model.parameters(), 
                           lr=0.1, 
                           momentum=0.9, 
                           weight_decay=1e-4)
```

#### Nesterov Momentum
**When to use:**
- When standard momentum isn't enough
- Theoretical improvements over standard momentum

```python
optimizer = torch.optim.SGD(model.parameters(), 
                           lr=0.1, 
                           momentum=0.9, 
                           nesterov=True)
```

### 2. Adam Family

#### Adam (Adaptive Moment Estimation)
**When to use:**
- Default choice for most problems
- Quick prototyping
- When you don't want to tune learning rates extensively

**How it works:**
- Adapts learning rate per parameter
- Uses both first moment (gradient) and second moment (squared gradient)
- Bias correction for early training steps

**Key parameters:**
- `lr`: 1e-3 (default), range: 1e-5 to 1e-2
- `betas`: (0.9, 0.999) - momentum parameters for gradient and squared gradient
- `eps`: 1e-8 - small constant for numerical stability

```python
optimizer = torch.optim.Adam(model.parameters(), 
                            lr=1e-3, 
                            betas=(0.9, 0.999), 
                            weight_decay=1e-4)
```

#### AdamW (Adam with Weight Decay)
**When to use:**
- Modern deep learning (especially transformers)
- When you need proper weight decay
- Generally preferred over Adam

**Key difference from Adam:**
- Proper weight decay implementation (decoupled from gradient)
- Better generalization in practice

**Parameters:**
- Same as Adam, but `weight_decay` works correctly
- Typical weight_decay: 1e-2 for transformers, 1e-4 for vision

```python
optimizer = torch.optim.AdamW(model.parameters(), 
                             lr=1e-3, 
                             weight_decay=1e-2)
```

### 3. RMSprop
**When to use:**
- RNNs and recurrent architectures
- When gradients have very different magnitudes
- Online learning scenarios

**How it works:**
- Adapts learning rate using moving average of squared gradients
- Good for non-stationary objectives

```python
optimizer = torch.optim.RMSprop(model.parameters(), 
                               lr=1e-3, 
                               alpha=0.99, 
                               eps=1e-8)
```

### 4. Specialized Optimizers

#### LBFGS (Limited-memory BFGS)
**When to use:**
- Small datasets where you can afford expensive steps
- When you need very precise optimization
- Scientific computing applications

**Characteristics:**
- Second-order method (uses curvature information)
- Requires closure function
- Memory efficient version of BFGS

```python
optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0)

def closure():
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    return loss

optimizer.step(closure)
```

---

## Parameter Selection Guide

### Learning Rate (lr)
**Most critical hyperparameter**

#### Typical Ranges by Optimizer:
- **Adam/AdamW**: 1e-5 to 1e-2 (default: 1e-3)
- **SGD**: 1e-3 to 1e-1 (default: 1e-2)
- **RMSprop**: 1e-4 to 1e-2 (default: 1e-3)

#### How to Find Good Learning Rate:
1. **Learning Rate Range Test:**
   - Start with very small lr (1e-7)
   - Exponentially increase each batch
   - Plot loss vs lr, choose lr where loss decreases fastest

2. **Signs of Wrong Learning Rate:**
   - Too high: Loss explodes or oscillates wildly
   - Too low: Very slow convergence, loss plateaus early

#### Domain-Specific Recommendations:
- **Computer Vision**: SGD (1e-1), Adam (1e-3 to 1e-4)
- **NLP**: Adam/AdamW (1e-4 to 5e-4)
- **Fine-tuning**: 1/10th of training lr
- **Large models**: Start smaller (1e-4 to 1e-5)

### Weight Decay
**L2 regularization strength - prevents overfitting**

#### Typical Values:
- **Vision**: 1e-4 to 1e-3
- **NLP/Transformers**: 1e-2 to 1e-1
- **Small datasets**: 1e-3 to 1e-2
- **Large datasets**: 1e-5 to 1e-4

#### When to Use:
- Model overfitting (train acc >> val acc)
- Large models with limited data
- Always use some amount (1e-5 minimum)

#### When NOT to Use:
- Underfit models
- Very small datasets where you need all model capacity

### Momentum (SGD)
**Smooths optimization trajectory**

#### Values:
- **Standard**: 0.9
- **Large datasets**: 0.95
- **Very large datasets**: 0.99

#### When to Adjust:
- Increase for noisy gradients
- Decrease if training becomes unstable

### Adam Betas
**Usually don't need to change**

#### Default (0.9, 0.999):
- Works for most cases
- beta1=0.9: gradient momentum
- beta2=0.999: squared gradient momentum

#### When to Adjust:
- **Sparse gradients**: Increase beta2 to 0.9999
- **Very noisy gradients**: Decrease beta1 to 0.8

---

## Training Strategies

### Learning Rate Scheduling
**Critical for final performance**

#### Step Decay
```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# Multiply lr by 0.1 every 30 epochs
```

#### Cosine Annealing
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
# Smooth decay following cosine curve
```

#### Plateau Reduction
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
# Reduce lr when validation loss plateaus
```

#### Warmup + Decay (Modern NLP)
```python
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### Gradient Clipping
**Prevents exploding gradients**

```python
# Clip by norm (most common)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
```

**When to use:**
- RNNs and transformers (almost always)
- When you see exploding gradients
- Typical max_norm: 1.0 to 5.0

---

## Optimizer Selection by Task

### Computer Vision

#### Training from Scratch:
```python
optimizer = torch.optim.SGD(model.parameters(), 
                           lr=0.1, 
                           momentum=0.9, 
                           weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
```

#### Fine-tuning:
```python
optimizer = torch.optim.Adam(model.parameters(), 
                            lr=1e-4, 
                            weight_decay=1e-4)
```

### Natural Language Processing

#### Transformer Training:
```python
optimizer = torch.optim.AdamW(model.parameters(), 
                             lr=5e-4, 
                             weight_decay=1e-2, 
                             betas=(0.9, 0.98))
# With warmup scheduler
```

#### RNN Training:
```python
optimizer = torch.optim.RMSprop(model.parameters(), 
                               lr=1e-3, 
                               alpha=0.95)
# With gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

### Reinforcement Learning
```python
optimizer = torch.optim.Adam(model.parameters(), 
                            lr=1e-4, 
                            eps=1e-5)  # Smaller eps for stability
```

---

## Advanced Techniques

### Differential Learning Rates
**Different lr for different parts of model**
```python
optimizer = torch.optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-4},
    {'params': model.classifier.parameters(), 'lr': 1e-3}
])
```

### Optimizer State Management
```python
# Save optimizer state
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
}, 'checkpoint.pth')

# Load optimizer state
checkpoint = torch.load('checkpoint.pth')
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### Custom Optimizers
```python
class CustomAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, custom_param=0.1):
        defaults = dict(lr=lr, custom_param=custom_param)
        super(CustomAdam, self).__init__(params, defaults)
    
    def step(self):
        # Custom optimization logic
        pass
```

---

## Troubleshooting Optimization

### Common Problems & Solutions:

#### Loss Not Decreasing:
1. **Check learning rate**: Too low? Increase by 10x
2. **Check gradients**: `torch.nn.utils.clip_grad_norm_` to see gradient norms
3. **Check data**: Verify labels and preprocessing
4. **Simplify model**: Overparameterized model might be hard to optimize

#### Loss Exploding:
1. **Reduce learning rate**: Divide by 10
2. **Add gradient clipping**: `max_norm=1.0`
3. **Check weight initialization**: Poor init can cause instability
4. **Reduce model size**: Might be too complex

#### Slow Convergence:
1. **Increase learning rate**: Multiply by 10
2. **Switch optimizer**: SGD → Adam for faster convergence
3. **Add learning rate scheduling**: Warmup + decay
4. **Check batch size**: Too small batch might be noisy

#### Overfitting During Training:
1. **Increase weight decay**: 1e-4 → 1e-3
2. **Add dropout**: If not already present
3. **Reduce learning rate**: Slower training might generalize better
4. **Early stopping**: Monitor validation loss

#### Poor Final Performance (Good Training):
1. **Switch to SGD**: Often finds better minima than Adam
2. **Longer training**: Sometimes need more epochs
3. **Learning rate scheduling**: Add decay schedule
4. **Ensemble methods**: Multiple models with different seeds

---

## Experimental Best Practices

### Hyperparameter Search:
1. **Start with learning rate**: Most important parameter
2. **Grid search key parameters**: lr, weight_decay, batch_size
3. **Random search**: For exploring parameter space
4. **Bayesian optimization**: For expensive experiments

### Monitoring:
```python
# Log key metrics
wandb.log({
    'train_loss': loss.item(),
    'learning_rate': optimizer.param_groups[0]['lr'],
    'gradient_norm': total_norm,
    'weight_decay': optimizer.param_groups[0]['weight_decay']
})
```

### Reproducibility:
```python
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
```

Remember: Optimization is both art and science. Start with proven defaults, then systematically adjust based on your specific problem characteristics and observed training dynamics.