# Production-Level Training Loop - Complete Design Guide

## Mental Template: Core Structure

Every production training script should follow this structure:

```
1. Configuration & Setup
2. Data Loading & Preprocessing
3. Model Creation & Initialization
4. Training Infrastructure (logging, checkpointing, etc.)
5. Training Loop Implementation
6. Validation Loop Implementation
7. Testing & Evaluation
8. Cleanup & Final Steps
```

---

## 1. Configuration & Argument Parsing

### What to Include:
- **Hyperparameters**: lr, batch_size, epochs, weight_decay
- **Model config**: architecture, hidden_dims, dropout_rate
- **Data config**: dataset_path, augmentations, train/val split
- **Training config**: device, mixed_precision, gradient_clipping
- **Logging config**: experiment_name, log_frequency, save_dir
- **Reproducibility**: random_seed, deterministic_mode

### Implementation Patterns:

#### Option 1: ArgumentParser (Simple)
```python
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # Training config
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--gradient_clipping', type=float, default=1.0)
    
    # Paths
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--experiment_name', type=str, required=True)
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()
```

#### Option 2: Config Files (Scalable)
```python
import yaml
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model
    model_name: str = "resnet50"
    num_classes: int = 10
    dropout: float = 0.1
    
    # Training
    lr: float = 1e-3
    batch_size: int = 32
    epochs: int = 100
    weight_decay: float = 1e-4
    
    # Infrastructure
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    
    # Paths
    data_dir: str = ""
    save_dir: str = "./checkpoints"
    experiment_name: str = ""
    
    # Reproducibility
    seed: int = 42

def load_config(config_path: str) -> TrainingConfig:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return TrainingConfig(**config_dict)
```

**When to use which:**
- **ArgumentParser**: Small projects, quick experiments
- **Config files**: Production systems, complex configurations, reproducibility

---

## 2. Reproducibility & Environment Setup

### Critical Setup Code:
```python
def set_reproducibility(seed: int, deterministic: bool = True):
    """Set seeds for reproducible training"""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Faster but non-deterministic
        torch.backends.cudnn.benchmark = True

def setup_device(device_name: str):
    """Setup and validate device"""
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(device_name)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    return device
```

---

## 3. Data Loading Strategy

### Key Decisions:

#### DataLoader Configuration:
```python
def create_dataloaders(config):
    # Dataset creation
    train_dataset = YourDataset(config.data_dir, split='train', transform=train_transform)
    val_dataset = YourDataset(config.data_dir, split='val', transform=val_transform)
    
    # DataLoader parameters
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # Always True for training
        num_workers=4,  # Adjust based on CPU cores
        pin_memory=True,  # Faster GPU transfer
        drop_last=True,  # Consistent batch sizes
        persistent_workers=True  # Faster epoch transitions
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,  # Larger batch for validation
        shuffle=False,  # Never shuffle validation
        num_workers=4,
        pin_memory=True,
        drop_last=False  # Use all validation data
    )
    
    return train_loader, val_loader
```

#### Data Augmentation Strategy:
```python
def get_transforms(config):
    """Separate transforms for train/validation"""
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # NO augmentation for validation
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform
```

**Critical Points:**
- **num_workers**: Start with 4, adjust based on CPU/bottleneck analysis
- **pin_memory**: Always True when using GPU
- **persistent_workers**: Reduces worker startup overhead
- **Validation batch size**: Can be 2x training batch size (no gradients)

---

## 4. Model Creation & Initialization

### Model Setup Pattern:
```python
def create_model(config, device):
    """Create and initialize model"""
    
    # Model creation
    if config.model_name == 'resnet50':
        model = models.resnet50(pretrained=config.pretrained)
        if config.num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, config.num_classes)
    else:
        model = YourCustomModel(config)
    
    # Weight initialization (if training from scratch)
    if not config.pretrained:
        model.apply(init_weights)
    
    # Move to device
    model = model.to(device)
    
    # Data parallel (if multiple GPUs)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    return model

def init_weights(m):
    """Custom weight initialization"""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
```

### Optimizer & Scheduler Setup:
```python
def create_optimizer_scheduler(model, config):
    """Create optimizer and learning rate scheduler"""
    
    # Different learning rates for different parts
    if hasattr(model, 'backbone') and config.differential_lr:
        optimizer = torch.optim.AdamW([
            {'params': model.backbone.parameters(), 'lr': config.lr * 0.1},
            {'params': model.classifier.parameters(), 'lr': config.lr}
        ], weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
    
    # Learning rate scheduler
    if config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs
        )
    elif config.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=0.1
        )
    elif config.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )
    
    return optimizer, scheduler
```

---

## 5. Training Infrastructure

### Logging Setup:
```python
import wandb
from torch.utils.tensorboard import SummaryWriter

def setup_logging(config):
    """Initialize logging systems"""
    
    # Create save directory
    save_dir = os.path.join(config.save_dir, config.experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Weights & Biases (recommended)
    if config.use_wandb:
        wandb.init(
            project=config.project_name,
            name=config.experiment_name,
            config=config.__dict__
        )
    
    # TensorBoard (alternative)
    if config.use_tensorboard:
        writer = SummaryWriter(log_dir=save_dir)
    else:
        writer = None
    
    return save_dir, writer

def log_metrics(metrics_dict, step, writer=None):
    """Log metrics to all configured systems"""
    
    if writer:
        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, step)
    
    if wandb.run:
        wandb.log(metrics_dict, step=step)
```

### Checkpointing System:
```python
class CheckpointManager:
    def __init__(self, save_dir, max_checkpoints=5):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics):
        """Save training checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'config': config.__dict__
        }
        
        # Save current checkpoint
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if metrics['val_accuracy'] == max(self.best_metrics):
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
        
        # Cleanup old checkpoints
        self.checkpoints.append(checkpoint_path)
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
    
    def load_checkpoint(self, model, optimizer, scheduler, checkpoint_path):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['metrics']
```

---

## 6. Training Loop Implementation

### Core Training Loop:
```python
def train_epoch(model, train_loader, optimizer, criterion, device, config):
    """Single training epoch"""
    
    model.train()  # Set to training mode
    
    # Metrics tracking
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    # Progress tracking
    pbar = tqdm(train_loader, desc='Training')
    
    for batch_idx, (data, targets) in enumerate(pbar):
        # Move data to device
        data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad()
        
        if config.mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = model(data)
                loss = criterion(outputs, targets)
        else:
            outputs = model(data)
            loss = criterion(outputs, targets)
        
        # Backward pass
        if config.mixed_precision:
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if config.gradient_clipping > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            
            # Gradient clipping
            if config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
            
            optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total_samples += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        accuracy = 100. * correct / total_samples
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy:.2f}%'
        })
        
        # Log batch metrics (optional, for debugging)
        if batch_idx % config.log_frequency == 0:
            step = epoch * len(train_loader) + batch_idx
            log_metrics({
                'batch_loss': loss.item(),
                'batch_accuracy': accuracy,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step)
    
    # Return epoch metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total_samples
    
    return {
        'train_loss': avg_loss,
        'train_accuracy': accuracy
    }
```

### Validation Loop:
```python
def validate_epoch(model, val_loader, criterion, device):
    """Single validation epoch"""
    
    model.eval()  # Set to evaluation mode
    
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    # No gradient computation for validation
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        
        for data, targets in pbar:
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_samples += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            accuracy = 100. * correct / total_samples
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total_samples
    
    return {
        'val_loss': avg_loss,
        'val_accuracy': accuracy
    }
```

---

## 7. Main Training Function

### Complete Training Pipeline:
```python
def train(config):
    """Main training function"""
    
    # Setup
    set_reproducibility(config.seed)
    device = setup_device(config.device)
    save_dir, writer = setup_logging(config)
    
    # Data
    train_loader, val_loader = create_dataloaders(config)
    
    # Model
    model = create_model(config, device)
    criterion = create_criterion(config)
    optimizer, scheduler = create_optimizer_scheduler(model, config)
    
    # Mixed precision
    if config.mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    
    # Checkpointing
    checkpoint_manager = CheckpointManager(save_dir)
    
    # Resume training if checkpoint exists
    start_epoch = 0
    best_val_accuracy = 0.0
    
    if config.resume_from:
        start_epoch, metrics = checkpoint_manager.load_checkpoint(
            model, optimizer, scheduler, config.resume_from
        )
        best_val_accuracy = metrics.get('val_accuracy', 0.0)
        print(f"Resumed training from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, config.epochs):
        print(f'\nEpoch {epoch+1}/{config.epochs}')
        print('-' * 50)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, config)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Combine metrics
        epoch_metrics = {**train_metrics, **val_metrics}
        
        # Learning rate scheduling
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['val_loss'])
            else:
                scheduler.step()
        
        # Log epoch metrics
        log_metrics(epoch_metrics, epoch, writer)
        
        # Print epoch summary
        print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
              f"Train Acc: {train_metrics['train_accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
              f"Val Acc: {val_metrics['val_accuracy']:.2f}%")
        
        # Save checkpoint
        is_best = val_metrics['val_accuracy'] > best_val_accuracy
        if is_best:
            best_val_accuracy = val_metrics['val_accuracy']
        
        checkpoint_manager.save_checkpoint(
            model, optimizer, scheduler, epoch, epoch_metrics
        )
        
        # Early stopping (optional)
        if config.early_stopping:
            if not is_best:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"Early stopping triggered after {config.patience} epochs without improvement")
                    break
            else:
                patience_counter = 0
    
    # Cleanup
    if writer:
        writer.close()
    
    print(f"Training completed! Best validation accuracy: {best_val_accuracy:.2f}%")
    return best_val_accuracy
```

---

## 8. Common Pitfalls & Solutions

### Critical Mistakes to Avoid:

#### 1. **Gradient Accumulation Errors**
```python
# WRONG: Forgetting to zero gradients
for batch in dataloader:
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()  # Gradients keep accumulating!

# CORRECT: Always zero gradients
for batch in dataloader:
    optimizer.zero_grad()  # Critical!
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

#### 2. **Train/Eval Mode Confusion**
```python
# WRONG: Forgetting to set modes
model.eval()
# ... validation loop ...
# ... training loop continues without model.train()

# CORRECT: Always set mode explicitly
def train_epoch(model, ...):
    model.train()  # Set training mode
    # ... training code ...

def validate_epoch(model, ...):
    model.eval()  # Set evaluation mode
    # ... validation code ...
```

#### 3. **Memory Leaks in Validation**
```python
# WRONG: Computing gradients during validation
def validate(model, val_loader):
    total_loss = 0
    for data, targets in val_loader:
        outputs = model(data)
        loss = criterion(outputs, targets)
        total_loss += loss  # Keeping computation graph!

# CORRECT: Use torch.no_grad() and .item()
def validate(model, val_loader):
    total_loss = 0
    with torch.no_grad():  # Disable gradient computation
        for data, targets in val_loader:
            outputs = model(data)
            loss = criterion(outputs, targets)
            total_loss += loss.item()  # Extract scalar value
```

#### 4. **Data Leakage in Augmentation**
```python
# WRONG: Augmenting validation data
val_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # NO! Random transforms in validation
    transforms.ToTensor()
])

# CORRECT: Deterministic validation transforms
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),  # Deterministic transforms only
    transforms.ToTensor()
])
```

#### 5. **Inconsistent Batch Sizes**
```python
# WRONG: Variable batch sizes affecting BatchNorm
train_loader = DataLoader(dataset, batch_size=32, drop_last=False)

# CORRECT: Consistent batch sizes for training
train_loader = DataLoader(dataset, batch_size=32, drop_last=True)
```

### Performance Pitfalls:

#### 1. **Inefficient Data Loading**
```python
# WRONG: Blocking data loading
train_loader = DataLoader(dataset, num_workers=0)  # Single-threaded

# CORRECT: Multi-threaded loading
train_loader = DataLoader(
    dataset, 
    num_workers=4,  # Use multiple workers
    pin_memory=True,  # Faster GPU transfer
    persistent_workers=True  # Avoid worker respawning
)
```

#### 2. **Unnecessary GPU-CPU Transfers**
```python
# WRONG: Moving to CPU for calculations
accuracy = (predictions == targets).float().mean().cpu().item()

# CORRECT: Keep calculations on GPU
accuracy = (predictions == targets).float().mean().item()
```

#### 3. **Synchronous Logging**
```python
# WRONG: Logging every batch (slows training)
for batch_idx, (data, targets) in enumerate(train_loader):
    # ... training code ...
    wandb.log({'loss': loss.item()})  # Every batch!

# CORRECT: Log periodically
if batch_idx % log_frequency == 0:
    wandb.log({'loss': loss.item()})
```

---

## 9. Production Best Practices

### Code Organization:
```
project/
├── configs/
│   ├── base_config.yaml
│   └── experiment_configs/
├── src/
│   ├── models/
│   ├── data/
│   ├── training/
│   └── utils/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── tests/
└── requirements.txt
```

### Error Handling:
```python
def train_with_error_handling(config):
    try:
        result = train(config)
        return result
    except KeyboardInterrupt:
        print("Training interrupted by user")
        # Save emergency checkpoint
        save_emergency_checkpoint(model, optimizer, epoch)
        return None
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU out of memory. Try reducing batch_size or model size.")
            torch.cuda.empty_cache()
        else:
            print(f"Runtime error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None
```

### Resource Monitoring:
```python
def monitor_resources():
    """Monitor GPU/CPU usage during training"""
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1e9
        memory_cached = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {memory_used:.1f}GB used, {memory_cached:.1f}GB cached")
```

### Testing Infrastructure:
```python
def test_training_loop():
    """Quick test with small data to verify training loop works"""
    config = get_test_config()  # Small dataset, few epochs
    config.epochs = 2
    config.batch_size = 4
    
    try:
        result = train(config)
        assert result is not None, "Training should return a result"
        print("Training loop test passed!")
    except Exception as e:
        print(f"Training loop test failed: {e}")
        raise
```

Remember: Production training code should be robust, reproducible, and maintainable. Always test your training loop with small data before running expensive experiments!