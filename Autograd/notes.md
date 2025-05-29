# Comprehensive Deep Learning Training Checklist & Templates

## 1. Training Loop Template & Checklist

### Pre-Training Setup Checklist
- [ ] **Data Pipeline**
  - [ ] DataLoader with proper `batch_size`, `shuffle=True` for training
  - [ ] `num_workers` set appropriately (typically 4-8)
  - [ ] `pin_memory=True` if using GPU
  - [ ] Validation DataLoader with `shuffle=False`
  - [ ] Data augmentation applied only to training set

- [ ] **Model Initialization**
  - [ ] Model moved to correct device (`model.to(device)`)
  - [ ] Proper weight initialization if needed
  - [ ] Model summary/parameter count logged

- [ ] **Loss Function Selection**
  - [ ] Appropriate criterion for task (CrossEntropyLoss, MSELoss, etc.)
  - [ ] Label smoothing considered if needed
  - [ ] Class weights for imbalanced datasets

### Optimizer Configuration Checklist

#### Core Optimizer Options
- [ ] **Optimizer Choice & Parameters**
  ```python
  # SGD Options
  optimizer = optim.SGD(
      model.parameters(),
      lr=0.01,                    # Learning rate
      momentum=0.9,               # Momentum factor (0.9 typical)
      weight_decay=1e-4,          # L2 regularization
      dampening=0,                # Dampening for momentum
      nesterov=True              # Nesterov momentum
  )
  
  # Adam Options  
  optimizer = optim.Adam(
      model.parameters(),
      lr=0.001,                   # Learning rate
      betas=(0.9, 0.999),        # Coefficients for running averages
      eps=1e-08,                 # Term for numerical stability
      weight_decay=1e-4,         # L2 regularization
      amsgrad=False              # Use AMSGrad variant
  )
  
  # AdamW (decoupled weight decay)
  optimizer = optim.AdamW(
      model.parameters(),
      lr=0.001,
      betas=(0.9, 0.999),
      eps=1e-08,
      weight_decay=0.01,         # Higher weight decay typical
      amsgrad=False
  )
  ```

- [ ] **Parameter Groups** (if needed)
  ```python
  # Different learning rates for different layers
  optimizer = optim.Adam([
      {'params': model.backbone.parameters(), 'lr': 1e-4},
      {'params': model.classifier.parameters(), 'lr': 1e-3}
  ])
  ```

#### Learning Rate Scheduling Options
- [ ] **Scheduler Selection**
  ```python
  # Step Decay
  scheduler = optim.lr_scheduler.StepLR(
      optimizer, 
      step_size=30,              # Decay every 30 epochs
      gamma=0.1                  # Multiply LR by 0.1
  )
  
  # Multi-Step Decay
  scheduler = optim.lr_scheduler.MultiStepLR(
      optimizer,
      milestones=[100, 150],     # Decay at these epochs
      gamma=0.1
  )
  
  # Exponential Decay
  scheduler = optim.lr_scheduler.ExponentialLR(
      optimizer,
      gamma=0.95                 # Decay factor per epoch
  )
  
  # Cosine Annealing
  scheduler = optim.lr_scheduler.CosineAnnealingLR(
      optimizer,
      T_max=200,                 # Maximum iterations
      eta_min=1e-6              # Minimum learning rate
  )
  
  # Reduce on Plateau (Adaptive)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(
      optimizer,
      mode='min',                # 'min' for loss, 'max' for accuracy
      factor=0.5,               # Reduction factor
      patience=10,              # Epochs to wait before reducing
      verbose=True,             # Print when LR changes
      threshold=1e-4,           # Threshold for measuring improvement
      cooldown=0,               # Epochs to wait after LR reduction
      min_lr=1e-6              # Lower bound on LR
  )
  
  # Warm Restart
  scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
      optimizer,
      T_0=10,                   # Initial restart period
      T_mult=2,                 # Factor to increase period
      eta_min=1e-6             # Minimum learning rate
  )
  ```

#### Gradient Management Options
- [ ] **Gradient Clipping**
  ```python
  # Clip by norm (most common)
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  
  # Clip by value
  torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
  ```

- [ ] **Gradient Accumulation** (if needed)
  ```python
  accumulation_steps = 4  # Effective batch size = batch_size * accumulation_steps
  ```

### Complete Training Loop Template

```python
def train_model(model, train_loader, val_loader, config):
    """
    Comprehensive training function with all best practices
    """
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Loss function
    criterion = config.get('criterion', nn.CrossEntropyLoss())
    
    # Optimizer
    optimizer = config['optimizer']
    
    # Scheduler
    scheduler = config.get('scheduler', None)
    use_scheduler = scheduler is not None
    
    # Training configuration
    epochs = config['epochs']
    accumulation_steps = config.get('accumulation_steps', 1)
    clip_grad_norm = config.get('clip_grad_norm', None)
    clip_grad_value = config.get('clip_grad_value', None)
    
    # Metrics tracking
    train_losses = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    
    # Early stopping
    best_val_loss = float('inf')
    patience = config.get('early_stopping_patience', None)
    patience_counter = 0
    
    # Checkpointing
    checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
    save_every = config.get('save_every', 10)
    
    for epoch in range(epochs):
        # ===================
        # TRAINING PHASE
        # ===================
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        # Reset gradients for accumulation
        optimizer.zero_grad()
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Scale loss for gradient accumulation
            if accumulation_steps > 1:
                loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                if clip_grad_value is not None:
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps
            num_batches += 1
        
        # Handle remaining gradients if batch doesn't divide evenly
        if num_batches % accumulation_steps != 0:
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            if clip_grad_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_grad_value)
            optimizer.step()
            optimizer.zero_grad()
        
        avg_train_loss = running_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # ===================
        # VALIDATION PHASE
        # ===================
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                val_loss += criterion(outputs, targets).item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        if use_scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # Logging
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Val Accuracy: {val_accuracy:.2f}%')
        print(f'  Learning Rate: {current_lr:.2e}')
        
        # Early stopping
        if patience is not None:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                save_checkpoint(model, optimizer, scheduler, epoch, avg_val_loss, 
                              f'{checkpoint_dir}/best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Regular checkpointing
        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, avg_val_loss,
                          f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth')
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'learning_rates': learning_rates
    }
```

### Training Loop Mental Checklist (Every Epoch)
1. [ ] `model.train()` - Set training mode
2. [ ] `optimizer.zero_grad()` - Clear gradients (before accumulation loop)
3. [ ] Forward pass: `outputs = model(data)`
4. [ ] Compute loss: `loss = criterion(outputs, targets)`
5. [ ] Scale loss if using gradient accumulation
6. [ ] Backward pass: `loss.backward()`
7. [ ] Gradient clipping (if enabled)
8. [ ] `optimizer.step()` - Update parameters
9. [ ] `optimizer.zero_grad()` - Clear for next iteration
10. [ ] `model.eval()` - Set evaluation mode for validation
11. [ ] `torch.no_grad()` context for validation
12. [ ] `scheduler.step()` - Update learning rate
13. [ ] Log metrics and save checkpoints

## 2. Checkpointing & State Management

### What to Save in Checkpoints
```python
def save_checkpoint(model, optimizer, scheduler, epoch, loss, filepath, **kwargs):
    """
    Comprehensive checkpoint saving function
    """
    checkpoint = {
        # Model state
        'model_state_dict': model.state_dict(),
        
        # Optimizer state (crucial for momentum-based optimizers)
        'optimizer_state_dict': optimizer.state_dict(),
        
        # Scheduler state
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        
        # Training progress
        'epoch': epoch,
        'loss': loss,
        
        # Random states for reproducibility
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
        
        # CUDA random state (if using GPU)
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        
        # Model architecture info (for dynamic models)
        'model_class': model.__class__.__name__,
        'model_args': getattr(model, '_init_args', {}),
        
        # Training configuration
        'config': kwargs.get('config', {}),
        
        # Additional metrics
        'train_losses': kwargs.get('train_losses', []),
        'val_losses': kwargs.get('val_losses', []),
        'val_accuracies': kwargs.get('val_accuracies', []),
        'learning_rates': kwargs.get('learning_rates', []),
        
        # Timestamp
        'timestamp': datetime.now().isoformat(),
        
        # PyTorch version for compatibility
        'pytorch_version': torch.__version__,
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")
```

### Loading and Resuming Training
```python
def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device='cpu'):
    """
    Comprehensive checkpoint loading function
    """
    print(f"Loading checkpoint: {filepath}")
    
    # Load checkpoint
    if device == 'cpu':
        checkpoint = torch.load(filepath, map_location='cpu')
    else:
        checkpoint = torch.load(filepath)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Move optimizer state to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    # Load scheduler state
    if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Restore random states for reproducibility
    if 'torch_rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['torch_rng_state'])
    if 'numpy_rng_state' in checkpoint:
        np.random.set_state(checkpoint['numpy_rng_state'])
    if 'python_rng_state' in checkpoint:
        random.setstate(checkpoint['python_rng_state'])
    if 'cuda_rng_state' in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
    
    # Return checkpoint info
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'train_losses': checkpoint.get('train_losses', []),
        'val_losses': checkpoint.get('val_losses', []),
        'val_accuracies': checkpoint.get('val_accuracies', []),
        'learning_rates': checkpoint.get('learning_rates', []),
        'config': checkpoint.get('config', {}),
    }

def resume_training(checkpoint_path, model, train_loader, val_loader, config):
    """
    Resume training from checkpoint
    """
    # Setup optimizer and scheduler
    optimizer = config['optimizer']
    scheduler = config.get('scheduler', None)
    device = config.get('device', 'cpu')
    
    # Load checkpoint
    checkpoint_info = load_checkpoint(
        checkpoint_path, model, optimizer, scheduler, device
    )
    
    # Update config with checkpoint info
    start_epoch = checkpoint_info['epoch'] + 1
    config['start_epoch'] = start_epoch
    
    # Continue training
    print(f"Resuming training from epoch {start_epoch}")
    return train_model(model, train_loader, val_loader, config)
```

### Checkpoint Management Best Practices

#### Checkpoint Strategy Checklist
- [ ] **Regular Checkpoints**
  - [ ] Save every N epochs (e.g., every 10 epochs)
  - [ ] Save at end of each epoch for critical experiments
  - [ ] Keep last K checkpoints to save disk space

- [ ] **Best Model Checkpoints**
  - [ ] Save when validation loss improves
  - [ ] Save when validation accuracy improves
  - [ ] Separate files for different metrics

- [ ] **Emergency Checkpoints**
  - [ ] Save before system shutdown
  - [ ] Save on training interruption
  - [ ] Auto-save on CUDA out-of-memory errors

#### File Organization
```
checkpoints/
├── experiment_name/
│   ├── best_model.pth          # Best validation performance
│   ├── latest_model.pth        # Most recent checkpoint
│   ├── checkpoint_epoch_10.pth # Regular intervals
│   ├── checkpoint_epoch_20.pth
│   └── final_model.pth         # End of training
```

#### Checkpoint Validation
```python
def validate_checkpoint(checkpoint_path):
    """
    Validate checkpoint integrity
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            print(f"Warning: Missing keys in checkpoint: {missing_keys}")
            return False
        
        print(f"✓ Checkpoint valid: epoch {checkpoint['epoch']}, loss {checkpoint.get('loss', 'N/A')}")
        return True
        
    except Exception as e:
        print(f"✗ Checkpoint invalid: {e}")
        return False
```

### Memory-Efficient Checkpointing
```python
# For large models, save only essential components
def save_lightweight_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save minimal checkpoint for memory efficiency
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {
            'state': optimizer.state_dict()['state'],
            'param_groups': optimizer.state_dict()['param_groups']
        },
        'epoch': epoch,
        'loss': loss,
    }
    torch.save(checkpoint, filepath)

# Clean up old checkpoints to save disk space
def cleanup_checkpoints(checkpoint_dir, keep_last=5):
    """
    Keep only the most recent N checkpoints
    """
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if len(checkpoints) > keep_last:
        for checkpoint in checkpoints[:-keep_last]:
            os.remove(checkpoint)
            print(f"Removed old checkpoint: {checkpoint}")
```

## Final Pre-Training Checklist

### Before Starting Training
- [ ] Data pipeline tested and validated
- [ ] Model architecture verified
- [ ] Loss function appropriate for task
- [ ] Optimizer and scheduler configured
- [ ] Gradient clipping parameters set
- [ ] Checkpoint directory created
- [ ] GPU memory usage estimated
- [ ] Baseline metrics established
- [ ] Logging and monitoring setup
- [ ] Early stopping criteria defined

### During Training Monitor
- [ ] Training and validation loss trends
- [ ] Learning rate schedule
- [ ] Gradient norms
- [ ] GPU memory usage
- [ ] Training speed (samples/sec)
- [ ] Checkpoint file sizes
- [ ] Model convergence indicators

### Post-Training Validation
- [ ] Final model performance evaluation
- [ ] Best checkpoint identified
- [ ] Training curves analyzed
- [ ] Model artifacts organized
- [ ] Results documented
- [ ] Model ready for deployment/inference