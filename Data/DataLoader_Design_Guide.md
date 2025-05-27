# Complete PyTorch DataLoader Guide for Computer Vision

## Table of Contents
1. [Core Architecture & Design Patterns](#core-architecture--design-patterns)
2. [Dataset Class Implementation](#dataset-class-implementation)
3. [Transform Pipeline Design](#transform-pipeline-design)
4. [DataLoader Configuration](#dataloader-configuration)
5. [Handling Common Challenges](#handling-common-challenges)
6. [Advanced Patterns & Optimizations](#advanced-patterns--optimizations)
7. [Production Considerations](#production-considerations)

---

## Core Architecture & Design Patterns

### Dataset vs DataLoader Responsibilities

**Dataset Class (`torch.utils.data.Dataset`)**
- **Purpose**: Defines how to access individual samples
- **Responsibilities**: Data loading, preprocessing, transformations, label mapping
- **Key Methods**: `__init__`, `__len__`, `__getitem__`
- **Design Philosophy**: Lazy loading, stateless operations, indexable interface

**DataLoader Class (`torch.utils.data.DataLoader`)**
- **Purpose**: Batching, shuffling, parallel processing
- **Responsibilities**: Sampling strategy, collation, multiprocessing coordination
- **Key Parameters**: `batch_size`, `shuffle`, `num_workers`, `sampler`
- **Design Philosophy**: Efficient batch generation, memory management

### Common Design Patterns

#### 1. **Map-Style vs Iterable-Style Datasets**
```python
# Map-Style (most common for CV)
class MapStyleDataset(Dataset):
    def __getitem__(self, index): pass
    def __len__(self): pass

# Iterable-Style (for streaming data)
class IterableStyleDataset(IterableDataset):
    def __iter__(self): pass
```

**When to Use:**
- **Map-Style**: Fixed-size datasets, random access needed, most CV tasks
- **Iterable-Style**: Streaming data, databases, real-time data feeds

#### 2. **Inheritance Patterns**
```python
# Base dataset for common functionality
class BaseVisionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
    
    def _load_image(self, path):
        # Common image loading logic
        pass

# Specific implementations
class ClassificationDataset(BaseVisionDataset): pass
class DetectionDataset(BaseVisionDataset): pass
```

---

## Dataset Class Implementation

### The `__init__` Method: Critical Design Decisions

#### File Discovery Strategies
```python
def __init__(self, root_dir, transform=None):
    # Strategy 1: Directory Walking (flexible)
    self.image_paths, self.labels = self._discover_files()
    
    # Strategy 2: Manifest File (faster, reproducible)
    self.samples = self._load_manifest("dataset_manifest.json")
    
    # Strategy 3: Database Index (scalable)
    self.db_connection = sqlite3.connect("dataset.db")
```

**Trade-offs:**
- **Directory Walking**: Flexible, handles new files automatically, slower initialization
- **Manifest Files**: Fast, reproducible, requires maintenance
- **Database**: Scalable, queryable, adds complexity

#### Label Mapping Design
```python
# Approach 1: String-to-Index Mapping (recommended)
self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(classes))}

# Approach 2: Pre-defined Mapping (for consistency across splits)
self.class_to_idx = self._load_class_mapping("classes.json")

# Approach 3: Hierarchical Labels
self.label_hierarchy = {
    "animals": {"mammals": ["cat", "dog"], "birds": ["eagle", "sparrow"]}
}
```

**Best Practices:**
- Always sort class names for reproducible mapping
- Store class mappings for inference consistency
- Consider hierarchical labels for fine-grained classification

#### Memory vs Speed Trade-offs
```python
# Lazy Loading (memory efficient)
self.image_paths = [...]  # Store paths only

# Eager Loading (speed optimized)
self.images = [self._load_image(path) for path in paths]  # Load all images

# Hybrid Approach (balanced)
self.image_cache = {}  # LRU cache for frequently accessed images
```

### The `__getitem__` Method: Robust Implementation

#### Error Handling Strategies
```python
def __getitem__(self, index):
    # Strategy 1: Retry with neighboring samples
    max_retries = 3
    for attempt in range(max_retries):
        try:
            actual_index = (index + attempt) % len(self.samples)
            return self._load_sample(actual_index)
        except Exception as e:
            continue
    raise RuntimeError(f"Failed to load sample after {max_retries} attempts")
    
    # Strategy 2: Return placeholder/default sample
    try:
        return self._load_sample(index)
    except:
        return self._get_default_sample()
    
    # Strategy 3: Fail fast (for debugging)
    return self._load_sample(index)  # Let it crash
```

**When to Use Each:**
- **Retry**: Production systems, occasional corruption expected
- **Placeholder**: When model can handle dummy samples
- **Fail Fast**: Development, debugging, data validation

#### Sample Loading Patterns
```python
def __getitem__(self, index):
    # Pattern 1: Single sample loading
    image, label = self._load_single(index)
    
    # Pattern 2: Multi-modal loading
    image = self._load_image(index)
    metadata = self._load_metadata(index)
    label = self._load_label(index)
    
    # Pattern 3: Sequence loading (for video/time series)
    sequence = self._load_sequence(index, sequence_length=16)
    
    return self._apply_transforms(image), label
```

---

## Transform Pipeline Design

### Transform Order: Critical Sequence
```python
# CORRECT ORDER
transforms.Compose([
    # 1. PIL Image Operations (spatial transforms)
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    
    # 2. Convert to Tensor (PIL -> Tensor)
    transforms.ToTensor(),  # Converts to [0,1] float tensor
    
    # 3. Tensor Operations (mathematical transforms)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

**Why This Order Matters:**
- PIL operations work on PIL Images (0-255 integer values)
- ToTensor() converts PIL -> Tensor and scales to [0,1]
- Normalize() expects tensors in [0,1] range

### Resize Strategies
```python
# Strategy 1: Fixed Resize (distorts aspect ratio)
transforms.Resize((224, 224))

# Strategy 2: Aspect-Preserving Resize + Crop
transforms.Resize(256),  # Resize shorter side to 256
transforms.CenterCrop(224)  # Extract 224x224 center

# Strategy 3: Random Resized Crop (data augmentation)
transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1))

# Strategy 4: Padding to Square
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        return transforms.functional.pad(image, (hp, vp, hp, vp))
```

**Decision Matrix:**
- **Fixed Resize**: Simple, consistent size, may distort objects
- **Resize + Crop**: Preserves aspect ratio, may lose information
- **RandomResizedCrop**: Great for augmentation, variable content per epoch
- **Padding**: Preserves all content, adds empty space

### Normalization Strategies
```python
# ImageNet Normalization (for pre-trained models)
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Custom Dataset Normalization
def calculate_dataset_stats(dataloader):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for data, _ in dataloader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
    mean /= len(dataloader.dataset)
    std /= len(dataloader.dataset)
    return mean, std

# No Normalization (for models trained from scratch)
# Just use ToTensor() which gives [0,1] range
```

**When to Use:**
- **ImageNet Stats**: Transfer learning with pre-trained models
- **Custom Stats**: Training from scratch on domain-specific data
- **No Normalization**: Simple models, debugging, specific architectures

### Augmentation Strategies by Domain

#### Natural Images/Photography
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
])
```

#### Medical Images
```python
transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),  # Careful with anatomical images
    # NO ColorJitter (preserve diagnostic information)
    # NO RandomVerticalFlip (anatomical orientation matters)
])
```

#### Satellite/Aerial Images
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),  # OK for overhead views
    transforms.RandomRotation(degrees=90, expand=True),  # Any orientation valid
    transforms.ColorJitter(brightness=0.3, contrast=0.3)  # Weather variations
])
```

#### Document/Text Images
```python
transforms.Compose([
    transforms.RandomAffine(degrees=2, translate=(0.02, 0.02)),  # Small variations
    # NO horizontal/vertical flips (text orientation matters)
    # NO strong color augmentation (preserve text readability)
])
```

---

## DataLoader Configuration

### Core Parameters Deep Dive

#### Batch Size Selection
```python
# Rule of thumb: Largest batch that fits in GPU memory
batch_sizes = {
    "development": 16,      # Fast iteration
    "training": 64,         # Good balance
    "inference": 128,       # Memory allows
    "mobile_deployment": 1  # Real-time processing
}

# Dynamic batch size calculation
def find_optimal_batch_size(model, dataloader, device):
    batch_size = 2
    while True:
        try:
            batch = next(iter(DataLoader(dataloader.dataset, batch_size=batch_size)))
            output = model(batch[0].to(device))
            loss = F.cross_entropy(output, batch[1].to(device))
            loss.backward()
            batch_size *= 2
        except RuntimeError:  # Out of memory
            return batch_size // 2
```

#### Worker Process Configuration
```python
# CPU cores available
num_workers_options = {
    "single_threaded": 0,           # Debugging, small datasets
    "conservative": min(4, os.cpu_count()),  # Safe default
    "aggressive": os.cpu_count(),   # Maximum parallelism
    "gpu_bound": 2,                 # When GPU is bottleneck
}

# Platform-specific considerations
import platform
if platform.system() == "Windows":
    num_workers = 0  # Multiprocessing issues on Windows
elif torch.cuda.is_available():
    num_workers = 4  # Balance CPU-GPU pipeline
else:
    num_workers = os.cpu_count()
```

#### Memory Optimization
```python
DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),  # Faster CPU->GPU transfer
    persistent_workers=True,               # Keep workers alive between epochs
    prefetch_factor=2,                     # Batches to prefetch per worker
    drop_last=False,                       # Keep smaller final batch
)
```

### Shuffling Strategies
```python
# Basic shuffling
DataLoader(dataset, shuffle=True)  # Random order each epoch

# Custom shuffling
class CustomSampler(Sampler):
    def __init__(self, dataset, seed=42):
        self.dataset = dataset
        self.seed = seed
    
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g)
        return iter(indices.tolist())

# No shuffling (for validation/testing)
DataLoader(dataset, shuffle=False)  # Consistent order
```

---

## Handling Common Challenges

### Class Imbalance Solutions

#### WeightedRandomSampler Implementation
```python
def create_weighted_sampler(dataset, sampling_strategy="inverse_frequency"):
    # Count samples per class
    class_counts = Counter(dataset.labels)
    
    if sampling_strategy == "inverse_frequency":
        # Weight = 1 / frequency
        class_weights = {cls: 1.0/count for cls, count in class_counts.items()}
    
    elif sampling_strategy == "balanced":
        # Weight to achieve equal representation
        total_samples = len(dataset)
        num_classes = len(class_counts)
        target_per_class = total_samples / num_classes
        class_weights = {cls: target_per_class/count for cls, count in class_counts.items()}
    
    elif sampling_strategy == "sqrt_inverse":
        # Softer rebalancing
        class_weights = {cls: 1.0/math.sqrt(count) for cls, count in class_counts.items()}
    
    # Create sample weights
    sample_weights = [class_weights[label] for label in dataset.labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True  # Allow sampling same image multiple times
    )
```

#### Alternative: Weighted Loss Functions
```python
# In your training loop
class_counts = Counter(train_dataset.labels)
total_samples = sum(class_counts.values())
class_weights = [total_samples / (len(class_counts) * count) 
                for count in class_counts.values()]
class_weights = torch.tensor(class_weights, dtype=torch.float)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

### Large Dataset Handling

#### Memory-Mapped Files
```python
class MemoryMappedDataset(Dataset):
    def __init__(self, data_file):
        # Memory map the file instead of loading into RAM
        self.data = np.memmap(data_file, dtype=np.uint8, mode='r')
        self.indices = self._build_index()
    
    def __getitem__(self, idx):
        start, end = self.indices[idx]
        image_bytes = self.data[start:end]
        image = Image.open(io.BytesIO(image_bytes))
        return image
```

#### Streaming from Cloud Storage
```python
class S3Dataset(Dataset):
    def __init__(self, bucket_name, prefix):
        self.s3_client = boto3.client('s3')
        self.bucket = bucket_name
        self.keys = self._list_objects(prefix)
    
    def __getitem__(self, idx):
        response = self.s3_client.get_object(
            Bucket=self.bucket, 
            Key=self.keys[idx]
        )
        image = Image.open(response['Body'])
        return image
```

### Multi-Modal Data
```python
class MultiModalDataset(Dataset):
    def __init__(self, image_dir, metadata_file):
        self.image_paths = self._get_image_paths(image_dir)
        self.metadata = pd.read_csv(metadata_file)
        self.image_transform = transforms.Compose([...])
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx])
        image = self.image_transform(image)
        
        # Load metadata
        metadata_row = self.metadata.iloc[idx]
        numerical_features = torch.tensor(metadata_row[['age', 'height', 'weight']].values)
        categorical_features = torch.tensor(metadata_row['category_encoded'])
        
        return {
            'image': image,
            'numerical': numerical_features,
            'categorical': categorical_features,
            'label': torch.tensor(metadata_row['label'])
        }

# Custom collate function for multi-modal data
def multimodal_collate_fn(batch):
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'numerical': torch.stack([item['numerical'] for item in batch]),
        'categorical': torch.stack([item['categorical'] for item in batch]),
        'label': torch.stack([item['label'] for item in batch])
    }

dataloader = DataLoader(dataset, collate_fn=multimodal_collate_fn)
```

---

## Advanced Patterns & Optimizations

### Custom Collate Functions
```python
# Variable-size images in same batch
def variable_size_collate_fn(batch):
    images, labels = zip(*batch)
    
    # Find max dimensions
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    
    # Pad all images to same size
    padded_images = []
    for img in images:
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        padded = F.pad(img, (0, pad_w, 0, pad_h))
        padded_images.append(padded)
    
    return torch.stack(padded_images), torch.tensor(labels)

# Mixed precision support
def mixed_precision_collate_fn(batch):
    images, labels = default_collate(batch)
    return images.half(), labels  # Convert images to float16
```

### Online Hard Example Mining
```python
class HardExampleDataset(Dataset):
    def __init__(self, base_dataset, model, device):
        self.base_dataset = base_dataset
        self.model = model
        self.device = device
        self.hard_examples = set()
        self.difficulty_scores = {}
    
    def update_difficulty_scores(self, epoch):
        """Update difficulty scores based on model predictions"""
        self.model.eval()
        with torch.no_grad():
            for idx in range(len(self.base_dataset)):
                image, label = self.base_dataset[idx]
                pred = self.model(image.unsqueeze(0).to(self.device))
                loss = F.cross_entropy(pred, label.unsqueeze(0).to(self.device))
                self.difficulty_scores[idx] = loss.item()
        
        # Mark top 20% as hard examples
        sorted_indices = sorted(self.difficulty_scores.keys(), 
                              key=lambda x: self.difficulty_scores[x], reverse=True)
        cutoff = len(sorted_indices) // 5
        self.hard_examples = set(sorted_indices[:cutoff])
    
    def __getitem__(self, idx):
        # Oversample hard examples
        if idx in self.hard_examples and random.random() < 0.7:
            return self.base_dataset[idx]
        else:
            return self.base_dataset[idx]
```

### Curriculum Learning
```python
class CurriculumDataset(Dataset):
    def __init__(self, base_dataset, difficulty_fn):
        self.base_dataset = base_dataset
        self.difficulty_fn = difficulty_fn
        self.current_difficulty = 0.0
        self.difficulties = [difficulty_fn(item) for item in base_dataset]
    
    def set_difficulty_threshold(self, threshold):
        self.current_difficulty = threshold
    
    def __len__(self):
        return sum(1 for d in self.difficulties if d <= self.current_difficulty)
    
    def __getitem__(self, idx):
        available_indices = [i for i, d in enumerate(self.difficulties) 
                           if d <= self.current_difficulty]
        actual_idx = available_indices[idx]
        return self.base_dataset[actual_idx]
```

### Caching Strategies
```python
class CachedDataset(Dataset):
    def __init__(self, base_dataset, cache_size=1000):
        self.base_dataset = base_dataset
        self.cache = {}
        self.cache_size = cache_size
        self.access_count = defaultdict(int)
    
    def __getitem__(self, idx):
        if idx in self.cache:
            self.access_count[idx] += 1
            return self.cache[idx]
        
        # Load from base dataset
        item = self.base_dataset[idx]
        
        # Add to cache if space available
        if len(self.cache) < self.cache_size:
            self.cache[idx] = item
        else:
            # Replace least frequently used item
            lfu_idx = min(self.cache.keys(), key=lambda x: self.access_count[x])
            del self.cache[lfu_idx]
            del self.access_count[lfu_idx]
            self.cache[idx] = item
        
        self.access_count[idx] += 1
        return item
```

---

## Production Considerations

### Validation and Testing Best Practices
```python
def create_train_val_test_splits(dataset, train_ratio=0.7, val_ratio=0.15):
    """Create stratified splits maintaining class distribution"""
    from sklearn.model_selection import train_test_split
    
    # Get labels for stratification
    labels = [dataset.labels[i] for i in range(len(dataset))]
    indices = list(range(len(dataset)))
    
    # Create train/temp split
    train_idx, temp_idx = train_test_split(
        indices, test_size=(1-train_ratio), 
        stratify=labels, random_state=42
    )
    
    # Create val/test split from temp
    temp_labels = [labels[i] for i in temp_idx]
    val_ratio_adjusted = val_ratio / (val_ratio + (1-train_ratio-val_ratio))
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1-val_ratio_adjusted),
        stratify=temp_labels, random_state=42
    )
    
    return train_idx, val_idx, test_idx

# Separate transforms for different splits
def get_transforms(split_type):
    if split_type == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:  # val/test
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
```

### Error Monitoring and Logging
```python
class MonitoredDataset(Dataset):
    def __init__(self, base_dataset, log_file="dataset_errors.log"):
        self.base_dataset = base_dataset
        self.logger = self._setup_logger(log_file)
        self.error_count = defaultdict(int)
        self.total_requests = 0
    
    def __getitem__(self, idx):
        self.total_requests += 1
        try:
            return self.base_dataset[idx]
        except Exception as e:
            error_type = type(e).__name__
            self.error_count[error_type] += 1
            self.logger.error(f"Error loading index {idx}: {str(e)}")
            
            # Fallback strategy
            return self._get_fallback_sample()
    
    def get_error_statistics(self):
        return {
            'total_requests': self.total_requests,
            'error_counts': dict(self.error_count),
            'error_rate': sum(self.error_count.values()) / self.total_requests
        }
```

### Performance Profiling
```python
import time
from contextlib import contextmanager

class ProfiledDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.timing_stats = defaultdict(list)
    
    @contextmanager
    def timer(self, operation):
        start = time.time()
        yield
        self.timing_stats[operation].append(time.time() - start)
    
    def __iter__(self):
        for batch in self.dataloader:
            with self.timer('batch_loading'):
                yield batch
    
    def print_stats(self):
        for operation, times in self.timing_stats.items():
            avg_time = sum(times) / len(times)
            print(f"{operation}: {avg_time:.4f}s avg, {len(times)} samples")

# Usage
profiled_loader = ProfiledDataLoader(train_loader)
for epoch in range(num_epochs):
    for batch in profiled_loader:
        # Training code
        pass
    profiled_loader.print_stats()
```

### Reproducibility
```python
def set_seed(seed=42):
    """Set seed for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    """Initialize each worker with different seed"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Use in DataLoader
DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    worker_init_fn=worker_init_fn,
    generator=torch.Generator().manual_seed(42)
)
```

---

## Decision Tree Summary

### When to Use Different Approaches

**Dataset Size:**
- **< 1GB**: Load all in memory, simple transforms
- **1GB - 100GB**: Lazy loading, efficient transforms, caching
- **> 100GB**: Streaming, distributed loading, memory mapping

**Data Type:**
- **Natural Images**: Standard augmentations, ImageNet normalization
- **Medical Images**: Careful augmentation, domain-specific normalization
- **Satellite Images**: Rotation-invariant augmentations
- **Documents**: Minimal augmentation, preserve text orientation

**Class Balance:**
- **Balanced**: Simple random sampling
- **Moderate Imbalance (1:10)**: WeightedRandomSampler or weighted loss
- **Severe Imbalance (1:100+)**: Combine sampling + loss weighting + evaluation metrics

**Hardware:**
- **Single GPU**: Batch size = GPU memory limit, num_workers = 4-8
- **Multi-GPU**: Larger batch sizes, DistributedSampler
- **CPU Only**: Smaller batch sizes, higher num_workers
- **Limited RAM**: Streaming datasets, memory mapping

**Training Stage:**
- **Development**: Small batch sizes, num_workers=0, simple transforms
- **Training**: Optimal batch sizes, full augmentation pipeline
- **Inference**: Larger batch sizes, no augmentation, deterministic

This guide covers the essential patterns and trade-offs for building robust, efficient DataLoaders for computer vision tasks. Choose components based on your specific requirements, and always profile your implementation to identify bottlenecks.