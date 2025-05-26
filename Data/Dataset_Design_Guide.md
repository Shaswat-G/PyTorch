# Complete PyTorch Dataset Design Guide: From Concepts to Implementation

## 1. Foundation: The Three Essential Methods

**Core Pattern:**
- `__init__`: Setup phase (load metadata, compute statistics, initialize state)
- `__getitem__`: Per-sample retrieval (return processed sample)
- `__len__`: Return total sample count

**Mental Model:** Setup → Access → Count

---

## 2. Data Loading Strategy Decision Tree

**Decision Point:** Eager vs Lazy Loading

### Eager Loading (Pattern A)
**When to use:**
- Dataset fits in memory (< 10% of available RAM)
- Fast repeated access needed
- Simple preprocessing pipeline

**Implementation:**
```python
def __init__(self):
    self.data = pd.read_csv(path)  # Load everything
    self.data = self._preprocess_all(self.data)  # Process everything

def __getitem__(self, idx):
    return self._apply_runtime_transforms(self.data.iloc[idx])  # Simple indexing
```

**Trade-offs:**
- ✅ Fast `__getitem__` calls
- ✅ Consistent performance
- ❌ High memory usage
- ❌ Slow initialization

### Lazy Loading (Pattern B)
**When to use:**
- Large datasets (> 50% of available RAM)
- Memory-constrained environments
- Infrequent access patterns

**Implementation:**
```python
def __init__(self):
    self.csv_path = path  # Store path only
    self.stats = self._compute_statistics()  # Compute stats once

def __getitem__(self, idx):
    sample = self._load_single_row(idx)  # Load on demand
    return self._apply_all_transforms(sample)  # Full processing pipeline
```

**Trade-offs:**
- ✅ Low memory usage
- ✅ Fast initialization
- ❌ Slower `__getitem__` calls
- ❌ Inconsistent performance

### Auto-Detection Strategy
```python
def _detect_strategy(self, csv_path, memory_threshold_mb=100):
    file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    estimated_memory = file_size_mb * 3  # Heuristic for DataFrame overhead
    return "eager" if estimated_memory < memory_threshold_mb else "lazy"
```

---

## 3. Statistics Computation Strategies

**Decision Point:** How to compute dataset-wide statistics (mean/std for normalization)

### Approach 1: Two-Pass (Recommended for learning)
**When:** Dataset too large for memory, need exact statistics
```python
# Pass 1: Compute statistics
stats = self._compute_stats_streaming(csv_path)
# Pass 2: Apply in __getitem__
normalized = (value - stats['mean']) / stats['std']
```

### Approach 2: Running Statistics
**When:** Online learning, streaming data
```python
class RunningStats:
    def update(self, value):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
```

### Approach 3: Sample-Based Estimation
**When:** Very large datasets, approximate statistics acceptable
```python
sample_df = pd.read_csv(path, skiprows=random_skip_indices)
stats = sample_df.describe()
```

### Approach 4: Pre-computed
**When:** Repeated dataset usage, statistics won't change
```python
# Compute once, save to disk
save_stats(dataset_stats, "dataset_stats.json")
# Load in __init__
self.stats = load_stats("dataset_stats.json")
```

---

## 4. Transform Architecture

**Key Insight:** Different transforms have different timing requirements

### Transform Categories:

1. **Data Cleaning** (once only)
   - Handle missing values, data type conversion
   - Apply in: `__init__` (eager) or `__getitem__` (lazy)

2. **Standardization** (once only, needs dataset stats)
   - Normalization, scaling
   - Compute stats in: `__init__` (always)
   - Apply in: `__init__` (eager) or `__getitem__` (lazy)

3. **Data Augmentation** (every access, often random)
   - Add noise, random transforms
   - Apply in: `__getitem__` (always)

### Implementation Pattern:
```python
def __init__(self, transforms=None):
    self.transforms = transforms or {}
    # Always compute dataset statistics
    self.stats = self._compute_stats()
    
    if self.processing_strategy == "eager":
        # Apply cleaning + standardization once
        self.data = self._apply_static_transforms(raw_data)

def __getitem__(self, idx):
    if self.processing_strategy == "eager":
        sample = self.data.iloc[idx]
        # Apply only augmentation transforms
        return self._apply_runtime_transforms(sample)
    else:
        # Apply all transforms: cleaning + standardization + augmentation
        sample = self._load_single_row(idx)
        return self._apply_all_transforms(sample)
```

---

## 5. Error Handling Strategy

**Decision Point:** How to handle invalid indices

### Options:
1. **Raise IndexError** - Fails fast, stops training
2. **Return None** - Can break downstream code
3. **Modulo wrap-around** - Prevents crashes, continues training ✅

**Implementation:**
```python
def __getitem__(self, idx):
    idx = idx % len(self)  # Wrap around
    # ... rest of method
```

---

## 6. Architecture Patterns by Domain

### Computer Vision
```python
# Eager loading common for image paths
def __init__(self):
    self.image_paths = glob.glob("images/*.jpg")
    self.labels = pd.read_csv("labels.csv")

def __getitem__(self, idx):
    image = Image.open(self.image_paths[idx])  # Load on demand
    return self.transforms(image), self.labels[idx]
```

### NLP
```python
# Mixed: metadata eager, text processing lazy/cached
def __init__(self):
    self.texts = pd.read_csv("texts.csv")  # If fits in memory
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)

def __getitem__(self, idx):
    tokens = self.tokenizer(self.texts.iloc[idx]['text'])
    return tokens, self.texts.iloc[idx]['label']
```

### Tabular
```python
# Usually eager loading (fits in memory)
def __init__(self):
    self.data = pd.read_csv(path)
    self.data = self._preprocess_tabular(self.data)  # Clean + encode

def __getitem__(self, idx):
    sample = self.data.iloc[idx]
    features = sample[self.feature_cols].values
    target = sample[self.target_col]
    return torch.tensor(features), torch.tensor(target)
```

### Time Series
```python
# Sliding windows - often computed in __init__
def __init__(self):
    raw_data = pd.read_csv(path)
    self.windows = self._create_sliding_windows(raw_data, window_size=50)

def __getitem__(self, idx):
    return self.windows[idx]['sequence'], self.windows[idx]['target']
```

---

## 7. Complete Implementation Checklist

### `__init__` Method Checklist:
- [ ] Parameter validation and defaults
- [ ] Strategy detection (eager vs lazy)
- [ ] Dataset length computation
- [ ] Statistics computation (for normalization)
- [ ] Conditional data loading based on strategy
- [ ] Transform pipeline initialization
- [ ] Metadata storage (column names, data types)

### `__getitem__` Method Checklist:
- [ ] Index validation and wrapping
- [ ] Conditional data loading (lazy mode only)
- [ ] Transform application based on strategy
- [ ] Proper tensor conversion
- [ ] Return format consistency (tuple vs dict)
- [ ] Error handling for corrupted data

### `__len__` Method Checklist:
- [ ] Return cached length (computed in `__init__`)

### Helper Methods Checklist:
- [ ] `_detect_strategy()` - Memory-based decision making
- [ ] `_compute_statistics()` - Dataset-wide stats
- [ ] `_load_single_row()` - Efficient single sample loading
- [ ] `_apply_static_transforms()` - One-time preprocessing
- [ ] `_apply_runtime_transforms()` - Per-access transforms
- [ ] `_validate_data()` - Data integrity checks

---

## 8. Performance and Memory Trade-offs Summary

| Aspect | Eager Loading | Lazy Loading |
|--------|---------------|--------------|
| **Memory Usage** | High (stores all data) | Low (stores metadata only) |
| **Init Time** | Slow (processes everything) | Fast (minimal setup) |
| **Access Time** | Fast (simple indexing) | Slower (processing per access) |
| **Consistency** | Consistent performance | Variable performance |
| **Best For** | Small-medium datasets, repeated access | Large datasets, memory-constrained |
| **Dataset Size** | < 10% available RAM | > 50% available RAM |

This comprehensive guide provides the decision framework and implementation patterns for designing robust, scalable PyTorch datasets across different domains and use cases.