# PyTorch Dataset/DataLoader Interview Quiz

## Instructions

This comprehensive quiz tests your understanding of PyTorch's data loading infrastructure, covering Dataset design, DataLoader configuration, and best practices for production machine learning pipelines.

### Direct-Recall

**Q1:** What are the three essential methods that must be implemented when creating a custom PyTorch Dataset class?
1. __init__ -> will validate parameters, initialize file paths. Then auto-detect loading and transformation strategy (Lazy vs eager basedon total dataset size and available ram). Then we compute data-set wide stats 
irrespective of laoding strategy to standardize (in the init method or getitem method depending on eager vs lazy), we apply transofmrations (Cleaning, Merging, standardizing), label mapping.
1. __len__ -> outputs the length of samples - could be len(labels), len(samples), etc or could be dataset wide statistic computed form the init -> total length
2. getitem ->given an index returns either a tuple of feature tesnor, label tensor or a dictionary if we need to return other data (metadata). Runtime transformations like augmentations always happen here and optionally the entire set of transformations (cleaning, pre-processing, standardizing and augmentations and tensor conversion) my happen here if we follow a lazy strategy. It is recommended to do a modulo wraparound to deal with index errors, and linear probing for 3-5 attempts if corrupted files.

**Q2:** What is the difference between Map-Style and Iterable-Style datasets? When would you use each?
for mapstyle we inherit form the torch.utils.data.Dataset and torch.utils.data.IterableDataset for streaming data. If our data is finite and indexable -> we use map style, if our data is not finite and is streaming, we use IterableDataset -> this could involve setting up DB connection engnies in __init__ and __iter__ method where we yield the feature and labels.

**Q3:** Name three key parameters of DataLoader that affect performance and explain their purpose.
1. Batch Size : It is the number of samples that are stacked into a batch after sampling from the dataset. Higher is better (your monte carlo gradient estiamte is better), but sometkmes you want noisy gradients for SGD to go out of local minimas and so try to achieve max without CUDA OOM error but less than 10% of dataset size.
2. Number of workers for multiprocessing -> 2-4 is commong, can be more = os.cpu_count() for max parallelism or eveng higerh for bigger GPUs -> 8
3. pin_memory -> faster CPU to GPU transfer. Others inlclude persisiting workers to prevent high init times for each process pool across epochs, prefetching samples and shuffiling -> where we can do weighted sampling.

**Q4:** What is the primary responsibility difference between Dataset and DataLoader classes?
Dataset class is to provide access to the dataset as a processed map-style object (Dataset) or an iterable (Streaming Dataset in IterableDataset). DataLoader handles batching (sampling, shuffling, collation), and multiprocessing arrangement to serve batches for training.

**Q5:** What are the trade-offs between eager loading vs lazy loading strategies in Dataset implementation?
Classic memory vs speed -> in-memry fully init dataset in eager loading take more time in init but getitem calls are much faster, whereas its the exact opposite in lazy loading.

### True/False

**T/F 1:** DataLoader can only work with Map-Style datasets that implement `__getitem__` and `__len__`. 

- **Answer:** False. DataLoader can work with both Map-Style and Iterable-Style datasets.

**T/F 2:** Setting `pin_memory=True` in DataLoader always improves performance when training on GPU.

- **Answer:** False. It only helps when transferring data to GPU and can increase memory usage.

**T/F 3:** The `num_workers` parameter in DataLoader should always be set to the number of CPU cores for optimal performance.

- **Answer:** False. Optimal number depends on I/O vs CPU bound operations and can cause overhead if too high.

### Short Answers

**Definition 1:** **Collate Function**
The collate function in DataLoader determines how individual samples are combined into batches. It takes a list of samples and returns a batched tensor, handling padding, stacking, and data type conversions as needed.

**Definition 2:** **WeightedRandomSampler**
A sampling strategy that allows controlling the probability of selecting each sample during training. Used for handling class imbalance by giving higher weights to underrepresented classes, ensuring balanced training.

**Definition 3:** **Transform Pipeline**
A sequence of data preprocessing operations (normalization, augmentation, type conversion) applied to samples. Typically separated into data transforms and target transforms, allowing modular and reusable preprocessing logic.

### Debug Challenges

**Debug Challenge 1:**

```python
class BrokenDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

# Usage
dataset = BrokenDataset([1, 2, 3, 4, 5])
dataloader = DataLoader(dataset, batch_size=2)
for batch in dataloader:
    print(batch)  # This will fail!
```

**Fix:** Missing `__len__` method. Add:

```python
def __len__(self):
    return len(self.data)
```

**Debug Challenge 2:**

```python
class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        return self.transform(image)  # Error with RGB images!
```

**Fix:** Normalization expects 3 channels for RGB. Change to:

```python
transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
```

### Design Prompt

**Design a complete Dataset class for a medical image classification task with the following requirements:**

Design `MedicalImageDataset` that should include:

**`__init__` method:**

- Accept `root_dir`, `csv_file` with image paths and labels
- Support optional `transform` and `target_transform`
- Implement error handling for missing files
- Cache metadata for performance

**`__len__` method:**

- Return total number of valid samples

**`__getitem__` method:**

- Load image from disk with retry mechanism
- Apply transforms with error handling
- Return (image_tensor, label_tensor) tuple
- Handle corrupted files gracefully

**Additional features:**

- Support for class balancing via WeightedRandomSampler
- Transforms for data augmentation (rotation, flip, normalization)
- Memory-efficient lazy loading for large datasets
- Logging for debugging and monitoring

### Match-The-Columns

Match each DataLoader method/parameter with its primary purpose:

| Parameter/Method | Purpose                                                |
| ---------------- | ------------------------------------------------------ |
| A. `batch_size`  | 1. Controls parallel data loading processes            |
| B. `shuffle`     | 2. Determines how samples are combined into batches    |
| C. `num_workers` | 3. Optimizes GPU memory transfer speed                 |
| D. `pin_memory`  | 4. Randomizes sample order between epochs              |
| E. `collate_fn`  | 5. Sets number of samples per training iteration       |
| F. `sampler`     | 6. Defines custom sampling strategy for data selection |

**Answers:** A-5, B-4, C-1, D-3, E-2, F-6

### One-Line Code

**Prompt:** Write a one-line transform that converts a PIL image to tensor and normalizes it with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

**Answer:**

```python
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
```

### Long-Form

**Question:** Explain the relationship between Dataset, DataLoader, and Sampler in PyTorch's data loading pipeline, and describe how they work together to enable efficient batch processing during model training.

**Answer:**
PyTorch's data loading architecture follows a clear separation of concerns where each component has distinct responsibilities that work together seamlessly. The Dataset class serves as the foundation, defining how individual samples are accessed and processed through `__getitem__` and `__len__` methods, encapsulating all data-specific logic like file I/O, preprocessing, and transformations. The DataLoader acts as an orchestrator that wraps the Dataset and handles batch-level operations including parallel processing via multiple workers, memory management with pin_memory optimization, and coordination between CPU and GPU memory spaces. The Sampler component (often working behind the scenes) determines the order and strategy for selecting samples from the Dataset, whether through simple sequential access, random shuffling, or sophisticated strategies like weighted sampling for handling class imbalance. Together, this design enables scalable, efficient data pipelines where the Dataset focuses on single-sample processing, the DataLoader manages batch creation and parallel loading, and the Sampler controls data distribution, allowing developers to optimize each component independently for their specific use case while maintaining clean, maintainable code that can scale from prototypes to production systems.
