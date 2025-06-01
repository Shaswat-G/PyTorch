# PyTorch: Zero to Hero

A comprehensive, hands-on curriculum for mastering PyTorch and deep learning fundamentals.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Course Overview

This repository provides a complete curriculum for learning PyTorch from fundamentals to production deployment. Each module builds systematically on previous concepts, with runnable code examples and detailed explanations throughout.

**Designed for:** Machine Learning Engineers, Researchers, and Students transitioning from theory to practical implementation

## Curriculum Structure

### 1. Tensors [`/Tensors`](Tensors/)

Foundation of all deep learning operations

- **Core Concepts**: Creation, manipulation, broadcasting, device management
- **Advanced Topics**: Memory optimization, GPU acceleration, autograd integration
- **Practice**: 100 progressive exercises with verified solutions

```python
# Tensor operations fundamentals
x = torch.randn(3, 4, device='cuda')
y = torch.softmax(x, dim=-1)
```

**Key Features:**

- Complete tensor operations reference
- Performance optimization techniques
- Real-world application examples

### 2. Data Pipeline [`/Data`](Data/)

Professional-grade data loading and preprocessing

- **Dataset Design**: Map-style vs Iterable datasets, lazy vs eager loading
- **DataLoader Configuration**: Batching, multiprocessing, memory optimization
- **Production Patterns**: Error handling, transforms, sampling strategies

```python
class SmartDataset(Dataset):
    def __init__(self, root_dir, strategy='auto'):
        self.strategy = self._detect_loading_strategy(root_dir)

    def __getitem__(self, idx):
        return self._load_with_retry(idx, max_attempts=3)
```

**Key Features:**

- Complete design guide with decision trees
- Interview preparation materials
- Production-ready implementations

### 3. Model Architecture [`/Model`](Model/)

Building blocks of modern neural networks

- **MLPs**: Flexible multi-layer perceptrons with ModuleList patterns
- **CNNs**: Modern architectures with attention mechanisms
- **Components**: Normalization, activation functions, pooling strategies

```python
class ModernCNNWithAttention(nn.Module):
    def __init__(self, num_classes, use_attention=True):
        super().__init__()
        self.extractor = self._build_backbone()
        self.attention = SpatialAttention(256) if use_attention else nn.Identity()
        self.classifier = self._build_classifier(num_classes)
```

**Key Features:**

- Modular, reusable architectures
- Transfer learning support
- Attention mechanisms

### 4. Loss Functions & Optimization [`/Loss`](Loss/)

Training neural networks effectively

- **Loss Selection**: Decision frameworks for regression, classification, embedding tasks
- **Custom Losses**: Asymmetric penalties, focal loss, label smoothing
- **Optimization**: Learning rate scheduling, gradient management

```python
def get_loss_function(task_type, class_weights=None, label_smoothing=0.1):
    if task_type == 'classification' and class_weights is not None:
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    return _get_default_loss(task_type)
```

**Key Features:**

- Comprehensive loss function reference
- Custom loss implementations
- Optimization best practices

### 5. Computational Graphs & Autograd [`/Autograd`](Autograd/)

Understanding automatic differentiation

- **Graph Mechanics**: Forward/backward passes, gradient accumulation
- **Memory Management**: Efficient computation, debugging tools
- **Custom Functions**: Extending PyTorch with custom autograd functions

```python
def analyze_computation_graph(model, input_tensor):
    output = model(input_tensor)
    make_dot(output, params=dict(model.named_parameters())).render("computation_graph")
```

**Key Features:**

- Interactive graph visualization
- Memory profiling tools
- Custom gradient implementations

### 6. Production Training [`/Production`](Production/)

Industry-grade training infrastructure

- **Training Loops**: Checkpointing, logging, reproducibility
- **Scalability**: Mixed precision, gradient accumulation
- **Monitoring**: Metrics tracking, early stopping

**Key Features:**

- Production-ready templates
- MLOps integration patterns
- Distributed training examples

## Quick Start

### Prerequisites

```bash
pip install torch torchvision torchaudio matplotlib jupyter
```

### Run Your First Example

```python
# Tensors/tensors.py - Test tensor operations
python Tensors/tensors.py

# Data/example.py - Custom dataset demo
python Data/example.py

# Model/practice_CNN.py - CNN with attention
python Model/practice_CNN.py
```

### Interactive Learning

```bash
# Launch tensor practice notebook
jupyter notebook Tensors/practice.ipynb

# Complete the quiz challenges
python -m pytest Data/quiz.md --doctest-modules
```

## Learning Paths

### Beginner Track (2-3 weeks)

1. **Tensors** - Master basic operations and GPU usage
2. **Data** - Build your first custom dataset
3. **Model** - Create MLPs and simple CNNs
4. **Loss** - Understand optimization fundamentals

### Intermediate Track (3-4 weeks)

1. **Advanced Models** - Attention mechanisms, transfer learning
2. **Custom Losses** - Task-specific optimization
3. **Autograd** - Computational graph mastery
4. **Production** - Scalable training loops

### Expert Track (4-6 weeks)

1. **Performance Optimization** - Memory, speed, distributed training
2. **Research Extensions** - Custom layers, novel architectures
3. **MLOps Integration** - Deployment, monitoring, versioning

## Key Features

- **100% Tested Code** - Every example runs out of the box
- **Progressive Complexity** - From basics to advanced concepts
- **Production Ready** - Industry best practices throughout
- **Interactive Learning** - Jupyter notebooks, quizzes, challenges
- **Comprehensive Coverage** - Everything needed for real projects
- **Modern PyTorch** - Latest features and patterns (PyTorch 2.0+)

## What You'll Build

By the end of this course, you'll have implemented:

- Flexible MLP architectures with attention and regularization
- Modern CNN designs with spatial attention mechanisms
- Production data pipelines with error handling and optimization
- Custom loss functions for specialized tasks
- Complete training infrastructure with checkpointing and monitoring
- Memory-efficient models ready for deployment

## Contributing

This course is designed to be a living resource. Contributions welcome!

- **Bug Reports**: Found an issue? Open an issue with reproducible code
- **Documentation**: Improve explanations, add examples
- **Code**: Add new architectures, optimize existing implementations
- **Exercises**: Create new practice problems and solutions

## License

MIT License - Feel free to use this content for learning, teaching, or commercial projects.

## Getting Started

Jump into any module that interests you, or follow the structured learning path. Every file is documented, tested, and ready to run.

```bash
git clone <repo-url>
cd PyTorch
python Tensors/tensors.py  # Your first step!
```
