# Tensors

Tensors are multi-dimensional arrays that are used to store, represent and manipulate data (input, output, model parameters, etc.) in deep learning. They are optimized for GPUs and hardware accelerators and automatic differentiation.

---

## What is a Tensor?

- A generalization of scalars (0D), vectors (1D), and matrices (2D) to N dimensions.
- PyTorch tensors are similar to NumPy arrays but with GPU acceleration and autograd support.
- Tensors can be indexed, sliced, reshaped, and broadcasted like NumPy arrays.

---

## Initialization

### From existing data

- Use `torch.tensor(data)` to create a tensor from Python lists or sequences.
- Example: `torch.tensor([[1, 2], [2, 3]])`

### From NumPy arrays

- Use `torch.from_numpy(numpy_array)` to convert a NumPy array to a tensor (shares memory!).
- Changes to one will reflect in the other unless `.clone()` is used.

### From another tensor

- Use `torch.ones_like(tensor)`, `torch.zeros_like(tensor)`, or `torch.rand_like(tensor)` to create new tensors with the same shape and dtype as an existing tensor.
- You can override dtype or device if needed.

### With random or constant values

- `torch.ones(shape, dtype=...)`, `torch.zeros(shape, dtype=...)`, `torch.rand(shape, dtype=...)` for constant or random initialization.
- `shape` is a tuple, e.g., `(2, 3)`.

---

## Crucial Tidbits & Tangents

- **Device Placement:**

  - Tensors can be moved to GPU with `.to('cuda')` or `.cuda()`, and back to CPU with `.to('cpu')`.
  - Example: `x = x.to('cuda')`

- **Dtype:**

  - Common dtypes: `torch.float32`, `torch.float64`, `torch.int64`, etc.
  - Use `tensor.dtype` to check, and `.to(dtype=...)` to convert.

- **Shape & Reshaping:**

  - Use `tensor.shape` or `tensor.size()` to get shape.
  - Use `tensor.view(new_shape)` or `tensor.reshape(new_shape)` to change shape.

- **Interoperability:**

  - PyTorch tensors and NumPy arrays can be converted back and forth easily.
  - `tensor.numpy()` returns a NumPy array (CPU only).
  - `torch.from_numpy(array)` creates a tensor from a NumPy array.

- **In-place Operations:**

  - Methods ending with `_` (e.g., `add_`, `zero_`) modify the tensor in-place.
  - Use with care to avoid unwanted side effects, especially with autograd.

- **Autograd:**

  - Tensors can track gradients if `requires_grad=True`.
  - Useful for neural network training and optimization.

- **Broadcasting:**

  - PyTorch supports broadcasting for arithmetic operations between tensors of different shapes, following NumPy rules.

- **Random Seed:**
  - For reproducibility, set seeds: `torch.manual_seed(42)` and `np.random.seed(42)`.

---

## Best Practices

- Prefer explicit dtype and device specification for reproducibility.
- Use `.clone()` when you need a copy that does not share memory.
- Avoid in-place operations unless necessary.
- Always check tensor shapes before operations to avoid broadcasting bugs.

---

> "Tensors are the fundamental building blocks of deep learning. Mastering their creation, manipulation, and properties is essential for any practitioner."

---

## CUDA: Accelerating Deep Learning

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and API for using GPUs. It enables massive speedups for deep learning by allowing tensor operations to run on thousands of GPU cores in parallel.

- **CUDA Toolkit:** Includes the CUDA compiler (`nvcc`), libraries (cuBLAS, cuDNN), and development tools for building GPU-accelerated applications.
- **PyTorch & CUDA:** PyTorch uses CUDA under the hood for `.to('cuda')` or `.cuda()` tensor operations, making deep learning training and inference much faster.
- **Why Important?** Training large neural networks on CPUs is impractical; GPUs (via CUDA) make modern deep learning feasible.

### Key CUDA Tools & Commands

- `nvcc`: The CUDA C/C++ compiler for building GPU code.
- `nvidia-smi`: Monitor GPU status, memory usage, temperature, and running processes.
- `nvidia-settings`: GUI for configuring GPU settings.
- `nvidia-debugdump`: Diagnostic tool for debugging GPU issues.
- `nvidia-cuda-toolkit`: The full suite of CUDA development tools and libraries.
- `nvidia-container-toolkit`, `nvidia-container-runtime`, `nvidia-docker`: Enable GPU access in Docker containers for reproducible, portable deep learning environments.
- `nvidia-persistenced`: Manages GPU persistence mode for faster context switching.
- `nvidia-modprobe`: Loads NVIDIA kernel modules.
- `nsys`, `nvprof`, `cuda-gdb`: Profiling and debugging tools for optimizing GPU code.

---

## Tensor Operations: Deep Learning Essentials

- **Indexing & Slicing:** Tensors support advanced slicing, similar to NumPy, for extracting sub-tensors or elements.
- **Joining:** Use `torch.cat` to concatenate tensors along an existing dimension, and `torch.stack` to add a new dimension.
- **Splitting:** `torch.split` and `torch.chunk` divide tensors into smaller pieces, useful for batching and data loading.
- **Shape Semantics:** Conventionally, tensors are shaped as (Batch, Channels, Height, Width) for images. Understanding and manipulating these dimensions is crucial for model design.
- **Reshaping:** `reshape`, `view`, and `flatten` allow you to change tensor shapes without copying data, as long as the total number of elements remains constant.
- **Squeeze/Unsqueeze:** Remove or add singleton dimensions to match expected input shapes for models or broadcasting.
- **Transpose/Permute:** Rearrange tensor dimensions for compatibility with different layers or frameworks.

---

## Advanced Tidbits & Tangents

- **Memory Sharing:** Tensors created from NumPy arrays share memory; use `.clone()` to avoid side effects.
- **In-place vs Out-of-place:** In-place ops (ending with `_`) save memory but can interfere with autograd. Use with caution.
- **Autograd:** Tensors with `requires_grad=True` track operations for automatic differentiation—essential for training neural networks.
- **Broadcasting:** PyTorch automatically expands tensors of different shapes for elementwise operations, following NumPy's broadcasting rules.
- **Device Transfers:** Use `.to('cuda')` and `.to('cpu')` to move tensors between devices. Always check `torch.cuda.is_available()` before using CUDA.
- **Profiling:** Use `torch.utils.bottleneck`, `nsys`, or `nvprof` to profile and optimize tensor operations and model training.

---

## Common Pitfalls

- Forgetting to move both data and model to the same device (CPU/GPU).
- Mismatched tensor shapes leading to runtime errors.
- Unintended in-place operations breaking autograd.
- Not setting random seeds, resulting in irreproducible results.

---

## Further Exploration

- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [PyTorch Tensor Operations](https://pytorch.org/docs/stable/tensors.html)
- [Deep Learning Book (Goodfellow et al.)](https://www.deeplearningbook.org/)

---

