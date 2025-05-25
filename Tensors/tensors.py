import torch
import numpy as np

# ---
# PyTorch Tensor Tutorial: Deep Learning Essentials
# ---

# For reproducibility: set random seeds for both PyTorch and NumPy
# This ensures that random numbers generated are the same every run (important for debugging and reproducibility)
torch.manual_seed(42)
np.random.seed(42)

# ---
# CUDA: GPU Support in PyTorch
# ---
# PyTorch can leverage NVIDIA GPUs for faster computation using CUDA.
# The torch.cuda module provides utilities to check and use GPU resources.

print(torch.cuda.is_available())  # Is CUDA (GPU) available?
print(torch.cuda.device_count())  # How many CUDA devices (GPUs) are there?
print(torch.cuda.get_device_name(0))  # Name of the first GPU
print(
    torch.cuda.get_device_capability(0)
)  # Compute capability (important for compatibility)
print(torch.cuda.get_device_properties(0))  # Full device properties

# ---
# Tensor Initialization
# ---

# 1. From existing data (Python lists, etc.)
data = [[1, 2], [2, 3]]
x_data = torch.tensor(data)  # Creates a tensor from a nested list
# print(x_data)

# 2. From NumPy arrays (shares memory, so changes reflect both ways unless cloned)
numpy_data = np.array(data)
x_data = torch.from_numpy(numpy_data)
# print(x_data)

# 3. From another tensor (retains shape/dtype unless overridden)
x_ones = torch.ones_like(x_data)  # All ones, same shape/dtype as x_data
x_rand = torch.rand_like(x_data, dtype=torch.float64)  # Random values, override dtype
# print(x_ones)
# print(x_rand)

# 4. With random or constant values
data_shape = (2, 3)
x_ones = torch.ones(data_shape, dtype=torch.int64)  # All ones, int64
gx_rand = torch.rand(data_shape, dtype=torch.float16)  # Random floats
x_zeros = torch.zeros(data_shape, dtype=torch.float16)  # All zeros
# print(x_ones)
# print(x_rand)
# print(x_zeros)

# ---
# Tensor Attributes: shape, dtype, device
# ---
tensor = torch.rand((3, 3), dtype=torch.float16, device="cuda")  # 3x3 tensor on GPU
# print(tensor)
# print(tensor.dtype)  # Data type
# print(tensor.shape)  # Shape (tuple)
# print(tensor.device)  # Device (CPU or CUDA)

# ---
# Operations on Tensors
# ---
# Most operations are much faster on GPU (if available)
if torch.cuda.is_available():
    tensor = tensor.to("cuda")  # Move tensor to GPU

# ---
# Indexing and Slicing
# ---
# Tensors can be indexed and sliced like NumPy arrays
# print(f"First row : {tensor[0,:]}")
# print(f"Third col : {tensor[:,2]}")
# print(f"Last Row, second-last column : {tensor[-1,-2].item()}")

# ---
# Joining Tensors: Concatenation and Stacking
# ---
# Concatenation: join along an existing dimension
to_cat = [
    torch.rand((1, 3), dtype=torch.float16, device="cuda"),
    torch.ones((1, 3), dtype=torch.int8, device="cuda"),
    torch.zeros((1, 3), dtype=torch.int8, device="cuda"),
]
concatenated_tensor = torch.cat(to_cat, dim=0)  # Shape: (3, 3)

# Stacking: join along a new dimension
stacked_tensor_1 = torch.stack(
    [concatenated_tensor, concatenated_tensor], dim=0
)  # (2, 3, 3)
stacked_tensor_2 = torch.stack(
    [concatenated_tensor, concatenated_tensor], dim=1
)  # (3, 2, 3)
stacked_tensor_3 = torch.stack(
    [concatenated_tensor, concatenated_tensor], dim=2
)  # (3, 3, 2)
# print(concatenated_tensor, concatenated_tensor.shape)
# print(stacked_tensor_1, stacked_tensor_1.shape)
# print(stacked_tensor_2, stacked_tensor_2.shape)
# print(stacked_tensor_3, stacked_tensor_3.shape)

# ---
# Splitting and Chunking
# ---
tensor = torch.rand((5, 2, 2), dtype=torch.float16, device="cuda")
concatenated_tensor = torch.concat([tensor, tensor], dim=0)  # (10, 2, 2)
splits = torch.split(concatenated_tensor, 2)  # Split into chunks of size 2 along dim 0
chunks = torch.chunk(concatenated_tensor, 5)  # Split into 5 equal chunks along dim 0
# print(tensor, tensor.shape)
# print(concatenated_tensor, concatenated_tensor.shape)
# print(splits, splits[0].shape)
# print(chunks, chunks[0].shape)

# ---
# Shape Semantics in Deep Learning
# ---
# Common convention: (Batch, Channels, Height, Width) = (B, C, H, W)
# Stack C images of HxW into a C-channel image: torch.stack(list_of_hxw_images, dim=0) -> (C, H, W)
# Stack B C-channel images into a batch: torch.stack(list_of_B_c_channel_images, dim=0) -> (B, C, H, W)
# Concatenate b mini-batches of size B: torch.concat(list_of_b_B-sized_batches, dim=0) -> (bB, C, H, W)
# Chunk into b' mini-batches: torch.chunk(tensor, b')

# ---
# Reshaping Tensors
# ---
# Reshape returns a view with a new shape (total elements must remain the same)
batch_of_three_channel_images = torch.rand(
    (8, 3, 32, 32), dtype=torch.float16, device="cuda"
)
flattened = batch_of_three_channel_images.reshape(
    8, 3, -1
)  # Flatten H and W into one dimension
print(
    batch_of_three_channel_images.shape, flattened.shape
)  # (8, 3, 32, 32) -> (8, 3, 1024)

# ---
# Squeezing and Unsqueezing
# ---
# Squeeze: remove dimensions of size 1 (singleton dims)
batch_of_three_channel_images = torch.rand(
    (8, 1, 32, 32), dtype=torch.float16, device="cuda"
)
squeezed = batch_of_three_channel_images.squeeze(dim=1)  # Remove channel dim if it's 1
print(
    batch_of_three_channel_images.shape, squeezed.shape
)  # (8, 1, 32, 32) -> (8, 32, 32)

# Unsqueeze: add a dimension of size 1 at a given position (useful for broadcasting)
batch_of_images = torch.rand((8, 32, 32), dtype=torch.float16, device="cuda")
# Suppose a CNN expects a channel dimension: (B, C, H, W)
new_batch_of_images = batch_of_images.unsqueeze(dim=1)  # Add channel dim at position 1
print(batch_of_images.shape, new_batch_of_images.shape)  # (8, 32, 32) -> (8, 1, 32, 32)

# ---
# Transposing and Permuting
# ---
# Transpose: swap two dimensions (e.g., (B, H, W, C) <-> (B, C, H, W))
batch_of_images = torch.rand((8, 32, 32, 3), dtype=torch.float16, device="cuda")
new_batch_of_images = batch_of_images.transpose(1, 3)  # Swap dims 1 and 3
print(
    batch_of_images.shape, new_batch_of_images.shape
)  # (8, 32, 32, 3) -> (8, 3, 32, 32)

# Permute: reorder all dimensions in one call
x = torch.randn(8, 28, 28, 3)  # (B, H, W, C)
x_perm = x.permute(0, 3, 1, 2)  # (B, C, H, W)
print(x.shape, x_perm.shape)  # (8, 28, 28, 3) -> (8, 3, 28, 28)

# ---
# Flattening
# ---
# Flatten spatial dimensions while retaining batch and channel structure
x = torch.randn(8, 3, 32, 32)
x_flat = x.flatten(start_dim=2)  # Flatten H and W into one dimension: (8, 3, 1024)
print(x.shape, x_flat.shape)

# ---
# Summary
# ---
# PyTorch tensors are flexible, efficient, and essential for deep learning workflows.
# Mastering their creation, manipulation, and device management is key to building performant models.


# Elementwise operations:
# ---
# Mathematical Operations with Tensors
# ---

# Basic arithmetic operations work with tensors just like with scalars
# These are applied element-wise to each value in the tensor
x = torch.randn(size=(8, 3, 2, 2))
# print(x)
x = x + 1  # Add 1 to every element (broadcasting scalar to tensor)
# print(x)
x = x * 0.5  # Multiply every element by 0.5
# print(x)

# ---
# Matrix Multiplication Operations
# ---

# Regular matrix multiplication with torch.matmul
# Matrix multiplication follows linear algebra rules for matrix-vector and matrix-matrix products
vector = torch.randn(size=(5, 1), dtype=torch.float16, device="cuda")  # column vector
weights = torch.randn(size=(5, 5), dtype=torch.float16, device="cuda")
y = torch.matmul(weights, vector)  # Matrix-vector multiplication: (5,5) @ (5,1) = (5,1)
print(y.shape)


# Batch matrix multiplication: torch.bmm is specifically for 3D tensors
# It performs matrix multiplication for each corresponding slice in the batch
# For torch.bmm(A, B):
# - A shape: (batch_size, n, m)
# - B shape: (batch_size, m, p)
# - Result: (batch_size, n, p)
# This is equivalent to doing matrix multiplication for each batch independently
x = torch.randn(
    size=(8, 5, 3), dtype=torch.float16, device="cuda"
)  # Shape: (batch_size, n, m)
y = torch.randn(
    size=(8, 3, 10), dtype=torch.float16, device="cuda"
)  # Shape: (batch_size, m, p)
z = torch.bmm(x, y)  # Batch matrix multiplication
print(z.shape)  # Result shape: (8, 5, 10)

# ---
# Element-wise Operations
# ---

# 1. Element-wise multiplication (Hadamard product)
x = torch.randn(size=(8, 5, 1), dtype=torch.float16, device="cuda")
y = torch.randn(size=(8, 5, 1), dtype=torch.float16, device="cuda")
z = torch.mul(x, y)  # Element-wise multiplication
# Shorthand: z = x * y

# 2. Other element-wise operations:
# - Addition: torch.add(x, y) or x + y
# - Subtraction: torch.sub(x, y) or x - y
# - Division: torch.div(x, y) or x / y
# - Exponentiation: torch.pow(x, n) or x ** n
# - Square root: torch.sqrt(x)

# Element-wise operations apply the operation to each corresponding pair
# of elements in the tensors. Broadcasting allows operations between tensors
# of different shapes by implicitly expanding the smaller tensor.


# ---
# Aggregation Operations (Reduction)
# ---
# Aggregation operations reduce tensors along specified dimensions

x = torch.randn(size=(8, 5, 2, 2), dtype=torch.float16, device="cuda")  # (B, C, H, W)

# Spatial average: reduce height and width dimensions (common in CNNs)
spatial_avg = x.mean(dim=(2, 3))  # Average over H and W dimensions
print(
    spatial_avg.shape
)  # (8, 5) - Each channel feature is averaged across spatial dims

# Batch loss reduction: average over batch dimension (common in loss functions)
batch_loss = x.mean(dim=0)  # Average over batch dimension
print(batch_loss.shape)  # (5, 2, 2) - Average feature map across all samples

# Global reductions - reduce to a single scalar value
global_avg = x.mean()  # Average over all elements
print(global_avg.item())  # Scalar value - useful for metrics

# Other global reduction operations
global_max = x.max()  # Maximum value across all elements
print(global_max.item())
global_min = x.min()  # Minimum value across all elements
print(global_min.item())
global_sum = x.sum()  # Sum of all elements
print(global_sum.item())
global_std = x.std()  # Standard deviation of all elements
print(global_std.item())
global_var = x.var()  # Variance of all elements
print(global_var.item())
global_median = x.median()  # Median of all elements
print(global_median.item())

# ---
# Statistical and Indexing Operations
# ---

# Classification use case: finding the predicted class
# argmax returns the indices of maximum values along a specified dimension
x = torch.randn(
    size=(8, 5), dtype=torch.float16, device="cuda"
)  # (batch_size, num_classes)
class_indices = torch.argmax(
    x, dim=1
)  # Get indices of max values along the class dimension
# This is commonly used to convert network outputs to class predictions

# amax returns the actual maximum values (not the indices)
max_values = torch.amax(x, dim=1)  # Get max values along the class dimension
print(class_indices.shape)  # (8,) - One class index per sample
print(max_values.shape)  # (8,) - One max value per sample

# ---
# Distance and Loss Computations
# ---

# Root Mean Square Error (RMSE) - common regression loss
x = torch.randn(size=(8, 5), dtype=torch.float16, device="cuda")  # Predictions
y = torch.randn(size=(8, 5), dtype=torch.float16, device="cuda")  # Targets
# RMSE calculation step by step:
# 1. Squared difference: (x - y)²
# 2. Mean of squared differences: mean((x - y)²)
# 3. Square root of the mean: sqrt(mean((x - y)²))
rmse = torch.sqrt(torch.mean((x - y) ** 2))
print(rmse.item())  # Single scalar value representing the RMSE loss

# ---
# Tensor Value Manipulation
# ---

# Clamping/Clipping - restrict values to a range
# Useful for:
# - Gradient clipping to prevent exploding gradients
# - Restricting activation values
# - Normalizing outputs to a specific range
x = torch.randn(size=(8, 5), dtype=torch.float16, device="cuda")
clamped_x = torch.clamp(x, min=-1.0, max=1.0)  # Restrict values between -1 and 1
print(clamped_x)  # All values will be between -1 and 1

# L2 norm (Euclidean norm)
# Used for:
# - Weight normalization
# - Feature normalization
# - Distance calculations
x = torch.randn(size=(8, 5, 2, 2), dtype=torch.float16, device="cuda")
l2_norm = torch.norm(
    x, p=2, dim=(1, 2, 3)
)  # L2 norm across channels, height, and width
print(l2_norm.shape)  # (8,) - One norm value per sample in the batch

# Cumulative sum - running total along a dimension
# Applications:
# - Cumulative probabilities
# - Prefix sums
# - Time-series analysis
x = torch.randn(size=(8, 5), dtype=torch.float16, device="cuda")
cumulative_sum = torch.cumsum(x, dim=1)  # Cumulative sum along the class dimension
print(cumulative_sum.shape)  # (8, 5) with running totals along dimension 1

# ---
# Conditional Operations
# ---

# torch.where - element-wise conditional selection
# Similar to: condition ? x : y in other languages
# Applications:
# - Implementing custom activation functions
# - Data preprocessing
# - Loss function components
x = torch.randn(size=(8, 5), dtype=torch.float16, device="cuda")
y = torch.randn(size=(8, 5), dtype=torch.float16, device="cuda")
selected_values = torch.where(x > 0, x, y)  # If x > 0, take value from x; else from y
print(selected_values.shape)  # (8, 5) - Shape matches input tensors

# ---
# Rounding Operations
# ---

# Rounding functions - for quantization, coordinate manipulation, and numerical stability
x = torch.randn(size=(8, 5), dtype=torch.float16, device="cuda")
rounded_x = torch.round(x)  # Round to nearest integer
floored_x = torch.floor(x)  # Round down to nearest integer
ceiled_x = torch.ceil(x)  # Round up to nearest integer
truncated_x = torch.trunc(x)  # Truncate toward zero (remove decimal part)
frac_x = torch.frac(x)  # Get fractional part only (x - floor(x))
print(rounded_x.shape, floored_x.shape, ceiled_x.shape, truncated_x.shape, frac_x.shape)

# ---
# Advanced Use Cases
# ---

# Example: Softmax operation (often used in classification)
logits = torch.randn(
    size=(8, 10), dtype=torch.float32, device="cuda"
)  # Raw model outputs
# Softmax converts raw logits to probabilities that sum to 1
softmax = torch.nn.functional.softmax(logits, dim=1)
print(softmax.sum(dim=1))  # Should be close to 1.0 for each sample

# Example: One-hot encoding (useful for classification targets)
indices = torch.randint(
    0, 10, (8,), device="cuda"
)  # Class indices (8 samples, 10 classes)
one_hot = torch.nn.functional.one_hot(indices, num_classes=10)
print(one_hot.shape)  # (8, 10) - One-hot encoded class labels

# Example: Batch normalization statistics (common in neural networks)
features = torch.randn(
    size=(32, 64, 7, 7), dtype=torch.float32, device="cuda"
)  # (B, C, H, W)
# Calculate batch statistics (mean and variance) for each channel
batch_mean = features.mean(
    dim=(0, 2, 3), keepdim=True
)  # Mean across batch and spatial dims
batch_var = features.var(
    dim=(0, 2, 3), keepdim=True
)  # Variance across batch and spatial dims
# Normalize the features
eps = 1e-5  # Small constant for numerical stability
normalized = (features - batch_mean) / torch.sqrt(batch_var + eps)
print(normalized.shape)  # (32, 64, 7, 7) - Same shape as input, but normalized

# ---
# Memory Efficiency
# ---

# In-place operations (modifying tensor directly)
# These save memory but can interfere with autograd
x = torch.randn(size=(1000, 1000), dtype=torch.float32, device="cuda")
x.add_(1.0)  # In-place addition (note the trailing underscore)
x.mul_(0.5)  # In-place multiplication
x.clamp_(min=-1.0, max=1.0)  # In-place clamping

# ---
# Summary
# ---
# PyTorch tensors provide a comprehensive set of operations for deep learning:
# - Basic arithmetic and matrix operations
# - Statistical functions and reductions
# - Shape manipulations
# - Indexing and conditional operations
# - Advanced mathematical functions
#
# These operations form the building blocks for:
# - Neural network layers
# - Loss functions
# - Optimization algorithms
# - Data preprocessing
# - Model evaluation metrics
