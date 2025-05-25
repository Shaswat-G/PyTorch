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
