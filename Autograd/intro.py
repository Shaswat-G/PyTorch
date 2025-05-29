import torch

print("=== PART 1: Understanding Tensors ===")

# Create different types of tensors
scalar = torch.tensor(5.0)
vector = torch.tensor([1.0, 2.0, 3.0])
matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

print(f"Scalar: {scalar}, shape: {scalar.shape}")
print(f"Vector: {vector}, shape: {vector.shape}")
print(f"Matrix: {matrix}, shape: {matrix.shape}")

print("\n=== PART 2: The Magic of requires_grad ===")

# This is where PyTorch becomes powerful for deep learning
x = torch.tensor(3.0, requires_grad=True)
print(f"x = {x}")
print(f"x.requires_grad = {x.requires_grad}")

# Let's create a simple function: y = x^2 + 2*x + 1
y = x**2 + 2 * x + 1
print(f"y = x^2 + 2*x + 1 = {y}")

print("\n=== PART 3: Computing Gradients (The Heart of Learning) ===")

# This is the magic! PyTorch automatically computes derivatives
y.backward()  # This computes dy/dx
print(f"Gradient dy/dx = {x.grad}")

# Let's verify this manually
# dy/dx = 2*x + 2
# At x=3: dy/dx = 2*3 + 2 = 8
print(f"Manual calculation: dy/dx at x=3 should be 2*3 + 2 = {2*3 + 2}")

print("\n=== PART 4: Why This Matters for Deep Learning ===")

# Imagine x is a weight in a neural network
# y is the loss function
# The gradient tells us how to adjust the weight to minimize loss!

weight = torch.tensor(0.5, requires_grad=True)
target = 10.0
prediction = 2 * weight  # Simple linear model
loss = (prediction - target) ** 2  # Mean squared error

print(f"Initial weight: {weight}")
print(f"Prediction: {prediction}")
print(f"Target: {target}")
print(f"Loss: {loss}")

loss.backward()
print(f"Gradient of loss w.r.t. weight: {weight.grad}")
print("This gradient tells us how to update the weight to reduce loss!")

weight.grad.zero_()  # Reset the gradient before the second backward pass
loss.backward()
print(f"Gradient of loss w.r.t. weight AGAIN: {weight.grad}")

print("\n=== EXPERIMENT FOR YOU ===")
print("Try changing the initial weight value and see how the gradient changes.")
print("Questions to think about:")
print("1. What happens if weight = 5.0? What should the gradient be?")
print("2. What if the target is 0 instead of 10?")
print("3. Why is the gradient negative when weight < target/2?")
