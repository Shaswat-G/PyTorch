"""
COMPREHENSIVE GUIDE: PyTorch Computational Graph & Autograd Internals

This file explains the inner workings of PyTorch's automatic differentiation system.
Understanding these concepts is crucial for:
1. Debugging gradient-related issues
2. Writing efficient training loops
3. Implementing custom neural network components
4. Optimizing memory usage during training

KEY CONCEPTS COVERED:
- How computational graphs are built node by node
- What grad_fn stores and why it's important
- Memory management and common pitfalls
- Custom autograd function implementation
- Gradient accumulation mechanics
- Debugging tools and best practices
"""

import torch
import torch.nn as nn
import gc
import weakref

# ==========================================
# DEEP DIVE: Computational Graph Mechanics
# ==========================================


def explore_computational_graph():
    """
    BEGINNER'S GUIDE: Understanding PyTorch's autograd internals

    WHAT IS A COMPUTATIONAL GRAPH?
    - A directed acyclic graph (DAG) that records all operations performed on tensors
    - Each node represents an operation (addition, multiplication, etc.)
    - Edges represent data flow between operations
    - PyTorch builds this graph automatically during forward pass
    - During backward pass, PyTorch traverses this graph to compute gradients

    WHY IS THIS IMPORTANT?
    - Enables automatic differentiation (autograd)
    - Allows us to compute gradients without manual calculus
    - Essential for training neural networks via backpropagation
    """
    print(
        "=== COMPUTATIONAL GRAPH EXPLORATION ===\n"
    )  # Step 1: Create leaf tensors (starting points of our computation)
    # requires_grad=True tells PyTorch: "Please track operations on this tensor"
    x = torch.tensor([2.0], requires_grad=True)  # This will be our input variable
    y = torch.tensor([3.0], requires_grad=True)  # Another input variable

    print(f"Initial tensors (leaf nodes in the computational graph):")
    print(f"x: {x}, grad_fn: {x.grad_fn}")  # grad_fn is None for leaf tensors
    print(f"y: {y}, grad_fn: {y.grad_fn}")  # grad_fn is None for leaf tensors
    print(f"x.is_leaf: {x.is_leaf}, y.is_leaf: {y.is_leaf}")  # Both are True

    # BEGINNER NOTE: 'Leaf' tensors are the starting points - they don't depend on other tensors
    # Think of them as variables in a mathematical equation: f(x, y) = some function of x and y    # Step 2: Build computation graph by performing operations
    # Each operation creates a new node in the graph with a grad_fn
    z1 = x * y  # MulBackward0 - remembers how to compute d(x*y)/dx and d(x*y)/dy
    z2 = z1 + x  # AddBackward0 - remembers how to compute d(z1+x)/dz1 and d(z1+x)/dx
    z3 = z2.pow(2)  # PowBackward0 - remembers how to compute d(z2^2)/dz2
    loss = z3.mean()  # MeanBackward0 - remembers how to compute d(mean(z3))/dz3

    print(f"\n=== GRAPH STRUCTURE (Step by Step) ===")
    print(f"z1 = x * y: {z1}, grad_fn: {z1.grad_fn}")
    print(f"z2 = z1 + x: {z2}, grad_fn: {z2.grad_fn}")
    print(f"z3 = z2^2: {z3}, grad_fn: {z3.grad_fn}")
    print(f"loss = mean(z3): {loss}, grad_fn: {loss.grad_fn}")

    # BEGINNER NOTE: Each grad_fn is like a recipe that knows:
    # 1. What operation was performed
    # 2. Which tensors were inputs to that operation
    # 3. How to compute the derivative with respect to each input    # Step 3: Examine the internal structure of grad_fn
    # grad_fn stores references to previous operations in the computational graph
    print(f"\n=== GRAD_FN INTERNALS (How the chain is built) ===")
    print(f"loss.grad_fn.next_functions: {loss.grad_fn.next_functions}")
    print(f"z3.grad_fn.next_functions: {z3.grad_fn.next_functions}")

    # BEGINNER EXPLANATION:
    # next_functions is a tuple of (function, input_index) pairs
    # This tells PyTorch: "When computing gradients, call these functions next"
    # It's like a linked list pointing backwards through the computation    # Step 4: Backward pass - this is where automatic differentiation happens!
    print(f"\n=== BACKWARD PASS (The Magic Moment) ===")
    print("About to call loss.backward()...")
    print("This will:")
    print("1. Start from 'loss' tensor")
    print("2. Work backwards through the computational graph")
    print("3. Apply chain rule at each step")
    print("4. Accumulate gradients in the .grad attributes of leaf tensors")

    loss.backward()

    print(f"\nResults after backward():")
    print(f"x.grad: {x.grad}")  # d(loss)/dx - how loss changes with respect to x
    print(f"y.grad: {y.grad}")  # d(loss)/dy - how loss changes with respect to y

    # MATHEMATICAL VERIFICATION:
    # Our function is: loss = ((x*y + x)^2).mean()
    # Since we have only one element, .mean() doesn't change the value
    # So: loss = (x*y + x)^2 = (x*(y+1))^2
    #
    # Using chain rule: d(loss)/dx = 2*(x*(y+1))*(y+1) = 2*x*(y+1)^2
    # With x=2, y=3: d(loss)/dx = 2*2*(3+1)^2 = 4*16 = 64
    print(f"\nManual verification (using calculus):")
    print(f"Our function: loss = (x*y + x)^2 = (x*(y+1))^2")
    print(
        f"d(loss)/dx = 2*x*(y+1)^2 = 2*{x.item()}*({y.item()}+1)^2 = {2 * x.item() * (y.item() + 1)**2}"
    )
    print(
        f"d(loss)/dy = 2*(x*y + x)*x = 2*x^2*(y+1) = 2*{x.item()}^2*({y.item()}+1) = {2 * x.item()**2 * (y.item() + 1)}"
    )
    print(f"PyTorch computed: x.grad={x.grad.item()}, y.grad={y.grad.item()}")
    print(f"Perfect match! ✓")


def understand_grad_accumulation():
    """
    CRITICAL CONCEPT: Why gradients accumulate and memory implications

    BEGINNER'S QUESTION: "Why do I need optimizer.zero_grad()?"
    ANSWER: Because PyTorch ADDS new gradients to existing ones by default!

    This behavior is actually useful for:
    1. Gradient accumulation (simulating larger batch sizes)
    2. RNN training (accumulating gradients across time steps)
    3. Multi-task learning (accumulating gradients from different losses)

    But it's also the #1 source of bugs for beginners!
    """
    print("\n=== GRADIENT ACCUMULATION MECHANICS ===\n")

    x = torch.tensor([1.0], requires_grad=True)

    # First computation and backward pass
    print("=== Experiment 1: First computation ===")
    y1 = x**2  # Simple function: f(x) = x^2, so df/dx = 2x = 2*1 = 2
    y1.backward()
    print(f"After first backward: x.grad = {x.grad}")
    print(f"Expected: 2*x = 2*1 = 2 ✓")

    # Second computation WITHOUT zero_grad() - this is the trap!
    print(f"\n=== Experiment 2: Second computation (WITHOUT zero_grad) ===")
    y2 = x**3  # New function: g(x) = x^3, so dg/dx = 3x^2 = 3*1 = 3
    y2.backward()
    print(f"After second backward (accumulated): x.grad = {x.grad}")
    print(f"Expected if accumulated: 2 + 3 = 5 ✓")
    print(f"❌ This is usually NOT what you want in training!")

    # Proper way: zero gradients before each backward pass
    print(f"\n=== Experiment 3: Proper way with zero_grad() ===")
    x.grad.zero_()  # Clear accumulated gradients
    y3 = x**4  # New function: h(x) = x^4, so dh/dx = 4x^3 = 4*1 = 4
    y3.backward()
    print(f"After zero_grad() and third backward: x.grad = {x.grad}")
    print(f"Expected: 4*x^3 = 4*1 = 4 ✓")
    print(f"✅ This is what you want in normal training!")


def tensor_item_deep_dive():
    """
    MEMORY TRAP: Understanding .item() vs direct tensor access

    BEGINNER'S QUESTION: "When should I use .item()?"
    ANSWER: When you want to extract a Python scalar and break the computational graph connection!

    This is crucial for:
    1. Logging/printing values without keeping the graph in memory
    2. Storing metrics that don't need gradients
    3. Debugging without memory leaks
    """
    print("\n=== TENSOR.ITEM() DEEP DIVE ===\n")

    x = torch.tensor([5.0], requires_grad=True)
    y = x**2  # y = 25, but still connected to computational graph

    print(f"=== Examining tensor y ===")
    print(f"y: {y}")
    print(f"y.item(): {y.item()}")  # Extracts Python float, breaks graph connection
    print(f"type(y): {type(y)}")  # PyTorch tensor
    print(f"type(y.item()): {type(y.item())}")  # Python float

    print(f"\n=== Graph connection status ===")
    print(f"y.requires_grad: {y.requires_grad}")  # True - part of computational graph
    print(f"y.grad_fn: {y.grad_fn}")  # Shows the operation that created y

    # .item() creates a Python scalar - no graph connection
    scalar_y = y.item()
    print(f"\n=== After using .item() ===")
    print(f"scalar_y = {scalar_y} (just a Python float)")
    print(f"scalar_y has no computational graph connection")

    # MEMORY EFFICIENCY DEMONSTRATION
    print(f"\n=== Memory Impact Demonstration ===")
    print("Creating lists with tensors vs scalars...")

    # This keeps the entire computational graph alive! Memory leak in training loops!
    tensor_list = [y for _ in range(1000)]  # 1000 references to the same tensor

    # This stores just Python floats - much more memory efficient
    scalar_list = [y.item() for _ in range(1000)]  # 1000 Python floats

    print(f"✅ Key insight: Using .item() for logging prevents memory leaks")
    print(f"❌ Storing tensor references keeps entire computational graph in memory")
    print(f"💡 Rule: Use .item() when you just need the value, not the gradient")


# ==========================================
# ADVANCED: Custom Autograd Functions
# ==========================================


class CustomSquare(torch.autograd.Function):
    """
    ADVANCED: Custom autograd function to understand internals

    This shows how PyTorch's autograd actually works under the hood.
    When you do x**2, PyTorch uses a similar mechanism internally.

    Key components:
    1. forward(): Define the operation (what happens during forward pass)
    2. backward(): Define the gradient computation (what happens during backward pass)
    3. ctx: Context object to save information for backward pass
    """

    @staticmethod
    def forward(ctx, input):
        """
        Forward pass: compute the result and save info for backward

        Args:
            ctx: Context object to save tensors for backward pass
            input: The input tensor

        Returns:
            The result of input^2
        """
        # Save tensors that we'll need for gradient computation
        ctx.save_for_backward(input)
        return input**2

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute gradients using chain rule

        Args:
            ctx: Context object containing saved tensors
            grad_output: Gradient flowing back from the next operation

        Returns:
            Gradient with respect to input
        """
        # Retrieve saved tensors
        (input,) = ctx.saved_tensors

        # Apply chain rule: d(x^2)/dx = 2x
        # grad_output is the gradient flowing from above (chain rule)
        return grad_output * 2 * input


def test_custom_autograd():
    """
    Testing our custom function vs PyTorch's built-in

    This demonstrates that our custom implementation produces
    identical results to PyTorch's built-in operations.
    """
    print("\n=== CUSTOM AUTOGRAD FUNCTION ===\n")

    x = torch.tensor([3.0], requires_grad=True)

    # Using our custom function
    print("=== Testing Custom Function ===")
    y_custom = CustomSquare.apply(x)  # Apply our custom square function
    print(f"Custom function result: {y_custom}")
    y_custom.backward()
    print(f"Custom function gradient: x.grad = {x.grad}")
    print(f"Expected: 2*x = 2*3 = 6 ✓")

    # Reset gradients and compare with built-in
    x.grad.zero_()
    print(f"\n=== Comparing with Built-in Function ===")
    y_builtin = x**2  # PyTorch's built-in square operation
    print(f"Built-in function result: {y_builtin}")
    y_builtin.backward()
    print(f"Built-in function gradient: x.grad = {x.grad}")
    print(f"Expected: 2*x = 2*3 = 6 ✓")

    print(f"\n✅ Both methods produce identical results!")
    print(f"💡 This shows how PyTorch's autograd works internally")


# ==========================================
# MEMORY AND PERFORMANCE CONSIDERATIONS
# ==========================================


def memory_pitfalls_demo():
    """
    CRITICAL FOR TRAINING: Common memory pitfalls that kill your training loops

    THE #1 MEMORY LEAK IN PYTORCH: Storing tensors instead of scalars in lists!

    This happens when you:
    1. Store loss values for plotting
    2. Save metrics during training
    3. Collect predictions for analysis

    The problem: Each tensor reference keeps the ENTIRE computational graph alive!
    """
    print("\n=== MEMORY PITFALLS ===\n")

    # SIMULATION: What happens during training loops
    losses_bad = []  # ❌ This will cause memory leaks!
    losses_good = []  # ✅ This is memory-efficient

    print("Simulating 5 training steps...")
    for i in range(5):
        # Simulate a forward pass with some computation
        x = torch.randn(1000, requires_grad=True)  # Simulate model parameters
        loss = (x**2).mean()  # Simulate loss computation

        print(f"\n--- Step {i+1} ---")
        if i == 0:
            print(f"Loss tensor details:")
            print(f"  Value: {loss.item():.6f}")
            print(f"  Shape: {loss.shape}")
            print(f"  Requires grad: {loss.requires_grad}")
            print(f"  Grad function: {loss.grad_fn}")

        # BAD: Storing the entire tensor (keeps computational graph!)
        losses_bad.append(loss)  # ❌ Memory leak!

        # GOOD: Storing just the scalar value
        losses_good.append(loss.item())  # ✅ Memory efficient!

    print(f"\n=== Memory Analysis ===")
    print(f"losses_bad[0] type: {type(losses_bad[0])}")
    print(f"losses_good[0] type: {type(losses_good[0])}")
    print(f"losses_bad[0] still has grad_fn: {losses_bad[0].grad_fn is not None}")
    print(f"losses_good[0] is just a number: {isinstance(losses_good[0], float)}")

    print(
        f"\n❌ PROBLEM: losses_bad keeps {len(losses_bad)} computational graphs in memory!"
    )
    print(f"   Each graph contains references to all intermediate tensors")
    print(f"   In real training, this grows until you run out of memory")

    print(f"\n✅ SOLUTION: Use .item() to extract values without graph references")
    print(f"   losses_good contains just {len(losses_good)} Python floats")
    print(f"   Memory usage stays constant regardless of training duration")

    print(f"\n💡 TRAINING LOOP RULE:")
    print(f"   For logging: loss.item(), accuracy.item(), etc.")
    print(f"   For backprop: loss.backward() (without storing the tensor)")


# ==========================================
# DEBUGGING TOOLS
# ==========================================


def debugging_tools():
    """
    Tools for debugging computational graphs
    """
    print("\n=== DEBUGGING TOOLS ===\n")

    x = torch.tensor([2.0], requires_grad=True)
    y = torch.tensor([3.0], requires_grad=True)

    z = x * y + x**2

    # Check if gradients are enabled
    print(f"Gradients enabled: {torch.is_grad_enabled()}")

    # Check tensor properties
    print(f"x.requires_grad: {x.requires_grad}")
    print(f"z.requires_grad: {z.requires_grad}")
    print(f"z.is_leaf: {z.is_leaf}")

    # Gradient checking (for debugging custom functions)
    from torch.autograd import gradcheck

    def simple_func(x):
        return (x**2).sum()

    # This verifies numerical vs analytical gradients
    test_input = torch.randn(3, dtype=torch.double, requires_grad=True)
    test_result = gradcheck(simple_func, test_input, eps=1e-6, atol=1e-4)
    print(f"Gradient check passed: {test_result}")


# Run all demonstrations
if __name__ == "__main__":
    explore_computational_graph()
    understand_grad_accumulation()
    tensor_item_deep_dive()
    test_custom_autograd()
    memory_pitfalls_demo()
    debugging_tools()

# ==========================================
# INTERVIEW QUESTIONS FOR YOU:
# ==========================================

"""
ADVANCED QUESTIONS:

1. MEMORY LEAK: You notice your training loop is consuming more and more memory 
   each epoch, even though batch size is constant. What are the top 3 most 
   likely causes and how would you debug this?

2. GRADIENT EXPLOSION: Your gradients are exploding. Beyond gradient clipping,
   what are 5 different root causes and their solutions?

3. COMPUTATIONAL EFFICIENCY: You have tensor operations inside a training loop
   that don't need gradients. How would you optimize this? What's the 
   performance difference between torch.no_grad() and .detach()?

4. CUSTOM LOSS DESIGN: You need to implement a loss function that requires
   computing gradients of gradients (second-order). How would you approach this?

5. DISTRIBUTED TRAINING: How does gradient accumulation work differently in 
   DataParallel vs DistributedDataParallel? What are the synchronization points?

6. MIXED PRECISION: When using automatic mixed precision (AMP), what happens 
   to the computational graph? How does GradScaler prevent underflow?
"""
