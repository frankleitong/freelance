# PyTorch Graph Mode: From Eager to Static Graph

## Table of Contents

- [1. Background: Two Execution Modes](#1-background-two-execution-modes)
- [2. Why Graph Mode Matters](#2-why-graph-mode-matters)
- [3. Graph Mode APIs in PyTorch](#3-graph-mode-apis-in-pytorch)
- [4. Hands-On Examples: Simple to Complex](#4-hands-on-examples-simple-to-complex)
  - [Example 1: Your First torch.compile](#example-1-your-first-torchcompile)
  - [Example 2: Scaling Up — Does Model Size Matter?](#example-2-scaling-up--does-model-size-matter)
  - [Example 3: The Real Speedup — Pointwise Fusion](#example-3-the-real-speedup--pointwise-fusion)
  - [Example 4: Realistic Transformer Block](#example-4-realistic-transformer-block)
  - [Example 5: TorchScript for Deployment](#example-5-torchscript-for-deployment)
- [5. When to Use What](#5-when-to-use-what)
- [6. Common Pitfalls](#6-common-pitfalls)
- [7. Summary](#7-summary)

---

## 1. Background: Two Execution Modes

PyTorch supports two fundamentally different execution paradigms:

### Eager Mode (Dynamic Graph)

This is PyTorch's default. Every operation executes **immediately** as Python runs, line by line.

```python
x = torch.randn(4, 4)
y = x + 2       # Executes NOW
z = y * y       # Executes NOW
```

- The computation graph is built and destroyed on every forward pass.
- Full Python control flow (if/else, loops, print, pdb) works naturally.
- Great for prototyping and debugging.

### Graph Mode (Static Graph)

The computation graph is captured **ahead of time**, then optimized and executed as a whole.

```python
@torch.compile
def f(x):
    y = x + 2
    z = y * y
    return z
```

- PyTorch traces the operations, builds a graph, and applies optimizations before running anything.
- The optimizer can see the full picture: fuse operations, eliminate redundancy, optimize memory.
- Harder to debug, but can be significantly faster.

Think of it this way:
- **Eager mode** = an interpreter running your code line by line.
- **Graph mode** = a compiler that reads your whole program, optimizes it, then runs the optimized version.

---

## 2. Why Graph Mode Matters

In eager mode, each operation (add, multiply, relu, etc.) launches a **separate GPU kernel**. Each kernel launch involves:

1. Read input tensors from GPU memory
2. Compute the result
3. Write output tensor back to GPU memory

For a chain of N pointwise operations, that means **N kernel launches** and **N round-trips to GPU memory**.

```
Eager:  [read → add → write] → [read → mul → write] → [read → relu → write]
                 ↑                       ↑                       ↑
            kernel #1               kernel #2               kernel #3
```

Graph mode can **fuse** these into a single kernel:

```
Compiled: [read → add → mul → relu → write]
                        ↑
                   kernel #1 (fused)
```

This reduces:
- **Kernel launch overhead**: fewer CPU→GPU dispatches
- **Memory bandwidth**: intermediate results stay in GPU registers instead of being written to and read from global memory

On modern GPUs, memory bandwidth (not compute) is often the bottleneck. This is why fusion matters so much.

---

## 3. Graph Mode APIs in PyTorch

PyTorch provides several APIs for graph mode:

| API | Introduced | Status | Use Case |
|-----|-----------|--------|----------|
| `torch.compile` | PyTorch 2.0 (2023) | **Recommended** | Training & inference speedup |
| `torch.export` | PyTorch 2.1 (2023) | Active | Deployment, edge, mobile |
| `torch.jit.trace` | PyTorch 1.0 (2018) | Legacy | Deployment without Python |
| `torch.jit.script` | PyTorch 1.0 (2018) | Legacy | Deployment without Python |

### torch.compile (TorchDynamo + TorchInductor)

The modern approach. Uses TorchDynamo to capture the graph from Python bytecode, then TorchInductor to generate optimized GPU kernels (via Triton).

```python
model = MyModel()
compiled_model = torch.compile(model)  # That's it
output = compiled_model(input)
```

### TorchScript (Legacy)

The older approach. Captures a serializable graph that can run without Python.

```python
# Tracing: records operations from a sample input
traced = torch.jit.trace(model, sample_input)

# Scripting: parses Python source into TorchScript IR
scripted = torch.jit.script(model)

# Save for deployment
traced.save("model.pt")
```

---

## 4. Hands-On Examples: Simple to Complex

### Prerequisites

```python
import torch
import torch.nn as nn
import time

def benchmark(fn, label, n=10000, warmup=100):
    """Benchmark a callable function."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.3f}s ({n} iters, {elapsed/n*1e6:.1f} us/iter)")

device = "cuda"
```

---

### Example 1: Your First torch.compile

**Goal**: See torch.compile in action on a simple model.

```python
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = SimpleNet().to(device)
x = torch.randn(64, 1024, device=device)

compiled_model = torch.compile(model)

with torch.no_grad():
    benchmark(lambda: model(x),          "Eager")
    benchmark(lambda: compiled_model(x), "Compiled")
```

**Expected result**: Almost no difference (~0-3%).

**Why?** This model is just 3 `nn.Linear` layers (matrix multiplications). Matrix multiplication already calls **cuBLAS**, NVIDIA's hand-tuned GEMM library. `torch.compile` cannot beat cuBLAS at what cuBLAS does best — there's simply nothing to optimize.

**Lesson**: `torch.compile` is not a universal "make everything faster" button. It optimizes specific types of operations.

---

### Example 2: Scaling Up — Does Model Size Matter?

**Goal**: Test whether a bigger model changes the picture.

```python
class BigNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            *[nn.Sequential(nn.Linear(4096, 4096), nn.ReLU()) for _ in range(10)]
        )

    def forward(self, x):
        return self.layers(x)

model = BigNet().to(device).eval()
x = torch.randn(256, 4096, device=device)

compiled_model = torch.compile(model)

with torch.no_grad():
    benchmark(lambda: model(x),          "Eager")
    benchmark(lambda: compiled_model(x), "Compiled")
```

**Expected result**: Still negligible difference.

**Why?** Even with 10 layers and larger dimensions, the compute is still dominated by matrix multiplications (cuBLAS). Scaling the size of `nn.Linear` just gives cuBLAS more work — it doesn't create more fusion opportunities.

**Lesson**: The **type** of operations matters more than the **size** of the model.

---

### Example 3: The Real Speedup — Pointwise Fusion

**Goal**: See a dramatic speedup by using operations that benefit from fusion.

```python
def pointwise_heavy(x):
    for _ in range(20):
        x = x * torch.sigmoid(x)                       # SiLU/Swish
        x = x / (x.std(dim=-1, keepdim=True) + 1e-6)   # manual norm
        x = x + torch.tanh(x)
        x = x * 0.5 + x ** 2 * 0.01
    return x

x = torch.randn(256, 4096, device=device)

compiled_fn = torch.compile(pointwise_heavy)

with torch.no_grad():
    benchmark(lambda: pointwise_heavy(x), "Eager")
    benchmark(lambda: compiled_fn(x),     "Compiled")
```

**Expected result**: **4-6x speedup** (or more).

**Why?** In eager mode, every single operation (`sigmoid`, `mul`, `std`, `div`, `tanh`, `add`, `pow`) launches a **separate GPU kernel**. Each kernel reads from and writes to GPU global memory. With 20 loop iterations and ~7 operations each, that's **~140 separate kernel launches** and memory round-trips.

`torch.compile` fuses the entire chain into a small number of kernels. Intermediate results stay in fast GPU registers instead of being written back to slow global memory.

```
Eager (per iteration):
  sigmoid: read x → compute → write tmp1
  mul:     read x, tmp1 → compute → write tmp2
  std:     read tmp2 → compute → write tmp3
  div:     read tmp2, tmp3 → compute → write tmp4
  tanh:    read tmp4 → compute → write tmp5
  add:     read tmp4, tmp5 → compute → write tmp6
  ...

Compiled (per iteration):
  fused:   read x → sigmoid → mul → std → div → tanh → add → ... → write result
```

**Lesson**: `torch.compile` shines when there are **chains of pointwise/elementwise operations** that can be fused, reducing memory bandwidth pressure.

---

### Example 4: Realistic Transformer Block

**Goal**: See how graph mode helps in a real-world architecture with a mix of matmul and pointwise ops.

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model=1024, n_heads=16, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-attention with residual + norm
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.dropout(attn_out)

        # FFN with residual + norm
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)
        return x

class SmallTransformer(nn.Module):
    def __init__(self, n_layers=6, d_model=1024, n_heads=16):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

model = SmallTransformer().to(device).eval()
x = torch.randn(32, 128, 1024, device=device)  # (batch, seq_len, d_model)

compiled_model = torch.compile(model)

with torch.no_grad():
    benchmark(lambda: model(x),          "Eager",    n=500)
    benchmark(lambda: compiled_model(x), "Compiled", n=500)
```

**Expected result**: 10-30% speedup (varies by GPU).

**Why?** A Transformer block mixes both types of operations:
- **Matmul-heavy**: Q/K/V projections, attention score computation, FFN layers → cuBLAS handles these, minimal gain
- **Pointwise-heavy**: LayerNorm, GELU, softmax, dropout, residual add → these get fused!

The overall speedup is a weighted average: the pointwise portions speed up dramatically, but the matmul portions stay the same.

**Lesson**: Real-world models see moderate but meaningful speedups. The more pointwise operations relative to matmuls, the bigger the gain.

---

### Example 5: TorchScript for Deployment

**Goal**: Export a model for production deployment without a Python runtime.

```python
class ProductionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

model = ProductionModel().eval()
sample_input = torch.randn(1, 784)

# --- Method 1: Tracing ---
# Records operations by running the model with sample input.
# WARNING: control flow (if/else based on input) is NOT captured.
traced_model = torch.jit.trace(model, sample_input)

# --- Method 2: Scripting ---
# Parses the Python source code into TorchScript IR.
# Supports control flow but has Python syntax limitations.
scripted_model = torch.jit.script(model)

# Save for deployment (can be loaded in C++, mobile, etc.)
traced_model.save("model_traced.pt")
scripted_model.save("model_scripted.pt")

# Load and verify
loaded_model = torch.jit.load("model_traced.pt")
with torch.no_grad():
    original_out = model(sample_input)
    loaded_out = loaded_model(sample_input)
    print(f"Output matches: {torch.allclose(original_out, loaded_out, atol=1e-5)}")
    # Output matches: True

# Inspect the graph
print(traced_model.graph)
```

**Tracing vs Scripting**:

| | `torch.jit.trace` | `torch.jit.script` |
|---|---|---|
| How it works | Runs model, records ops | Parses Python source |
| Control flow | NOT captured (flattened) | Supported (limited) |
| Dynamic shapes | Fixed to trace-time shapes | Supported |
| Ease of use | Easy | May need code changes |

**Note**: For new projects, prefer `torch.export` over TorchScript for deployment:

```python
# Modern alternative (PyTorch 2.1+)
exported = torch.export.export(model, (sample_input,))
```

---

## 5. When to Use What

| Scenario | Recommended Approach |
|----------|---------------------|
| Prototyping & debugging | Eager mode (default) |
| Speed up training | `torch.compile(model)` |
| Speed up inference | `torch.compile(model, mode="reduce-overhead")` |
| Deploy without Python | `torch.export` (or TorchScript) |
| Mobile / edge deployment | `torch.export` → ExecuTorch |
| Maximum inference optimization | `torch.compile` + quantization |

### torch.compile modes

```python
# Default: good balance of compile time and speedup
model = torch.compile(model)

# Reduce overhead: minimize CPU overhead, best for small batches
model = torch.compile(model, mode="reduce-overhead")

# Max autotune: try many kernel configs, slower compile, faster runtime
model = torch.compile(model, mode="max-autotune")
```

---

## 6. Common Pitfalls

### 1. Expecting speedup on matmul-heavy models

```python
# This will NOT be faster with torch.compile:
def matmul_only(x, w1, w2, w3):
    x = x @ w1
    x = x @ w2
    x = x @ w3
    return x
```

cuBLAS already handles matrix multiplication optimally. `torch.compile` adds value by fusing the **other** operations.

### 2. Graph breaks

When `torch.compile` encounters code it can't trace (e.g., data-dependent control flow, unsupported Python features), it inserts a **graph break** — splitting the graph into smaller pieces and falling back to eager mode for the untraceable parts.

```python
@torch.compile
def f(x):
    x = x * 2
    print(x.shape)  # graph break! print is a Python side effect
    x = x + 1
    return x
```

Use `torch._dynamo.explain(f, x)` to diagnose graph breaks.

### 3. First-call overhead

`torch.compile` compiles on the **first call**, which can take seconds to minutes. Always warm up before benchmarking:

```python
compiled_model = torch.compile(model)

# BAD: includes compilation time
start = time.time()
compiled_model(x)  # First call: compiles AND runs
print(time.time() - start)  # Misleadingly slow

# GOOD: warm up first
for _ in range(3):
    compiled_model(x)  # Compilation happens here
torch.cuda.synchronize()

start = time.time()
for _ in range(1000):
    compiled_model(x)  # Pure runtime
torch.cuda.synchronize()
print(time.time() - start)
```

### 4. Dynamic shapes cause recompilation

If input shapes change between calls, `torch.compile` may recompile the graph each time. Use `dynamic=True` to handle varying shapes:

```python
compiled_model = torch.compile(model, dynamic=True)
```

---

## 7. Summary

### The Core Idea

PyTorch graph mode captures your computation as a **static graph** and optimizes it before execution. The primary optimization is **kernel fusion** — combining multiple operations into one GPU kernel to reduce memory bandwidth usage.

### What Speeds Up

| Operation Type | Eager | Compiled | Reason |
|---------------|-------|----------|--------|
| Matrix multiply (`nn.Linear`) | Fast | Same | Already uses cuBLAS |
| Pointwise chains (norm, activation, etc.) | Slow (many kernels) | **Fast (fused)** | Fewer kernel launches, less memory I/O |
| Mixed (real models) | Baseline | **10-30% faster** | Pointwise portions get fused |
| Pure pointwise workloads | Baseline | **4-6x faster** | Everything fuses |

### The Evolution

```
TorchScript (2018)  →  torch.compile (2023)  →  torch.export (2023+)
   (legacy)              (training)              (deployment)
```

### One-Line Takeaway

> `torch.compile` doesn't make individual operations faster — it makes **chains of operations** faster by fusing them into fewer GPU kernel launches, reducing memory bandwidth overhead.
