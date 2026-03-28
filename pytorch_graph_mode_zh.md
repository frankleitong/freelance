# PyTorch 图模式：从即时执行到静态图

## 目录

- [1. 背景：两种执行模式](#1-背景两种执行模式)
- [2. 为什么图模式很重要](#2-为什么图模式很重要)
- [3. PyTorch 中的图模式 API](#3-pytorch-中的图模式-api)
- [4. 实战示例：从简单到复杂](#4-实战示例从简单到复杂)
  - [示例 1：初识 torch.compile](#示例-1初识-torchcompile)
  - [示例 2：增大模型——模型大小有影响吗？](#示例-2增大模型模型大小有影响吗)
  - [示例 3：真正的加速——逐点运算融合](#示例-3真正的加速逐点运算融合)
  - [示例 4：真实的 Transformer 模块](#示例-4真实的-transformer-模块)
  - [示例 5：TorchScript 部署](#示例-5torchscript-部署)
- [5. 何时使用什么](#5-何时使用什么)
- [6. 常见陷阱](#6-常见陷阱)
- [7. 总结](#7-总结)

---

## 1. 背景：两种执行模式

PyTorch 支持两种根本不同的执行范式：

### 即时模式（动态图）

这是 PyTorch 的默认模式。每个操作在 Python 执行时**立即运行**，逐行执行。

```python
x = torch.randn(4, 4)
y = x + 2       # 立即执行
z = y * y       # 立即执行
```

- 计算图在每次前向传播时构建并销毁。
- 完全支持 Python 控制流（if/else、循环、print、pdb）。
- 非常适合原型开发和调试。

### 图模式（静态图）

计算图被**提前捕获**，然后作为整体进行优化和执行。

```python
@torch.compile
def f(x):
    y = x + 2
    z = y * y
    return z
```

- PyTorch 追踪操作，构建计算图，在执行前应用优化。
- 优化器可以看到全局：融合操作、消除冗余、优化内存。
- 调试更困难，但可以显著加速。

可以这样理解：
- **即时模式** = 解释器逐行运行你的代码。
- **图模式** = 编译器读取整个程序，优化后再运行优化版本。

---

## 2. 为什么图模式很重要

在即时模式下，每个操作（加法、乘法、relu 等）都会启动一个**独立的 GPU 内核**。每次内核启动涉及：

1. 从 GPU 显存读取输入张量
2. 计算结果
3. 将输出张量写回 GPU 显存

对于 N 个逐点操作的链，这意味着 **N 次内核启动**和 **N 次 GPU 显存往返**。

```
即时模式:  [读取 → 加法 → 写入] → [读取 → 乘法 → 写入] → [读取 → relu → 写入]
                   ↑                       ↑                       ↑
              内核 #1                  内核 #2                  内核 #3
```

图模式可以将这些**融合**为单个内核：

```
编译模式: [读取 → 加法 → 乘法 → relu → 写入]
                        ↑
                   内核 #1（融合后）
```

这减少了：
- **内核启动开销**：更少的 CPU→GPU 调度
- **显存带宽**：中间结果保留在 GPU 寄存器中，而不是写入和读取全局显存

在现代 GPU 上，显存带宽（而非计算能力）通常是瓶颈。这就是融合如此重要的原因。

---

## 3. PyTorch 中的图模式 API

PyTorch 提供了多个图模式 API：

| API | 引入时间 | 状态 | 用途 |
|-----|---------|------|------|
| `torch.compile` | PyTorch 2.0 (2023) | **推荐** | 训练和推理加速 |
| `torch.export` | PyTorch 2.1 (2023) | 活跃 | 部署、边缘计算、移动端 |
| `torch.jit.trace` | PyTorch 1.0 (2018) | 旧版 | 无 Python 环境部署 |
| `torch.jit.script` | PyTorch 1.0 (2018) | 旧版 | 无 Python 环境部署 |

### torch.compile（TorchDynamo + TorchInductor）

现代方法。使用 TorchDynamo 从 Python 字节码捕获计算图，然后用 TorchInductor 生成优化的 GPU 内核（通过 Triton）。

```python
model = MyModel()
compiled_model = torch.compile(model)  # 就这么简单
output = compiled_model(input)
```

### TorchScript（旧版）

较老的方法。捕获可序列化的计算图，可以在没有 Python 的环境中运行。

```python
# 追踪：通过样本输入记录操作
traced = torch.jit.trace(model, sample_input)

# 脚本化：将 Python 源码解析为 TorchScript IR
scripted = torch.jit.script(model)

# 保存用于部署
traced.save("model.pt")
```

---

## 4. 实战示例：从简单到复杂

### 前置准备

```python
import torch
import torch.nn as nn
import time

def benchmark(fn, label, n=10000, warmup=100):
    """基准测试函数。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.3f}s ({n} 次迭代, {elapsed/n*1e6:.1f} us/次)")

device = "cuda"
```

---

### 示例 1：初识 torch.compile

**目标**：在简单模型上体验 torch.compile。

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
    benchmark(lambda: model(x),          "即时模式")
    benchmark(lambda: compiled_model(x), "编译模式")
```

**预期结果**：几乎没有差异（约 0-3%）。

**为什么？** 这个模型只有 3 个 `nn.Linear` 层（矩阵乘法）。矩阵乘法已经调用了 **cuBLAS**——NVIDIA 手工调优的 GEMM 库。`torch.compile` 无法在 cuBLAS 最擅长的领域超越它——根本没有什么可优化的。

**结论**：`torch.compile` 不是万能的"让一切变快"按钮。它优化的是特定类型的操作。

---

### 示例 2：增大模型——模型大小有影响吗？

**目标**：测试更大的模型是否改变结果。

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
    benchmark(lambda: model(x),          "即时模式")
    benchmark(lambda: compiled_model(x), "编译模式")
```

**预期结果**：差异仍然可以忽略。

**为什么？** 即使有 10 层和更大的维度，计算仍然由矩阵乘法（cuBLAS）主导。增大 `nn.Linear` 的尺寸只是给 cuBLAS 更多工作——并没有创造更多融合机会。

**结论**：操作的**类型**比模型的**大小**更重要。

---

### 示例 3：真正的加速——逐点运算融合

**目标**：通过使用可受益于融合的操作来观察显著加速。

```python
def pointwise_heavy(x):
    for _ in range(20):
        x = x * torch.sigmoid(x)                       # SiLU/Swish
        x = x / (x.std(dim=-1, keepdim=True) + 1e-6)   # 手动归一化
        x = x + torch.tanh(x)
        x = x * 0.5 + x ** 2 * 0.01
    return x

x = torch.randn(256, 4096, device=device)

compiled_fn = torch.compile(pointwise_heavy)

with torch.no_grad():
    benchmark(lambda: pointwise_heavy(x), "即时模式")
    benchmark(lambda: compiled_fn(x),     "编译模式")
```

**预期结果**：**4-6 倍加速**（甚至更多）。

**为什么？** 在即时模式下，每个操作（`sigmoid`、`mul`、`std`、`div`、`tanh`、`add`、`pow`）都会启动一个**独立的 GPU 内核**。每个内核都需要从 GPU 全局显存读写。20 次循环迭代，每次约 7 个操作，总共约 **140 次独立的内核启动**和显存往返。

`torch.compile` 将整个操作链融合为少量内核。中间结果保留在快速的 GPU 寄存器中，而不是写回慢速的全局显存。

```
即时模式（每次迭代）:
  sigmoid: 读取 x → 计算 → 写入 tmp1
  mul:     读取 x, tmp1 → 计算 → 写入 tmp2
  std:     读取 tmp2 → 计算 → 写入 tmp3
  div:     读取 tmp2, tmp3 → 计算 → 写入 tmp4
  tanh:    读取 tmp4 → 计算 → 写入 tmp5
  add:     读取 tmp4, tmp5 → 计算 → 写入 tmp6
  ...

编译模式（每次迭代）:
  融合:   读取 x → sigmoid → mul → std → div → tanh → add → ... → 写入结果
```

**结论**：`torch.compile` 在存在**逐点/逐元素操作链**时表现突出，通过融合减少显存带宽压力。

---

### 示例 4：真实的 Transformer 模块

**目标**：观察图模式如何帮助包含矩阵乘法和逐点操作混合的真实架构。

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
        # 带残差连接和归一化的自注意力
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.dropout(attn_out)

        # 带残差连接和归一化的前馈网络
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
    benchmark(lambda: model(x),          "即时模式", n=500)
    benchmark(lambda: compiled_model(x), "编译模式", n=500)
```

**预期结果**：10-30% 加速（因 GPU 而异）。

**为什么？** Transformer 模块混合了两种类型的操作：
- **矩阵乘法为主**：Q/K/V 投影、注意力分数计算、FFN 层 → cuBLAS 处理，增益很小
- **逐点操作为主**：LayerNorm、GELU、softmax、dropout、残差加法 → 这些会被融合！

总体加速是加权平均：逐点部分大幅加速，但矩阵乘法部分保持不变。

**结论**：真实模型会获得适度但有意义的加速。逐点操作相对于矩阵乘法越多，增益越大。

---

### 示例 5：TorchScript 部署

**目标**：导出模型用于无需 Python 运行时的生产部署。

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

# --- 方法 1：追踪 ---
# 通过样本输入运行模型来记录操作。
# 注意：基于输入的控制流（if/else）不会被捕获。
traced_model = torch.jit.trace(model, sample_input)

# --- 方法 2：脚本化 ---
# 将 Python 源码解析为 TorchScript IR。
# 支持控制流，但 Python 语法有限制。
scripted_model = torch.jit.script(model)

# 保存用于部署（可在 C++、移动端等环境加载）
traced_model.save("model_traced.pt")
scripted_model.save("model_scripted.pt")

# 加载并验证
loaded_model = torch.jit.load("model_traced.pt")
with torch.no_grad():
    original_out = model(sample_input)
    loaded_out = loaded_model(sample_input)
    print(f"输出匹配: {torch.allclose(original_out, loaded_out, atol=1e-5)}")
    # 输出匹配: True

# 查看计算图
print(traced_model.graph)
```

**追踪 vs 脚本化**：

| | `torch.jit.trace` | `torch.jit.script` |
|---|---|---|
| 工作方式 | 运行模型，记录操作 | 解析 Python 源码 |
| 控制流 | 不会捕获（被展平） | 支持（有限制） |
| 动态形状 | 固定为追踪时的形状 | 支持 |
| 易用性 | 简单 | 可能需要修改代码 |

**注意**：新项目建议使用 `torch.export` 替代 TorchScript 进行部署：

```python
# 现代替代方案（PyTorch 2.1+）
exported = torch.export.export(model, (sample_input,))
```

---

## 5. 何时使用什么

| 场景 | 推荐方法 |
|------|---------|
| 原型开发和调试 | 即时模式（默认） |
| 加速训练 | `torch.compile(model)` |
| 加速推理 | `torch.compile(model, mode="reduce-overhead")` |
| 无 Python 部署 | `torch.export`（或 TorchScript） |
| 移动端/边缘部署 | `torch.export` → ExecuTorch |
| 最大推理优化 | `torch.compile` + 量化 |

### torch.compile 模式

```python
# 默认：编译时间和加速的良好平衡
model = torch.compile(model)

# 减少开销：最小化 CPU 开销，适合小批量
model = torch.compile(model, mode="reduce-overhead")

# 最大自动调优：尝试多种内核配置，编译更慢，运行更快
model = torch.compile(model, mode="max-autotune")
```

---

## 6. 常见陷阱

### 1. 期望在矩阵乘法密集模型上获得加速

```python
# 使用 torch.compile 不会更快：
def matmul_only(x, w1, w2, w3):
    x = x @ w1
    x = x @ w2
    x = x @ w3
    return x
```

cuBLAS 已经以最优方式处理矩阵乘法。`torch.compile` 通过融合**其他**操作来增加价值。

### 2. 图断裂

当 `torch.compile` 遇到无法追踪的代码（如依赖数据的控制流、不支持的 Python 特性），它会插入**图断裂**——将计算图分割成更小的部分，对无法追踪的部分回退到即时模式。

```python
@torch.compile
def f(x):
    x = x * 2
    print(x.shape)  # 图断裂！print 是 Python 副作用
    x = x + 1
    return x
```

使用 `torch._dynamo.explain(f, x)` 诊断图断裂。

### 3. 首次调用开销

`torch.compile` 在**首次调用**时编译，可能需要数秒到数分钟。基准测试前务必预热：

```python
compiled_model = torch.compile(model)

# 错误：包含了编译时间
start = time.time()
compiled_model(x)  # 首次调用：编译并运行
print(time.time() - start)  # 结果会有误导性

# 正确：先预热
for _ in range(3):
    compiled_model(x)  # 编译在这里发生
torch.cuda.synchronize()

start = time.time()
for _ in range(1000):
    compiled_model(x)  # 纯运行时间
torch.cuda.synchronize()
print(time.time() - start)
```

### 4. 动态形状导致重新编译

如果调用之间输入形状改变，`torch.compile` 可能每次都重新编译计算图。使用 `dynamic=True` 处理变化的形状：

```python
compiled_model = torch.compile(model, dynamic=True)
```

---

## 7. 总结

### 核心思想

PyTorch 图模式将你的计算捕获为**静态图**，在执行前进行优化。主要优化是**内核融合**——将多个操作合并为一个 GPU 内核，减少显存带宽使用。

### 什么会被加速

| 操作类型 | 即时模式 | 编译模式 | 原因 |
|---------|---------|---------|------|
| 矩阵乘法（`nn.Linear`） | 快 | 相同 | 已使用 cuBLAS |
| 逐点操作链（归一化、激活等） | 慢（多个内核） | **快（融合）** | 更少的内核启动，更少的显存 I/O |
| 混合（真实模型） | 基准线 | **快 10-30%** | 逐点部分被融合 |
| 纯逐点工作负载 | 基准线 | **快 4-6 倍** | 所有操作都被融合 |

### 发展历程

```
TorchScript (2018)  →  torch.compile (2023)  →  torch.export (2023+)
    （旧版）              （训练加速）              （部署）
```

### 一句话总结

> `torch.compile` 不会让单个操作变快——它通过将**操作链**融合为更少的 GPU 内核启动来加速，减少显存带宽开销。
