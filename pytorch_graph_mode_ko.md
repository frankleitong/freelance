# PyTorch 그래프 모드: Eager 모드에서 정적 그래프까지

## 목차

- [1. 배경: 두 가지 실행 모드](#1-배경-두-가지-실행-모드)
- [2. 그래프 모드가 중요한 이유](#2-그래프-모드가-중요한-이유)
- [3. PyTorch의 그래프 모드 API](#3-pytorch의-그래프-모드-api)
- [4. 실습 예제: 단순에서 복잡으로](#4-실습-예제-단순에서-복잡으로)
  - [예제 1: 첫 번째 torch.compile](#예제-1-첫-번째-torchcompile)
  - [예제 2: 스케일 업 — 모델 크기가 영향을 미칠까?](#예제-2-스케일-업--모델-크기가-영향을-미칠까)
  - [예제 3: 진짜 속도 향상 — 포인트와이즈 연산 융합](#예제-3-진짜-속도-향상--포인트와이즈-연산-융합)
  - [예제 4: 실제 Transformer 블록](#예제-4-실제-transformer-블록)
  - [예제 5: TorchScript 배포](#예제-5-torchscript-배포)
- [5. 상황별 가이드](#5-상황별-가이드)
- [6. 흔한 실수](#6-흔한-실수)
- [7. 요약](#7-요약)

---

## 1. 배경: 두 가지 실행 모드

PyTorch는 근본적으로 다른 두 가지 실행 패러다임을 지원합니다:

### Eager 모드 (동적 그래프)

PyTorch의 기본 모드입니다. 모든 연산이 Python 실행에 따라 **즉시 실행**됩니다.

```python
x = torch.randn(4, 4)
y = x + 2       # 지금 바로 실행
z = y * y       # 지금 바로 실행
```

- 계산 그래프가 매 순전파마다 생성되고 파괴됩니다.
- Python 제어 흐름(if/else, 루프, print, pdb)이 자연스럽게 동작합니다.
- 프로토타이핑과 디버깅에 최적입니다.

### 그래프 모드 (정적 그래프)

계산 그래프가 **사전에 캡처**되고, 전체적으로 최적화되어 실행됩니다.

```python
@torch.compile
def f(x):
    y = x + 2
    z = y * y
    return z
```

- PyTorch가 연산을 추적하고, 그래프를 구성하고, 실행 전에 최적화를 적용합니다.
- 옵티마이저가 전체 그림을 파악: 연산 융합, 중복 제거, 메모리 최적화.
- 디버깅이 어려워지지만, 상당히 빨라질 수 있습니다.

이렇게 생각하세요:
- **Eager 모드** = 코드를 한 줄씩 실행하는 인터프리터.
- **그래프 모드** = 전체 프로그램을 읽고, 최적화한 후 실행하는 컴파일러.

---

## 2. 그래프 모드가 중요한 이유

Eager 모드에서는 각 연산(덧셈, 곱셈, relu 등)이 **별도의 GPU 커널**을 실행합니다. 각 커널 실행에는 다음이 포함됩니다:

1. GPU 메모리에서 입력 텐서 읽기
2. 결과 계산
3. 출력 텐서를 GPU 메모리에 다시 쓰기

N개의 포인트와이즈 연산 체인에서는 **N번의 커널 실행**과 **N번의 GPU 메모리 왕복**이 필요합니다.

```
Eager:  [읽기 → 덧셈 → 쓰기] → [읽기 → 곱셈 → 쓰기] → [읽기 → relu → 쓰기]
                ↑                       ↑                       ↑
           커널 #1                 커널 #2                 커널 #3
```

그래프 모드는 이들을 단일 커널로 **융합**할 수 있습니다:

```
컴파일됨: [읽기 → 덧셈 → 곱셈 → relu → 쓰기]
                       ↑
                  커널 #1 (융합됨)
```

이를 통해 다음이 줄어듭니다:
- **커널 실행 오버헤드**: CPU→GPU 디스패치 감소
- **메모리 대역폭**: 중간 결과가 글로벌 메모리 대신 GPU 레지스터에 유지됨

현대 GPU에서는 메모리 대역폭(연산 능력이 아닌)이 병목이 되는 경우가 많습니다. 그래서 융합이 매우 중요합니다.

---

## 3. PyTorch의 그래프 모드 API

PyTorch는 그래프 모드를 위한 여러 API를 제공합니다:

| API | 도입 시기 | 상태 | 용도 |
|-----|---------|------|------|
| `torch.compile` | PyTorch 2.0 (2023) | **권장** | 학습 및 추론 가속 |
| `torch.export` | PyTorch 2.1 (2023) | 활성 | 배포, 엣지, 모바일 |
| `torch.jit.trace` | PyTorch 1.0 (2018) | 레거시 | Python 없이 배포 |
| `torch.jit.script` | PyTorch 1.0 (2018) | 레거시 | Python 없이 배포 |

### torch.compile (TorchDynamo + TorchInductor)

현대적인 접근법. TorchDynamo로 Python 바이트코드에서 그래프를 캡처하고, TorchInductor로 최적화된 GPU 커널을 생성합니다(Triton 경유).

```python
model = MyModel()
compiled_model = torch.compile(model)  # 이게 전부입니다
output = compiled_model(input)
```

### TorchScript (레거시)

이전 접근법. 직렬화 가능한 그래프를 캡처하여 Python 없이 실행할 수 있습니다.

```python
# 추적: 샘플 입력으로 연산을 기록
traced = torch.jit.trace(model, sample_input)

# 스크립팅: Python 소스 코드를 TorchScript IR로 파싱
scripted = torch.jit.script(model)

# 배포용으로 저장
traced.save("model.pt")
```

---

## 4. 실습 예제: 단순에서 복잡으로

### 사전 준비

```python
import torch
import torch.nn as nn
import time

def benchmark(fn, label, n=10000, warmup=100):
    """벤치마크 함수."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.3f}s ({n}회, {elapsed/n*1e6:.1f} us/회)")

device = "cuda"
```

---

### 예제 1: 첫 번째 torch.compile

**목표**: 간단한 모델에서 torch.compile을 체험한다.

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
    benchmark(lambda: compiled_model(x), "컴파일됨")
```

**예상 결과**: 거의 차이 없음 (약 0-3%).

**왜?** 이 모델은 3개의 `nn.Linear` 레이어(행렬 곱셈)뿐입니다. 행렬 곱셈은 이미 **cuBLAS**(NVIDIA가 직접 튜닝한 GEMM 라이브러리)를 호출합니다. `torch.compile`은 cuBLAS가 가장 잘하는 것을 능가할 수 없습니다 — 최적화할 것이 없습니다.

**교훈**: `torch.compile`은 만능 "모든 것을 빠르게" 버튼이 아닙니다. 특정 유형의 연산을 최적화합니다.

---

### 예제 2: 스케일 업 — 모델 크기가 영향을 미칠까?

**목표**: 더 큰 모델이 결과를 바꾸는지 테스트한다.

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
    benchmark(lambda: compiled_model(x), "컴파일됨")
```

**예상 결과**: 여전히 무시할 수 있는 차이.

**왜?** 10개 레이어와 더 큰 차원이 있어도, 연산은 여전히 행렬 곱셈(cuBLAS)이 지배적입니다. `nn.Linear` 크기를 키워도 cuBLAS에 더 많은 작업을 줄 뿐 — 융합 기회가 늘어나지 않습니다.

**교훈**: 연산의 **유형**이 모델의 **크기**보다 중요합니다.

---

### 예제 3: 진짜 속도 향상 — 포인트와이즈 연산 융합

**목표**: 융합의 혜택을 받는 연산으로 극적인 속도 향상을 관찰한다.

```python
def pointwise_heavy(x):
    for _ in range(20):
        x = x * torch.sigmoid(x)                       # SiLU/Swish
        x = x / (x.std(dim=-1, keepdim=True) + 1e-6)   # 수동 정규화
        x = x + torch.tanh(x)
        x = x * 0.5 + x ** 2 * 0.01
    return x

x = torch.randn(256, 4096, device=device)

compiled_fn = torch.compile(pointwise_heavy)

with torch.no_grad():
    benchmark(lambda: pointwise_heavy(x), "Eager")
    benchmark(lambda: compiled_fn(x),     "컴파일됨")
```

**예상 결과**: **4-6배 속도 향상** (또는 그 이상).

**왜?** Eager 모드에서는 모든 연산(`sigmoid`, `mul`, `std`, `div`, `tanh`, `add`, `pow`)이 **별도의 GPU 커널**을 실행합니다. 각 커널이 GPU 글로벌 메모리에서 읽고 씁니다. 20회 루프 × 약 7개 연산으로, 약 **140번의 별도 커널 실행**과 메모리 왕복이 발생합니다.

`torch.compile`은 전체 체인을 소수의 커널로 융합합니다. 중간 결과가 느린 글로벌 메모리에 쓰이지 않고 빠른 GPU 레지스터에 유지됩니다.

```
Eager (반복당):
  sigmoid: x 읽기 → 계산 → tmp1 쓰기
  mul:     x, tmp1 읽기 → 계산 → tmp2 쓰기
  std:     tmp2 읽기 → 계산 → tmp3 쓰기
  div:     tmp2, tmp3 읽기 → 계산 → tmp4 쓰기
  tanh:    tmp4 읽기 → 계산 → tmp5 쓰기
  add:     tmp4, tmp5 읽기 → 계산 → tmp6 쓰기
  ...

컴파일됨 (반복당):
  융합:   x 읽기 → sigmoid → mul → std → div → tanh → add → ... → 결과 쓰기
```

**교훈**: `torch.compile`은 **포인트와이즈/요소별 연산 체인**이 있을 때 빛을 발하며, 융합을 통해 메모리 대역폭 압력을 줄입니다.

---

### 예제 4: 실제 Transformer 블록

**목표**: 행렬 곱셈과 포인트와이즈 연산이 혼합된 실제 아키텍처에서 그래프 모드가 어떻게 도움이 되는지 확인한다.

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
        # 잔차 연결과 정규화가 포함된 셀프 어텐션
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.dropout(attn_out)

        # 잔차 연결과 정규화가 포함된 피드포워드
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
    benchmark(lambda: compiled_model(x), "컴파일됨", n=500)
```

**예상 결과**: 10-30% 속도 향상 (GPU에 따라 다름).

**왜?** Transformer 블록은 두 가지 유형의 연산을 혼합합니다:
- **행렬 곱셈 중심**: Q/K/V 프로젝션, 어텐션 점수 계산, FFN 레이어 → cuBLAS가 처리, 개선 최소
- **포인트와이즈 중심**: LayerNorm, GELU, softmax, dropout, 잔차 덧셈 → 이것들이 융합됨!

전체 속도 향상은 가중 평균입니다: 포인트와이즈 부분은 크게 빨라지지만, 행렬 곱셈 부분은 그대로입니다.

**교훈**: 실제 모델에서는 적당하지만 의미 있는 속도 향상을 얻습니다. 행렬 곱셈 대비 포인트와이즈 연산이 많을수록 개선이 큽니다.

---

### 예제 5: TorchScript 배포

**목표**: Python 런타임 없이 프로덕션 배포를 위해 모델을 내보낸다.

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

# --- 방법 1: 추적 ---
# 샘플 입력으로 모델을 실행하여 연산을 기록합니다.
# 주의: 입력에 의존하는 제어 흐름(if/else)은 캡처되지 않습니다.
traced_model = torch.jit.trace(model, sample_input)

# --- 방법 2: 스크립팅 ---
# Python 소스 코드를 TorchScript IR로 파싱합니다.
# 제어 흐름을 지원하지만 Python 구문에 제한이 있습니다.
scripted_model = torch.jit.script(model)

# 배포용으로 저장 (C++, 모바일 등에서 로드 가능)
traced_model.save("model_traced.pt")
scripted_model.save("model_scripted.pt")

# 로드 및 검증
loaded_model = torch.jit.load("model_traced.pt")
with torch.no_grad():
    original_out = model(sample_input)
    loaded_out = loaded_model(sample_input)
    print(f"출력 일치: {torch.allclose(original_out, loaded_out, atol=1e-5)}")
    # 출력 일치: True

# 그래프 확인
print(traced_model.graph)
```

**추적 vs 스크립팅**:

| | `torch.jit.trace` | `torch.jit.script` |
|---|---|---|
| 작동 방식 | 모델을 실행하고 연산 기록 | Python 소스를 파싱 |
| 제어 흐름 | 캡처 안 됨 (평탄화) | 지원 (제한적) |
| 동적 형상 | 추적 시 형상에 고정 | 지원 |
| 사용 편의성 | 쉬움 | 코드 수정 필요할 수 있음 |

**참고**: 새 프로젝트에서는 배포에 TorchScript 대신 `torch.export`를 권장합니다:

```python
# 현대적 대안 (PyTorch 2.1+)
exported = torch.export.export(model, (sample_input,))
```

---

## 5. 상황별 가이드

| 시나리오 | 권장 접근법 |
|---------|-----------|
| 프로토타이핑과 디버깅 | Eager 모드 (기본값) |
| 학습 가속 | `torch.compile(model)` |
| 추론 가속 | `torch.compile(model, mode="reduce-overhead")` |
| Python 없이 배포 | `torch.export` (또는 TorchScript) |
| 모바일/엣지 배포 | `torch.export` → ExecuTorch |
| 최대 추론 최적화 | `torch.compile` + 양자화 |

### torch.compile 모드

```python
# 기본: 컴파일 시간과 속도 향상의 좋은 균형
model = torch.compile(model)

# 오버헤드 감소: CPU 오버헤드 최소화, 작은 배치에 최적
model = torch.compile(model, mode="reduce-overhead")

# 최대 자동 튜닝: 다양한 커널 구성 시도, 컴파일은 느리지만 실행은 빠름
model = torch.compile(model, mode="max-autotune")
```

---

## 6. 흔한 실수

### 1. 행렬 곱셈 중심 모델에서 속도 향상을 기대하기

```python
# torch.compile로 빨라지지 않습니다:
def matmul_only(x, w1, w2, w3):
    x = x @ w1
    x = x @ w2
    x = x @ w3
    return x
```

cuBLAS는 이미 행렬 곱셈을 최적으로 처리합니다. `torch.compile`은 **다른** 연산을 융합하여 가치를 더합니다.

### 2. 그래프 브레이크

`torch.compile`이 추적할 수 없는 코드(데이터 의존 제어 흐름, 지원되지 않는 Python 기능 등)를 만나면 **그래프 브레이크**를 삽입합니다 — 그래프를 작은 조각으로 분할하고 추적할 수 없는 부분은 Eager 모드로 폴백합니다.

```python
@torch.compile
def f(x):
    x = x * 2
    print(x.shape)  # 그래프 브레이크! print는 Python 부작용
    x = x + 1
    return x
```

`torch._dynamo.explain(f, x)`으로 그래프 브레이크를 진단할 수 있습니다.

### 3. 첫 번째 호출 오버헤드

`torch.compile`은 **첫 번째 호출** 시 컴파일합니다. 수 초에서 수 분이 걸릴 수 있습니다. 벤치마크 전에 반드시 워밍업하세요:

```python
compiled_model = torch.compile(model)

# 나쁜 예: 컴파일 시간이 포함됨
start = time.time()
compiled_model(x)  # 첫 번째 호출: 컴파일하고 실행
print(time.time() - start)  # 오해의 소지가 있는 결과

# 좋은 예: 먼저 워밍업
for _ in range(3):
    compiled_model(x)  # 여기서 컴파일이 일어남
torch.cuda.synchronize()

start = time.time()
for _ in range(1000):
    compiled_model(x)  # 순수 실행 시간
torch.cuda.synchronize()
print(time.time() - start)
```

### 4. 동적 형상으로 인한 재컴파일

호출 사이에 입력 형상이 변하면, `torch.compile`이 매번 그래프를 재컴파일할 수 있습니다. `dynamic=True`로 가변 형상을 처리하세요:

```python
compiled_model = torch.compile(model, dynamic=True)
```

---

## 7. 요약

### 핵심 아이디어

PyTorch 그래프 모드는 계산을 **정적 그래프**로 캡처하고 실행 전에 최적화합니다. 주요 최적화는 **커널 융합** — 여러 연산을 하나의 GPU 커널로 결합하여 메모리 대역폭 사용을 줄이는 것입니다.

### 무엇이 빨라지는가

| 연산 유형 | Eager | 컴파일됨 | 이유 |
|---------|-------|---------|------|
| 행렬 곱셈 (`nn.Linear`) | 빠름 | 동일 | 이미 cuBLAS 사용 |
| 포인트와이즈 연산 체인 (정규화, 활성화 등) | 느림 (다수의 커널) | **빠름 (융합)** | 커널 실행 감소, 메모리 I/O 감소 |
| 혼합 (실제 모델) | 기준선 | **10-30% 빠름** | 포인트와이즈 부분이 융합됨 |
| 순수 포인트와이즈 워크로드 | 기준선 | **4-6배 빠름** | 모든 것이 융합됨 |

### 발전 과정

```
TorchScript (2018)  →  torch.compile (2023)  →  torch.export (2023+)
    (레거시)              (학습 가속)              (배포)
```

### 한 줄 요약

> `torch.compile`은 개별 연산을 빠르게 하는 것이 아니라, **연산 체인**을 더 적은 GPU 커널 실행으로 융합하여 메모리 대역폭 오버헤드를 줄임으로써 빠르게 합니다.
