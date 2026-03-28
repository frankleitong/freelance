# PyTorch グラフモード：Eager モードから静的グラフへ

## 目次

- [1. 背景：2つの実行モード](#1-背景2つの実行モード)
- [2. なぜグラフモードが重要なのか](#2-なぜグラフモードが重要なのか)
- [3. PyTorch のグラフモード API](#3-pytorch-のグラフモード-api)
- [4. 実践例：シンプルから複雑へ](#4-実践例シンプルから複雑へ)
  - [例 1：初めての torch.compile](#例-1初めての-torchcompile)
  - [例 2：スケールアップ——モデルサイズは影響するか？](#例-2スケールアップモデルサイズは影響するか)
  - [例 3：本当の高速化——ポイントワイズ演算の融合](#例-3本当の高速化ポイントワイズ演算の融合)
  - [例 4：実際の Transformer ブロック](#例-4実際の-transformer-ブロック)
  - [例 5：TorchScript によるデプロイ](#例-5torchscript-によるデプロイ)
- [5. 使い分けガイド](#5-使い分けガイド)
- [6. よくある落とし穴](#6-よくある落とし穴)
- [7. まとめ](#7-まとめ)

---

## 1. 背景：2つの実行モード

PyTorch は根本的に異なる2つの実行パラダイムをサポートしています：

### Eager モード（動的グラフ）

PyTorch のデフォルトモードです。すべての演算が Python の実行に合わせて**即座に実行**されます。

```python
x = torch.randn(4, 4)
y = x + 2       # 今すぐ実行
z = y * y       # 今すぐ実行
```

- 計算グラフはフォワードパスの度に構築・破棄されます。
- Python の制御フロー（if/else、ループ、print、pdb）がそのまま動作します。
- プロトタイピングやデバッグに最適です。

### グラフモード（静的グラフ）

計算グラフが**事前にキャプチャ**され、全体として最適化・実行されます。

```python
@torch.compile
def f(x):
    y = x + 2
    z = y * y
    return z
```

- PyTorch が演算をトレースし、グラフを構築し、実行前に最適化を適用します。
- オプティマイザが全体像を把握：演算の融合、冗長性の除去、メモリの最適化。
- デバッグは難しくなりますが、大幅に高速化できます。

こう考えてください：
- **Eager モード** = コードを1行ずつ実行するインタプリタ。
- **グラフモード** = プログラム全体を読み込み、最適化してから実行するコンパイラ。

---

## 2. なぜグラフモードが重要なのか

Eager モードでは、各演算（加算、乗算、relu など）が**個別の GPU カーネル**を起動します。各カーネルの起動には以下が含まれます：

1. GPU メモリから入力テンソルを読み込む
2. 結果を計算する
3. 出力テンソルを GPU メモリに書き戻す

N 個のポイントワイズ演算のチェーンでは、**N 回のカーネル起動**と **N 回の GPU メモリ往復**が必要です。

```
Eager:  [読込 → 加算 → 書込] → [読込 → 乗算 → 書込] → [読込 → relu → 書込]
                ↑                       ↑                       ↑
           カーネル #1             カーネル #2             カーネル #3
```

グラフモードはこれらを単一のカーネルに**融合**できます：

```
コンパイル済: [読込 → 加算 → 乗算 → relu → 書込]
                         ↑
                    カーネル #1（融合済み）
```

これにより以下が削減されます：
- **カーネル起動オーバーヘッド**：CPU→GPU ディスパッチが減少
- **メモリ帯域幅**：中間結果がグローバルメモリではなく GPU レジスタに保持される

現代の GPU では、メモリ帯域幅（計算能力ではなく）がボトルネックになることが多いです。だからこそ融合が非常に重要なのです。

---

## 3. PyTorch のグラフモード API

PyTorch はグラフモード用に複数の API を提供しています：

| API | 導入時期 | ステータス | 用途 |
|-----|---------|-----------|------|
| `torch.compile` | PyTorch 2.0 (2023) | **推奨** | 学習・推論の高速化 |
| `torch.export` | PyTorch 2.1 (2023) | アクティブ | デプロイ、エッジ、モバイル |
| `torch.jit.trace` | PyTorch 1.0 (2018) | レガシー | Python なしでのデプロイ |
| `torch.jit.script` | PyTorch 1.0 (2018) | レガシー | Python なしでのデプロイ |

### torch.compile（TorchDynamo + TorchInductor）

現代的なアプローチ。TorchDynamo で Python バイトコードからグラフをキャプチャし、TorchInductor で最適化された GPU カーネルを生成します（Triton 経由）。

```python
model = MyModel()
compiled_model = torch.compile(model)  # これだけです
output = compiled_model(input)
```

### TorchScript（レガシー）

古いアプローチ。シリアライズ可能なグラフをキャプチャし、Python なしで実行できます。

```python
# トレース：サンプル入力から演算を記録
traced = torch.jit.trace(model, sample_input)

# スクリプト化：Python ソースコードを TorchScript IR に解析
scripted = torch.jit.script(model)

# デプロイ用に保存
traced.save("model.pt")
```

---

## 4. 実践例：シンプルから複雑へ

### 準備

```python
import torch
import torch.nn as nn
import time

def benchmark(fn, label, n=10000, warmup=100):
    """ベンチマーク関数。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed:.3f}s ({n} 回, {elapsed/n*1e6:.1f} us/回)")

device = "cuda"
```

---

### 例 1：初めての torch.compile

**目標**：シンプルなモデルで torch.compile を体験する。

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
    benchmark(lambda: compiled_model(x), "コンパイル済")
```

**予想結果**：ほぼ差なし（約 0-3%）。

**なぜ？** このモデルは 3 つの `nn.Linear` 層（行列乗算）だけです。行列乗算はすでに **cuBLAS**（NVIDIA が手動チューニングした GEMM ライブラリ）を呼び出しています。`torch.compile` は cuBLAS が最も得意とする処理を超えることはできません——最適化すべきものがないのです。

**教訓**：`torch.compile` は万能の「すべてを高速化する」ボタンではありません。特定の種類の演算を最適化します。

---

### 例 2：スケールアップ——モデルサイズは影響するか？

**目標**：より大きなモデルで結果が変わるか検証する。

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
    benchmark(lambda: compiled_model(x), "コンパイル済")
```

**予想結果**：依然として無視できる程度の差。

**なぜ？** 10 層でより大きな次元でも、計算は依然として行列乗算（cuBLAS）が支配的です。`nn.Linear` のサイズを大きくしても、cuBLAS により多くの仕事を与えるだけで、融合の機会は増えません。

**教訓**：演算の**種類**がモデルの**サイズ**より重要です。

---

### 例 3：本当の高速化——ポイントワイズ演算の融合

**目標**：融合の恩恵を受ける演算で劇的な高速化を観察する。

```python
def pointwise_heavy(x):
    for _ in range(20):
        x = x * torch.sigmoid(x)                       # SiLU/Swish
        x = x / (x.std(dim=-1, keepdim=True) + 1e-6)   # 手動正規化
        x = x + torch.tanh(x)
        x = x * 0.5 + x ** 2 * 0.01
    return x

x = torch.randn(256, 4096, device=device)

compiled_fn = torch.compile(pointwise_heavy)

with torch.no_grad():
    benchmark(lambda: pointwise_heavy(x), "Eager")
    benchmark(lambda: compiled_fn(x),     "コンパイル済")
```

**予想結果**：**4〜6 倍の高速化**（またはそれ以上）。

**なぜ？** Eager モードでは、各演算（`sigmoid`、`mul`、`std`、`div`、`tanh`、`add`、`pow`）が**個別の GPU カーネル**を起動します。各カーネルが GPU グローバルメモリから読み書きします。20 回のループ × 約 7 演算で、約 **140 回のカーネル起動**とメモリ往復が発生します。

`torch.compile` はチェーン全体を少数のカーネルに融合します。中間結果は低速なグローバルメモリに書き戻されず、高速な GPU レジスタに保持されます。

```
Eager（1イテレーションあたり）:
  sigmoid: x 読込 → 計算 → tmp1 書込
  mul:     x, tmp1 読込 → 計算 → tmp2 書込
  std:     tmp2 読込 → 計算 → tmp3 書込
  div:     tmp2, tmp3 読込 → 計算 → tmp4 書込
  tanh:    tmp4 読込 → 計算 → tmp5 書込
  add:     tmp4, tmp5 読込 → 計算 → tmp6 書込
  ...

コンパイル済（1イテレーションあたり）:
  融合:   x 読込 → sigmoid → mul → std → div → tanh → add → ... → 結果書込
```

**教訓**：`torch.compile` は**ポイントワイズ/要素単位の演算チェーン**がある場合に威力を発揮し、融合によりメモリ帯域幅の圧力を軽減します。

---

### 例 4：実際の Transformer ブロック

**目標**：行列乗算とポイントワイズ演算が混在する実世界のアーキテクチャでグラフモードがどう役立つか確認する。

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
        # 残差接続と正規化付き自己注意
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.dropout(attn_out)

        # 残差接続と正規化付きフィードフォワード
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
    benchmark(lambda: model(x),          "Eager",       n=500)
    benchmark(lambda: compiled_model(x), "コンパイル済", n=500)
```

**予想結果**：10〜30% の高速化（GPU により異なる）。

**なぜ？** Transformer ブロックは両方の種類の演算を含みます：
- **行列乗算中心**：Q/K/V 射影、アテンションスコア計算、FFN 層 → cuBLAS が処理、改善は最小限
- **ポイントワイズ中心**：LayerNorm、GELU、softmax、dropout、残差加算 → これらが融合される！

全体の高速化は加重平均です：ポイントワイズ部分は大幅に高速化しますが、行列乗算部分はそのままです。

**教訓**：実世界のモデルでは控えめだが有意義な高速化が得られます。行列乗算に対してポイントワイズ演算が多いほど、改善は大きくなります。

---

### 例 5：TorchScript によるデプロイ

**目標**：Python ランタイムなしの本番デプロイ用にモデルをエクスポートする。

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

# --- 方法 1：トレース ---
# サンプル入力でモデルを実行し演算を記録。
# 注意：入力に依存する制御フロー（if/else）はキャプチャされません。
traced_model = torch.jit.trace(model, sample_input)

# --- 方法 2：スクリプト化 ---
# Python ソースコードを TorchScript IR に解析。
# 制御フローをサポートしますが、Python の構文に制限があります。
scripted_model = torch.jit.script(model)

# デプロイ用に保存（C++、モバイルなどでロード可能）
traced_model.save("model_traced.pt")
scripted_model.save("model_scripted.pt")

# ロードして検証
loaded_model = torch.jit.load("model_traced.pt")
with torch.no_grad():
    original_out = model(sample_input)
    loaded_out = loaded_model(sample_input)
    print(f"出力一致: {torch.allclose(original_out, loaded_out, atol=1e-5)}")
    # 出力一致: True

# グラフの確認
print(traced_model.graph)
```

**トレース vs スクリプト化**：

| | `torch.jit.trace` | `torch.jit.script` |
|---|---|---|
| 仕組み | モデルを実行し演算を記録 | Python ソースを解析 |
| 制御フロー | キャプチャされない（展開される） | サポート（制限あり） |
| 動的シェイプ | トレース時の形状に固定 | サポート |
| 使いやすさ | 簡単 | コード変更が必要な場合あり |

**注意**：新規プロジェクトでは、デプロイには TorchScript より `torch.export` を推奨します：

```python
# モダンな代替手段（PyTorch 2.1+）
exported = torch.export.export(model, (sample_input,))
```

---

## 5. 使い分けガイド

| シナリオ | 推奨アプローチ |
|---------|--------------|
| プロトタイピングとデバッグ | Eager モード（デフォルト） |
| 学習の高速化 | `torch.compile(model)` |
| 推論の高速化 | `torch.compile(model, mode="reduce-overhead")` |
| Python なしでのデプロイ | `torch.export`（または TorchScript） |
| モバイル/エッジデプロイ | `torch.export` → ExecuTorch |
| 最大限の推論最適化 | `torch.compile` + 量子化 |

### torch.compile モード

```python
# デフォルト：コンパイル時間と高速化の良いバランス
model = torch.compile(model)

# オーバーヘッド削減：CPU オーバーヘッドを最小化、小バッチに最適
model = torch.compile(model, mode="reduce-overhead")

# 最大自動チューニング：多くのカーネル構成を試行、コンパイルは遅いが実行は速い
model = torch.compile(model, mode="max-autotune")
```

---

## 6. よくある落とし穴

### 1. 行列乗算中心のモデルでの高速化を期待する

```python
# torch.compile では速くなりません：
def matmul_only(x, w1, w2, w3):
    x = x @ w1
    x = x @ w2
    x = x @ w3
    return x
```

cuBLAS はすでに行列乗算を最適に処理しています。`torch.compile` は**その他の**演算を融合することで価値を発揮します。

### 2. グラフブレーク

`torch.compile` がトレースできないコード（データ依存の制御フロー、サポートされていない Python 機能など）に遭遇すると、**グラフブレーク**を挿入します——グラフを小さな部分に分割し、トレースできない部分は Eager モードにフォールバックします。

```python
@torch.compile
def f(x):
    x = x * 2
    print(x.shape)  # グラフブレーク！print は Python の副作用
    x = x + 1
    return x
```

`torch._dynamo.explain(f, x)` でグラフブレークを診断できます。

### 3. 初回呼び出しのオーバーヘッド

`torch.compile` は**初回呼び出し時**にコンパイルします。数秒から数分かかることがあります。ベンチマーク前に必ずウォームアップしてください：

```python
compiled_model = torch.compile(model)

# 悪い例：コンパイル時間が含まれる
start = time.time()
compiled_model(x)  # 初回呼び出し：コンパイルして実行
print(time.time() - start)  # 誤解を招く結果

# 良い例：先にウォームアップ
for _ in range(3):
    compiled_model(x)  # ここでコンパイルが行われる
torch.cuda.synchronize()

start = time.time()
for _ in range(1000):
    compiled_model(x)  # 純粋な実行時間
torch.cuda.synchronize()
print(time.time() - start)
```

### 4. 動的シェイプによる再コンパイル

呼び出し間で入力シェイプが変わると、`torch.compile` は毎回グラフを再コンパイルする可能性があります。`dynamic=True` で可変シェイプに対応してください：

```python
compiled_model = torch.compile(model, dynamic=True)
```

---

## 7. まとめ

### コアアイデア

PyTorch グラフモードは計算を**静的グラフ**としてキャプチャし、実行前に最適化します。主な最適化は**カーネル融合**——複数の演算を1つの GPU カーネルに結合し、メモリ帯域幅の使用を削減することです。

### 何が高速化されるか

| 演算タイプ | Eager | コンパイル済 | 理由 |
|-----------|-------|------------|------|
| 行列乗算（`nn.Linear`） | 高速 | 同じ | すでに cuBLAS を使用 |
| ポイントワイズ演算チェーン（正規化、活性化など） | 低速（多数のカーネル） | **高速（融合）** | カーネル起動減少、メモリ I/O 削減 |
| 混合（実モデル） | ベースライン | **10〜30% 高速** | ポイントワイズ部分が融合 |
| 純ポイントワイズワークロード | ベースライン | **4〜6 倍高速** | すべてが融合 |

### 進化の過程

```
TorchScript (2018)  →  torch.compile (2023)  →  torch.export (2023+)
  （レガシー）           （学習高速化）            （デプロイ）
```

### 一言まとめ

> `torch.compile` は個々の演算を速くするのではなく、**演算のチェーン**をより少ない GPU カーネル起動に融合することで高速化し、メモリ帯域幅のオーバーヘッドを削減します。
