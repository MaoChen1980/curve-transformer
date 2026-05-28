# Curve Transformer — 语言是曲线

> **"Language is a curve, not a sequence."**

[→ English Version](#english-version)

---

<a id="chinese-version"></a>

## 中文版

### 想法起源

大多数语言模型把文本当作序列处理：给定前 N 个 token，预测第 N+1 个。我们问了一个不同的问题：

**如果语义是一个连续空间里的点，这个空间本身有几何结构呢？**

这个想法的起点来自一个类比——**贝塞尔曲线**：

- 贝塞尔曲线用一组控制点 `P_i` 定义一条平滑路径，每个控制点通过伯恩斯坦基函数 `B_i^n(t)` 影响整条曲线
- 文本生成可以看作在高维语义空间里"画一条曲线"
- **参数 = 控制点**，模型参数决定了曲线走向
- **encode = 从 prompt 算出控制点位置**
- **decode = 用控制点重绘曲线**
- **attention 权重 = 伯恩斯坦权重**，每个输入 token 对当前输出的影响

从神经网络的角度看，这只是一个拟合函数 — 没有"基函数"的强假设，只有连续映射。贝塞尔提供了理解"为什么这样能 work"的直觉，而不是架构约束。

从这个视角出发，我们推导出了几个关键判断：

1. **有限注意力就够了** — 伯恩斯坦权重在局部最大，远处趋近于零。曲线天生局部平滑，不需要全局长注意力。滑动窗口在数学上就是够的。

2. **分段并发训练可行** — 高阶贝塞尔不稳定（龙格现象），工程解法是分段。对应到训练：模型太大不好并发，那就切段独立训练，边界对齐。精度损失在曲线拟合框架下是可接受的。

3. **大纲模式** — 画曲线时不看全局，需要时切到控制多边形看趋势。对应到生成：大部分时候用局部注意力，偶尔切到"大纲模式"看看全局结构。

这三个判断驱动了后续所有的实验。

---

### 实验过程

#### v1：LSTM 基线

第一个实现用 LSTM + 正弦位置编码：

```
Embedding → PE → LSTM → FC → vocab
```

结果：能跑。但生成重复严重——小数据集 + 简单 RNN 的经典问题。

还撞上了 **E_in / E_out 维度不匹配问题**：embedding 维度和 hidden 维度不同，不断引发 shape 错误。

**教训**：统一维度设计是必要的。

---

#### v2：选择性状态 + 并行训练

从 v1 的教训出发，做了两个优化：

**思路 1 — 并行化**：增大 batch_size（16→64），多句话同时训练。精度有损失但换来了训练速度。

**思路 2 — 有限上下文**：LSTM → SelectiveState，受 Mamba（状态空间模型）启发：

```python
# LSTM: h[t] = f(W·h[t-1], U·x[t]) — 无差别记住一切
# SelectiveState:
gate  = sigmoid(W_g·cat(x, h))      # 保留多少旧状态
cand  = tanh(W_c·cat(x, h))          # 新信息候选
h_new = gate * h + (1 - gate) * cand # 选择性融合
```

**核心改变**：模型自己决定记住什么、忘掉什么。相当于一个有限容量工作记忆，解决了梯度衰减和局部重复问题。

架构：
```
Embedding(vocab, E=64) → PE → SelectiveState × 2 → FC → vocab
```

结果：记忆改善了。生成在边界案例上仍有重复。根本问题没变：模型没有学到结构化语义，它只是记忆了。

**教训**：并行训练是对的，但选择性状态的表达力有限。需要更强的上下文机制。

---

#### v3：Transformer（转折）

这时候做了一个关键转向。既然：

1. **并行训练是自然的** — PyTorch 天然并行 batch 维度，不需要手动分块
2. **Attention 处理上下文** — 不需要在训练时手动管理隐藏状态，直接对整序列跑 attention
3. **统一维度** — 去掉 E_in/E_out 分离，全程 D=256

架构变成：

```
Embedding(vocab, E=64) + Sinusoidal PE → Linear(E→D=256)
  → Block × 2
  │   ├── PreNorm + MHA (Q/K/V/O 全部 D→D)
  │   ├── PreNorm + FFN (GELU)
  │   └── 残差连接
  → LayerNorm → FC → vocab
```

每个 Block 的 `step(x, h)` 方法用于自回归生成：

```python
gate = sigmoid(q)                    # 保留多少旧状态？
h_new = h * gate + v * (1 - gate)    # 门控状态更新
x = x + o_proj(h_new)                # 残差
x = x + ffn(norm(x))
return x, h_new
```

**关键区别**：训练时用标准双向 MHA，每个位置看到完整句子；推理时扔掉 attention，只用 `step()` 的门控状态自回归。

这对应了贝塞尔的思想：训练时"画完整条曲线确定控制点"，推理时"只看控制点推下一段"。

在 3000 条模板生成的句子上训练后：

| Step | Loss |
|------|------|
| 0 | 5.95 |
| 300 | 0.187 |
| 1000 | 0.013 |
| 5000 | 0.000 |
| 8000 | 0.000 |

loss 归零 — 模型把每条训练句子都背下来了。

**但语义插值实验有效：**

```
Traverse: '今天' → '未来'

α=0.0  '天'  conf=0.999
α=0.5  '天'  conf=0.827
α=0.6  '未'  conf=0.602
α=1.0  '未'  conf=0.947
```

从"天"到"未"的字符过渡是平滑的。即使生成完全崩溃了，latent space 的几何结构仍然编码了语义关系。

**生成崩溃**（因为过拟合）：
```
'今天天气' → '今天天气年年年年年年年年年年年年年年年'
'我爱'     → '我爱爱爱爱爱爱爱爱爱爱爱爱爱爱爱爱'
```

但这是数据量问题，不是架构问题。关键发现：**表示学习和生成是两码事**。latent space 学到了真实的语义几何，但生成头只是记忆了。

---

#### v3 延续：真实语料

当前阶段：v3 架构不变，但用 GPT-2 生成的真实中文句子代替模板语料。`gen_corpus.py` 从 HuggingFace 加载 `uer/gpt2-distil-chinese-cluecorpussmall`，用多样化 prompt 生成句子，目标是让模型从记忆走向泛化。

优化：
- 设置 `HF_ENDPOINT=https://hf-mirror.com` 保证国内下载稳定
- 限制生成量：20 prompts × 2 temps × 2 top_p × 2 repeats，最多 1000 句

---

### 理论分析：与标准 Transformer 的区别

你的模型和标准小 Transformer 的核心区别：

| 维度 | 标准 Transformer | Curve Transformer |
|------|-----------------|-------------------|
| 训练注意力 | Causal mask（只看过去） | 双向全量（看完整句子） |
| 生成注意力 | 每步关注所有历史 KV | 不用 attention，只用状态 h |
| 推理复杂度 | O(L) — 随序列长度增长 | O(1) — 固定 |
| 状态更新 | 无显式状态 | `h_new = h * gate + v * (1-gate)` |
| 训练/推理一致性 | 相同 | 不同路径共享权重 |

#### 理论优势

**1. 训练信号更强**

双向 attention 让每个 token 直接看到完整句子梯度，没有 causal mask 的信息遮蔽。相当于画曲线时能看到整条线再落笔。梯度信号更多、更直接。

**2. 推理 O(1) 代替 O(L)**

不需要 KV cache，没有 context window 上限。推理成本不随序列长度增长。Scale 上去就是质的区别。

**3. 训练/推理解耦**

双向 attention 负责"学"（把曲线形状压缩到状态 h），门控状态负责"推"（只用状态生成下一段）。计算分配和执行分配是分开的 — 和贝塞尔先确定控制点再画曲线的逻辑一致。

**本质上在赌一件事**：门控状态 h 可以在双向 attention 的监督下，学到足够好的动力学来近似 attention 的输出。

#### 理论缺陷

**1. 训练/推理分布偏移 — 核心问题**

损失函数只优化 `forward()`（双向 attention），但推理跑的是 `step()`（只有状态 h）。这两个路径共享权重，但 `q_proj` 在 forward 里学做 attention query，在 step 里却被用来算 gate。**同一个权重被优化去做两件不同的事。**

`step()` 从未被显式走过 loss，自回归推理越往后误差积累越严重。

**2. 信息瓶颈**

D=256 的 h 要压缩整个句子的信息。短句可行（~50 token），但上下文越长，压缩损失越大。标准 Transformer 的 KV cache 是 O(L)，你的是 O(1) — 既是优势也是约束。

**3. 门控只看当前 token**

```python
gate = sigmoid(q)  # q 只依赖当前 token 的 embedding
```

"保留多少旧状态"这个决策只取决于当前 token 本身，不取决于已经积累的上下文 h。模型无法根据"我已经记了什么"来调整吸收新信息的力度。这在语义上挺奇怪的 — 同一个字在不同语境下对状态的更新应该不同。

**4. 双向 attention 的"作弊"问题**

训练时预测下一个 token，但 attention 能看到未来的 token。等于考你"下一句是什么"但先把答案给你看了。attention 层学到了很强的未来信息模式，但这些模式在推理时用 `step()` 根本复现不出来。低 loss 不代表 `step()` 的推理能力。

#### 理论优势/缺陷总结

| 维度 | 优势 | 代价 |
|------|------|------|
| 推理速度 | O(1)，和序列长度无关 | h 的容量限制上下文长度 |
| 训练信号 | 双向，梯度更强 | 训练学的能力推理时用不上 |
| 状态更新 | 简洁优雅 | 门控只看当前 token，表达力有限 |

#### 从实用角度看

以上是理论分析。从实用出发，这个思路有几个更直接的观察：

**1. 如果状态 h 足够，KV cache 就是拐杖**

标准 transformer 需要 KV cache 是因为架构没有显式记忆，每一步都回头看所有历史。如果固定大小的状态 h 能编码生成所需的上下文，那么 O(L) 的 KV cache 只是一个工程 hack，不是架构进步。

**这个方向的隐含结论很炸裂**：整个 LLM 推理基础设施都在围绕 KV cache 做优化 — PagedAttention、KV cache 量化、cache 调度、显存管理…… 如果状态模型真的 work，这些东西全部不需要了。推理成本从 O(L²) 降到 O(L)，而且没有上下文窗口硬上限。这指向一个更深层的问题：**你真的需要记住所有过去才能生成下一个 token 吗？**

**2. 大部分场景的瓶颈是局部一致，不是长上下文**

- 对话 — 前后句连贯比记住 100 轮前的细节重要
- 代码生成 — 函数内的逻辑一致比跨文件引用重要
- 翻译 — 几句话内的术语统一比全文记忆重要
- 写作辅助 — 段落流畅比全书结构重要

长上下文是锦上添花，局部稳定是雪中送炭。**如果局部稳定能做到可靠，就已经覆盖了绝大多数实际需求。**

**3. 大纲 + 章节 + 句子 — 认知的自然粒度**

金庸写《天龙八部》也是先定章回大纲，再一章一章写。人类写作本就是分粒度的：
- **大纲模式**：偶尔看全局，确认方向
- **章节模式**：专注当前块，局部注意力
- **句子模式**：逐句生成，依赖前几句的语境

模型架构不需要每一层都做全局长注意力。它只需要在局部稳定生成，同时保留切换到大纲模式的能力。这和"局部为主，全局偶尔"的直觉完全一致。

---

### 项目文件

```
myllm/
├── main.py          # v1: LSTM 基线 — E_in/E_out 不匹配
├── main_v2.py       # v2: SelectiveState + 并行训练
├── main_v3.py       # v3: Transformer, 统一维度 (当前)
├── gen_corpus.py    # 用 GPT-2 生成训练语料
├── model_v3.pt      # v3 权重 (~4.7MB)
├── checkpoint_v3.pt # v3 检查点 (可恢复训练)
├── real_corpus.txt  # GPT-2 生成的真实语料
├── compare_v1_v2.py # v1 vs v2 对比
├── gen_v2*.py       # v2 生成脚本
├── diagnose_v2.py   # v2 诊断输出
└── check_model.py   # 架构验证
```

### 经验总结

1. **统一维度简化了一切** — E_in/E_out 分离在 v1 中造成持续 shape 错误，v3 统一 D=256 后消失
2. **注意力是强大的学习信号** — 双向 attention 让训练收敛更快，但学到的能力在推理时不一定能复现
3. **Latent space 几何是真实存在的** — 即使过拟合到 loss=0，语义插值仍然有效。表示学习和生成可以分离
4. **更多数据 > 更复杂架构** — 当前最大瓶颈是语料量（3000 句），不是模型设计
5. **贝塞尔是直觉工具，不是架构约束** — 曲线类比帮助推导了有限注意力和分段并发，但最终的模型就是标准的函数拟合

---

### 训练

```bash
# v3 (当前)
python main_v3.py

# 生成训练语料
cd E:\claude\myllm && py gen_corpus.py
```

- **优化器**: AdamW, lr=2e-3, weight_decay=0.01
- **损失函数**: CrossEntropyLoss, ignore PAD (index 0)
- **检查点**: 每 500 步 → `checkpoint_v3.pt`
- **训练时间**: ~5 分钟 CPU 跑 8000 steps

### 引用

```bibtex
@software{curve_transformer_2026,
  title={Curve Transformer: Language as a Curve — Bezier-Inspired Semantic Interpolation},
  author={Mao Chen},
  year={2026},
  url={https://github.com/MaoChen1980/curve-transformer}
}
```

---

<a id="english-version"></a>

## English Version

### Origin of the Idea

Most language models treat text as a sequence: given N tokens, predict token N+1. We asked a different question:

**What if meaning is a point in a continuous space — and the space itself has geometry?**

This came from an analogy with **Bezier curves**:

- A Bezier curve is defined by control points `P_i`, each influencing the curve through Bernstein basis functions `B_i^n(t)`
- Text generation = "drawing a curve" in a high-dimensional semantic space
- **Parameters = control points** — they determine where the curve goes
- **Encode = compute control points from a prompt**
- **Decode = reconstruct the curve from control points**
- **Attention weights = Bernstein weights** — how each input token shapes the output

But ultimately, a neural network is just a function approximator. The Bezier analogy provides intuition — why locality matters, why segmentation works — not architectural constraints. The model is a fitting function, nothing more.

From this perspective, we derived three principles:

1. **Local attention is sufficient** — Bernstein weights peak locally and decay to zero at a distance. Curves are locally smooth. Full global attention is unnecessary.

2. **Segmented parallel training is feasible** — High-order Bezier curves are unstable (Runge's phenomenon). The fix is segmentation. For training: split the model, train segments independently, align boundaries. Precision loss is acceptable in a curve-fitting framework.

3. **Outline mode** — Use local attention for generation; switch to a global view only when needed. Same as checking the control polygon instead of the rendered curve.

---

### The Journey

#### v1: LSTM Baseline

```
Embedding → PE → LSTM → FC → vocab
```

It worked, but generation was repetitive — the classic small-dataset RNN problem.

We also hit **E_in / E_out dimension mismatch**: embedding dim ≠ hidden dim caused constant shape errors.

**Lesson**: Unified dimensions are necessary.

---

#### v2: Selective State + Parallel Training

Two optimizations:

**Parallelization**: batch_size 16 → 64. Acceptable precision loss for speed.

**Bounded context**: LSTM → SelectiveState (inspired by Mamba/SSM):

```python
gate  = sigmoid(W_g·cat(x, h))      # how much old state to keep
cand  = tanh(W_c·cat(x, h))          # new information candidate
h_new = gate * h + (1 - gate) * cand # selective fusion
```

The model decides what to remember and forget — a bounded working memory that mitigates gradient decay and repetition.

**Lesson**: Parallel training was right, but selective states lack expressiveness. We needed a stronger context mechanism.

---

#### v3: Transformer — The Pivot

The key realization: PyTorch natively parallelizes over batches, attention naturally handles context, and unified dimensions eliminate shape errors.

Architecture:
```
Embedding(vocab, E=64) + Sinusoidal PE → Linear(E→D=256)
  → Block × 2 (MHA + FFN) → LayerNorm → FC → vocab
```

The `step(x, h)` method for generation:
```python
gate = sigmoid(q)
h_new = h * gate + v * (1 - gate)
x = x + o_proj(h_new)
x = x + ffn(norm(x))
return x, h_new
```

**Key difference**: Training uses full bidirectional MHA (every position sees the whole sentence); inference discards attention entirely, using only the gated state `h`.

Training on 3000 template-generated sentences:

| Step | Loss |
|------|------|
| 0 | 5.95 |
| 300 | 0.187 |
| 1000 | 0.013 |
| 5000 | 0.000 |
| 8000 | 0.000 |

Loss hit zero — the model memorized every sentence.

**But semantic interpolation still worked:**
```
Traverse: '今天' → '未来'
α=0.0  '天'  conf=0.999  →  α=1.0  '未'  conf=0.947
```

A smooth transition from "today/present" to "future." The latent space geometry encodes semantic relationships even under complete overfitting.

Generation collapsed (from overfitting):
```
'今天天气' → '今天天气年年年年年年年年年年年年年年年'
```

**Key finding**: Representation and generation are separable. The latent space learned real geometry; the generation head just memorized.

---

#### Current Phase: Real Corpus

Same v3 architecture, but template data has been replaced with real Chinese sentences from GPT-2 (`uer/gpt2-distil-chinese-cluecorpussmall`). The goal: push the model from memorization toward generalization.

---

### How This Differs From a Standard Transformer

| Aspect | Standard Transformer | Curve Transformer |
|--------|--------------------|-------------------|
| Training attention | Causal mask (past only) | Full bidirectional |
| Generation attention | Full KV history | State-only (no attention) |
| Inference complexity | O(L) | O(1) |
| State mechanism | None (KV cache is passive) | Active gated state |
| Train/inference parity | Identical | Different paths, shared weights |

#### Theoretical Advantages

**1. Stronger training signal** — Bidirectional attention gives every position access to the full gradient. No information masking.

**2. O(1) inference** — No KV cache, no context window limit. Constant cost regardless of sequence length.

**3. Decoupled compute** — Attention learns (compresses curve shape into state h); the gated state executes (extends the curve using state only). This separates learning from generation.

#### Theoretical Weaknesses

**1. Train/inference distribution shift — the core problem**

The loss only optimizes `forward()` (bidirectional attention), but inference runs `step()` (state only). `q_proj` learns to be an attention query in `forward()` but serves as a gate in `step()`. **The same weight is optimized for two different purposes.**

Since `step()` never sees a gradient, autoregressive errors accumulate over time.

**2. Information bottleneck**

A D=256 state `h` must compress the entire sentence. KV cache is O(L); your state is O(1) — both a strength and a constraint.

**3. Gating ignores accumulated context**

```python
gate = sigmoid(q)  # q depends only on the current token
```

"How much to keep from the old state" depends only on the current token, not on the accumulated context `h`.

**4. Bidirectional attention "cheats"**

Training predicts the next token while attention sees the future. The attention layer learns strong patterns that `step()` can never reproduce at inference.

#### Summary

| Dimension | Advantage | Cost |
|-----------|-----------|------|
| Inference speed | O(1), independent of sequence length | h's capacity limits context |
| Training signal | Bidirectional, stronger gradients | Learned patterns may not transfer |
| State update | Simple, elegant | Gate ignores context, limited expressiveness |

#### A Practical Perspective

Beyond the theory, the architecture suggests several practical observations:

**1. If state h is sufficient, KV cache is a crutch**

Standard transformers need KV cache because the architecture has no explicit memory — every step must look back at the full history. If a fixed-size state h can encode the context needed for generation, then O(L) KV cache is an engineering hack, not an architectural advance.

**The implication is explosive**: the entire LLM inference infrastructure is built around KV cache optimization — PagedAttention, KV cache quantization, cache scheduling, memory management. If state-based models actually work, none of it is needed. Inference cost drops from O(L²) to O(L), with no hard context window limit. This raises a deeper question: **do you really need to remember everything to generate the next token?**

**2. Most real-world bottlenecks are local coherence, not long context**

- Dialogue — sentence-to-sentence flow matters more than 100-turn-old details
- Code generation — function-level consistency matters more than cross-file references
- Translation — term uniformity across a few sentences matters more than full-document memory
- Writing — paragraph fluency matters more than book-length structure

Long context is a nice-to-have; local stability is a must-have. **If local stability can be made reliable, it already covers the vast majority of practical needs.**

**3. Outline + chapter + sentence — cognitive granularity**

Jin Yong didn't write "The Demi-Gods and Semi-Devils" in one pass; he outlined chapters, then wrote one chapter at a time. Human writing is inherently hierarchical:

- **Outline mode**: occasional global view to verify direction
- **Chapter mode**: focused attention on the current block
- **Sentence mode**: step-by-step generation, depending on recent context

A model architecture doesn't need full global attention at every layer. It needs stable local generation with the ability to occasionally zoom out. This aligns with the intuition of "local first, global when needed."

---

### Project Files

```
myllm/
├── main.py          # v1: LSTM baseline — E_in/E_out mismatch
├── main_v2.py       # v2: SelectiveState + parallel training
├── main_v3.py       # v3: Transformer, unified D (current)
├── gen_corpus.py    # GPT-2 corpus generation
├── model_v3.pt      # v3 weights (~4.7MB)
├── checkpoint_v3.pt # v3 checkpoint (resumable)
├── real_corpus.txt  # GPT-2 generated corpus
├── compare_v1_v2.py # v1 vs v2 comparison
├── gen_v2*.py       # v2 generation scripts
├── diagnose_v2.py   # v2 diagnostics
└── check_model.py   # architecture verification
```

### Key Takeaways

1. **Unified dimensions simplify everything** — E_in/E_out separation caused constant errors in v1; unified D=256 fixed them
2. **Attention is a powerful learning signal** — bidirectional attention converges fast, but learned patterns may not transfer to state-only inference
3. **Latent space geometry is real** — even at loss=0, semantic interpolation works. Representation and generation are separable
4. **More data > better architecture** — the bottleneck is corpus size (3000 sentences), not model design
5. **Bezier is an intuition tool, not an architectural constraint** — the analogy helped derive local attention and segmented training, but the final model is just function fitting

### Training

```bash
# v3 (current)
python main_v3.py

# Generate training corpus
cd E:\claude\myllm && py gen_corpus.py
```

- **Optimizer**: AdamW, lr=2e-3, weight_decay=0.01
- **Loss**: CrossEntropyLoss, ignore PAD (index 0)
- **Checkpoints**: Every 500 steps → `checkpoint_v3.pt`
- **Training time**: ~5 min CPU for 8000 steps

### Citation

```bibtex
@software{curve_transformer_2026,
  title={Curve Transformer: Language as a Curve — Bezier-Inspired Semantic Interpolation},
  author={Mao Chen},
  year={2026},
  url={https://github.com/MaoChen1980/curve-transformer}
}
```

---

## License

MIT — this is a research prototype, not a production model.
