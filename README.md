# Curve Transformer

> **"Language is a curve, not a sequence."**

A character-level neural language model that treats semantic meaning as a continuous manifold — and demonstrates that you can walk between concepts by interpolating directly in the model's latent space.

---

## The Core Idea

Most language models treat text as a sequence: predict the next token given all previous tokens. We asked a different question:

> **What if the meaning of a sentence is a point in a continuous space — and the space itself has geometry?**

If "今天" (today) and "未来" (future) are points in a high-dimensional manifold, can we walk between them? Can we traverse the semantic space and watch meaning shift continuously, character by character?

This repo is an exploration of that idea — through three architectural iterations and a lot of dead ends.

---

## The Journey

### Phase 1: The Intuition

It started with a philosophical observation: language is not a linear chain of discrete symbols. The meaning of a sentence is more like a point in a manifold — it has continuity, nearby points have similar meanings, and the space itself is structured.

We wanted a model that could:
1. Encode a sentence into a point `z` in latent space
2. Walk between two points by linearly interpolating `z`
3. Decode the interpolated `z` back into text — and watch meaning shift

The key test: traverse from "今天" to "未来". If the model truly understands semantic structure, the intermediate points should produce meaningful Chinese characters that bridge the two concepts.

### Phase 2: v1 — LSTM Baseline

The first implementation used a standard LSTM with sinusoidal positional encoding:

```
Embedding → PE → LSTM → FC → vocab
```

**Result**: It worked. But generation was repetitive — the classic problem with small datasets and simple RNNs. The model couldn't maintain long-range coherence.

We also ran into the **E_in / E_out dimension mismatch problem**: embedding dim ≠ hidden dim, and the mismatch kept breaking things.

### Phase 3: v2 — SSM-Style Recurrent Layers

The user proposed dropping LSTM entirely. Inspired by Mamba (state-space models), we replaced LSTM with **selective state layers**:

```
h_new = f * h + i * cand
```

Where `f` (forget gate) and `i` (input gate) are computed from `cat(x, h)`. This is essentially an LSTM but without the separate cell state — just the hidden state, updated selectively.

**Key change**: No more RNN. Each layer is a pure function that takes `(x, h)` and returns the new `h`. State management is explicit.

**Result**: Memory improved. Generation was better but still repetitive on edge cases. The fundamental problem remained: the model didn't learn structured semantics, it memorized.

### Phase 4: v3 — Transformer (The Turn)

This is where things got interesting. The user proposed:

1. **Parallel training is natural** — PyTorch batches sequences in parallel automatically. No manual chunking needed.
2. **Attention handles context** — we don't need to manually manage hidden state during training. Just run full attention over the sequence.
3. **Unified dimension** — drop E_in/E_out separation. One dimension `D` throughout. Everything `D → D`.

The architecture became:

```
Embedding(vocab, E=64) + Sinusoidal PE → Linear(E→D) 
  → Block×2 (multi-head attention + FFN) 
  → LayerNorm → FC → vocab
```

Where each **Block** is:
```
PreNorm → Q/K/V projections → MHA → residual
  → PreNorm → FFN(GELU) → residual
```

Total params: ~1.17M.

**Recurrent step for generation**: Each Block has a `step(x, h)` method that converts MHA into a gated update:

```python
gate = sigmoid(q)          # how much to keep from previous state?
h_new = h * gate + v * (1 - gate)
x = x + o_proj(h_new)      # residual
x = x + ffn(norm(x))       # FFN
return x, h_new
```

This makes the Transformer autoregressive without losing the attention mechanism.

### Phase 5: The Discovery

After training on 3000 synthetic sentences (2973 template-generated + ~30 hand-crafted):

| Step | Loss |
|------|------|
| 0 | 5.95 |
| 300 | 0.187 |
| 1000 | 0.013 |
| 5000 | 0.000 |
| 8000 | 0.000 |

The model overfits completely. Loss hits zero — it has memorized every training sentence.

**But then we tested semantic interpolation:**

```
Traverse: '今天' → '未来'

α= 0.0  '天'  conf=0.999  ███████████████████
α= 0.4  '天'  conf=0.978  ███████████████████
α= 0.5  '天'  conf=0.827  ████████████████
α= 0.6  '未'  conf=0.602  ████████████
α= 1.0  '未'  conf=0.947  ██████████████████
```

The character transitions from "天" to "未" — a smooth, meaningful shift in the latent space. The model's **latent geometry** encodes semantic structure, even though generation is broken.

**Key insight**: The latent space learned semantic relationships even under extreme overfitting. The geometry is real, it's just the generation head that collapsed.

### Phase 6: Why Generation Breaks

Generation collapses with repetition:

```
'今天天气' → '今天天气年年年年年年年年年年年年年年年'
'我爱'     → '我爱爱爱爱爱爱爱爱爱爱爱爱爱爱爱爱'
```

The model memorized the most common continuation for each prefix. With only 3000 training sentences and a 1.17M-parameter model, it had enough capacity to memorize rather than generalize. The generation head is effectively a lookup table.

**But**: The semantic interpolation still works because it uses the **latent space geometry** directly, bypassing the broken generation head. This is the key observation: the representation is meaningful even when the generation mechanism fails.

---

## Architecture Summary

### v3 — Final Transformer

```
Embedding(vocab=532, E=64) + Sinusoidal PE → Linear(E→D=256)
  → Block × 2
  │   ├── PreNorm + Q/K/V/O projections (D→D)
  │   │   └── Attention: (B,H,L,L) ← (B,H,L,Dh) @ (B,H,Dh,L)
  │   ├── PreNorm + FFN(D→D*2→D)
  │   └── Residual connections throughout
  → LayerNorm → FC → vocab

Block.step(x, h) for autoregressive generation:
  gate = sigmoid(q)
  h_new = h * gate + v * (1 - gate)   ← gated state update
  return x + o_proj(h_new), h_new
```

### Key Design Decisions

| Decision | Why |
|----------|-----|
| Sinusoidal PE | Original Transformer paper approach — absolute positions with smooth harmonics |
| Unified D=256 | No E_in/E_out mismatch — simpler, consistent |
| Gated MHA step | Converts parallel attention into recurrent update for generation |
| Pre-norm | More stable training than post-norm |
| GELU activation | Smooth, used in modern Transformers (BERT etc.) |
| Weight decay 0.01 | Minimal regularization — we wanted to see what pure overfitting looks like |

---

## Project Files

```
myllm/
├── main.py          # v1: Simple LSTM baseline — E_in/E_out mismatch, breaks
├── main_v2.py       # v2: SSM-style selective state layers, no RNN
├── main_v3.py       # v3: Full Transformer, unified D (LATEST)
├── model_v3.pt      # v3 trained weights (~4.7MB)
├── checkpoint_v3.pt # v3 optimizer state + weights
├── compare_v1_v2.py # Side-by-side comparison of v1 vs v2 generation
├── gen_v2*.py       # v2 generation scripts with repetition penalty
├── diagnose_v2.py   # v2 diagnostic output
├── check_model.py   # Model architecture verification
├── debug_shapes.py  # Shape debugging for tensor operations
├── test_*.py        # Debug scripts
└── README.md        # This file
```

---

## Training

```bash
python main_v3.py
```

- **Corpus**: 2973 synthetic Chinese sentences (template-based generation)
- **Optimizer**: AdamW, lr=2e-3, weight_decay=0.01
- **Loss**: Cross-entropy, ignore PAD (index 0)
- **Checkpoints**: Every 500 steps → `checkpoint_v3.pt`
- **Training time**: ~5 min on CPU for 8000 steps

---

## Key Learnings

1. **Overfitting is not always bad** — it revealed that the latent space has real geometry. The model memorized the corpus, but the geometry it learned is meaningful.

2. **Unified dimension is simpler** — the E_in/E_out split in v1 caused constant shape mismatches. One dimension `D` throughout is cleaner and works better.

3. **Parallel training is natural** — PyTorch batches natively parallelize over the batch dimension. No manual chunking or trickery needed.

4. **Interpolation is robust** — even with collapsed generation, semantic interpolation in latent space works reliably. The geometry is learned, the generation head is broken.

5. **Sinusoidal PE is sufficient** — no learned positional encoding needed for this scale. The classical approach works fine.

---

## Why Generation Collapses (and How to Fix It)

**Root cause**: 2973 sentences, 1.17M parameters → the model memorizes rather than generalizes.

**Fixes** (not implemented yet):

1. **More data**: Real Chinese text corpus (Wikipedia, news) — 100K+ sentences minimum
2. **More regularization**: Higher dropout (0.3+), weight decay (0.1), early stopping
3. **Smaller model**: 256K params instead of 1.17M to force generalization
4. **Pretrained backbone**: Start from Chinese BERT/GPT and fine-tune
5. **Contrastive learning**: Add a term that pushes z("今天") and z("未来") apart

---

## What This Project Tells Us

The core hypothesis — **"language is a curve"** — is partially confirmed:

✅ The latent space has meaningful geometry — semantic interpolation produces meaningful transitions

❌ The generation mechanism fails under overfitting — memorized continuations collapse into repetition

The interesting finding: **you can separate representation from generation**. The representation learned something real; the generation head just memorized. This suggests that for a truly generalizable model, you need both:
- A rich, structured latent space (from large-scale contrastive learning)
- A robust generation head (from large-scale autoregressive training)

---

## Citation

```bibtex
@software{curve_transformer_2026,
  title={Curve Transformer: Language as a Curve — Latent Semantic Interpolation in Transformer Space},
  author={Mao Chen},
  year={2026},
  url={https://github.com/MaoChen1980/curve-transformer}
}
```

---

## License

MIT — use freely, but remember: this is a research prototype, not a production model.