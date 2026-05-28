# Curve Transformer

A character-level Transformer for Chinese text generation with sinusoidal positional encoding and semantic interpolation along the model's latent curve.

> "Language is a curve, not a sequence."

## Status

This repo captures the evolution of the Curve Model project — from simple LSTM (v1) → SSM-style RNN (v2) → full Transformer (v3).

**Latest: v3** trains successfully and demonstrates meaningful semantic interpolation, but generation collapses due to overfitting on small synthetic data. See [Results](#results) for details.

## Architecture (v3 — Transformer)

```
Embedding (vocab, 64) → Sinusoidal PE → Linear(64→256)
  → Block × 2 (multi-head attention + FFN)
  → LayerNorm → Linear(256→vocab)
```

- **Embed + PE**: Sinusoidal positional encoding (raw tensor, no nn.Module)
- **Block**: Pre-norm MHA (4 heads, 64 dim/head) → AddNorm → FFN(GELU) → AddNorm
- **Recurrent step**: Gated MHA state update for autoregressive generation
- **Total params**: ~1.17M

## Training

```bash
python main_v3.py
```

- **Corpus**: 2973 synthetic Chinese sentences (template-based generation)
- **Optimizer**: AdamW, lr=2e-3, weight_decay=0.01
- **Loss**: Cross-entropy, ignore PAD
- **Checkpoints**: Every 500 steps → `checkpoint_v3.pt`

## Results (v3)

### Training
| Step | Loss |
|------|------|
| 0 | 5.95 |
| 300 | 0.187 |
| 1000 | 0.013 |
| 5000 | 0.000 |
| 8000 | 0.000 |

Loss converges to near-zero — the model overfits the training corpus completely.

### Semantic Interpolation

Smooth transitions between semantic concepts in latent space:

```
Traverse: '今天' → '未来'

α= 0.0  '天'  conf=0.999  ███████████████████
α= 0.4  '天'  conf=0.978  ███████████████████
α= 0.5  '天'  conf=0.827  ████████████████
α= 0.6  '未'  conf=0.602  ████████████
α= 1.0  '未'  conf=0.947  ██████████████████
```

The model learns a meaningful geometry: semantic concepts are linearly interpolatable in the latent space.

### Generation (broken)

```
'今天天气' → '今天天气年年年年年年年年年年年年年年年'
'我爱'     → '我爱爱爱爱爱爱爱爱爱爱爱爱爱爱爱爱'
```

Extreme repetition collapse — the model memorized the most common continuations from training.

## Evolution

| Version | Model | Key Change | Result |
|---------|-------|-----------|--------|
| v1 | LSTM | Baseline | Works but repetitive |
| v2 | SSM-RNN | State-space inspired layers | Memory better, generation still weak |
| v3 | Transformer | Full attention + PE | Interpolation works, generation collapses |

## Project Files

```
myllm/
├── main.py          # v1: Simple LSTM baseline
├── main_v2.py       # v2: SSM-style recurrent layers
├── main_v3.py       # v3: Full Transformer (latest)
├── model_v3.pt      # v3 trained model weights
├── checkpoint_v3.pt # v3 optimizer state + weights
├── compare_v1_v2.py # Comparative analysis script
├── test_*.py       # Debug / verification scripts
└── README.md        # This file
```

## Key Learnings

1. **Parallel training works**: PyTorch parallelizes over batch natively — no manual chunking needed
2. **Unified dimension is simpler**: Separating E_in/E_out led to shape mismatches; single D throughout is cleaner
3. **Overfitting is the main enemy**: With small synthetic data, loss=0 means memorizing, not learning
4. **Interpolation is robust**: Even with collapsed generation, semantic interpolation in latent space still works — the geometry is learned, just the autoregressive head is broken

## Next Steps (to fix generation)

- **More data**: Real Chinese text corpus instead of synthetic templates
- **More regularization**: Dropout, weight decay, early stopping
- **Larger model + fewer steps**: Prevent memorization
- **Or**: Start from a pretrained Chinese model (e.g.,BERT/GPT) and fine-tune

## Citation

```bibtex
@software{curve_transformer_2026,
  title={Curve Transformer: Transformer with Latent Semantic Interpolation},
  author={Mao Chen},
  year={2026},
  url={https://github.com/MaoChen1980/curve-transformer}
}
```