"""
Curve Model v3 — Clean Transformer (unified dimension)
=======================================================
Fixed: single unified dimension D throughout.
Attention: D→D projections, multi-head, residual + norm + FFN
No more E_in/E_out confusion.

Training data: REAL corpus from GPT2-distil-chinese (not synthetic templates).
"""

# USERNAME fix — MUST be set BEFORE any torch import.
# Python 3.14 on Windows: torch._inductor needs getpass.getuser() which fails
# if USERNAME/USER env vars are missing. Set both env vars here at module entry.
import os, getpass
os.environ.setdefault("USERNAME", "user")
os.environ.setdefault("USER", "user")

import random, math, torch, torch.nn as nn, torch.nn.functional as F
# ══════════════════════════════════════════════════════════════════════════════
# 1. Data — load REAL corpus from GPT2-distil-chinese
# ══════════════════════════════════════════════════════════════════════════════
CORPUS_FILE = "E:/claude/myllm/real_corpus.txt"
with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
    REAL_CORPUS = [line.strip() for line in f if line.strip()]

# Use real corpus directly (no template substitution)
CORPUS = REAL_CORPUS

chars = set("".join(CORPUS))
c2i = {"<PAD>":0,"<UNK>":1}
for c in sorted(chars): c2i[c] = len(c2i)
i2c = {v:k for k,v in c2i.items()}
MAXLEN = max(len(s) for s in CORPUS)
VOCAB = len(c2i)
print(f"Corpus: {len(CORPUS)} sentences, max_len={MAXLEN}")

def enc(s):
    t = [c2i.get(c,1) for c in s]
    return torch.tensor(t + [0]*(MAXLEN-len(t)), dtype=torch.long)

# PE tensor: (MAXLEN, D)
def make_pe(D, L=MAXLEN):
    p = torch.zeros(L, D)
    for i in range(L):
        for j in range(D):
            p[i,j] = math.sin(i/10000**(2*j/D)) if j%2==0 else math.cos(i/10000**(2*j/D))
    return p

# ══════════════════════════════════════════════════════════════════════════════
# 2. Model — Unified dimension D throughout
# ══════════════════════════════════════════════════════════════════════════════
class Block(nn.Module):
    """Standard Transformer block: MHA → AddNorm → FFN → AddNorm"""
    def __init__(self, D, heads=4):
        super().__init__()
        Dh = D // heads
        self.attn_norm = nn.LayerNorm(D)
        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, D)
        self.v_proj = nn.Linear(D, D)
        self.o_proj = nn.Linear(D, D)
        self.heads = heads
        self.Dh = Dh

        self.ffn_norm = nn.LayerNorm(D)
        self.ffn = nn.Sequential(
            nn.Linear(D, D*2),
            nn.GELU(),
            nn.Linear(D*2, D),
        )

    def forward(self, x):
        """x: (L, B, D) → (L, B, D)"""
        L, B, D = x.shape
        H, Dh = self.heads, self.Dh

        # MHA with pre-norm
        x_norm = self.attn_norm(x)
        Q = self.q_proj(x_norm).view(L, B, H, Dh).permute(1, 2, 0, 3)   # (B,H,L,Dh)
        K = self.k_proj(x_norm).view(L, B, H, Dh).permute(1, 2, 0, 3)
        V = self.v_proj(x_norm).view(L, B, H, Dh).permute(1, 2, 0, 3)

        # Attention: (B,H,L,L) ← (B,H,L,Dh) @ (B,H,Dh,L)
        attn = F.softmax(torch.matmul(Q, K.transpose(-2,-1))/math.sqrt(Dh), dim=-1)
        C = torch.matmul(attn, V)                                   # (B,H,L,Dh)
        C = C.permute(2, 0, 1, 3).reshape(L, B, D)                  # (L,B,D)
        x = x + self.o_proj(C)                                       # residual

        # FFN with pre-norm
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def step(self, x, h):
        """x: (B, D), h: (B, D) → (B, D), (B, D) — recurrent for generation"""
        B = x.size(0)
        H, Dh = self.heads, self.Dh

        x_n = self.attn_norm(x)
        q = self.q_proj(x_n).view(B, H, Dh)
        v = self.v_proj(x_n).view(B, H, Dh)

        gate = torch.sigmoid(q)
        h_h = h.view(B, H, Dh)
        h_new = h_h * gate + v * (1 - gate)
        h_new = h_new.reshape(B, -1)      # (B, D)

        x = x + self.o_proj(h_new)
        x = x + self.ffn(self.ffn_norm(x))
        return x, h_new


class CurveTransformer(nn.Module):
    """Embed → PE → Blocks → Norm → FC"""
    def __init__(self, vocab, E=64, D=256, n_layers=2, heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, E, padding_idx=0)
        self.pe = make_pe(E)                     # (MAXLEN, E)
        self.proj = nn.Linear(E, D)              # E → D
        self.blocks = nn.ModuleList([Block(D, heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(D)
        self.fc = nn.Linear(D, vocab)

    def encode(self, tok):
        """tok: (L, B) → h: (L, B, D)"""
        L, B = tok.shape
        x = self.embed(tok) + self.pe[:L].unsqueeze(1)   # (L, B, E)
        x = self.proj(x)                                  # (L, B, D)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, tok):
        h = self.encode(tok)
        return self.fc(h)[:-1], h                       # (L-1, B, V)

    def init_h(self, B, dev):
        return torch.zeros(B, self.blocks[0].heads * self.blocks[0].Dh, device=dev)

    def gen_step(self, tok, h):
        """Full-context generation: attend over all tokens so far.
        tok: (L, B), h: (B, D) unused — kept for API compat."""
        L, B = tok.shape
        x = self.embed(tok) + self.pe[:L].unsqueeze(1)
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        h_new = x[-1]                                        # update recurrent state
        return self.fc(x[-1]), h_new                       # (B, V)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Training
# ══════════════════════════════════════════════════════════════════════════════
CKPT = "E:/claude/myllm/checkpoint_v3.pt"

def train(steps=8000, lr=2e-3, B=32, print_every=300):
    dev = torch.device("cpu")
    model = CurveTransformer(VOCAB, E=64, D=256, n_layers=2, heads=4).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")
    data = torch.stack([enc(s) for s in CORPUS]).to(dev)

    start = 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer_state"])
        start = ck["step"] + 1
        print(f"Continue from step {start}")

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Corpus: {len(CORPUS)} sentences, vocab={VOCAB}\n")

    for s in range(start, steps):
        model.train()
        opt.zero_grad()

        idx = torch.randint(0, len(CORPUS), (B,))
        tok = data[idx].transpose(0, 1)    # (L, B)

        logits, _ = model(tok)
        tgt = tok[1:].contiguous()
        loss = crit(logits.reshape(-1, VOCAB), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if s % print_every == 0:
            print(f"step {s:5d} | loss {loss.item():.4f}")

        if s > 0 and s % 500 == 0:
            torch.save({"model_state": model.state_dict(),
                       "optimizer_state": opt.state_dict(),
                       "step": s, "c2i": c2i, "i2c": i2c}, CKPT)
            print(f"saved step {s}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4. Generation
# ══════════════════════════════════════════════════════════════════════════════
def generate(model, prompt, max_new=15, temp=0.9, rep=2.5):
    dev = next(model.parameters()).device
    model.eval()
    tok = torch.tensor([c2i.get(c,1) for c in prompt], dtype=torch.long).to(dev)

    # Encode prompt: embed → proj → blocks → norm
    with torch.no_grad():
        L = tok.size(0)
        x = model.embed(tok.unsqueeze(1)) + model.pe[:L].unsqueeze(1)
        x = model.proj(x)
        for blk in model.blocks:
            x = blk(x)
        x = model.norm(x)
        # h: last position, last block, (D,)
        h = x[-1, 0]    # (D,)

    result = list(prompt)
    for _ in range(max_new):
        with torch.no_grad():
            logits, h = model.gen_step(tok, h)
            p = F.softmax(logits / temp, dim=-1).squeeze(0)
            for c in set(result[-5:]):
                i = c2i.get(c, -1)
                if i >= 0: p[i] = p[i] ** 0.7
            nxt = p.argmax().item()
        if nxt == 0: break
        result.append(i2c.get(nxt, ""))
        tok = torch.cat([tok, tok.new_tensor([nxt])])

    return "".join(result)


def traverse(model, a, b, steps=10):
    dev = next(model.parameters()).device
    model.eval()
    def _z(p):
        t = torch.tensor([c2i.get(c,1) for c in p], dtype=torch.long).to(dev)
        x = model.embed(t).unsqueeze(1) + model.pe[:len(t)].unsqueeze(1)
        x = model.proj(x)
        for blk in model.blocks:
            x = blk(x)
        return model.norm(x)[-1, 0]
    z1, z2 = _z(a), _z(b)
    print(f"\n  Traverse: '{a}' → '{b}'")
    for i in range(steps+1):
        al = i/steps
        z = z1*(1-al) + z2*al
        p = F.softmax(model.fc(z), dim=-1)
        c = i2c.get(p.argmax().item(), ".")
        conf = p.max().item()
        print(f"  alpha={al:.2f}  '{c}'  conf={conf:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("Curve v3 — Transformer + REAL GPT2 Corpus")
    print(f"Corpus: {len(CORPUS)} sentences, vocab={VOCAB}")
    print("="*60)
    model = train(steps=8000, lr=2e-3, B=32, print_every=300)

    print("\n=== Generation ===")
    for p in ["今天天气","我爱","宇宙","健康","人工智能"]:
        g = generate(model, p, rep=2.5)
        print(f"  '{p}' -> '{g}'")

    print("\n=== Interpolation ===")
    traverse(model, "今天", "未来")

    torch.save({"model_state": model.state_dict(), "c2i": c2i, "i2c": i2c},
               "E:/claude/myllm/model_v3.pt")
    print("\nSaved -> model_v3.pt")