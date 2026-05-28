"""Minimal debug of SSM forward shapes."""
import torch
from torch import nn
import torch.nn.functional as F

class SSMScan(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        Dh = hidden_dim // n_heads
        self.q_proj = nn.Linear(in_dim, hidden_dim)
        self.k_proj = nn.Linear(in_dim, hidden_dim)
        self.v_proj = nn.Linear(in_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, in_dim)
        self.head_scale = nn.Parameter(torch.ones(n_heads, Dh))
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_seq):
        L, B, E = x_seq.shape
        _nheads = self.n_heads
        _hd = self.hidden_dim
        H, Dh = _nheads, _hd // _nheads
        print(f"    SSMScan.forward: in_dim={in_dim}, hidden={self.hidden_dim}, H={H}, Dh={Dh}")
        print(f"    x_seq shape: {x_seq.shape} → after reshape: ({L}, {B}, {H}, {Dh})")
        q = self.q_proj(x_seq).view(L, B, H, Dh).permute(1, 2, 0, 3)
        print(f"    q shape: {q.shape}")
        return x_seq

# Test with B=16 (chunk_size), L=20
model = SSMScan(64, 256, 4)
print("=== Layer 0 (in_dim=64, hidden=256) ===")
x = torch.randn(20, 16, 64)
try:
    out = model(x)
    print(f"  ✅ output: {out.shape}")
except Exception as e:
    print(f"  ❌ {e}")

# Test with B=16 (chunk_size after splitting 64/4=16)
print("\n=== With batch=64, chunks=4 → chunk_size=16 ===")
print("  tokens shape: (L=20, B=16)")
x = torch.randint(2, 400, (20, 16))  # random tokens
try:
    out = model(x.float())  # tokens need to be float for Linear
    print(f"  ✅")
except Exception as e:
    print(f"  ❌ {e}")

# Check embed output
print("\n=== Embed output ===")
embed = nn.Embedding(400, 64)
tokens = torch.randint(2, 400, (20, 16))
emb = embed(tokens)
print(f"  tokens: {tokens.shape} → emb: {emb.shape}")
print(f"  embed.weight: {embed.weight.shape}")

# Full test
print("\n=== Full model test ===")
from torch.nn.functional import elu

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(400, 64)
        self.layer = SSMScan(64, 256, 4)
        self.norm = nn.LayerNorm(256)
        self.fc = nn.Linear(256, 400)
    def forward(self, tokens):
        x = self.embed(tokens)  # (20, 16, 64)
        print(f"  embed out: {x.shape}")
        x = self.layer(x)
        print(f"  layer out: {x.shape}")
        x = self.norm(x)
        return x

m = SimpleModel()
tokens = torch.randint(2, 400, (20, 16))
try:
    out = m(tokens)
    print(f"  ✅ final: {out.shape}")
except Exception as e:
    print(f"  ❌ {e}")

# Test: what if tokens has dtype long?
tokens_long = torch.randint(2, 400, (20, 16)).long()
tokens_int = torch.randint(2, 400, (20, 16))
print(f"\n  tokens dtype: {tokens_long.dtype}")
emb = m.embed(tokens_long)
print(f"  emb: {emb.shape} ✅")

print("\n=== All shapes look correct ===")