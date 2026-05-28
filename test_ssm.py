"""Quick test SSMScan forward + recurrent."""
import math, torch, torch.nn as nn, torch.nn.functional as F

class SSMScan(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        head_dim = hidden_dim // n_heads
        self.query_proj = nn.Linear(embed_dim, hidden_dim)
        self.key_proj = nn.Linear(embed_dim, hidden_dim)
        self.val_proj = nn.Linear(embed_dim, hidden_dim)
        self.A = nn.Parameter(torch.randn(n_heads, head_dim, head_dim) * 0.01)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x_seq):
        """x_seq: (L, B, E) → (L, B, E)"""
        L, B, E = x_seq.shape
        H = self.n_heads
        Dh = self.hidden_dim // H
        q = self.query_proj(x_seq).view(L, B, H, Dh).permute(1, 2, 0, 3)  # (B,H,L,Dh)
        k = self.key_proj(x_seq).view(L, B, H, Dh).permute(1, 2, 0, 3)
        v = self.val_proj(x_seq).view(L, B, H, Dh).permute(1, 2, 0, 3)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(Dh)
        attn = F.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, v)                                     # (B,H,L,Dh)
        ctx = ctx.permute(2, 0, 1, 3).reshape(L, B, self.hidden_dim)    # (L,B,D)
        out = x_seq + self.out_proj(ctx)
        return out

    def forward_recurrent(self, x, h):
        """x: (B, E), h: (B, D) → logits (B, E), h_new (B, D)"""
        B, E = x.shape
        H, Dh = self.n_heads, self.hidden_dim // self.n_heads
        q = self.query_proj(x).view(B, H, Dh)
        v = self.val_proj(x).view(B, H, Dh)
        a_scale = torch.sigmoid(self.A.mean(dim=[1, 2]))                    # (H,) not (H,Dh)
        a_scale = a_scale.unsqueeze(1)                                      # (H,1)
        h_h = h.view(B, H, Dh)                                              # (B,D)→(B,H,Dh)
        gate = torch.sigmoid(q)
        h_new = a_scale * h_h + gate * v                                    # broadcasting
        h_new = h_new.reshape(B, self.hidden_dim)                       # (B,D)
        logits = x + self.out_proj(h_new)
        return logits, h_new

# Test
B, L, E, D = 16, 20, 64, 256
layer = SSMScan(E, D, 4)
x_seq = torch.randn(L, B, E)
out = layer(x_seq)
print(f"forward: {x_seq.shape} → {out.shape}")

# Test recurrent
h = torch.zeros(B, D)
x = torch.randn(B, E)
logits, h_new = layer.forward_recurrent(x, h)
print(f"recurrent: x={x.shape}, h={h.shape} → logits={logits.shape}, h_new={h_new.shape}")
print("✅ All OK")