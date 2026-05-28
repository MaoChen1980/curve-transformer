"""Check model state dict shapes."""
import torch, os
from torch import nn

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

class CurveSSM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256, n_layers=2, n_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.ssm_layers = nn.ModuleList([
            SSMScan(embed_dim if i == 0 else hidden_dim, hidden_dim, n_heads)
            for i in range(n_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

model = CurveSSM(500, 64, 256, 2, 4)

print("=== Fresh model state_dict ===")
for k, v in model.state_dict().items():
    print(f"  {k}: {v.shape}")

if os.path.exists('E:/claude/myllm/checkpoint_v3.pt'):
    ckpt = torch.load('E:/claude/myllm/checkpoint_v3.pt', map_location='cpu', weights_only=False)
    step = ckpt['step']
    print(f"\n=== Checkpoint keys (step {step}) ===")
    for k, v in ckpt['model_state'].items():
        print(f"  {k}: {v.shape}")
else:
    print("\nNo checkpoint_v3.pt")

# Forward pass test
tokens = torch.randint(2, 500, (20, 16))  # (L, B)
print(f"\nForward test: tokens={tokens.shape}")
try:
    logits, h = model(tokens)
    print(f"  logits={logits.shape}, h={h.shape} ✅")
except Exception as e:
    print(f"  Error: {e}")