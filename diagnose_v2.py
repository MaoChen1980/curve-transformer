"""Diagnose selective state gate outputs."""
import math, torch, torch.nn as nn, torch.nn.functional as F

class SelectiveState(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.i_gate = nn.Linear(in_dim + hidden_dim, hidden_dim)
        self.f_gate = nn.Linear(in_dim + hidden_dim, hidden_dim)
        self.c_cand = nn.Linear(in_dim + hidden_dim, hidden_dim)
    def forward(self, x, h):
        combined = torch.cat([x, h], dim=-1)
        f = torch.sigmoid(self.f_gate(combined))
        i = torch.sigmoid(self.i_gate(combined))
        cand = torch.tanh(self.c_cand(combined))
        return f * h + i * cand

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=20):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        for pos in range(max_len):
            for i in range(dim):
                angle = pos / (10000 ** (2 * i / dim))
                pe[pos, i] = math.sin(angle) if i % 2 == 0 else math.cos(angle)
        self.register_buffer('pe', pe)

class CurveRNNv2(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256, n_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pe = PositionalEncoding(embed_dim)
        self.state_layers = nn.ModuleList([
            SelectiveState(embed_dim if i==0 else hidden_dim, hidden_dim)
            for i in range(n_layers)])
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def init_state(self, batch_size, device):
        return [torch.zeros(batch_size, self.hidden_dim, device=device) for _ in self.state_layers]
    def step(self, x, state):
        for li, sl in enumerate(self.state_layers):
            state[li] = sl(x, state[li])
            x = state[li]
        return state

device = torch.device('cpu')
model = CurveRNNv2(532, 64, 256, 2).to(device)
ckpt = torch.load('E:/claude/myllm/checkpoint_v2.pt', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state'])
char2idx = ckpt['char2idx']
idx2char = {v:k for k,v in char2idx.items()}
model.eval()

# Check gate outputs for a simple prompt
token_ids = [char2idx.get(c, 1) for c in "今天天气"]
tokens = torch.tensor(token_ids, dtype=torch.long)
state = model.init_state(1, device)

with torch.no_grad():
    for t in range(len(tokens)):
        x = model.embed(tokens[t:t+1]) + model.pe.pe[t:t+1]
        x = x.squeeze(1)
        state = model.step(x, state)

# Check final state
h = state[-1]
logits = model.fc(h)
top5_idx = logits[0].topk(5).indices.tolist()
top5_char = [idx2char.get(t, '?') for t in top5_idx]
print(f"Prompt: '今天天气'")
print(f"Top 5 predictions: {top5_char}")
print(f"Top 5 probs: {[f'{p:.3f}' for p in logits[0].topk(5).values.tolist()]}")
print(f"State norm: {h.norm().item():.4f}")

# What is the FC layer doing?
fc_out = model.fc.weight[:10, :5]
print(f"FC weight norm: {model.fc.weight.norm().item():.4f}")
print(f"FC bias: {model.fc.bias[:5].tolist()}")

# Simple: what would the ORIGINAL v1 model produce?
print("\n--- Compare with v1 model ---")
ckpt_v1 = torch.load('E:/claude/myllm/model.pt', map_location=device, weights_only=False)
print(f"v1 trained steps: loaded")