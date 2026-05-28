"""Quick load + generate from checkpoint_v2"""
import os, math, torch, torch.nn as nn, torch.nn.functional as F

# Same model definition (abbreviated)
class SelectiveState(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.i_gate = nn.Linear(embed_dim + hidden_dim, hidden_dim)
        self.f_gate = nn.Linear(embed_dim + hidden_dim, hidden_dim)
        self.c_cand = nn.Linear(embed_dim + hidden_dim, hidden_dim)
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
        self.register_buffer("pe", pe)
    def forward(self, x):
        return x + self.pe[:x.size(0)]

class CurveRNNv2(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256, n_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pe = PositionalEncoding(embed_dim)
        self.state_layers = nn.ModuleList([
            SelectiveState(embed_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(n_layers)])
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def init_state(self, batch_size, device):
        return [torch.zeros(batch_size, self.hidden_dim, device=device)
                for _ in self.state_layers]
    def step(self, x, state):
        for li, sl in enumerate(self.state_layers):
            state[li] = sl(x, state[li])
            x = state[li]
        return state

device = torch.device("cpu")
model = CurveRNNv2(532, embed_dim=64, hidden_dim=256, n_layers=2).to(device)

ckpt = torch.load("E:/claude/myllm/checkpoint_v2.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state"])
char2idx = ckpt["char2idx"]
idx2char = ckpt["idx2char"]
step = ckpt["step"]

print(f"✅ Loaded checkpoint from step {step}")
model.eval()

def generate(prompt, max_new=15, temp=0.8):
    tokens = torch.tensor([char2idx.get(c, 1) for c in prompt], dtype=torch.long).to(device)
    state = model.init_state(1, device)
    with torch.no_grad():
        for t in range(len(tokens)):
            x = model.embed(tokens[t]) + model.pe.pe[t]
            state = model.step(x, state)
    result = list(prompt)
    for _ in range(max_new):
        x = model.embed(tokens[-1:])                   # (1, E)
        x = x + model.pe.pe[0:1]                       # (1, E) — keep 2D
        state = model.step(x, state)
        logits = model.fc(state[-1])                   # (1, D) → (1, V)
        probs = F.softmax(logits / temp, dim=-1)
        tok = probs.argmax(dim=-1).item()
        if tok == 0:
            break
        result.append(idx2char.get(tok, ""))
        tokens = torch.cat([tokens, torch.tensor([tok], dtype=torch.long).to(device)])
    return "".join(result)

print("\nGeneration (step", step, "):")
for prompt in ["今天天气", "我爱", "宇宙", "月亮", "未来", "健康", "诗歌"]:
    outs = [generate(prompt, max_new=15) for _ in range(2)]
    print(f"  '{prompt}' → '{outs[0]}' | '{outs[1]}'")