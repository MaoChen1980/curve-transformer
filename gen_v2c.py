"""Fix: position tracking + repetition penalty in generation."""
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
print(f"Loaded v2 from step {ckpt['step']}")

def generate(prompt, max_new=15, temp=0.9, repetition_penalty=2.0):
    tokens = torch.tensor([char2idx.get(c, 1) for c in prompt], dtype=torch.long).to(device)
    state = model.init_state(1, device)

    # Encode prompt with correct positions
    with torch.no_grad():
        for t in range(len(tokens)):
            x = model.embed(tokens[t:t+1]) + model.pe.pe[t:t+1]  # correct position
            x = x.squeeze(1)
            state = model.step(x, state)

    result = list(prompt)
    # Track absolute position for PE (after prompt)
    gen_pos = len(tokens)

    for _ in range(max_new):
        with torch.no_grad():
            # Use current position PE
            pos_idx = min(gen_pos, 19)
            x = model.embed(tokens[-1:]) + model.pe.pe[pos_idx:pos_idx+1]
            x = x.squeeze(1)
            state = model.step(x, state)
            logits = model.fc(state[-1])  # (1, V)

            # Repetition penalty
            for prev_char in set(result[-5:]):
                idx = char2idx.get(prev_char, -1)
                if idx >= 0:
                    logits[0, idx] /= repetition_penalty

            probs = F.softmax(logits / temp, dim=-1)
            tok = probs.argmax(dim=-1).item()

        if tok == 0:
            break
        result.append(idx2char.get(tok, ''))
        tokens = torch.cat([tokens, torch.tensor([tok], dtype=torch.long).to(device)])
        gen_pos += 1

    return ''.join(result)

print("\n=== Generation v2 (with position tracking + repetition penalty) ===\n")
for prompt in ["今天天气", "我爱", "宇宙", "月亮", "未来", "健康", "诗歌"]:
    g1 = generate(prompt, max_new=15, repetition_penalty=2.0)
    g2 = generate(prompt, max_new=15, repetition_penalty=2.0, temp=1.2)
    print(f"  '{prompt}' | temp=0.9 → '{g1}'")
    print(f"  '{prompt}' | temp=1.2 → '{g2}'")
    print()