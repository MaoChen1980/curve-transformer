"""Compare v1 vs v2 generation — apples to apples, same prompts."""
import math, torch, torch.nn as nn, torch.nn.functional as F

# === v1 model (original) ===
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=20):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        for pos in range(max_len):
            for i in range(dim):
                angle = pos / (10000 ** (2 * i / dim))
                pe[pos, i] = math.sin(angle) if i % 2 == 0 else math.cos(angle)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:x.size(0)]

class CurveRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256, n_layers=2):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pe = PositionalEncoding(embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, batch_first=False)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def init_state(self, batch_size, device):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)
    def step(self, x, hidden):
        x = x.unsqueeze(0)
        out, new_hidden = self.rnn(x, hidden)
        return out.squeeze(0), new_hidden

device = torch.device('cpu')

# Load v1
v1 = CurveRNN(532, 64, 256, 2).to(device)
ckpt_v1 = torch.load('E:/claude/myllm/model.pt', map_location=device, weights_only=False)
v1.load_state_dict(ckpt_v1['model_state'])
v1.eval()
char2idx = ckpt_v1['char2idx']
idx2char = {v:k for k,v in char2idx.items()}

def gen_v1(prompt, max_new=15, temp=0.8, rep_penalty=1.5):
    tokens = torch.tensor([char2idx.get(c,1) for c in prompt], dtype=torch.long).to(device)
    hidden = v1.init_state(1, device)
    with torch.no_grad():
        tokens_b = tokens.unsqueeze(-1)
        L = tokens_b.shape[0]
        emb = v1.embed(tokens_b) + v1.pe.pe[:L].unsqueeze(1)
        _, hidden = v1.rnn(emb)
    result = list(prompt)
    for _ in range(max_new):
        x = v1.embed(tokens[-1:]) + v1.pe.pe[0]
        x = x.unsqueeze(0)
        out, hidden = v1.rnn(x, hidden)
        logits = v1.fc(out.squeeze(0))
        for char in set(result[-4:]):
            idx = char2idx.get(char, -1)
            if idx >= 0:
                logits[0, idx] -= rep_penalty
        probs = F.softmax(logits / temp, dim=-1)
        tok = probs.argmax(dim=-1).item()
        if tok == 0: break
        result.append(idx2char.get(tok, ''))
        tokens = torch.cat([tokens, torch.tensor([tok], dtype=torch.long).to(device)])
    return ''.join(result)

print("=== v1 generation (top-1, rep_penalty=1.5) ===")
for prompt in ["今天天气", "我爱", "健康", "诗歌"]:
    print(f"  '{prompt}' → '{gen_v1(prompt, max_new=15, rep_penalty=1.5)}'")

print("\n=== v1 generation (top-1, rep_penalty=3.0) ===")
for prompt in ["今天天气", "我爱", "健康", "诗歌"]:
    print(f"  '{prompt}' → '{gen_v1(prompt, max_new=15, rep_penalty=3.0)}'")

# Load v2
from copy import deepcopy
v2 = deepcopy(v1)
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

# v2 has no rnn/lstm, only state_layers + fc
ckpt_v2 = torch.load('E:/claude/myllm/checkpoint_v2.pt', map_location=device, weights_only=False)
# Need to reload full v2 model architecture
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

v2_new = CurveRNNv2(532, 64, 256, 2).to(device)
v2_new.load_state_dict(ckpt_v2['model_state'])
v2_new.eval()

def gen_v2(prompt, max_new=15, temp=0.9, rep_penalty=3.0):
    tokens = torch.tensor([char2idx.get(c,1) for c in prompt], dtype=torch.long).to(device)
    state = v2_new.init_state(1, device)
    for t in range(len(tokens)):
        x = v2_new.embed(tokens[t:t+1]) + v2_new.pe.pe[t:t+1]
        x = x.squeeze(1)
        state = v2_new.step(x, state)
    result = list(prompt)
    gen_pos = len(tokens)
    for _ in range(max_new):
        pos_idx = min(gen_pos, 19)
        x = v2_new.embed(tokens[-1:]) + v2_new.pe.pe[pos_idx:pos_idx+1]
        x = x.squeeze(1)
        state = v2_new.step(x, state)
        logits = v2_new.fc(state[-1])
        for char in set(result[-4:]):
            idx = char2idx.get(char, -1)
            if idx >= 0:
                logits[0, idx] -= rep_penalty
        probs = F.softmax(logits / temp, dim=-1)
        tok = probs.argmax(dim=-1).item()
        if tok == 0: break
        result.append(idx2char.get(tok, ''))
        tokens = torch.cat([tokens, torch.tensor([tok], dtype=torch.long).to(device)])
        gen_pos += 1
    return ''.join(result)

print("\n=== v2 generation (rep_penalty=3.0) ===")
for prompt in ["今天天气", "我爱", "健康", "诗歌"]:
    print(f"  '{prompt}' → '{gen_v2(prompt, max_new=15, rep_penalty=3.0)}'")

print("\n=== v2 generation (rep_penalty=5.0, temp=1.2) ===")
for prompt in ["今天天气", "我爱", "健康", "诗歌"]:
    print(f"  '{prompt}' → '{gen_v2(prompt, max_new=15, rep_penalty=5.0, temp=1.2)}'")