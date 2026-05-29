"""
Train v4 - C: Pure GRU baseline.
No attention, just stacked GRU with embedding + projection.
"""
import sys, os, re, json, glob, math, random
os.environ["USERNAME"] = "user"
os.environ["USER"] = "user"
import torch, torch.nn as nn, torch.nn.functional as F

POETRY_DIR = "E:/claude/myllm/corpora/tang_poetry/"
DORAEMON_DIR = "E:/claude/myllm/corpora/doraemon/"
CKPT = "E:/claude/myllm/checkpoint_v4_gru.pt"

def load_poetry():
    texts = []
    for fpath in sorted(glob.glob(os.path.join(POETRY_DIR, "poet.tang.*.json"))):
        with open(fpath, 'r') as f:
            for poem in json.load(f):
                t = "".join(poem["paragraphs"])
                if len(t) >= 4: texts.append(t)
    return texts

def load_doraemon():
    texts = []; jp = set('のひとつがくなんいむぶあいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん')
    for fname in sorted(os.listdir(DORAEMON_DIR)):
        if not fname.endswith('.txt') or fname in {"伴我同行.txt"}: continue
        with open(os.path.join(DORAEMON_DIR, fname), 'r') as f:
            t = re.sub(r'\{[^}]*\}', '', f.read())
        lines = [l.strip() for l in t.split('\n') if l.strip()]
        joined = "".join(lines)
        if sum(1 for c in joined if c in jp)/max(len(joined),1) > 0.1: continue
        texts.extend(lines)
    return texts

print("Loading data...", flush=True)
all_t = load_poetry() + load_doraemon()
random.shuffle(all_t)

chars = set(c for t in all_t for c in t)
c2i = {"<PAD>":0, "<UNK>":1, "<BOS>":2, "<EOS>":3}
for c in sorted(chars): c2i[c] = len(c2i)
MAXLEN = 100; VOCAB = len(c2i)

def enc(s):
    t = [2] + [c2i.get(c,1) for c in s] + [3]
    if len(t) > MAXLEN: t = t[:MAXLEN]
    return torch.tensor(t + [0]*(MAXLEN-len(t)), dtype=torch.long)

data = torch.stack([enc(s) for s in all_t])
print(f"Samples: {len(all_t)}, vocab={VOCAB}, data={data.shape}", flush=True)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(VOCAB,128,padding_idx=0)
        pe=torch.zeros(MAXLEN,128)
        for i in range(MAXLEN):
            for j in range(128):
                pe[i,j]=math.sin(i/10000**(2*j/128)) if j%2==0 else math.cos(i/10000**(2*j/128))
        self.register_buffer('pe',pe);self.proj=nn.Linear(128,256)
        self.gru=nn.GRU(256,256,num_layers=3,batch_first=False)
        self.norm=nn.LayerNorm(256);self.fc=nn.Linear(256,VOCAB)
    def forward(self,tok):
        L,B=tok.shape;x=self.emb(tok)+self.pe[:L].unsqueeze(1)
        x=self.proj(x);x,_=self.gru(x)
        return self.fc(self.norm(x))[:-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

model = Model().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
crit = nn.CrossEntropyLoss(ignore_index=0)

start = 0
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"])
    opt.load_state_dict(ck["optimizer_state"])
    start = ck["step"] + 1
    print(f"Resuming from step {start}", flush=True)

print(f"Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

last_losses = []
TOTAL = 10000

for s in range(start, TOTAL):
    model.train(); opt.zero_grad()
    idx = torch.randint(0, len(all_t), (16,))
    tok = data[idx].transpose(0, 1).to(device)
    logits = model(tok)
    loss = crit(logits.reshape(-1, VOCAB), tok[1:].reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    if s % 100 == 0:
        print(f"step {s:4d}/{TOTAL} | loss {loss.item():.4f}", flush=True)
    if s > 0 and s % 500 == 0:
        torch.save({"model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
                    "step": s, "c2i": c2i, "i2c": {v:k for k,v in c2i.items()}}, CKPT)
        print(f"  checkpoint saved step {s}", flush=True)

torch.save({"model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
            "step": s, "c2i": c2i, "i2c": {v:k for k,v in c2i.items()}}, CKPT)
print(f"Final checkpoint saved at step {s}", flush=True)

model_path = "E:/claude/myllm/model_v4_gru.pt"
torch.save({"model_state": model.state_dict(), "c2i": c2i,
            "i2c": {v:k for k,v in c2i.items()}}, model_path)
print(f"Model saved -> {model_path}", flush=True)
