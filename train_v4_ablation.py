"""
v4 ablation study: A/B/C/D variants of D3 GRUCell step model.
Trains from scratch on poetry. Supports all combinations.
"""
import sys, os, re, json, glob, math, random
os.environ["USERNAME"] = "user"
os.environ["USER"] = "user"
import torch, torch.nn as nn, torch.nn.functional as F

# ─── Config ──────────────────────────────────────────────
VARIANT = os.environ.get("VARIANT", "base")
# Options: base A B C D AC ABC ABCD

POETRY_DIR = "E:/claude/myllm/corpora/tang_poetry/"
CKPT     = f"E:/claude/myllm/checkpoint_v4_ablation_{VARIANT}.pt"
OUT      = f"E:/claude/myllm/model_v4_ablation_{VARIANT}.pt"

HAS_A = 'A' in VARIANT
HAS_B = 'B' in VARIANT
HAS_C = 'C' in VARIANT
HAS_D = 'D' in VARIANT

# ─── Data ────────────────────────────────────────────────
def load_poetry():
    texts = []
    for fpath in sorted(glob.glob(os.path.join(POETRY_DIR, "poet.tang.*.json"))):
        with open(fpath, 'r') as f:
            for poem in json.load(f):
                t = "".join(poem["paragraphs"])
                if len(t) >= 4: texts.append(t)
    return texts

print(f"Variant={VARIANT} | A={HAS_A} B={HAS_B} C={HAS_C} D={HAS_D}", flush=True)
print("Loading Tang poetry...", flush=True)
all_t = load_poetry()
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

# ─── Block ───────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, bidx=0):
        super().__init__()
        D=256;h=4;dh=64;self.bidx=bidx;self.D=D
        self.an=nn.LayerNorm(D);self.qp=nn.Linear(D,D);self.kp=nn.Linear(D,D)
        self.vp=nn.Linear(D,D);self.op=nn.Linear(D,D);self.h=h;self.dh=dh
        self.fn=nn.LayerNorm(D);self.ff=nn.Sequential(nn.Linear(D,D*2),nn.GELU(),nn.Linear(D*2,D))

        # B: residual injection
        if HAS_B:
            self.hp = nn.Linear(D, D)

        # C: cross-attention to h history buffer
        if HAS_C:
            self.cq = nn.Linear(D, D)
            self.ck = nn.Linear(D, D)
            self.cv = nn.Linear(D, D)

        # D: dual-scale — fast GRU + slow GRU
        if HAS_D:
            self.gru_fast = nn.GRUCell(D, D)
            self.gru_slow = nn.GRUCell(D, D)
            self.slow_interval = 3
        else:
            self.gru = nn.GRUCell(D, D)

    def forward(self, x):
        L,B,D=x.shape;H,Dh=self.h,self.dh
        xn=self.an(x);Q=self.qp(xn).view(L,B,H,Dh).permute(1,2,0,3)
        K=self.kp(xn).view(L,B,H,Dh).permute(1,2,0,3);V=self.vp(xn).view(L,B,H,Dh).permute(1,2,0,3)
        s=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(Dh)
        mask=torch.triu(torch.full((L,L),float('-inf'),device=x.device),diagonal=1)
        a=F.softmax(s+mask,dim=-1)
        C=torch.matmul(a,V).permute(2,0,1,3).reshape(L,B,D)
        x=x+self.op(C);x=x+self.ff(self.fn(x));return x

    def step(self, x, h, h_buffer=None, slow_counter=0, h_slow=None):
        D=self.D
        xn = self.an(x).squeeze(0)  # [B, D]

        # C: cross-attend to h_buffer
        if HAS_C and h_buffer is not None and len(h_buffer) > 0:
            buf = torch.stack(h_buffer, dim=0)  # [K, B, D]
            q = self.cq(xn).unsqueeze(0)
            k = self.ck(buf); v = self.cv(buf)
            s = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(D)
            a = F.softmax(s, dim=-1)
            xn = xn + (a * v).sum(dim=0)

        if HAS_D:
            h_fast = self.gru_fast(xn, h)
            if h_slow is None:
                h_slow = torch.zeros_like(h)
            new_sc = slow_counter + 1
            if new_sc >= self.slow_interval:
                h_slow_new = self.gru_slow(h_fast, h_slow)
                new_sc = 0
            else:
                h_slow_new = h_slow
            h_new = h_fast + h_slow_new
        else:
            h_new = self.gru(xn, h)
            new_sc = None; h_slow_new = None

        if HAS_B:
            x = x.squeeze(0) + self.op(h_new) + self.hp(h_new)
        else:
            x = x.squeeze(0) + self.op(h_new)

        x = x + self.ff(self.fn(x))
        if HAS_D:
            return x.unsqueeze(0), h_new, new_sc, h_slow_new
        return x.unsqueeze(0), h_new

# ─── Model ──────────────────────────────────────────────
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(VOCAB,128,padding_idx=0)
        pe=torch.zeros(MAXLEN,128)
        for i in range(MAXLEN):
            for j in range(128):
                pe[i,j]=math.sin(i/10000**(2*j/128)) if j%2==0 else math.cos(i/10000**(2*j/128))
        self.register_buffer('pe',pe);self.proj=nn.Linear(128,256)
        self.blocks=nn.ModuleList([Block(bidx=i) for i in range(3)])
        self.norm=nn.LayerNorm(256);self.fc=nn.Linear(256,VOCAB)

    def forward(self, tok):
        L,B=tok.shape;x=self.emb(tok)+self.pe[:L].unsqueeze(1)
        x=self.proj(x)
        for b in self.blocks: x=b(x)
        return self.fc(self.norm(x))[:-1]

    def step_generate(self, tok, h, h_slow_list=None):
        x=self.emb(tok)+self.pe[:1];x=self.proj(x)
        if HAS_D:
            if h_slow_list is None:
                h_slow_list = [None] * len(self.blocks)
            for i, b in enumerate(self.blocks):
                h_i = h[i] if HAS_A else h
                x, h_i, _, h_slow_list[i] = b.step(x, h_i, h_slow=h_slow_list[i])
                if HAS_A: h[i] = h_i
                else: h = h_i
            return self.fc(self.norm(x)), h, h_slow_list
        else:
            for i, b in enumerate(self.blocks):
                h_i = h[i] if HAS_A else h
                x, h_i = b.step(x, h_i)
                if HAS_A: h[i] = h_i
                else: h = h_i
            return self.fc(self.norm(x)), h

# ─── Training ────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

model = Model().to(device)
total = sum(p.numel() for p in model.parameters())
print(f"Params: {total:,}", flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
crit = nn.CrossEntropyLoss(ignore_index=0)

start = 0; STEP_LEN = 3
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state"], strict=False)
    opt.load_state_dict(ck["optimizer_state"])
    start = ck["step"] + 1
    print(f"Resuming from step {start}", flush=True)

TOTAL = int(os.environ.get("TOTAL", "5000"))

for s in range(start, TOTAL):
    model.train(); opt.zero_grad()
    idx = torch.randint(0, len(all_t), (16,))
    tok = data[idx].transpose(0, 1).to(device)

    logits = model(tok)
    loss_fwd = crit(logits.reshape(-1, VOCAB), tok[1:].reshape(-1))

    cut = random.randint(15, min(MAXLEN - STEP_LEN - 1, 60))
    with torch.no_grad():
        x = model.emb(tok[:cut]) + model.pe[:cut].unsqueeze(1)
        x = model.proj(x)
        for b in model.blocks: x = b(x)
        h_fwd = model.norm(x)[-1].detach()

    if HAS_A:
        h = [h_fwd.clone() for _ in range(3)]
    else:
        h = h_fwd

    h_slow_list = [None] * 3 if HAS_D else None
    step_logits = []

    for si in range(STEP_LEN):
        inp = tok[cut+si:cut+si+1]
        if HAS_D:
            logit, h, h_slow_list = model.step_generate(inp, h, h_slow_list)
        else:
            logit, h = model.step_generate(inp, h)
        step_logits.append(logit)

    step_targets = tok[cut+1:cut+1+STEP_LEN]
    loss_step = crit(torch.cat(step_logits, dim=0).reshape(-1, VOCAB), step_targets.reshape(-1))

    loss = loss_fwd
    if not torch.isnan(loss_step) and not torch.isinf(loss_step):
        loss = loss_fwd + loss_step * 0.5
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()

    if s % 100 == 0:
        print(f"step {s:4d}/{TOTAL} | fwd {loss_fwd.item():.4f} | step {loss_step.item():.4f}", flush=True)
    if s > 0 and s % 1000 == 0:
        torch.save({"model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
                    "step": s, "c2i": c2i, "i2c": {v:k for k,v in c2i.items()}}, CKPT)
        print(f"  checkpoint saved step {s}", flush=True)

torch.save({"model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
            "step": s, "c2i": c2i, "i2c": {v:k for k,v in c2i.items()}}, CKPT)
torch.save({"model_state": model.state_dict(), "c2i": c2i,
            "i2c": {v:k for k,v in c2i.items()}}, OUT)
print(f"Done -> {OUT}", flush=True)
