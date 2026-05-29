"""
Train v4: 全唐诗 + 哆啦A梦 dialogue.
Architecture: E=128, D=256, layers=3, heads=4.
Trains until convergence (3× improvement < 0.1) or max 2500 steps.
Saves checkpoint every 500 steps.
"""
import sys, os, re, json, glob, math, random
os.environ["USERNAME"] = "user"
os.environ["USER"] = "user"
import torch, torch.nn as nn, torch.nn.functional as F

print("Starting segmented v4 training...", flush=True)

POETRY_DIR = "E:/claude/myllm/corpora/tang_poetry/"
DORAEMON_DIR = "E:/claude/myllm/corpora/doraemon/"
CKPT = "E:/claude/myllm/checkpoint_v4.pt"

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

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        D=256;h=4;dh=64
        self.an=nn.LayerNorm(D);self.qp=nn.Linear(D,D);self.kp=nn.Linear(D,D)
        self.vp=nn.Linear(D,D);self.op=nn.Linear(D,D);self.h=h;self.dh=dh
        self.fn=nn.LayerNorm(D);self.ff=nn.Sequential(nn.Linear(D,D*2),nn.GELU(),nn.Linear(D*2,D))
    def forward(self,x):
        L,B,D=x.shape;H,Dh=self.h,self.dh
        xn=self.an(x);Q=self.qp(xn).view(L,B,H,Dh).permute(1,2,0,3)
        K=self.kp(xn).view(L,B,H,Dh).permute(1,2,0,3);V=self.vp(xn).view(L,B,H,Dh).permute(1,2,0,3)
        a=F.softmax(torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(Dh),dim=-1)
        C=torch.matmul(a,V).permute(2,0,1,3).reshape(L,B,D)
        x=x+self.op(C);x=x+self.ff(self.fn(x));return x
    def step(self,x,h):
        B,H,Dh=x.size(1),self.h,self.dh
        xn=self.an(x);q=self.qp(xn).view(B,H,Dh);v=self.vp(xn).view(B,H,Dh)
        gate=torch.sigmoid(q);h_new=h.view(B,H,Dh)*gate+v*(1-gate)
        h_new=torch.tanh(h_new).reshape(B,-1)
        x=x.squeeze(0)+self.op(h_new);x=x+self.ff(self.fn(x));return x.unsqueeze(0),h_new

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(VOCAB,128,padding_idx=0)
        pe=torch.zeros(MAXLEN,128)
        for i in range(MAXLEN):
            for j in range(128):
                pe[i,j]=math.sin(i/10000**(2*j/128)) if j%2==0 else math.cos(i/10000**(2*j/128))
        self.register_buffer('pe',pe);self.proj=nn.Linear(128,256)
        self.blocks=nn.ModuleList([Block() for _ in range(3)])
        self.norm=nn.LayerNorm(256);self.fc=nn.Linear(256,VOCAB)
    def forward(self,tok):
        L,B=tok.shape;x=self.emb(tok)+self.pe[:L].unsqueeze(1)
        x=self.proj(x)
        for b in self.blocks: x=b(x)
        return self.fc(self.norm(x))[:-1]
    def step_generate(self,tok,h):
        x=self.emb(tok)+self.pe[:1];x=self.proj(x)
        for b in self.blocks: x,h=b.step(x,h)
        return self.fc(self.norm(x)),h

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

# Early stop: 3 consecutive improvements < 0.1 means converged
last_losses = []
TOTAL = 3000

STEP_LEN = 2  # step() unroll length

for s in range(start, TOTAL):
    model.train(); opt.zero_grad()
    idx = torch.randint(0, len(all_t), (16,))
    tok = data[idx].transpose(0, 1).to(device)
    logits = model(tok)
    loss_fwd = crit(logits.reshape(-1, VOCAB), tok[1:].reshape(-1))
    loss = loss_fwd

    # Step path training: unroll step() on a suffix
    if random.random() < 0.5 and tok.size(0) > 55:
        cut = random.randint(15, 50)
        with torch.no_grad():
            x = model.emb(tok[:cut]) + model.pe[:cut].unsqueeze(1)
            x = model.proj(x)
            for b in model.blocks: x = b(x)
            h = model.norm(x)[-1].detach()
        h_curr = h; step_logits = []
        for i in range(STEP_LEN):
            logit, h_curr = model.step_generate(tok[cut+i:cut+i+1], h_curr)
            step_logits.append(logit)
        step_logits = torch.cat(step_logits, dim=0)
        targets = tok[cut+1:cut+1+STEP_LEN]
        loss_step = crit(step_logits.reshape(-1, VOCAB), targets.reshape(-1))
        if not torch.isnan(loss_step) and not torch.isinf(loss_step):
            loss = loss_fwd + 0.3 * loss_step

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    if s % 100 == 0:
        msg = f"step {s:4d}/{TOTAL} | loss {loss_fwd.item():.4f}"
        if 'loss_step' in dir() and isinstance(loss_step, torch.Tensor):
            msg += f" | step_loss {loss_step.item():.4f}"
        print(msg, flush=True)
        last_losses.append(loss_fwd.item())
        if len(last_losses) >= 4:
            imps = [last_losses[i-1] - last_losses[i] for i in range(len(last_losses)-3, len(last_losses))]
            if all(0 <= d < 0.1 for d in imps):
                print(f"Converged at step {s} (3× improvement < 0.1)", flush=True)
                break
    if s > 0 and s % 500 == 0:
        torch.save({"model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
                    "step": s, "c2i": c2i, "i2c": {v:k for k,v in c2i.items()}}, CKPT)
        print(f"  checkpoint saved step {s}", flush=True)

torch.save({"model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
            "step": s, "c2i": c2i, "i2c": {v:k for k,v in c2i.items()}}, CKPT)
print(f"Final checkpoint saved at step {s}", flush=True)

model_v4_path = "E:/claude/myllm/model_v4.pt"
torch.save({"model_state": model.state_dict(), "c2i": c2i,
            "i2c": {v:k for k,v in c2i.items()}}, model_v4_path)
print(f"Model saved → {model_v4_path}", flush=True)
