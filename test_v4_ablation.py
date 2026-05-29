"""
Test v4 ablation variants. Tests both forward() and step() generation.
"""
import sys, os, math, random
os.environ["USERNAME"] = "user"
os.environ["USER"] = "user"
import torch, torch.nn as nn, torch.nn.functional as F

VARIANT = os.environ.get("VARIANT", "base")
HAS_A = 'A' in VARIANT
HAS_B = 'B' in VARIANT
HAS_C = 'C' in VARIANT
HAS_D = 'D' in VARIANT

CKPT = f"E:/claude/myllm/model_v4_ablation_{VARIANT}.pt"
if not os.path.exists(CKPT):
    print(f"No model found: {CKPT}")
    sys.exit(1)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
c2i = ck["c2i"]; i2c = {int(k):v for k,v in ck["i2c"].items()}
VOCAB = len(c2i); MAXLEN = 100

def enc(s): return [c2i.get(c, 1) for c in s]
def dec(t): return "".join(i2c.get(int(i), "?") for i in t if int(i) not in {0,1,2,3})

class Block(nn.Module):
    def __init__(self, bidx=0):
        super().__init__()
        D=256;h=4;dh=64;self.bidx=bidx;self.D=D
        self.an=nn.LayerNorm(D);self.qp=nn.Linear(D,D);self.kp=nn.Linear(D,D)
        self.vp=nn.Linear(D,D);self.op=nn.Linear(D,D);self.h=h;self.dh=dh
        self.fn=nn.LayerNorm(D);self.ff=nn.Sequential(nn.Linear(D,D*2),nn.GELU(),nn.Linear(D*2,D))
        if HAS_B: self.hp = nn.Linear(D, D)
        if HAS_C:
            self.cq = nn.Linear(D, D); self.ck = nn.Linear(D, D); self.cv = nn.Linear(D, D)
        if HAS_D:
            self.gru_fast = nn.GRUCell(D, D); self.gru_slow = nn.GRUCell(D, D)
        else: self.gru = nn.GRUCell(D, D)

    def forward(self, x):
        L,B,D=x.shape;H,Dh=self.h,self.dh
        xn=self.an(x);Q=self.qp(xn).view(L,B,H,Dh).permute(1,2,0,3)
        K=self.kp(xn).view(L,B,H,Dh).permute(1,2,0,3);V=self.vp(xn).view(L,B,H,Dh).permute(1,2,0,3)
        s=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(Dh)
        mask=torch.triu(torch.full((L,L),float('-inf'),device=x.device),diagonal=1)
        a=F.softmax(s+mask,dim=-1)
        C=torch.matmul(a,V).permute(2,0,1,3).reshape(L,B,D)
        x=x+self.op(C);x=x+self.ff(self.fn(x));return x

    def step(self, x, h, h_buffer=None, h_slow=None):
        D=self.D
        xn = self.an(x).squeeze(0)
        if HAS_C and h_buffer is not None and len(h_buffer) > 0:
            buf = torch.stack(h_buffer, dim=0)  # [K, B, D]
            q = self.cq(xn).unsqueeze(0); k = self.ck(buf); v = self.cv(buf)
            a = F.softmax(torch.matmul(q, k.transpose(-2,-1))/math.sqrt(D), dim=-1)
            xn = xn + (a * v).sum(dim=0)
        if HAS_D:
            h_fast = self.gru_fast(xn, h)
            if h_slow is None: h_slow = torch.zeros_like(h)
            h_slow_new = self.gru_slow(h_fast, h_slow)
            h_new = h_fast + h_slow_new
        else:
            h_new = self.gru(xn, h)
            h_slow_new = None
        if HAS_B: x = x.squeeze(0) + self.op(h_new) + self.hp(h_new)
        else: x = x.squeeze(0) + self.op(h_new)
        x = x + self.ff(self.fn(x))
        if HAS_D: return x.unsqueeze(0), h_new, h_slow_new
        return x.unsqueeze(0), h_new

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
    def forward(self,tok):
        L,B=tok.shape;x=self.emb(tok)+self.pe[:L].unsqueeze(1)
        x=self.proj(x)
        for b in self.blocks: x=b(x)
        return self.fc(self.norm(x))
    def step_generate(self,tok,h,h_buffer=None,h_slow_list=None):
        x=self.emb(tok)+self.pe[:1];x=self.proj(x)
        if HAS_D:
            if h_slow_list is None: h_slow_list = [None]*3
            for i,b in enumerate(self.blocks):
                h_i = h[i] if HAS_A else h
                x, h_i, h_slow_list[i] = b.step(x, h_i, h_buffer, h_slow_list[i])
                if HAS_A: h[i] = h_i
                else: h = h_i
            return self.fc(self.norm(x)), h, h_slow_list
        else:
            for i,b in enumerate(self.blocks):
                h_i = h[i] if HAS_A else h
                x, h_i = b.step(x, h_i, h_buffer)
                if HAS_A: h[i] = h_i
                else: h = h_i
            return self.fc(self.norm(x)), h

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
model.load_state_dict(ck["model_state"], strict=False)
model.eval()
print(f"\nVariant {VARIANT} | {sum(p.numel() for p in model.parameters()):,} params | {device}")

prompts = ["今天天气","春风吹","明月","登高","朝辞"]

def gen_forward(prompt, max_new=30, temperature=0.8, top_k=20):
    t = [2] + enc(prompt)
    for _ in range(max_new):
        logits = model(torch.tensor(t, dtype=torch.long).unsqueeze(1).to(device))
        logits = logits[-1, 0] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        if nxt in (0, 3): break
        t.append(nxt)
    return dec(t[1:])

def gen_step(prompt, max_new=30, temperature=0.8, top_k=20):
    t = [2] + enc(prompt)
    if HAS_A: h = [torch.zeros(1,256).to(device) for _ in range(3)]
    else: h = torch.zeros(1,256).to(device)
    h_buffer = []; h_slow = [None]*3 if HAS_D else None
    for _ in range(max_new):
        inp = torch.tensor([t[-1]], dtype=torch.long).unsqueeze(1).to(device)
        if HAS_D: logits, h, h_slow = model.step_generate(inp, h, h_buffer, h_slow)
        else: logits, h = model.step_generate(inp, h, h_buffer)
        logits = logits[-1,0]/temperature
        if top_k>0:
            v,_=torch.topk(logits,min(top_k,logits.size(-1)))
            logits[logits<v[-1]]=float('-inf')
        nxt=torch.multinomial(F.softmax(logits,dim=-1),1).item()
        if nxt in(0,3): break
        t.append(nxt)
        if HAS_C:
            h_buffer.append(h[0].detach() if HAS_A else h.detach())
            if len(h_buffer)>4: h_buffer.pop(0)
    return dec(t[1:])

print("--- forward() ---")
for p in prompts:
    print(f"  '{p}' -> '{gen_forward(p)}'")
print("--- step() ---")
for p in prompts:
    print(f"  '{p}' -> '{gen_step(p)}'")
print()
