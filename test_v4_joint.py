"""
Test v4 - D2: Joint model. Tests BOTH forward() and step() generation.
"""
import sys, os, math
os.environ["USERNAME"] = "user"
os.environ["USER"] = "user"
import torch, torch.nn as nn, torch.nn.functional as F

CKPT = "E:/claude/myllm/model_v4_joint.pt"
if not os.path.exists(CKPT):
    print("No Joint model found!")
    sys.exit(1)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
c2i = ck["c2i"]; i2c = {int(k):v for k,v in ck["i2c"].items()}
VOCAB = len(c2i); MAXLEN = 100

def enc(s): return [c2i.get(c, 1) for c in s]
def dec(t): return "".join(i2c.get(int(i), "?") for i in t if int(i) not in {0,1,2,3})

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        D=256;h=4;dh=64
        self.an=nn.LayerNorm(D);self.qp=nn.Linear(D,D);self.kp=nn.Linear(D,D)
        self.vp=nn.Linear(D,D);self.op=nn.Linear(D,D);self.h=h;self.dh=dh
        self.fn=nn.LayerNorm(D);self.ff=nn.Sequential(nn.Linear(D,D*2),nn.GELU(),nn.Linear(D*2,D))
        self.gp=nn.Linear(D,D);self.sp=nn.Linear(D,D)
    def forward(self,x):
        L,B,D=x.shape;H,Dh=self.h,self.dh
        xn=self.an(x);Q=self.qp(xn).view(L,B,H,Dh).permute(1,2,0,3)
        K=self.kp(xn).view(L,B,H,Dh).permute(1,2,0,3);V=self.vp(xn).view(L,B,H,Dh).permute(1,2,0,3)
        s=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(Dh)
        mask=torch.triu(torch.full((L,L),float('-inf'),device=x.device),diagonal=1)
        a=F.softmax(s+mask,dim=-1)
        C=torch.matmul(a,V).permute(2,0,1,3).reshape(L,B,D)
        x=x+self.op(C);x=x+self.ff(self.fn(x));return x
    def step(self,x,h):
        B,H,Dh=x.size(1),self.h,self.dh
        xn=self.an(x);g=self.gp(xn).view(B,H,Dh);s=self.sp(xn).view(B,H,Dh)
        gate=torch.sigmoid(g);h_new=h.view(B,H,Dh)*gate+s*(1-gate)
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
        return self.fc(self.norm(x))
    def step_generate(self,tok,h):
        x=self.emb(tok)+self.pe[:1];x=self.proj(x)
        for b in self.blocks: x,h=b.step(x,h)
        return self.fc(self.norm(x)),h

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
model.load_state_dict(ck["model_state"], strict=False)
model.eval()
print(f"Loaded Joint model | Device: {device} | Params: {sum(p.numel() for p in model.parameters()):,}")

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
    h = torch.zeros(1, 256).to(device)
    for _ in range(max_new):
        logits, h = model.step_generate(torch.tensor([t[-1]], dtype=torch.long).unsqueeze(1).to(device), h)
        logits = logits[-1, 0] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        if nxt in (0, 3): break
        t.append(nxt)
    return dec(t[1:])

print("\n" + "="*60)
print("Joint model generations")
print("="*60)

prompts = ["今天天气","我爱","春风吹","明月","人工智能","大雄",
           "静夜思","春晓","登高","望月"]
print("\n--- forward() path (causal attention) ---")
for p in prompts:
    out = gen_forward(p)
    print(f"  '{p}' -> '{out}'")

print("\n--- step() path (gated state) ---")
for p in prompts:
    out = gen_step(p)
    print(f"  '{p}' -> '{out}'")

print("\nDone.")
