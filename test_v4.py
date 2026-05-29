"""
Test v4 trained model: generation + interpolation.
Loads checkpoint_v4.pt and runs inference using forward() and step().
"""
import sys, os, re, json, glob, math
os.environ["USERNAME"] = "user"
os.environ["USER"] = "user"
import torch, torch.nn as nn, torch.nn.functional as F

CKPT = "E:/claude/myllm/checkpoint_v4.pt"
if not os.path.exists(CKPT):
    print("No checkpoint found!")
    sys.exit(1)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
c2i = ck["c2i"]
i2c = {int(k):v for k,v in ck["i2c"].items()}
VOCAB = len(c2i)
MAXLEN = 100

def enc(s):
    return [c2i.get(c, 1) for c in s]

def dec(t):
    return "".join(i2c.get(int(i), "?") for i in t if int(i) not in {0,1,2,3})

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
        self.register_buffer('pe', pe)
        self.proj=nn.Linear(128,256)
        self.blocks=nn.ModuleList([Block() for _ in range(3)])
        self.norm=nn.LayerNorm(256);self.fc=nn.Linear(256,VOCAB)
    def forward(self,tok):
        L,B=tok.shape;x=self.emb(tok)+self.pe[:L].unsqueeze(1)
        x=self.proj(x)
        for b in self.blocks: x=b(x)
        return self.fc(self.norm(x))[:-1]
    def encode(self,tok):
        L,B=tok.shape;x=self.emb(tok)+self.pe[:L].unsqueeze(1)
        x=self.proj(x)
        for b in self.blocks: x=b(x)
        h=self.norm(x)[-1];return self.fc(self.norm(x)),h
    def generate_step(self,tok,h):
        x=self.emb(tok)+self.pe[:1];x=self.proj(x)
        for b in self.blocks: x,h=b.step(x,h)
        return self.fc(self.norm(x)),h

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
model.load_state_dict(ck["model_state"], strict=False)
model.eval()
print(f"Loaded (step {ck.get('step','?')}) | Device: {device} | Params: {sum(p.numel() for p in model.parameters()):,}")

# ─── forward() generation ───
print("\n" + "="*60)
print("forward() generation (full attention each step)")
print("="*60)

def gen_fwd(prompt, max_new=30):
    t = [2]+enc(prompt)
    for _ in range(max_new):
        logits = model(torch.tensor(t,dtype=torch.long).unsqueeze(1).to(device))
        nxt = logits[-1,0].argmax().item()
        if nxt in (0,3): break
        t.append(nxt)
    return dec(t[1:])

for p in ["今天天气","我爱","春风吹","明月","人工智能","大雄"]:
    print(f"  '{p}' → '{gen_fwd(p)}'")

# ─── step() generation ───
print("\n" + "="*60)
print("step() generation (gated state, never explicitly trained)")
print("="*60)

def gen_step(prompt, max_new=30):
    t = torch.tensor([2]+enc(prompt),dtype=torch.long).unsqueeze(1).to(device)
    with torch.no_grad(): _,h = model.encode(t)
    tok = t[-1:]
    result = list(prompt)
    for _ in range(max_new):
        logits,h = model.generate_step(tok,h)
        nxt = logits[0].argmax().item()
        if nxt in (0,3): break
        result.append(i2c.get(nxt,"?"))
        tok = torch.tensor([[nxt]],dtype=torch.long).to(device)
    return "".join(result)

for p in ["今天天气","我爱","春风吹","明月","人工智能"]:
    print(f"  '{p}' → '{gen_step(p)}'")

# ─── Interpolation ───
print("\n" + "="*60)
print("Latent interpolation")
print("="*60)

def get_z(text):
    t = torch.tensor([2]+enc(text),dtype=torch.long).unsqueeze(1).to(device)
    with torch.no_grad():
        x=model.emb(t)+model.pe[:t.size(0)];x=model.proj(x)
        for b in model.blocks: x=b(x)
        return model.norm(x)[-1,0]

for a,b in [("春天","冬天"),("大雄","宜静"),("今天","未来")]:
    z1,z2=get_z(a),get_z(b)
    print(f"  '{a}' → '{b}'")
    for i in range(11):
        al=i/10;z=z1*(1-al)+z2*al
        p=F.softmax(model.fc(z),dim=-1)
        print(f"  α={al:.2f}  '{i2c.get(p.argmax().item(),'.')}'  conf={p.max().item():.3f}")

print("\nDone.")
