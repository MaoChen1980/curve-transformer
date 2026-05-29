"""
Test v4 - C: Pure GRU generation.
"""
import sys, os, math
os.environ["USERNAME"] = "user"
os.environ["USER"] = "user"
import torch, torch.nn as nn, torch.nn.functional as F

CKPT = "E:/claude/myllm/model_v4_gru.pt"
if not os.path.exists(CKPT):
    print("No GRU model found!")
    sys.exit(1)

ck = torch.load(CKPT, map_location="cpu", weights_only=False)
c2i = ck["c2i"]; i2c = {int(k):v for k,v in ck["i2c"].items()}
VOCAB = len(c2i); MAXLEN = 100

def enc(s): return [c2i.get(c, 1) for c in s]
def dec(t): return "".join(i2c.get(int(i), "?") for i in t if int(i) not in {0,1,2,3})

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
model = Model().to(device)
model.load_state_dict(ck["model_state"], strict=False)
model.eval()
print(f"Loaded GRU model | Device: {device} | Params: {sum(p.numel() for p in model.parameters()):,}")

def gen(prompt, max_new=30, temperature=0.8, top_k=20):
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

print("\n" + "="*60)
print("GRU generations")
print("="*60)

prompts = ["今天天气","我爱","春风吹","明月","人工智能","大雄",
           "静夜思","春晓","登高","望月"]
for p in prompts:
    out = gen(p)
    print(f"  '{p}' -> '{out}'")

print("\nDone.")
