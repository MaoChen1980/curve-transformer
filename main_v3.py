"""
Curve Model v3 — Clean Transformer (unified dimension)
=======================================================
Fixed: single unified dimension D throughout.
Attention: D→D projections, multi-head, residual + norm + FFN
No more E_in/E_out confusion.

Key idea (from user):
  1. Parallel training  ← PyTorch batches natively parallel
  2. Bounded context   ← attention handles long range, but we cap context
  3. Large corpus       ← 3000 sentences, varied templates
"""

import random, math, os, torch, torch.nn as nn, torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════════════
# 1. Data
# ══════════════════════════════════════════════════════════════════════════════
SUBJECTS = ["我","你","他","她","它","我们","你们","他们",
    "春天","夏天","秋天","冬天","天空","大地","海洋","山川",
    "时间","空间","宇宙","银河","地球","月亮","太阳","星星",
    "森林","草原","沙漠","河流","湖泊","大海","山顶","平原",
    "科学家","艺术家","作家","音乐家","画家","老师","医生",
    "孩子","老人","青年","少年","女孩","男孩","朋友","家人",
    "人工智能","机器人","量子计算机","5G网络","区块链","云计算"]
PREDICATES = ["是","有","在","爱","喜欢","想要","需要","知道","认为",
    "看见","听见","感觉","记得","想象","创造","发现",
    "发展","改变","成长","学习","探索","追求","实现","完成",
    "感受","理解","表达","记录","保护","建设","改进","提升","突破",
    "照耀","滋润","养育","陪伴","支持","帮助",
    "唱歌","跳舞","绘画","写作","旅行","阅读","思考"]
OBJECTS = ["世界上","时间里","宇宙中","大自然","社会中","历史上",
    "和平","自由","希望","梦想","知识","智慧","爱情","友情",
    "亲情","健康","财富","幸福","快乐","美丽","真理","正义",
    "科技","艺术","文化","教育","经济","政治","环境","资源",
    "天空","大地","海洋","山川","森林","草原","沙漠","河流",
    "阳光","月光","星光","书","音乐","电影","画","诗","歌",
    "家","城市","村庄","国家","学校","公园","图书馆",
    "今天","明天","昨天","现在","过去","未来"]
MODIFIERS = ["非常","特别","十分","真的","确实","当然",
    "慢慢地","静静地","轻轻地","悄悄地","渐渐地"]
TEMPLATES = [
    "{S}在{O}","{S}是{O}","{S}有{O}","{S}爱{O}","{S}喜欢{O}",
    "{S}需要{O}","{S}知道{O}","{S}想要{O}","{S}认为{O}","{S}创造{O}",
    "{M}{S}是{O}","{S}和{S}在{O}","{S}与{S}一起{O}",
    "关于{S}的{O}","{M}的{O}是{S}",
    "{S}说{O}","{S}看{O}","{S}听{O}","{S}感到{O}",
    "{S}正在{O}","{S}已经{O}","{S}将要{O}"]

def gen_corpus(n=3000):
    out = []
    for _ in range(n):
        t = random.choice(TEMPLATES)
        s = (t.replace("{S}", random.choice(SUBJECTS))
              .replace("{O}", random.choice(OBJECTS))
              .replace("{M}", random.choice(MODIFIERS)))
        if 4 <= len(s) <= 16 and s not in out:
            out.append(s)
    out += [
        "今天天气真好","我想去看看外面的世界","风吹过来很凉爽",
        "星星在天上闪烁","河水向东流去","春天来了花开了",
        "你是我最重要的人","每天都要保持微笑",
        "阳光照在窗台上","雨后的空气清新",
        "我爱自然和自由","夜晚的城市灯火通明",
        "鸟儿在枝头唱歌","山很高云很白",
        "海浪拍打着沙滩","月光洒在湖面上",
        "未来充满希望","梦想就在前方",
        "努力就会有收获","坚持就是胜利",
        "勇敢面对困难","世界很大很美好",
        "知识就是力量","时间是宝贵的",
        "健康是最大的财富","友情珍贵难得",
        "爱情让人成长","亲情温暖人心",
        "音乐让人放松","读书让人明智",
        "旅行让人开阔","绘画表达情感",
        "诗歌抒发心意","人工智能改变世界",
        "量子计算突破","网络安全重要",
        "爱是最美的语言","家是最温暖的港湾",
        "朋友是一面镜子","微笑是最好的名片",
        "学习改变命运","感恩让人幸福",
        "世界因你而精彩","珍惜当下的每一刻",
    ]
    return list(set(out))

CORPUS = gen_corpus(3000)
random.shuffle(CORPUS)

chars = set("".join(CORPUS))
c2i = {"<PAD>":0,"<UNK>":1}
for c in sorted(chars): c2i[c] = len(c2i)
i2c = {v:k for k,v in c2i.items()}
VOCAB, MAXLEN = len(c2i), 20

def enc(s):
    t = [c2i.get(c,1) for c in s]
    return torch.tensor(t + [0]*(MAXLEN-len(t)), dtype=torch.long)

# PE tensor: (MAXLEN, D)
def make_pe(D, L=MAXLEN):
    p = torch.zeros(L, D)
    for i in range(L):
        for j in range(D):
            p[i,j] = math.sin(i/10000**(2*j/D)) if j%2==0 else math.cos(i/10000**(2*j/D))
    return p

# ══════════════════════════════════════════════════════════════════════════════
# 2. Model — Unified dimension D throughout
# ══════════════════════════════════════════════════════════════════════════════
class Block(nn.Module):
    """Standard Transformer block: MHA → AddNorm → FFN → AddNorm"""
    def __init__(self, D, heads=4):
        super().__init__()
        Dh = D // heads
        self.attn_norm = nn.LayerNorm(D)
        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, D)
        self.v_proj = nn.Linear(D, D)
        self.o_proj = nn.Linear(D, D)
        self.heads = heads
        self.Dh = Dh

        self.ffn_norm = nn.LayerNorm(D)
        self.ffn = nn.Sequential(
            nn.Linear(D, D*2),
            nn.GELU(),
            nn.Linear(D*2, D),
        )

    def forward(self, x):
        """x: (L, B, D) → (L, B, D)"""
        L, B, D = x.shape
        H, Dh = self.heads, self.Dh

        # MHA with pre-norm
        x_norm = self.attn_norm(x)
        Q = self.q_proj(x_norm).view(L, B, H, Dh).permute(1, 2, 0, 3)   # (B,H,L,Dh)
        K = self.k_proj(x_norm).view(L, B, H, Dh).permute(1, 2, 0, 3)
        V = self.v_proj(x_norm).view(L, B, H, Dh).permute(1, 2, 0, 3)

        # Attention: (B,H,L,L) ← (B,H,L,Dh) @ (B,H,Dh,L)
        attn = F.softmax(torch.matmul(Q, K.transpose(-2,-1))/math.sqrt(Dh), dim=-1)
        C = torch.matmul(attn, V)                                   # (B,H,L,Dh)
        C = C.permute(2, 0, 1, 3).reshape(L, B, D)                  # (L,B,D)
        x = x + self.o_proj(C)                                       # residual

        # FFN with pre-norm
        x = x + self.ffn(self.ffn_norm(x))
        return x

    def step(self, x, h):
        """x: (B, D), h: (B, D) → (B, D), (B, D) — recurrent for generation"""
        B = x.size(0)
        H, Dh = self.heads, self.Dh

        x_n = self.attn_norm(x)
        q = self.q_proj(x_n).view(B, H, Dh)
        v = self.v_proj(x_n).view(B, H, Dh)

        gate = torch.sigmoid(q)
        h_h = h.view(B, H, Dh)
        h_new = h_h * gate + v * (1 - gate)
        h_new = h_new.reshape(B, -1)      # (B, D)

        x = x + self.o_proj(h_new)
        x = x + self.ffn(self.ffn_norm(x))
        return x, h_new


class CurveTransformer(nn.Module):
    """Embed → PE → Blocks → Norm → FC"""
    def __init__(self, vocab, E=64, D=256, n_layers=2, heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, E, padding_idx=0)
        self.pe = make_pe(E)                     # (MAXLEN, E)
        self.proj = nn.Linear(E, D)              # E → D
        self.blocks = nn.ModuleList([Block(D, heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(D)
        self.fc = nn.Linear(D, vocab)

    def encode(self, tok):
        """tok: (L, B) → h: (L, B, D)"""
        L, B = tok.shape
        x = self.embed(tok) + self.pe[:L].unsqueeze(1)   # (L, B, E)
        x = self.proj(x)                                  # (L, B, D)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward(self, tok):
        h = self.encode(tok)
        return self.fc(h)[:-1], h                       # (L-1, B, V)

    def init_h(self, B, dev):
        return torch.zeros(B, self.blocks[0].heads * self.blocks[0].Dh, device=dev)

    def gen_step(self, tok, h):
        x = self.embed(tok) + self.pe[0]                # (B, E)
        x = self.proj(x)                                 # (B, D)
        for blk in self.blocks:
            x, h = blk.step(x, h)
        return self.fc(self.norm(x)), h                 # (B, V)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Training
# ══════════════════════════════════════════════════════════════════════════════
CKPT = "E:/claude/myllm/checkpoint_v3.pt"

def train(steps=8000, lr=2e-3, B=32, print_every=300):
    dev = torch.device("cpu")
    model = CurveTransformer(VOCAB, E=64, D=256, n_layers=2, heads=4).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")
    data = torch.stack([enc(s) for s in CORPUS]).to(dev)

    start = 0
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer_state"])
        start = ck["step"] + 1
        print(f"Continue from step {start}")

    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Corpus: {len(CORPUS)} sentences, vocab={VOCAB}\n")

    for s in range(start, steps):
        model.train()
        opt.zero_grad()

        idx = torch.randint(0, len(CORPUS), (B,))
        tok = data[idx].transpose(0, 1)    # (L, B)

        logits, _ = model(tok)
        tgt = tok[1:].contiguous()
        loss = crit(logits.reshape(-1, VOCAB), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if s % print_every == 0:
            print(f"step {s:5d} | loss {loss.item():.4f}")

        if s > 0 and s % 500 == 0:
            torch.save({"model_state": model.state_dict(),
                       "optimizer_state": opt.state_dict(),
                       "step": s, "c2i": c2i, "i2c": i2c}, CKPT)
            print(f"💾 ckpt step {s}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4. Generation
# ══════════════════════════════════════════════════════════════════════════════
def generate(model, prompt, max_new=15, temp=0.9, rep=2.5):
    dev = next(model.parameters()).device
    model.eval()
    tok = torch.tensor([c2i.get(c,1) for c in prompt], dtype=torch.long).to(dev)

    # Encode prompt: embed → proj → blocks → norm
    with torch.no_grad():
        L = tok.size(0)
        x = model.embed(tok.unsqueeze(1)) + model.pe[:L].unsqueeze(1)
        x = model.proj(x)
        for blk in model.blocks:
            x = blk(x)
        x = model.norm(x)
        # h: last position, last block, (D,)
        h = x[-1, 0]    # (D,)

    result = list(prompt)
    for _ in range(max_new):
        with torch.no_grad():
            logits, h = model.gen_step(tok[-1:], h)
            p = F.softmax(logits / temp, dim=-1).squeeze(0)
            for c in set(result[-5:]):
                i = c2i.get(c, -1)
                if i >= 0: p[i] = p[i] ** 0.7
            nxt = p.argmax().item()
        if nxt == 0: break
        result.append(i2c.get(nxt, ""))
        tok = torch.cat([tok, tok.new_tensor([nxt])])

    return "".join(result)


def traverse(model, a, b, steps=10):
    dev = next(model.parameters()).device
    model.eval()
    def _z(p):
        t = torch.tensor([c2i.get(c,1) for c in p], dtype=torch.long).to(dev)
        x = model.embed(t).unsqueeze(1) + model.pe[:len(t)].unsqueeze(1)
        x = model.proj(x)
        for blk in model.blocks:
            x = blk(x)
        return model.norm(x)[-1, 0]
    z1, z2 = _z(a), _z(b)
    print(f"\n  Traverse: '{a}' → '{b}'")
    for i in range(steps+1):
        al = i/steps
        z = z1*(1-al) + z2*al
        p = F.softmax(model.fc(z), dim=-1)
        c = i2c.get(p.argmax().item(), "·")
        conf = p.max().item()
        print(f"  α={al:4.1f}  '{c}'  conf={conf:.3f}  {'█'*int(conf*20)}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("Curve v3 — Transformer + Large Corpus")
    print(f"Corpus: {len(CORPUS)} sentences, vocab={VOCAB}")
    print("="*60)
    model = train(steps=8000, lr=2e-3, B=32, print_every=300)

    print("\n=== Generation ===")
    for p in ["今天天气","我爱","宇宙","健康","人工智能"]:
        g = generate(model, p, rep=2.5)
        print(f"  '{p}' → '{g}'")

    print("\n=== Interpolation ===")
    traverse(model, "今天", "未来")

    torch.save({"model_state": model.state_dict(), "c2i": c2i, "i2c": i2c},
               "E:/claude/myllm/model_v3.pt")
    print("\n✅ Saved → model_v3.pt")