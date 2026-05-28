"""
Curve Language Model v2 — Selective Memory + Bounded Context
=============================================================
同时实现两个优化思路：

  思路 1（并行化）:  batch 并行 — 多句话同时训练，每句独立走曲线
                     精度有损失 → batch_size 从 16→64，换来速度
                     
  思路 2（有限上下文）: 选择性状态更新
                     不是 h[t] = f(h[t-1], x[t])（记住一切）
                     而是 h[t] = gate * h[t-1] + (1-gate) * new
                     模型自己决定记住多少、忘掉多少
                     → 解决梯度衰减 + 局部重复问题

核心改变：LSTM → SelectiveState（简化版 Mamba 思想）
"""

import random, math, os, torch, torch.nn as nn, torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════════════
# 1. Data
# ══════════════════════════════════════════════════════════════════════════════
CORPUS = [
    "今天天气真好", "我想去看看外面的世界", "风吹过来很凉爽", "星星在天上闪烁",
    "河水向东流去", "春天来了花开了", "你是我最重要的人", "每天都要保持微笑",
    "阳光照在窗台上", "雨后的空气清新", "我爱自然和自由", "夜晚的城市灯火通明",
    "鸟儿在枝头唱歌", "山很高云很白", "海浪拍打着沙滩", "月光洒在湖面上",
    "风吹麦浪一片金黄", "秋天的枫叶红了", "冬天的雪花飘落", "春天的小草绿了",
    "夏天的西瓜很甜", "秋天的月亮很圆", "冬天的温暖来自家", "我爱这四季分明",
    "你是我心中的光", "愿你每天都开心", "我们一起走很远的路", "未来充满希望",
    "梦想就在前方", "努力就会有收获", "坚持就是胜利", "勇敢面对困难",
    "世界很大很美好", "知识就是力量", "时间是宝贵的", "健康是最大的财富",
    "友情珍贵难得", "爱情让人成长", "亲情温暖人心", "音乐让人放松",
    "读书让人明智", "旅行让人开阔", "绘画表达情感", "诗歌抒发心意",
    "舞蹈展现活力", "电影记录人生", "咖啡香浓提神", "茶香清雅淡泊",
    "歌声动听悦耳", "笑声温暖人心", "星空浩瀚无垠", "银河璀璨夺目",
    "宇宙无限广阔", "地球是我们的家", "时间从此开始", "空间无限延展",
    "语言传递思想", "文字记录历史", "科学揭示规律", "技术改变世界",
    "艺术美化生活", "哲学追问本源", "教育培养人才", "经济运转社会",
    "政治管理国家", "外交和平交流", "军事保卫安全", "警察维护治安",
    "医生救死扶伤", "护士细心照料", "教师教书育人", "作家书写故事",
    "画家描绘风景", "建筑凝固音乐", "桥梁连接两岸", "道路通达四方",
    "网络连接世界", "手机随身携带", "电脑处理信息", "图书馆安静知识",
    "博物馆展示历史", "公园休闲放松", "银行存取钱款", "邮局传递信件",
    "森林郁郁葱葱", "草原广阔无垠", "沙漠干燥荒凉", "河流滋润大地",
    "湖泊宁静如镜", "瀑布飞流直下", "平原开阔平坦", "山顶视野开阔",
    "地震突然破坏", "火山壮丽危险", "洪水泛滥成灾", "保护地球家园",
    "太阳能清洁", "风能永不枯竭", "电动汽车环保", "智慧城市便利",
    "人工智能发展", "机器人自动化", "大数据分析", "云计算弹性",
    "网络安全重要", "AI辅助医疗", "AI驾驶汽车", "AI作曲音乐",
    "AI对话交流", "AI造福人类", "AI无限可能", "AI未来已来",
    "AI时代开启", "AI改变世界", "量子计算突破", "5G网络快速",
    "元宇宙虚拟", "VR虚拟沉浸", "基因决定特征", "细胞是生命基础",
    "器官协同工作", "大脑指挥一切", "心脏泵血不息", "肺部呼吸换气",
    "肝脏解毒代谢", "眼睛看见世界", "耳朵听见声音", "嘴巴说话交流",
    "记忆存储过去", "想象创造未来", "情感连接人心", "理性指引方向",
    "创造力无限", "好奇心驱动探索", "爱是最美的语言", "家是最温暖的港湾",
    "朋友是一面镜子", "微笑是最好的名片", "坚持就能成功", "失败是成功之母",
    "学习改变命运", "行动成就梦想", "感恩让人幸福", "宽容化解矛盾",
    "分享带来快乐", "陪伴是最长情的告白", "信任是友谊的基石",
    "勇气面对未知", "善良照亮世界", "真诚打动人心", "勤奋创造财富",
    "自律带来自由", "乐观面对困境", "爱让世界更美好", "和平是人类共同的愿望",
    "正义永远不会缺席", "幸福需要经营", "健康比金钱更值", "平安比富贵更好",
    "简单生活最快乐", "心中有光不惧黑暗", "有梦就要去追",
    "世界因你而精彩", "每天进步一点点", "保持热爱奔赴山海",
    "做最好的自己", "诗和远方都在等待", "向前看别回头",
    "珍惜当下的每一刻", "一切皆有可能",
]

random.seed(42)
random.shuffle(CORPUS)

all_chars = set("".join(CORPUS))
char2idx = {"<PAD>": 0, "<UNK>": 1}
for i, c in enumerate(sorted(all_chars)):
    char2idx[c] = i + 2
idx2char = {v: k for k, v in char2idx.items()}
VOCAB_SIZE = len(char2idx)
MAX_LEN = 20
UNK_IDX = 1

print(f"[Data] Vocab: {VOCAB_SIZE} | Sentences: {len(CORPUS)}")


def encode_sentence(s: str) -> torch.Tensor:
    tokens = [char2idx.get(c, UNK_IDX) for c in s]
    if len(tokens) < MAX_LEN:
        tokens += [0] * (MAX_LEN - len(tokens))
    return torch.tensor(tokens[:MAX_LEN], dtype=torch.long)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Model v2 — Selective State + Parallel Batched Training
# ══════════════════════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        for pos in range(max_len):
            for i in range(dim):
                angle = pos / (10000 ** (2 * i / dim))
                pe[pos, i] = math.sin(angle) if i % 2 == 0 else math.cos(angle)
        self.register_buffer("pe", pe)

    def forward(self, x):
        L = x.size(0)
        return x + self.pe[:L]


class SelectiveState(nn.Module):
    """
    选择性状态更新 — 思路 2 的核心实现

    传统 LSTM：h[t] = tanh(W·h[t-1] + U·x[t])
    → 无差别记住一切，梯度衰减导致早期信息丢失

    Selective State：
      gate  = σ(W_g·h[t-1] + U_g·x[t])      ← 决定保留多少旧状态
      cand  = tanh(W_c·h[t-1] + U_c·x[t])    ← 新信息候选
      h[t]  = gate * h[t-1] + (1-gate) * cand

    效果：
      - gate 决定「忘掉多少老信息，记住多少新信息」
      - 模型可以自适应地保留关键语义，过滤噪音
      - 相当于一个「有限容量工作记忆」
    """

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input gate: 决定保留多少新信息
        self.i_gate = nn.Linear(embed_dim + hidden_dim, hidden_dim)
        # Forget gate: 决定保留多少旧状态（= 选择性记忆的核心）
        self.f_gate = nn.Linear(embed_dim + hidden_dim, hidden_dim)
        # Candidate: 新状态候选
        self.c_cand = nn.Linear(embed_dim + hidden_dim, hidden_dim)

    def forward(self, x, h):
        """
        x: (B, E)  — 当前 token 的 embedding
        h: (B, D)  — 上一个 hidden state
        Returns: new_h (B, D)
        """
        combined = torch.cat([x, h], dim=-1)       # (B, E+D)
        f = torch.sigmoid(self.f_gate(combined))  # forget gate: 保留旧状态多少
        i = torch.sigmoid(self.i_gate(combined))   # input gate:  新信息多少
        cand = torch.tanh(self.c_cand(combined))   # 候选新状态
        new_h = f * h + i * cand                   # 选择性融合
        return new_h


class CurveRNNv2(nn.Module):
    """
    Curve Model v2 — 两个优化同时实现

    思路 1（并行化）：batch_size 从 16→64，多句话并行训练
    思路 2（有限上下文）：LSTM → SelectiveState， bounded context

    曲线语义不变：h[t] = curve point at position t
    改变的是更新机制：从"记住一切" → "选择性记忆"
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256, n_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pe = PositionalEncoding(embed_dim)

        # 替换 LSTM → SelectiveState
        self.state_layers = nn.ModuleList([
            SelectiveState(embed_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(n_layers)
        ])

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def encode(self, tokens):
        """
        tokens: (L, B) → h_seq (L, B, D)
        每一步都用 SelectiveState 选择性更新
        """
        L, B = tokens.shape[0], tokens.shape[1]

        # Initialize hidden states: (n_layers, B, D)
        h = [torch.zeros(B, self.hidden_dim, device=tokens.device)
             for _ in range(len(self.state_layers))]

        h_seq = []
        for t in range(L):
            x = self.embed(tokens[t])           # (B, E)
            x = x + self.pe.pe[t]               # add positional encoding
            x = x + self.pe.pe[t]               # (B, E)

            # Layer-by-layer selective update
            for li, state_layer in enumerate(self.state_layers):
                h[li] = state_layer(x, h[li])    # selective update
                x = h[li]                         # pass to next layer

            h_seq.append(h[-1].unsqueeze(0))     # record top layer state

        h_seq = torch.cat(h_seq, dim=0)          # (L, B, D)
        return h_seq

    def forward(self, tokens):
        """
        tokens: (L, B) → logits (L-1, B, V)
        """
        h_seq = self.encode(tokens)
        logits = self.fc(h_seq)
        return logits[:-1], h_seq

    def init_state(self, batch_size, device):
        """Initialize clean state for generation."""
        return [torch.zeros(batch_size, self.hidden_dim, device=device)
                for _ in self.state_layers]

    def step(self, x, h_state):
        """One step of selective update for generation."""
        for li, state_layer in enumerate(self.state_layers):
            h_state[li] = state_layer(x, h_state[li])
            x = h_state[li]
        return h_state


# ══════════════════════════════════════════════════════════════════════════════
# 3. Training — parallel batch + checkpoint
# ══════════════════════════════════════════════════════════════════════════════
CHECKPOINT_PATH = "E:/claude/myllm/checkpoint_v2.pt"
SAVE_EVERY = 500


def train_model(steps=5000, lr=1e-3, batch_size=64, print_every=200):
    """
    思路 1 体现在这里：
      batch_size=64（比之前 v1 的 16 大 4 倍）
      → 每步并行训练 64 句话，效率 4x
      → 精度有损失（batch 增大的方差），但换来训练速度
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CurveRNNv2(VOCAB_SIZE, embed_dim=64, hidden_dim=256, n_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")

    all_tokens = torch.stack([encode_sentence(s) for s in CORPUS]).to(device)
    N = len(CORPUS)

    start_step = 0
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt["step"] + 1
        print(f"✅ Loaded checkpoint from step {ckpt['step']} (continue from {start_step})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model v2] ~{n_params:,} params | batch_size={batch_size}")
    print(f"[Train] {steps} steps (from {start_step})\n")

    for step in range(start_step, steps):
        model.train()
        optimizer.zero_grad()

        # ── Batch parallel: 64 句话同时训练 ─────────────────────────────
        idx = torch.randint(0, N, (batch_size,))
        tokens = all_tokens[idx].transpose(0, 1)    # (L, B=64)

        logits, _ = model(tokens)                  # (L-1, B, V)
        targets = tokens[1:].contiguous()           # (L-1, B)
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step % print_every == 0:
            print(f"  step {step:5d} | loss {loss.item():.4f} | lr {optimizer.param_groups[0]['lr']:.6f}")

        if step > 0 and step % SAVE_EVERY == 0:
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "step": step,
                "char2idx": char2idx,
                "idx2char": idx2char,
            }, CHECKPOINT_PATH)
            print(f"💾 Checkpoint saved at step {step}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4. Generation — selective memory walk
# ══════════════════════════════════════════════════════════════════════════════
def generate(model, prompt, max_new=15, temp=0.8):
    """
    用选择性状态生成：
      - 每一步更新状态（gate 决定记住/忘掉）
      - 状态始终是固定大小的向量（bounded context）
      - 不会无限积累旧信息 → 减少重复
    """
    device = next(model.parameters()).device
    model.eval()

    tokens = torch.tensor([char2idx.get(c, UNK_IDX) for c in prompt], dtype=torch.long).to(device)

    # Initialize selective state with prompt
    state = model.init_state(1, device)
    with torch.no_grad():
        for t in range(len(tokens)):
            x = model.embed(tokens[t]) + model.pe.pe[t]
            state = model.step(x, state)

    result = list(prompt)

    for _ in range(max_new):
        with torch.no_grad():
            x = model.embed(tokens[-1:]) + model.pe.pe[0]
            state = model.step(x, state)
            h = state[-1]                            # top layer state (D,)
            logits = model.fc(h.unsqueeze(0))        # (1, V)
            probs = F.softmax(logits / temp, dim=-1)

            # Simple top-1 sampling
            next_tok = probs.argmax(dim=-1).item()

        if next_tok == 0:
            break
        result.append(idx2char.get(next_tok, ""))
        tokens = torch.cat([tokens, torch.tensor([next_tok], dtype=torch.long).to(device)])

    return "".join(result)


def traverse(model, p1, p2, steps=10):
    """Walk between two curve points (bounded latent space)."""
    device = next(model.parameters()).device
    model.eval()

    def get_z(p):
        tokens = torch.tensor([char2idx.get(c, UNK_IDX) for c in p], dtype=torch.long).to(device)
        state = model.init_state(1, device)
        with torch.no_grad():
            for t in range(len(tokens)):
                x = model.embed(tokens[t]) + model.pe.pe[t]
                state = model.step(x, state)
        return state[-1]                             # top layer final state

    z1, z2 = get_z(p1), get_z(p2)
    print(f"\n  Traversal: '{p1}' → '{p2}'")
    for i in range(steps + 1):
        alpha = i / steps
        z = z1 * (1 - alpha) + z2 * alpha
        probs = F.softmax(model.fc(z.unsqueeze(0)), dim=-1).squeeze(0)
        tok = probs.argmax().item()
        c = idx2char.get(tok, "·") if tok != 0 else "·"
        conf = probs[tok].item()
        print(f"  α={alpha:4.1f}  '{c}'  conf={conf:.3f}  {'█' * int(conf * 20)}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Curve Model v2 — Selective Memory + Parallel Batch")
    print("  思路 1: batch_size=64 并行训练")
    print("  思路 2: SelectiveState (bounded context)")
    print(f"Device: {device}")
    print("=" * 60)

    model = train_model(steps=6000, lr=1e-3, batch_size=64, print_every=300)

    print("\n" + "=" * 60)
    print("Generation — selective memory walk (repeat check: less '好好好'?)")
    print("=" * 60)
    for prompt in ["今天天气", "我爱", "宇宙", "月亮", "未来", "健康", "诗歌"]:
        outs = [generate(model, prompt, max_new=15) for _ in range(2)]
        print(f"\n  Prompt: '{prompt}'")
        for o in outs:
            print(f"    → '{o}'")

    print("\n" + "=" * 60)
    print("Interpolation — bounded latent space walk")
    print("=" * 60)
    traverse(model, "今天", "月亮", steps=10)
    traverse(model, "我爱", "诗歌", steps=10)

    # Save final
    torch.save({
        "model_state": model.state_dict(),
        "char2idx": char2idx,
        "idx2char": idx2char,
        "VOCAB_SIZE": VOCAB_SIZE,
        "MAX_LEN": MAX_LEN,
    }, "E:/claude/myllm/model_v2.pt")
    print("\n✅ Done! Model saved → model_v2.pt")