"""
Curve Language Model — PyTorch + Incremental Checkpoint
====================================================
Key features:
  - Train + save checkpoint every N steps (resumable on timeout)
  - Load from checkpoint if exists
  - Curve paradigm: hidden state = "curve point"
  - Generate: walk forward along the curve
  - Interpolate: move between two curve points in latent space
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
# 2. Model
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


class CurveRNN(nn.Module):
    """
    "Curve" = trajectory through hidden-state space.

    Design philosophy (Bezier-inspired):
      - h[t] is the "curve point" at position t
      - The sequence {h[0], h[1], ...} is the "language curve"
      - Adding a token = extending the curve by one point
      - Generation = walking forward along the learned curve
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256, n_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pe = PositionalEncoding(embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers, batch_first=False)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def encode(self, tokens):
        """tokens (L, B) → h_seq (L, B, D). h[t] is the curve point at t."""
        emb = self.embed(tokens)
        L, B = emb.shape[0], emb.shape[1]
        emb = emb + self.pe.pe[:L].unsqueeze(1)
        out, _ = self.rnn(emb)
        return out

    def decode(self, h):
        """h: (D,) or (B,D) → logits"""
        if h.dim() == 1:
            h = h.unsqueeze(0)
        return self.fc(h)

    def forward(self, tokens):
        """
        tokens: (L, B) → logits (L-1, B, V)
        Predict next token from each position's curve point.
        """
        h_seq = self.encode(tokens)
        logits = self.fc(h_seq)
        return logits[:-1], h_seq


# ══════════════════════════════════════════════════════════════════════════════
# 3. Training with checkpoint
# ══════════════════════════════════════════════════════════════════════════════
def save_checkpoint(model, optimizer, step, save_path):
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "char2idx": char2idx,
        "idx2char": idx2char,
        "VOCAB_SIZE": VOCAB_SIZE,
        "MAX_LEN": MAX_LEN,
    }, save_path)
    print(f"\n💾 Checkpoint saved at step {step} → {save_path}")


CHECKPOINT_PATH = "E:/claude/myllm/checkpoint.pt"
SAVE_EVERY = 500


def train_model(steps=3000, lr=1e-3, batch_size=16, print_every=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CurveRNN(VOCAB_SIZE, embed_dim=64, hidden_dim=256, n_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")

    all_tokens = torch.stack([encode_sentence(s) for s in CORPUS]).to(device)
    N = len(CORPUS)
    start_step = 0

    # ── Load checkpoint if exists ────────────────────────────────────────────
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt["step"] + 1
        print(f"✅ Loaded checkpoint from step {ckpt['step']} (continuing from step {start_step})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] ~{n_params:,} params")
    print(f"[Train] {steps} steps total (starting from {start_step})\n")

    for step in range(start_step, steps):
        model.train()
        optimizer.zero_grad()

        idx = torch.randint(0, N, (batch_size,))
        tokens = all_tokens[idx].transpose(0, 1)      # (L, B)

        logits, _ = model(tokens)
        targets = tokens[1:].contiguous()
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if step % print_every == 0:
            print(f"  step {step:5d} | loss {loss.item():.4f} | lr {optimizer.param_groups[0]['lr']:.6f}")

        # ── Save checkpoint every N steps ──────────────────────────────────
        if step > 0 and step % SAVE_EVERY == 0:
            save_checkpoint(model, optimizer, step, CHECKPOINT_PATH)

    return model


# ══════════════════════════════════════════════════════════════════════════════
# 4. Generation
# ══════════════════════════════════════════════════════════════════════════════
def generate(model, prompt, max_new=15, temp=0.8, top_k=10):
    """Walk forward along the curve: encode prompt → hidden state → sample next → repeat."""
    device = next(model.parameters()).device
    model.eval()

    tokens = torch.tensor([char2idx.get(c, UNK_IDX) for c in prompt], dtype=torch.long).to(device)

    # Initialize LSTM hidden state by encoding full prompt
    with torch.no_grad():
        tokens_batch = tokens.unsqueeze(-1)                  # (L, 1)
        L = tokens_batch.shape[0]
        emb = model.embed(tokens_batch) + model.pe.pe[:L].unsqueeze(1)
        _, hidden = model.rnn(emb)

    result = list(prompt)

    for _ in range(max_new):
        with torch.no_grad():
            x = model.embed(tokens[-1:])
            x = x + model.pe.pe[0]
            x = x.unsqueeze(0)                               # (1, 1, E)
            out, hidden = model.rnn(x, hidden)               # (1, 1, D)
            h = out.squeeze(0).squeeze(0)                    # (D,)
            logits = model.fc(h.unsqueeze(0))                # (1, V)
            probs = F.softmax(logits / temp, dim=-1)

            top_vals, top_idx = probs.topk(top_k, dim=-1)
            mask = torch.zeros_like(probs)
            mask.scatter_(1, top_idx, top_vals)
            mask = mask / mask.sum(dim=-1, keepdims=True).clamp(min=1e-10)
            next_tok = torch.multinomial(mask, 1).item()

        if next_tok == 0:
            break
        result.append(idx2char.get(next_tok, ""))
        tokens = torch.cat([tokens, torch.tensor([next_tok], dtype=torch.long).to(device)])

    return "".join(result)


def traverse(model, p1, p2, steps=10):
    """Walk between two curve points — shows smooth semantic shift in latent space."""
    device = next(model.parameters()).device
    model.eval()

    def get_z(p):
        t = torch.tensor([char2idx.get(c, UNK_IDX) for c in p], dtype=torch.long).to(device)
        t = t.unsqueeze(-1)
        L = t.shape[0]
        emb = model.embed(t) + model.pe.pe[:L].unsqueeze(1)
        out, _ = model.rnn(emb)
        return out[-1].squeeze(0)

    z1, z2 = get_z(p1), get_z(p2)
    print(f"\n  Traversal: '{p1}' → '{p2}'")
    for i in range(steps + 1):
        alpha = i / steps
        z = z1 * (1 - alpha) + z2 * alpha
        logits = model.fc(z.unsqueeze(0))
        probs = F.softmax(logits, dim=-1).squeeze(0)
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
    print("Curve Language Model — PyTorch + Checkpoint")
    print(f"Device: {device}")
    print("Checkpoint saves every", SAVE_EVERY, "steps →", CHECKPOINT_PATH)
    print("=" * 60)

    model = train_model(steps=5000, lr=1e-3, batch_size=16, print_every=300)

    print("\n" + "=" * 60)
    print("Generation: autonomous curve traversal")
    print("=" * 60)
    for prompt in ["今天天气", "我爱", "宇宙", "月亮", "未来", "健康", "诗歌"]:
        outs = [generate(model, prompt, max_new=12) for _ in range(2)]
        print(f"\n  Prompt: '{prompt}'")
        for o in outs:
            print(f"    → '{o}'")

    print("\n" + "=" * 60)
    print("Interpolation: walking between two curve points")
    print("=" * 60)
    traverse(model, "今天", "月亮", steps=10)
    traverse(model, "我爱", "诗歌", steps=10)

    # Save final model
    save_path = "E:/claude/myllm/model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "char2idx": char2idx,
        "idx2char": idx2char,
        "VOCAB_SIZE": VOCAB_SIZE,
        "MAX_LEN": MAX_LEN,
    }, save_path)
    print(f"\n✅ Final model saved → {save_path}")