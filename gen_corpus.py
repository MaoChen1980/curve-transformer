"""
Generate real Chinese sentences using pretrained uer/gpt2-distil-chinese-cluecorpussmall.
Then use these as training data for Curve Transformer v4.
"""
import os
# HuggingFace mirror for fast downloads in China
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"

from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
import random

# Load pretrained model
print("Loading pretrained model...")
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-distil-chinese-cluecorpussmall")
pipe = TextGenerationPipeline(model, tokenizer, device=-1)  # CPU
print("Model loaded.")

# Diverse seed prompts to get varied sentences
PROMPTS = [
    # Time & seasons
    "今天天气", "今日阳光", "春天来了", "夏天很热", "秋风吹过", "冬天寒冷",
    "清晨的阳光", "午后的阳光", "夜幕降临", "夜深人静", "星光闪烁", "月光洒在",
    "昨天下雨", "明天下雪", "时间流逝", "岁月如梭", "光阴似箭", "四季轮回",

    # Emotions & feelings
    "我爱", "你爱", "他爱", "她爱", "心中有", "心里感到", "感到幸福", "感到悲伤",
    "梦想是", "希望是", "心中梦想", "心中希望", "自由是", "和平是",
    "幸福在于", "友情是", "亲情是", "爱情是", "感恩的心",
    "微笑是", "眼泪是", "坚强是", "勇敢是", "善良是",

    # Nature & world
    "天空是", "大地是", "海洋是", "山川是", "河流是", "森林是", "草原是",
    "大海很大", "高山很高", "河水向东", "风吹过来", "雨后空气",
    "星星在天", "太阳升起", "月亮挂在", "云在天上", "鸟儿在",

    # Knowledge & culture
    "人工智能", "科学是", "技术是", "艺术是", "音乐是", "诗歌是", "绘画是",
    "知识是", "智慧是", "真理是", "文化是", "教育是", "科技是",
    "科学探索", "技术创新", "艺术创作", "音乐欣赏", "诗歌朗诵",
    "人工智能改变", "量子计算", "网络安全", "数据科学",

    # Life & philosophy
    "人生是", "生活是", "生命是", "世界是", "宇宙是", "时间是",
    "人生如", "生活像", "世界很大", "宇宙无限", "人生短暂",
    "活在当下", "珍惜当下", "珍惜时间", "珍惜生命", "珍惜友情",
    "努力工作", "努力学习", "努力生活", "努力追求", "努力实现",

    # People & relationships
    "老师是", "医生是", "朋友是", "家人是", "孩子是", "老人是",
    "科学家在", "艺术家在", "音乐家在", "作家在", "画家在",
    "我有一个", "你有一个", "他有一个", "我们一起",
    "我和你是", "我和你在", "我和我的",

    # Actions & activities
    "我喜欢", "我想要", "我在", "我想去", "我去过", "我看过",
    "他在做", "她在想", "它在飞", "他们在讨论",
    "学习是", "工作是", "旅行是", "阅读是", "写作是", "思考是",
    "唱歌是", "跳舞是", "绘画是", "拍照是", "做饭是",

    # Abstract concepts
    "和平是", "自由是", "正义是", "公平是", "民主是",
    "希望在于", "力量在于", "快乐在于", "美丽在于", "成功在于",
    "爱是", "恨是", "善是", "恶是", "真与假",

    # More diverse seeds
    "关于人生的", "关于爱情的", "关于友谊的", "关于梦想的",
    "未来的世界", "未来的科技", "未来的生活", "未来的教育",
    "中国的文化", "中国的历史", "中国的科技", "中国的艺术",
    "互联网的", "数字时代的", "信息时代的", "智能时代的",

    # Specific topics
    "健康是", "财富是", "成功是", "失败是", "成长是",
    "最好的", "最美的", "最真的", "最好的时光", "最美的风景",
    "每天都要", "每年都是", "每次都是", "每个人都是", "每个地方都是",

    # Short completions
    "风吹", "雨落", "花开", "叶落", "日出", "月升",
    "想你", "爱你", "恨你", "帮你", "教你", "养你",
    "真好", "真美", "真香", "真好听", "真好看", "真好玩",
    "可以吗", "可以吗", "怎么办", "为什么", "怎么用", "怎么学",
]

# Limit: use a diverse subset of prompts
import random as _rnd
_rnd.seed(42)
SAMPLE_SIZE = min(20, len(PROMPTS))
SAMPLED = _rnd.sample(PROMPTS, SAMPLE_SIZE)
MAX_SENTENCES = 1000

print(f"\nUsing {SAMPLE_SIZE} seed prompts (sampled from {len(PROMPTS)}), max {MAX_SENTENCES} sentences...")

ALL_SENTENCES = set()
# Generate multiple sentences per prompt with varied parameters
for p in SAMPLED:
    if len(ALL_SENTENCES) >= MAX_SENTENCES:
        break
    for temp in [0.8, 0.9]:
        if len(ALL_SENTENCES) >= MAX_SENTENCES:
            break
        for top_p in [0.9, 0.95]:
            if len(ALL_SENTENCES) >= MAX_SENTENCES:
                break
            for _ in range(2):
                if len(ALL_SENTENCES) >= MAX_SENTENCES:
                    break
                try:
                    out = pipe(
                        p,
                        max_new_tokens=random.randint(15, 30),
                        do_sample=True,
                        temperature=temp,
                        top_p=top_p,
                        repetition_penalty=1.2,
                    )
                    text = out[0]['generated_text'].replace(" ", "").replace("\u3000", "").strip()
                    # Clean: remove padding tokens, keep 4-25 chars
                    if 4 <= len(text) <= 25:
                        ALL_SENTENCES.add(text)
                except Exception as e:
                    pass

print(f"Generated {len(ALL_SENTENCES)} unique sentences before filtering")

# Filter: keep only meaningful Chinese sentences
def is_clean(s):
    if len(s) < 4 or len(s) > 25:
        return False
    # Reject if > 60% same char
    if len(s) >= 6:
        most_common = max(set(s), key=s.count)
        if s.count(most_common) / len(s) > 0.6:
            return False
    # Reject if contains obvious model artifacts
    bad = ["####", "....", "~~~~", "////", "++++", "----", "````", "===="]
    if any(b in s for b in bad):
        return False
    return True

CLEAN = [s for s in ALL_SENTENCES if is_clean(s)]
print(f"After cleaning: {len(CLEAN)} sentences")

# Sort for reproducibility
CLEAN.sort()
OUTPUT = "E:/claude/myllm/real_corpus.txt"
with open(OUTPUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(CLEAN))

print(f"\n✅ Saved {len(CLEAN)} sentences → {OUTPUT}")

# Show sample
print("\nSample sentences:")
for s in CLEAN[:20]:
    print(f"  {s}")