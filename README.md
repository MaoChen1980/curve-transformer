# Curve Transformer — 语言是曲线

> **"Language is a curve, not a sequence."**

[→ English Version](#english-version)

---

<a id="chinese-version"></a>

## 中文版

### 想法起源

大多数语言模型把文本当作序列处理：给定前 N 个 token，预测第 N+1 个。我们问了一个不同的问题：

**如果语义是一个连续空间里的点，这个空间本身有几何结构呢？**

这个想法的起点来自一个类比——**贝塞尔曲线**：

- 贝塞尔曲线用一组控制点 `P_i` 定义一条平滑路径，每个控制点通过伯恩斯坦基函数 `B_i^n(t)` 影响整条曲线
- 文本生成可以看作在高维语义空间里"画一条曲线"
- **参数 = 控制点**，模型参数决定了曲线走向
- **encode = 从 prompt 算出控制点位置**
- **decode = 用控制点重绘曲线**
- **attention 权重 = 伯恩斯坦权重**，每个输入 token 对当前输出的影响

从神经网络的角度看，这只是一个拟合函数 — 没有"基函数"的强假设，只有连续映射。贝塞尔提供了理解"为什么这样能 work"的直觉，而不是架构约束。

从这个视角出发，我们推导出了几个关键判断：

1. **有限注意力就够了** — 伯恩斯坦权重在局部最大，远处趋近于零。曲线天生局部平滑，不需要全局长注意力。滑动窗口在数学上就是够的。

2. **分段并发训练可行** — 高阶贝塞尔不稳定（龙格现象），工程解法是分段。对应到训练：模型太大不好并发，那就切段独立训练，边界对齐。精度损失在曲线拟合框架下是可接受的。

3. **大纲模式** — 画曲线时不看全局，需要时切到控制多边形看趋势。对应到生成：大部分时候用局部注意力，偶尔切到"大纲模式"看看全局结构。

这三个判断驱动了后续所有的实验。

---

### 实验过程

#### v1：LSTM 基线

第一个实现用 LSTM + 正弦位置编码：

```
Embedding → PE → LSTM → FC → vocab
```

结果：能跑。但生成重复严重——小数据集 + 简单 RNN 的经典问题。

还撞上了 **E_in / E_out 维度不匹配问题**：embedding 维度和 hidden 维度不同，不断引发 shape 错误。

**教训**：统一维度设计是必要的。

---

#### v2：选择性状态 + 并行训练

从 v1 的教训出发，做了两个优化：

**思路 1 — 并行化**：增大 batch_size（16→64），多句话同时训练。精度有损失但换来了训练速度。

**思路 2 — 有限上下文**：LSTM → SelectiveState，受 Mamba（状态空间模型）启发：

```python
# LSTM: h[t] = f(W·h[t-1], U·x[t]) — 无差别记住一切
# SelectiveState:
gate  = sigmoid(W_g·cat(x, h))      # 保留多少旧状态
cand  = tanh(W_c·cat(x, h))          # 新信息候选
h_new = gate * h + (1 - gate) * cand # 选择性融合
```

**核心改变**：模型自己决定记住什么、忘掉什么。相当于一个有限容量工作记忆，解决了梯度衰减和局部重复问题。

架构：
```
Embedding(vocab, E=64) → PE → SelectiveState × 2 → FC → vocab
```

结果：记忆改善了。生成在边界案例上仍有重复。根本问题没变：模型没有学到结构化语义，它只是记忆了。

**教训**：并行训练是对的，但选择性状态的表达力有限。需要更强的上下文机制。

---

#### v3：Transformer（转折）

这时候做了一个关键转向。既然：

1. **并行训练是自然的** — PyTorch 天然并行 batch 维度，不需要手动分块
2. **Attention 处理上下文** — 不需要在训练时手动管理隐藏状态，直接对整序列跑 attention
3. **统一维度** — 去掉 E_in/E_out 分离，全程 D=256

架构变成：

```
Embedding(vocab, E=64) + Sinusoidal PE → Linear(E→D=256)
  → Block × 2
  │   ├── PreNorm + MHA (Q/K/V/O 全部 D→D)
  │   ├── PreNorm + FFN (GELU)
  │   └── 残差连接
  → LayerNorm → FC → vocab
```

每个 Block 的 `step(x, h)` 方法用于自回归生成：

```python
gate = sigmoid(q)                    # 保留多少旧状态？
h_new = h * gate + v * (1 - gate)    # 门控状态更新
x = x + o_proj(h_new)                # 残差
x = x + ffn(norm(x))
return x, h_new
```

**关键区别**：训练时用标准双向 MHA，每个位置看到完整句子；推理时扔掉 attention，只用 `step()` 的门控状态自回归。

这对应了贝塞尔的思想：训练时"画完整条曲线确定控制点"，推理时"只看控制点推下一段"。

在 3000 条模板生成的句子上训练后：

| Step | Loss |
|------|------|
| 0 | 5.95 |
| 300 | 0.187 |
| 1000 | 0.013 |
| 5000 | 0.000 |
| 8000 | 0.000 |

loss 归零 — 模型把每条训练句子都背下来了。

**但语义插值实验有效：**

```
Traverse: '今天' → '未来'

α=0.0  '天'  conf=0.999
α=0.5  '天'  conf=0.827
α=0.6  '未'  conf=0.602
α=1.0  '未'  conf=0.947
```

从"天"到"未"的字符过渡是平滑的。即使生成完全崩溃了，latent space 的几何结构仍然编码了语义关系。

**生成崩溃**（因为过拟合）：
```
'今天天气' → '今天天气年年年年年年年年年年年年年年年'
'我爱'     → '我爱爱爱爱爱爱爱爱爱爱爱爱爱爱爱爱'
```

但这是数据量问题，不是架构问题。关键发现：**表示学习和生成是两码事**。latent space 学到了真实的语义几何，但生成头只是记忆了。

---

#### v3 延续：真实语料

当前阶段：v3 架构不变，但用 GPT-2 生成的真实中文句子代替模板语料。`gen_corpus.py` 从 HuggingFace 加载 `uer/gpt2-distil-chinese-cluecorpussmall`，用多样化 prompt 生成句子，目标是让模型从记忆走向泛化。

优化：
- 设置 `HF_ENDPOINT=https://hf-mirror.com` 保证国内下载稳定
- 限制生成量：20 prompts × 2 temps × 2 top_p × 2 repeats，最多 1000 句

---

#### v4：三步实验 + 门控状态优化

从 v3 的教训出发，v4 是一个系统性实验：在真实数据上对比多种架构。

**注意 — 两套语料，不要混淆**：
- **混合语料（B/C/D/D2 使用）**：全唐诗 + 哆啦A梦对话，41580 条样本，vocab=6243
- **纯唐诗（D2-poetry-only / D3 使用）**：仅全唐诗约 3000 首，vocab ~3100
- B/C/D/D2 先用混合语料训练至 10000 步；为了排除数据干扰、聚焦架构对比，后续 D2 和 D3 统一切换到纯唐诗 from scratch 训练
- **这意味着 D2/D3 的 fwd loss（~0.7-1.0）不能直接与 B/C（~3.5-3.8）比较** — vocab 大小不同，loss 基数不同

**数据集详情**：
- **全唐诗**：`poet.tang.*.json`，3000+ 首，提取 `paragraphs` 字段拼接，过滤 < 4 字
- **哆啦A梦**：27 部电影剧本 `.txt` 文件，过滤日文含量 > 10% 的段落，跳过《伴我同行》
- 字符级分词，maxlen=100，BOS/EOS/PAD/UNK 控制符

**通用架构**（E=128, D=256, h=4, 3层, ~4M 参数）：
```
Embedding(V, 128) + Sinusoidal PE → Linear(128→256)
  → Block × 3 (PreNorm + MHA + FFN) → LayerNorm → FC(V)
```

---

##### B：因果 Transformer（baseline）

标准 decoder-only，triangular causal mask，纯 `forward()` 训练，无 `step()` 路径。

训练 3000 步后 loss 仍 ~4.5 远未收敛，扩展至 10000 步后最终 loss 3.57。

**生成示例**（temp=0.8, top_k=20）：
```
春风吹 → 春风吹 滿 ，波玉比呂得。感君人恆意，，君淚恩中。。
春晓   → 春晓光壇至，，八神天麾。。金不鸞來，，風花動一植。。
```
有诗歌结构，但重复模式明显。

---

##### C：纯 GRU（无注意力对照）

用 `nn.GRU(256, 256, num_layers=3)` 替换 MHA Block，其余相同。验证"没有注意力只有 GRU 能做什么"。

10000 步后 loss ~3.8。

**生成示例**：
```
春风吹 → 春风吹備，，歸風有光。。黃有今仙月，，無不萬可明明。。
登高   → 登高眇北霧方，，萬夜條緒北得霜。。
```
GRU 也能产出诗歌结构，但输出较短。

---

##### D：二阶段训练（冻结 attention，独立门控投影）

在因果 Transformer 基础上为每个 Block 增加独立门控参数：

```python
self.gp = nn.Linear(D, D)   # gate projection
self.sp = nn.Linear(D, D)   # state projection
```

加载 B 的权重后冻结所有 `qp/kp/vp/op/ff`，**只训练 gp/sp**（6/56 个参数，占 9%）。

**step() 门控状态更新**：
```python
gate = sigmoid(gp(xn))
h_new = tanh(h * gate + sp(xn) * (1 - gate))
```

2000 步后 step_loss ~0.93-5.7，容量严重不足。10000 步后仍不稳定，**step 路径生成极短**（2-5 词即停）。

---

##### D2：联合训练（不冻结，独立门控投影）

同 D 的架构（独立 gp/sp），但**不冻结任何参数**，forward + step 联合优化：

```
loss = loss_fwd + 0.5 * loss_step    # NaN 时跳过 step
```

- `STEP_LEN = 1`（1 步展开），NaN guard 防止梯度爆炸
- gp/sp 初始化 `N(0, 0.01)` 避免起步溢出
- 从因果 checkpoint 加载权重

**结果**：forward 路径生成质量好（混合唐诗 + 哆啦A梦风格），但 **step 路径仍然短**——泄漏积分器 `h_new = h*gate + s*(1-gate)` 表达力不足。

---

##### D3：GRUCell 状态转移 ⭐（最终方案）

保留 forward() 的 causal attention 不变，**step() 路径用完整 GRUCell 替换泄漏积分器**：

```python
# Block.__init__
self.gru = nn.GRUCell(D, D)

# Block.step()
xn = self.an(x).squeeze(0)    # [B, D]
h_new = self.gru(xn, h)       # 完整 GRU 门控 (reset + update)
x = x.squeeze(0) + self.op(h_new)
x = x + self.ff(self.fn(x))
return x.unsqueeze(0), h_new
```

- 参数量：4.6M（比 D2 多 0.8M，来自 GRUCell 的 `W_ih` + `W_hh`）
- 训练：纯唐诗 3000 首，from scratch，10000 步
- 最终 fwd loss 0.68-0.82，step loss 3.19-4.77

**D3 vs D2 对比（纯唐诗训练 10000 步）**：

| 指标 | D2 (leaky integrator) | D3 (GRUCell) ⭐ |
|------|----------------------|-----------------|
| 参数量 | 3.8M | 4.6M |
| fwd loss | 0.74-1.05 | 0.68-0.82 |
| step loss | 1.91-5.69 | 3.19-4.77 |
| **step 生成** | **2-5 词即停** | **10-30 词连续** |

**关键生成对比（step 路径）**：
```
D2 step:  春风吹 → 春吹。
          明月   → 明月下君爲臺。
          登高   → 登高臺。

D3 step:  春风吹 → 春吹不盡，一朝，何處，今歲，金爐燭雲飛燕趙。
          明月   → 明月落，一聲，勞思曲，何處所以樂，郭，不知豪俠骨
          望月   → 望月落。何處，憂出轉化，何以流簫宴，不見山朱顏色不可尋
          登高   → 登高樓臺側，所思，非復低遠行
```

**结论**：两个线性投影做 leaky integration 不足以维持自回归生成。一个完整 GRU 细胞做状态转移后，step 路径首次实现了有意义长度的诗歌生成。D3 被选为最终架构。

---

##### D3 消融实验：A/B/C/D 变体

基于 D3 的 GRUCell 架构，我们测试了 4 个改进方向（及组合），看它们能否缩小 forward/step 差距：

**A — 逐层独立状态**：原版 D3 所有 Block 共享同一个 h，改为每个 Block 维护自己的 h。参数变化：+0（只是拆开用）。

**B — 残差注入**：step 路径在 residual 中加入 `hp(h_new)` 投影，让残差流更直接地收到状态信息。参数增加：~197K。

**C — 历史状态交叉注意力**：保留最近 4 步的 h 到 buffer，step 时用 cross-attention 查询历史状态。参数增加：~590K。

**D — 双尺度 GRU**：原 GRUCell 拆为 fast GRU + slow GRU，slow GRU 每 3 步更新一次，输出相加。参数增加：~1.19M。

**训练设置**：8 个变体（base + A + B + C + D + AC + ABC + ABCD），纯唐诗 from scratch，base 训练 5000 步，其余 2000 步（变体之间已经可对比）。

```
训练步数 | 变体     | fwd loss | step loss | 参数
2000     | base      | 4.62     | 5.48      | 4.62M
2000     | A         | 5.10     | 5.34      | 4.62M
2000     | B         | 6.31     | 7.06      | 4.82M
2000     | C         | 5.10     | 5.87      | 5.22M
2000     | D         | 5.02     | 4.35      | 5.81M
2000     | AC        | 4.19     | 5.22      | 5.22M  ← 最佳 fwd
2000     | ABC       | 4.88     | 5.49      | 5.41M
2000     | ABCD      | 5.10     | 5.18      | 6.60M
```

**step() 生成对比（温度 0.8, top_k 20）**：
```
              base                   | ABCD
今天天气 →   今天天。                | 今天天羣門。不復平陽期。
春风吹   →   春吹斷鮫綃。莫問城...   | 春吹花。
明月     →   明月明朝。莫問身居...   | 明月，白馬頭。日日白馬...
登高     →   登高鳥，忽斷鮫綃昏...   | 登高堂。
朝辞     →   朝。相逢春深殿前...     | 朝馬羣。
```

**key observation**：
- **D（双尺度 GRU）单变体最佳 step loss（4.35）**，说明双尺度结构有利于自回归稳定性
- **AC（A+C）forward loss 最低（4.19）**，逐层独立 + 交叉注意力有协同效应
- **B 残差注入表现最差**，可能因为额外的注入路径干扰了 residual 流的原有学习
- **ABCD 全开没有胜过 D 单变体**，说明更多参数不是更好 — 各改进方向之间有干扰
- 所有变体的 **step loss >> fwd loss** 的差距模式与原版 D3 一致，说明 forward/step gap 是架构层面的根本挑战

---

#### 关键发现

1. **门控状态需要足够表达力** — leaky integrator（`h*gate + s*(1-gate)`）在 2-5 步后坍缩到 EOS，GRUCell 的 reset + update 门控组合可以维持 10-30 步生成
2. **step 路径必须参与训练** — D2 的联合训练比 D 的冻结训练效果好，但架构本身的表达力才是根本限制
3. **温度 + top-k 采样消除重复** — 用 `temperature=0.8, top_k=20` 替换 argmax 后，所有模型的生成多样性显著提升
4. **训练从 3000→10000 步有显著改善** — fwd loss 从 ~4.5 降至 ~0.7（纯唐诗），说明之前远未收敛

---

### 假设验证与最终结论

回到最初的问题：**双向 MHA 训练 + O(1) 门控状态推理，能替代标准 Transformer 的 O(L) KV cache 吗？**

**已验证（部分成立）**：
- GRUCell 做状态转移，在联合训练下可以稳定生成 10-30 步 — 证明 O(1) state 路径**能够学习自回归生成**
- step 路径必须参与训练、状态门控需要足够表达力，这两个必要条件已确认
- 在纯唐诗这样风格高度一致的数据上，step 路径可以产生有诗歌结构的内容

**未验证（仍在赌）**：
- **forward 路径仍然明显强于 step 路径** — 即使 D3，fwd loss 0.7 vs step loss 3.2-4.8，差距>4倍。forward 生成的诗歌也更流畅连贯
- 只在纯唐诗上验证了，**混合语料（多样化数据）上 D3 的 step 路径表现未知**
- 只在 3 层 D=256 的小规模上验证了，**scale up 后 state 还够不够用未知**
- **没有解决"门控只看当前 token"的固有限制** — GRUCell 的输入仍然是 `x_norm`，不是 `cat(x_norm, h)`

**所以核心结论是**：O(1) state 推理的方向有初步证据支持（D3），但在质量上仍有显著差距。目前的位置是"证明了这条路径可以走，但远没走到头"。

---

### 下一步方向

基于以上结论，几个值得探索的方向：

**1. D3 混合语料训练** — 把 D3 放到唐诗+哆啦A梦混合语料上训练，看 step 路径能否处理多样化数据。这是最直接的下一步，成本最低。

**2. Forward/step gap 分析** — 逐 token 对比两个路径的 logits 分布，精确定位 divergence 从哪里开始、哪些 token 类别最容易偏离。为架构改进提供依据。

**3. 状态机制改进** — 当前 GRUCell 的输入是 `x_norm`（只看当前 token），可以改为 `cat(x_norm, h)`（让门控看到已积累的上下文），或者引入更复杂的交叉注意力到 state。

**4. Scale up** — 6 层 / D=512，看更大容量下 state 路径是否获益更多。如果 gap 随规模缩小，说明瓶颈在容量；如果 gap 不变，说明问题在架构。

**5. Post-training 对齐** — 先用 forward 路径生成大量高质量文本，再用这些文本 distill step 路径，或者用 RL 直接优化 step 路径的生成质量。

---

### 理论分析：与标准 Transformer 的区别

你的模型和标准小 Transformer 的核心区别：

| 维度 | 标准 Transformer | Curve Transformer (D3) |
|------|-----------------|----------------------|
| 训练注意力 | Causal mask（只看过去） | 双向全量（看完整句子） |
| 生成注意力 | 每步关注所有历史 KV | 不用 attention，只用 GRU 状态 h |
| 推理复杂度 | O(L) — 随序列长度增长 | O(1) — 固定 |
| 状态更新 | 无显式状态 | `h_new = GRUCell(x_norm, h)` |
| 训练/推理一致性 | 相同 | 不同路径，独立权重 |

#### 理论优势

**1. 训练信号更强**

双向 attention 让每个 token 直接看到完整句子梯度，没有 causal mask 的信息遮蔽。相当于画曲线时能看到整条线再落笔。梯度信号更多、更直接。

**2. 推理 O(1) 代替 O(L)**

不需要 KV cache，没有 context window 上限。推理成本不随序列长度增长。Scale 上去就是质的区别。

**3. 训练/推理解耦**

双向 attention 负责"学"（把曲线形状压缩到状态 h），门控状态负责"推"（只用状态生成下一段）。计算分配和执行分配是分开的 — 和贝塞尔先确定控制点再画曲线的逻辑一致。

**本质上在赌一件事**：门控状态 h 可以在双向 attention 的监督下，学到足够好的动力学来近似 attention 的输出。

#### 理论缺陷

**1. 训练/推理分布偏移 — 核心问题（v4 D3 已缓解）**

损失函数只优化 `forward()`（双向 attention），但推理跑的是 `step()`（只有状态 h）。v3 中这两个路径共享权重——`q_proj` 在 forward 里学做 attention query，在 step 里却被用来算 gate。**同一个权重被优化去做两件不同的事。**

v4 D3 的缓解方案：
- **独立状态权重**：`step()` 用独立的 GRUCell，不再与 `qp/kp/vp` 共享
- **step 参与训练**：`loss = loss_fwd + 0.5 * loss_step`，step 路径直接接收梯度
- **结果**：GRUCell 收敛后能稳定生成 10-30 词，虽然 fwd 路径仍然更强大

**2. 信息瓶颈**

D=256 的 h 要压缩整个句子的信息。短句可行（~50 token），但上下文越长，压缩损失越大。标准 Transformer 的 KV cache 是 O(L)，你的是 O(1) — 既是优势也是约束。

**3. 门控只看当前 token**

```python
gate = sigmoid(q)  # q 只依赖当前 token 的 embedding
```

"保留多少旧状态"这个决策只取决于当前 token 本身，不取决于已经积累的上下文 h。模型无法根据"我已经记了什么"来调整吸收新信息的力度。这在语义上挺奇怪的 — 同一个字在不同语境下对状态的更新应该不同。

**4. 双向 attention 的"作弊"问题**

训练时预测下一个 token，但 attention 能看到未来的 token。等于考你"下一句是什么"但先把答案给你看了。attention 层学到了很强的未来信息模式，但这些模式在推理时用 `step()` 根本复现不出来。低 loss 不代表 `step()` 的推理能力。

#### 理论优势/缺陷总结

| 维度 | 优势 | 代价 |
|------|------|------|
| 推理速度 | O(1)，和序列长度无关 | h 的容量限制上下文长度 |
| 训练信号 | 双向，梯度更强 | 训练学的能力推理时用不上 |
| 状态更新 | 简洁优雅 | 门控只看当前 token，表达力有限 |

#### 从实用角度看

以上是理论分析。从实用出发，这个思路有几个更直接的观察：

**1. 如果状态 h 足够，KV cache 就是拐杖**

标准 transformer 需要 KV cache 是因为架构没有显式记忆，每一步都回头看所有历史。如果固定大小的状态 h 能编码生成所需的上下文，那么 O(L) 的 KV cache 只是一个工程 hack，不是架构进步。

**这个方向的隐含结论很炸裂**：整个 LLM 推理基础设施都在围绕 KV cache 做优化 — PagedAttention、KV cache 量化、cache 调度、显存管理…… 如果状态模型真的 work，这些东西全部不需要了。推理成本从 O(L²) 降到 O(L)，而且没有上下文窗口硬上限。

不止如此。**状态模型的每步推理是真正的 O(1)** — `step()` 的矩阵乘法量是固定的，不管前面生成了一百个字还是一百万字。标准 transformer 即使有 KV cache，每步仍然需要做 attention 读取（O(L) 读取 + 计算），只是省了重复编码。而状态模型从计算到读取都是常数时间。这意味着：局部稳定生成不只是省显存，推理速度本身也会极大提升。这指向一个更深层的问题：**你真的需要记住所有过去才能生成下一个 token 吗？**

**2. 大部分场景的瓶颈是局部一致，不是长上下文**

- 对话 — 前后句连贯比记住 100 轮前的细节重要
- 代码生成 — 函数内的逻辑一致比跨文件引用重要
- 翻译 — 几句话内的术语统一比全文记忆重要
- 写作辅助 — 段落流畅比全书结构重要

长上下文是锦上添花，局部稳定是雪中送炭。**如果局部稳定能做到可靠，就已经覆盖了绝大多数实际需求。**

**3. 大纲 + 章节 + 句子 — 认知的自然粒度**

金庸写《天龙八部》也是先定章回大纲，再一章一章写。人类写作本就是分粒度的：
- **大纲模式**：偶尔看全局，确认方向
- **章节模式**：专注当前块，局部注意力
- **句子模式**：逐句生成，依赖前几句的语境

模型架构不需要每一层都做全局长注意力。它只需要在局部稳定生成，同时保留切换到大纲模式的能力。这和"局部为主，全局偶尔"的直觉完全一致。

---

### 项目文件

```
myllm/
├── main.py          # v1: LSTM 基线 — E_in/E_out 不匹配
├── main_v2.py       # v2: SelectiveState + 并行训练
├── main_v3.py       # v3: Transformer, 统一维度 (当前)
├── gen_corpus.py    # 用 GPT-2 生成训练语料
├── model_v3.pt      # v3 权重 (~4.7MB)
├── checkpoint_v3.pt # v3 检查点 (可恢复训练)
├── real_corpus.txt  # GPT-2 生成的真实语料
├── compare_v1_v2.py # v1 vs v2 对比
├── gen_v2*.py       # v2 生成脚本
├── diagnose_v2.py   # v2 诊断输出
└── check_model.py   # 架构验证
```

### 经验总结

1. **统一维度简化了一切** — E_in/E_out 分离在 v1 中造成持续 shape 错误，v3 统一 D=256 后消失
2. **注意力是强大的学习信号** — 双向 attention 让训练收敛更快，但学到的能力在推理时不一定能复现
3. **Latent space 几何是真实存在的** — 即使过拟合到 loss=0，语义插值仍然有效。表示学习和生成可以分离
4. **更多数据 > 更复杂架构** — 从 3000→10000 步训练带来的提升大于架构改动，模型远未饱和
5. **贝塞尔是直觉工具，不是架构约束** — 曲线类比帮助推导了有限注意力和分段并发，但最终的模型就是标准的函数拟合
6. **门控状态表达力是关键瓶颈** — leaky integrator（`h*gate + s*(1-gate)`）不足以维持 step 生成，GRUCell 可以
7. **独立化训练/推理权重解决了根本冲突** — forward 和 step 路径不再共享 q_proj 后，两者可以各自优化
8. **step 路径必须参与训练** — 冻结 step 权重（D）或使用表达力不足的门控（D2）都导致生成坍缩；GRUCell + 联合训练是首个有效方案

---

### 训练

```bash
# v3
python main_v3.py

# v4 B — 因果 Transformer（混合语料）
"/c/Users/savyc/miniconda3/python.exe" train_v4_causal.py

# v4 C — 纯 GRU（混合语料）
"/c/Users/savyc/miniconda3/python.exe" train_v4_gru.py

# v4 D — 二阶段冻结（混合语料）
"/c/Users/savyc/miniconda3/python.exe" train_v4_phase2.py

# v4 D2 — 联合训练（混合语料）
"/c/Users/savyc/miniconda3/python.exe" train_v4_joint.py

# v4 D3 — GRUCell step（纯唐诗 from scratch）⭐
"/c/Users/savyc/miniconda3/python.exe" train_v4_poetry_d3.py
```

- **优化器**: AdamW, lr=2e-3, weight_decay=0.01
- **损失函数**: CrossEntropyLoss, ignore PAD (index 0)
- **检查点**: 每 500 步 → `checkpoint_v4_*.pt`
- **训练时间**: ~10-12 min GPU (MX250) 跑 10000 步
- **D3 推理**: step() 是 O(1) — 无 KV cache，每步计算量固定
- **采样**: temperature=0.8 + top_k=20（argmax 导致严重重复）
- **语料注意**: B/C/D/D2 混合语料（vocab=6243），D2-poetry 和 D3 纯唐诗（vocab~3100），两组 loss 不可直接比较

### 引用

```bibtex
@software{curve_transformer_2026,
  title={Curve Transformer: Language as a Curve — Bezier-Inspired Semantic Interpolation},
  author={Mao Chen},
  year={2026},
  url={https://github.com/MaoChen1980/curve-transformer}
}
```

---

<a id="english-version"></a>

## English Version

### Origin of the Idea

Most language models treat text as a sequence: given N tokens, predict token N+1. We asked a different question:

**What if meaning is a point in a continuous space — and the space itself has geometry?**

This came from an analogy with **Bezier curves**:

- A Bezier curve is defined by control points `P_i`, each influencing the curve through Bernstein basis functions `B_i^n(t)`
- Text generation = "drawing a curve" in a high-dimensional semantic space
- **Parameters = control points** — they determine where the curve goes
- **Encode = compute control points from a prompt**
- **Decode = reconstruct the curve from control points**
- **Attention weights = Bernstein weights** — how each input token shapes the output

But ultimately, a neural network is just a function approximator. The Bezier analogy provides intuition — why locality matters, why segmentation works — not architectural constraints. The model is a fitting function, nothing more.

From this perspective, we derived three principles:

1. **Local attention is sufficient** — Bernstein weights peak locally and decay to zero at a distance. Curves are locally smooth. Full global attention is unnecessary.

2. **Segmented parallel training is feasible** — High-order Bezier curves are unstable (Runge's phenomenon). The fix is segmentation. For training: split the model, train segments independently, align boundaries. Precision loss is acceptable in a curve-fitting framework.

3. **Outline mode** — Use local attention for generation; switch to a global view only when needed. Same as checking the control polygon instead of the rendered curve.

---

### The Journey

#### v1: LSTM Baseline

```
Embedding → PE → LSTM → FC → vocab
```

It worked, but generation was repetitive — the classic small-dataset RNN problem.

We also hit **E_in / E_out dimension mismatch**: embedding dim ≠ hidden dim caused constant shape errors.

**Lesson**: Unified dimensions are necessary.

---

#### v2: Selective State + Parallel Training

Two optimizations:

**Parallelization**: batch_size 16 → 64. Acceptable precision loss for speed.

**Bounded context**: LSTM → SelectiveState (inspired by Mamba/SSM):

```python
gate  = sigmoid(W_g·cat(x, h))      # how much old state to keep
cand  = tanh(W_c·cat(x, h))          # new information candidate
h_new = gate * h + (1 - gate) * cand # selective fusion
```

The model decides what to remember and forget — a bounded working memory that mitigates gradient decay and repetition.

**Lesson**: Parallel training was right, but selective states lack expressiveness. We needed a stronger context mechanism.

---

#### v3: Transformer — The Pivot

The key realization: PyTorch natively parallelizes over batches, attention naturally handles context, and unified dimensions eliminate shape errors.

Architecture:
```
Embedding(vocab, E=64) + Sinusoidal PE → Linear(E→D=256)
  → Block × 2 (MHA + FFN) → LayerNorm → FC → vocab
```

The `step(x, h)` method for generation:
```python
gate = sigmoid(q)
h_new = h * gate + v * (1 - gate)
x = x + o_proj(h_new)
x = x + ffn(norm(x))
return x, h_new
```

**Key difference**: Training uses full bidirectional MHA (every position sees the whole sentence); inference discards attention entirely, using only the gated state `h`.

Training on 3000 template-generated sentences:

| Step | Loss |
|------|------|
| 0 | 5.95 |
| 300 | 0.187 |
| 1000 | 0.013 |
| 5000 | 0.000 |
| 8000 | 0.000 |

Loss hit zero — the model memorized every sentence.

**But semantic interpolation still worked:**
```
Traverse: '今天' → '未来'
α=0.0  '天'  conf=0.999  →  α=1.0  '未'  conf=0.947
```

A smooth transition from "today/present" to "future." The latent space geometry encodes semantic relationships even under complete overfitting.

Generation collapsed (from overfitting):
```
'今天天气' → '今天天气年年年年年年年年年年年年年年年'
```

**Key finding**: Representation and generation are separable. The latent space learned real geometry; the generation head just memorized.

---

#### Current Phase: Real Corpus

Same v3 architecture, but template data has been replaced with real Chinese sentences from GPT-2 (`uer/gpt2-distil-chinese-cluecorpussmall`). The goal: push the model from memorization toward generalization.

---

#### v4: Three-Path Experiments + Gated State Optimization

Building on v3, v4 is a systematic comparison of architectures on real data.

**Dataset note — two corpora were used, do not conflate**:
- **Mixed (B/C/D/D2)**: Tang poetry + Doraemon dialogues, 41,580 samples, vocab=6243
- **Poetry-only (D2-poetry / D3)**: Tang poetry only, ~3,000 poems, vocab ~3100
- B/C/D/D2 trained on mixed data for 10,000 steps; later D2 and D3 switched to poetry-only (from scratch) to isolate architecture effects
- **D2/D3 fwd loss (~0.7-1.0) is NOT comparable to B/C (~3.5-3.8)** — different vocab sizes change the loss baseline

**Common architecture** (E=128, D=256, h=4, 3 layers, ~4M params):
```
Embedding(V, 128) + Sinusoidal PE → Linear(128→256)
  → Block × 3 (PreNorm + MHA + FFN) → LayerNorm → FC(V)
```

**B: Causal Transformer (baseline)** — Standard decoder-only with triangular mask. Pure `forward()` training, no `step()` path. Trained on mixed data. Final loss 3.57.

**C: Pure GRU (no-attention control)** — Replaced MHA blocks with `nn.GRU(256, 256, 3)`. Loss ~3.8 at 10K steps. Produces poetic structure but shorter output.

**D: Two-phase training (frozen attention, independent gate/state)** — Added independent `gp`/`sp` projections per block, loaded B's weights, froze all `qp/kp/vp/op/ff`. Trained only gp/sp (9% of params). Step path generated only 2-5 tokens — severe capacity bottleneck.

**D2: Joint training (no freeze, independent gate/state)** — Same independent gp/sp but no frozen weights. Forward + step joint optimization: `loss = loss_fwd + 0.5 * loss_step`. Trained on mixed data from B checkpoint. Step path still collapsed to 2-5 tokens — leaky integrator `h*gate + s*(1-gate)` lacks expressiveness.

**D3: GRUCell step path ⭐ (final architecture)** — Replaced leaky integrator with `nn.GRUCell(D, D)`:

```python
# Block.step()
xn = self.an(x).squeeze(0)
h_new = self.gru(xn, h)            # full GRU with reset + update gates
x = x.squeeze(0) + self.op(h_new)
x = x + self.ff(self.fn(x))
return x.unsqueeze(0), h_new
```

- 4.6M params (0.8M more than D2, from GRUCell's W_ih + W_hh)
- Trained on pure Tang poetry from scratch, 10K steps
- Final fwd loss 0.68-0.82, step loss 3.19-4.77
- **Step path generates 10-30 tokens** — first time step() achieves meaningful-length poetry generation

**D3 vs D2 comparison (poetry-only, 10K steps)**:

| Metric | D2 (leaky integrator) | D3 (GRUCell) ⭐ |
|--------|----------------------|-----------------|
| Params | 3.8M | 4.6M |
| fwd loss | 0.74-1.05 | 0.68-0.82 |
| step loss | 1.91-5.69 | 3.19-4.77 |
| **step generation** | **2-5 tokens** | **10-30 tokens** |

**Key generation comparison (step path)**:
```
D2 step:  Spring breeze → spring breeze.
          Bright moon  → under the moon, for you.

D3 step:  Spring breeze → unending spring breeze, one morning, where, this year,
          golden furnace candle smoke swallows Zhao.
```

**Conclusion**: Two linear projections doing leaky integration cannot sustain autoregressive generation. A full GRU cell for state transition enables meaningful-length poetry generation from the step path for the first time. D3 is selected as the final architecture.

---

##### D3 Ablation Study: A/B/C/D Variants

Based on D3's GRUCell architecture, we tested 4 improvement directions (and combinations) to see which (if any) could narrow the forward/step gap:

**A — Per-layer state**: Original D3 shares one `h` across all blocks; changed to each block maintaining its own `h`. Parameter delta: +0.

**B — Residual injection**: Added `hp(h_new)` projection to the residual stream in the step path, giving residual more direct state info. Parameter delta: +197K.

**C — Cross-attention to h buffer**: Keep last 4 hidden states in a buffer; apply cross-attention during step(). Parameter delta: +590K.

**D — Dual-scale GRU**: Split GRUCell into fast GRU + slow GRU (slow updates every 3 steps); outputs summed. Parameter delta: +1.19M.

**Training**: 8 variants (base + A + B + C + D + AC + ABC + ABCD), pure Tang poetry from scratch, base trained 5000 steps, others 2000 steps.

```
Steps | Variant | fwd loss | step loss | Params
2000  | base    | 4.62     | 5.48      | 4.62M
2000  | A       | 5.10     | 5.34      | 4.62M
2000  | B       | 6.31     | 7.06      | 4.82M
2000  | C       | 5.10     | 5.87      | 5.22M
2000  | D       | 5.02     | 4.35      | 5.81M  ← best step loss
2000  | AC      | 4.19     | 5.22      | 5.22M  ← best fwd loss
2000  | ABC     | 4.88     | 5.49      | 5.41M
2000  | ABCD    | 5.10     | 5.18      | 6.60M
```

**Key observations**:
- **D (dual-scale GRU) has the best step loss (4.35)** — dual timescale helps autoregressive stability
- **AC (A+C) has the best fwd loss (4.19)** — per-layer state + cross-attention shows synergy
- **B (residual injection) performs worst** — extra injection may disrupt the residual flow
- **ABCD doesn't beat D alone** — more parameters isn't better; the improvements interfere
- All variants show **step loss >> fwd loss**, confirming the forward/step gap is a fundamental architectural challenge

---

### Hypothesis Verification & Final Verdict

**Original hypothesis**: Can bidirectional MHA training + O(1) gated state inference replace a standard Transformer's O(L) KV cache?

**Partially validated**:
- GRUCell state transition + joint training enables stable 10-30 step generation — O(1) state can learn autoregressive generation
- Step path must participate in training; state gating needs sufficient expressiveness — both confirmed necessary
- On stylistically consistent data (Tang poetry), step path produces recognizably poetic output

**Still unverified**:
- **Forward path significantly outperforms step path** — D3 fwd loss 0.7 vs step loss 3.2-4.8, >4x gap
- Only tested on poetry; **mixed-corpus (diverse data) D3 behavior unknown**
- Only tested at 3-layer D=256; **scale-up behavior unknown**
- **GRUCell input is still only `x_norm`, not `cat(x_norm, h)`** — the "gate ignores accumulated context" limitation persists

**Bottom line**: O(1) state inference has preliminary evidence (D3), but quality gap remains significant. The path is viable but far from complete.

### Next Directions

1. **D3 mixed-corpus training** — cheapest next step, tests if step path handles diverse data
2. **Forward/step gap analysis** — token-level logit comparison to locate divergence origins
3. **State mechanism improvements** — feed `cat(x_norm, h)` to GRUCell so gating sees accumulated context
4. **Scale up** — 6 layers / D=512 to test if larger capacity narrows the gap
5. **Post-training alignment** — distill forward quality into step path via RL or generated data distillation

### How This Differs From a Standard Transformer

| Aspect | Standard Transformer | Curve Transformer (D3) |
|--------|--------------------|----------------------|
| Training attention | Causal mask (past only) | Full bidirectional |
| Generation attention | Full KV history | State-only via GRU h (no attention) |
| Inference complexity | O(L) — grows with sequence | O(1) — fixed |
| State mechanism | None (KV cache is passive) | `h_new = GRUCell(x_norm, h)` |
| Train/inference paths | Identical | Different paths, **independent weights** |

#### Theoretical Advantages

**1. Stronger training signal** — Bidirectional attention gives every position access to the full gradient. No information masking.

**2. O(1) inference** — No KV cache, no context window limit. Constant cost regardless of sequence length.

**3. Decoupled compute** — Attention learns (compresses curve shape into state h); the gated state executes (extends the curve using state only). This separates learning from generation.

D3's independent GRUCell (instead of shared q_proj) is key: forward and step paths no longer compete for the same weights.

#### Theoretical Weaknesses

**1. Train/inference distribution shift — the core problem (D3 partially mitigates)**

The loss primarily optimizes `forward()`, but inference runs `step()`. D3's mitigation: independent GRUCell weights for step (no weight sharing with qp/kp/vp), and joint training (`loss = loss_fwd + 0.5 * loss_step`). Result: step path generates 10-30 tokens, but forward path still significantly outperforms it (loss gap >4x).

**2. Information bottleneck**

A D=256 state `h` must compress the entire sentence. KV cache is O(L); state is O(1) — both a strength and a constraint.

**3. Gate still ignores accumulated context**

```python
h_new = GRUCell(x_norm, h)  # x_norm comes from current token only
```

GRUCell's reset and update gates operate on `x_norm` (current token + LayerNorm) — the input to step() does not include the accumulated state `h` in the attention-residual path. While GRU's own hidden-to-hidden weights partially address this (unlike the original linear gate), the residual stream `x` still receives no direct information about what has been accumulated.

**4. Bidirectional attention "cheats" — still an open problem**

Training predicts the next token while attention sees the future. D3's independent weights and step training help, but the underlying asymmetry remains: attention learns patterns that step() cannot reproduce, because step() never sees the full context matrix.

#### Summary

| Dimension | Advantage | Cost |
|-----------|-----------|------|
| Inference speed | O(1), independent of sequence length | h's capacity limits effective context |
| Training signal | Bidirectional, stronger gradients | Learned patterns partially transfer (D3 mitigates) |
| State update | GRUCell, expressive reset+update gates | Gate still sees only current token input |

#### A Practical Perspective

Beyond the theory, the architecture suggests several practical observations:

**1. If state h is sufficient, KV cache is a crutch**

Standard transformers need KV cache because the architecture has no explicit memory — every step must look back at the full history. If a fixed-size state h can encode the context needed for generation, then O(L) KV cache is an engineering hack, not an architectural advance.

**The implication is explosive**: the entire LLM inference infrastructure is built around KV cache optimization — PagedAttention, KV cache quantization, cache scheduling, memory management. If state-based models actually work, none of it is needed. Inference cost drops from O(L²) to O(L), with no hard context window limit.

Better yet: **each `step()` is truly O(1)** — the matrix multiply cost is fixed regardless of whether you've generated 100 tokens or 1 million. Even with KV cache, a standard transformer still does O(L) attention reads per step. A state model reads and computes in constant time. This means: local stability doesn't just save memory — inference speed itself improves dramatically. This raises a deeper question: **do you really need to remember everything to generate the next token?**

**2. Most real-world bottlenecks are local coherence, not long context**

- Dialogue — sentence-to-sentence flow matters more than 100-turn-old details
- Code generation — function-level consistency matters more than cross-file references
- Translation — term uniformity across a few sentences matters more than full-document memory
- Writing — paragraph fluency matters more than book-length structure

Long context is a nice-to-have; local stability is a must-have. **If local stability can be made reliable, it already covers the vast majority of practical needs.**

**3. Outline + chapter + sentence — cognitive granularity**

Jin Yong didn't write "The Demi-Gods and Semi-Devils" in one pass; he outlined chapters, then wrote one chapter at a time. Human writing is inherently hierarchical:

- **Outline mode**: occasional global view to verify direction
- **Chapter mode**: focused attention on the current block
- **Sentence mode**: step-by-step generation, depending on recent context

A model architecture doesn't need full global attention at every layer. It needs stable local generation with the ability to occasionally zoom out. This aligns with the intuition of "local first, global when needed."

---

### Project Files

```
myllm/
├── main.py              # v1: LSTM baseline — E_in/E_out mismatch
├── main_v2.py           # v2: SelectiveState + parallel training
├── main_v3.py           # v3: Transformer, unified D (current)
├── gen_corpus.py        # GPT-2 corpus generation
├── model_v3.pt          # v3 weights (~4.7MB)
├── checkpoint_v3.pt     # v3 checkpoint (resumable)
├── real_corpus.txt      # GPT-2 generated corpus
├── compare_v1_v2.py     # v1 vs v2 comparison
├── gen_v2*.py           # v2 generation scripts
├── diagnose_v2.py       # v2 diagnostics
├── check_model.py       # architecture verification
│
├── train_v4_causal.py   # B: Causal Transformer (10000 steps)
├── train_v4_gru.py      # C: Pure GRU baseline (10000 steps)
├── train_v4_phase2.py   # D: Two-phase frozen (10000 steps)
├── train_v4_joint.py    # D2: Joint forward+step (10000 steps)
├── train_v4_poetry_joint.py  # D2 poetry-only (10000 steps)
├── train_v4_poetry_d3.py     # D3: GRUCell step, poetry-only ⭐
│
├── test_v4_causal.py    # B test
├── test_v4_gru.py       # C test
├── test_v4_phase2.py    # D test (forward + step)
├── test_v4_joint.py     # D2 test (forward + step)
├── test_v4_poetry_joint.py  # D2 poetry test
├── test_v4_poetry_d3.py     # D3 poetry test ⭐
│
├── train_v4_ablation.py    # 消融实验: 统一训练脚本 (VARIANT=A/B/C/D/AC/ABC/ABCD)
├── test_v4_ablation.py     # 消融实验: 统一测试脚本
├── run_ablation_sequential.sh  # 消融实验: 顺序运行所有变体
├── run_ablations.sh        # 消融实验: 简易运行脚本
├── run_ablations.bat       # 消融实验: Windows批处理版本
│
├── model_v4_causal.pt   # B weights (~16MB)
├── model_v4_gru.pt      # C weights (~15MB)
├── model_v4_phase2.pt   # D weights (~18MB)
├── model_v4_joint.pt    # D2 weights (~18MB)
├── model_v4_poetry_joint.pt  # D2 poetry weights (~16MB)
├── model_v4_poetry_d3.pt     # D3 weights (~19MB) ⭐
│
├── corpora/
│   ├── tang_poetry/     # 全唐诗 JSON 数据
│   └── doraemon/        # 哆啦A梦电影剧本
│
└── README.md

### Key Takeaways

1. **Unified dimensions simplify everything** — E_in/E_out separation caused constant errors in v1; unified D=256 fixed them
2. **Attention is a powerful learning signal** — bidirectional attention converges fast, but learned patterns may not transfer to state-only inference
3. **Latent space geometry is real** — even at loss=0, semantic interpolation works. Representation and generation are separable
4. **More data > better architecture** — the bottleneck is corpus size, not model design
5. **Bezier is an intuition tool, not an architectural constraint** — the analogy helped derive local attention and segmented training, but the final model is just function fitting
6. **Gate expressiveness is the critical bottleneck** — leaky integrator (`h*gate + s*(1-gate)`) collapses in 2-5 steps; GRUCell with reset+update gates sustains 10-30 steps
7. **Independent train/inference weights solve the fundamental conflict** — decoupling forward (attention) weights from step (gating) weights lets each path optimize separately
8. **Step path must participate in training** — frozen step (D) or underpowered gating (D2) both collapse; GRUCell + joint training is the first working combination

### Training

```bash
# v3 (current)
python main_v3.py

# v4 — Causal Transformer (B, mixed corpus)
"/c/Users/savyc/miniconda3/python.exe" train_v4_causal.py

# v4 — Pure GRU (C, mixed corpus)
"/c/Users/savyc/miniconda3/python.exe" train_v4_gru.py

# v4 — Phase 2 frozen step (D, mixed corpus)
"/c/Users/savyc/miniconda3/python.exe" train_v4_phase2.py

# v4 — Joint forward+step (D2, mixed corpus)
"/c/Users/savyc/miniconda3/python.exe" train_v4_joint.py

# v4 — D3 GRUCell step, poetry-only (from scratch) ⭐
"/c/Users/savyc/miniconda3/python.exe" train_v4_poetry_d3.py
```

**Optimizer**: AdamW, lr=2e-3, weight_decay=0.01
**Loss**: CrossEntropyLoss, ignore PAD (index 0)
**Checkpoints**: Every 500 steps → `checkpoint_v4_*.pt`
**Training time**: ~10-12 min GPU (MX250) for 10,000 steps
**D3 inference**: step() is O(1) — no KV cache, constant cost per token
**Sampling**: temperature=0.8 + top_k=20 (argmax causes repetition)
**Corpus note**: B/C/D/D2 trained on mixed (poetry+Doraemon, vocab=6243); D2-poetry and D3 trained on pure Tang poetry from scratch (vocab ~3100). Do not cross-compare losses between the two groups.

### Citation

```bibtex
@software{curve_transformer_2026,
  title={Curve Transformer: Language as a Curve — Bezier-Inspired Semantic Interpolation},
  author={Mao Chen},
  year={2026},
  url={https://github.com/MaoChen1980/curve-transformer}
}
```

---

## License

MIT — this is a research prototype, not a production model.
