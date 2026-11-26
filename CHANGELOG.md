# CHANGELOG — 相对上游 PicoLLM 的修改清单

本次修改的目标：让 PicoLLM 完整支持 **Qwen2.5 系列 GGUF 模型**，
同时保持对原有 Llama 系模型（SentencePiece 分词）的完全兼容。

测试基线：Qwen2.5-0.5B-Instruct-Q4_K_M 与 TinyLlama-1.1B-Chat-Q4_K_M，
环境 Debian bookworm armv7l（4 核 / 981MB RAM）。

---

## 1. quant.h / quant.c — 新增 Q5_0 量化格式

**问题**：Qwen2.5-0.5B 的 GGUF 中含 133 个 Q5_0 张量（attn_q/k/v、
ffn 矩阵等），上游 quant.h 只注册了 F32/F16/Q4_0/Q4_1/Q5_1/Q8_0/Q4_K/Q6_K，
加载时行大小计算为 0，权重被直接丢弃。

**修改**：
- `quant.h`：新增 `ggml_block_q5_0` 结构体
  （32 元素 × 4bit + 16×1bit = 22 字节，`block_size=32, size=22`）
- `quant.c`：新增 `dequantize_row_q5_0()`（QK5_0 特例：高半字节为
  单独的 4bit 半值、d 为 fp16）
- `gguf_type_block_size()` / `gguf_type_quant_size()` 分发表补 Q5_0
- `vec_dot` 分发链未注册 Q5_0，走 `dequantize_row` → 点积的兜底路径
  （正确但无融合优化，见性能一节）

**验证**：`tools/q5-check.c` 对孪生张量 `token_embd`（Q8_0）与
`output.weight`（Q5_0）逐行反量化后做全排列点积对比，
Q5_0 vs F32 参考与 Q8_0 vs F32 参考误差同量级（≈1e-3），反量化正确。

---

## 2. model.h / model.c — QKV bias 支持

**问题**：Qwen2.5 每层注意力都有 Q/K/V 三个 bias 向量（24 层 × 3 = 72 个，
head_dim 大小），上游完全忽略 `blk.N.attn_{q,k,v}.bias`，导致输出乱码。

**修改**：
- `model.h`：每层新增 `attn_q/k/v_bias` 指针与 `type_attn_*_bias` 字段
- `model.c` 加载：解析 `blk.N.attn_{q,k,v}.bias`（F32，与权重同布局）
- `model.c` 前向：q/k/v 投影后分别 `vec_add`；
  **k bias 必须加在 RoPE 旋转之前**（HF 的 apply_rotary_pos_emb
  作用于 bias 加完之后的结果）

**验证**：`PICOLM_DEBUG_TOKENS=1` 启动时打印
`layers with all 3 biases: 24/24`，指针与 type（0=F32）运行时确认。

---

## 3. model.h / model.c — GGUF 元数据加载扩展

**问题**：上游不解析分词所需的 `tokenizer.ggml.merges` / `pre` /
`token_type`，也不识别 Qwen 的 `qwen2.*` 键前缀。

**修改**：
- 新增解析键（均为 GGUF 数组类型，正确跳过/挂载）：
  - `tokenizer.ggml.merges` → `tok_merges_data` / `tok_n_merges`
  - `tokenizer.ggml.pre` → `tok_pre_data` / `tok_pre_len`
  - `tokenizer.ggml.token_type` → `tok_types_data` / `tok_n_types`（u32 数组）
- **键前缀归一化**：读取前把 `qwen2.attention.*` / `qwen2.block.*`
  等替换为 `llama.*`，使 Qwen GGUF 复用 llama 分支
  （rope_freq_base、layer_norm_rms_epsilon 等由此正确读取）
- **head_dim 除零保护**：`head_dim = n_embd / n_heads` 为 0 时回退到
  `attn_q_dim / n_heads`
- 新增读取 `llama.attention.layer_norm_rms_epsilon` → 全局 `g_rms_eps`
  （Qwen 为 1e-6，上游硬编码 1e-5；缺键时保持 1e-5 不影响 Llama 系）

---

## 4. tokenizer.h / tokenizer.c — 分词器重写（GPT-2 字节级 BPE）

**问题**：上游分词器只支持 SentencePiece（以 0x2581 表示空格），
而 Qwen 是 GPT-2 字节级 BPE（以 U+0120 表示空格，且词表没有
merge/scores 元数据）。原编码器输出完全错误的 token，
是输出乱码的根本原因之一。

**修改**（tokenizer.c 整体重写，原文件为备份 tokenizer.c.bak）：
- **字节渲染**：GPT-2 `bytes_to_unicode` 规则——空格→U+0120、
  换行→U+010A；可打印字节(33-126/161-172/174-255)原样；
  其余字节渲染为 U+0100+n 的可打印形式
- **解码逆映射**：码点 256..323 查表还原原始字节，
  CJK 等 >323 码点原样透传
- **merges rank 哈希表**：读取 `tokenizer.ggml.merges` 构建
  (a,b)→rank 线性探测哈希表（key=(a+1)<<32|(b+1)），按 rank
  从低到高贪心合并——Qwen 无 scores 元数据，rank 即合并顺序；
  有 scores 的模型（Llama SP）保留原路径
- **风格自动检测**：`tokenizer.ggml.pre` 匹配 gpt2/qwen2/jina/
  refact/starcoder 等→GPT 风格；否则扫描词表统计
  U+0120 与 0x2581 前缀数量判定
- **特殊 token 字面匹配**：`token_type` 为 2/3/4 的 token
  （本模型 3 个核心特殊 token + 共 22 个）在编码时优先按字面
  匹配——`<|im_start|>` 无 merges 规则，BPE 无法从字符拼出，
  必须整串识别（否则被拆成 6 个碎片，模型无法识别对话边界）
- **GPT 风格不加前导 BOS**，与 HF `apply_chat_template` 一致
- SentencePiece 路径原样保留，TinyLlama 回归测试通过

---

## 5. tensor.c — RoPE 改为 NeoX（rotate-half）风格

**问题**：上游按交错式配对 (2i, 2i+1) 旋转，而 HF 的
Llama/Qwen 实现为 NeoX 旋转半式 (i, i+half)。两种约定变换不同，
造成持续性质量退化（表现为回声式输出、答案缺失）。

**修改**：`rope()` 对 q 与 k 的每个头改为
`pair(i, i + head_dim/2)`，角度仍为 pos / base^(2i/d)。
此为最后一处修复，改后两模型输出同时转正。

---

## 6. tensor.h / tensor.c — rms eps 可配置

新增全局 `g_rms_eps`（默认 1e-5），`rmsnorm()` 使用之；
model.c 读取 `llama.attention.layer_norm_rms_epsilon` 覆盖
（Qwen 1e-6）。

---

## 7. 调试功能与工具

- `picolm.c` / `model.c` / `tokenizer.c`：环境变量
  `PICOLM_DEBUG_TOKENS=1` 时打印 prompt 分词结果、首步
  top-5 logits、bias 加载状态、token_type 信息（默认静默）
- `tools/gguf-info.c`：GGUF 元数据与张量结构检查器
  （踩坑记录：bool 为 1 字节、张量区不加填充、
  张量 offset 为 u64、基址需按最大块对齐）
- `tools/q5-check.c`：Q5_0 反量化正确性对比工具

---

## 8. 测试结果

- TinyLlama-1.1B-Chat-Q4_K_M（SentencePiece 回归）：
  prompt "The capital of France is" → 正确输出 Paris，style=sentencepiece
- Qwen2.5-0.5B-Instruct-Q4_K_M（最终验收）：
  模板 prompt "What is the capital of France?" → 正确输出
  "The capital of France is Paris."（EOS 自然收尾）；
  分词 style=gpt2，15 tokens 与 HF apply_chat_template 完全一致

性能（armv7l 4核 981MB，-j 4 -t 0）：
- Qwen2.5-0.5B：Prefill 约 0.5 tok/s，生成约 0.5 tok/s，内存 393MB
- 瓶颈：Q5_0 权重走 dequantize 兜底路径，无 NEON 融合 vec_dot

开发期备份（未包含在本包内，留在服务器 /root/.picolm/src/）：
model.c.bak、model.c.fwd.bak、model.h.bak、quant.c.bak、
quant.h.bak、tokenizer.c.bak —— 各为对应文件修改前的版本。
