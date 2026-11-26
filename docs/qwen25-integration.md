# Qwen2.5 GGUF 集成开发文档

> 目标：让 PicoLLM（原生只支持 Llama 系 SentencePiece 模型）
> 完整支持 Qwen2.5 系列 GGUF，记录全部根因、定位方法与验证过程。

## 1. 背景与环境

- 硬件/系统：Debian bookworm armv7l，4 核 / 981MB RAM，无 GPU
- 软件：仅 gcc/make（无 Python、无 llama.cpp 可对照）
- 网络：huggingface.co / deb.debian.org / github 不通，
  仅 hf-mirror.com、mirrors.aliyun.com 可用
- 目标模型：Qwen2.5-0.5B-Instruct-Q4_K_M.gguf（约 500MB）

## 2. Qwen2.5 与上游支持范围的差异

| 维度 | PicoLLM 原生（Llama 系） | Qwen2.5 |
|---|---|---|
| 分词器 | SentencePiece，空格=0x2581 | GPT-2 字节级 BPE，空格=U+0120 |
| merges/scores | 有 scores，按分数合并 | 无 scores，按 merges 顺序（rank）合并 |
| 特殊 token | 词表内可拼出 | `<|im_start|>` 等无 merges，只能整串识别 |
| QKV bias | 无 | 72 个（24 层 × q/k/v，F32） |
| RoPE | 交错式（原实现） | NeoX 旋转半式（HF 实现） |
| rms eps | 硬编码 1e-5 | 1e-6 |
| 权重量化 | Q4_K/Q6_K/Q8_0 等 | 含 133 个 Q5_0 张量 |
| GGUF 键前缀 | llama.* | qwen2.*（需归一化） |

任一差异未处理都会导致输出退化；本次共修复 6 项。
## 3. 调试方法论

没有 Python/llama.cpp 对照时的定位手段：
1. 自写 GGUF 解析器（tools/gguf-info.c）核对元数据与张量表
2. 分词对照：PICOLM_DEBUG_TOKENS=1 打印逐步 token，与 HF 预期比对
3. logits 检查：打印首步 top-5，判断前向质量
4. 孪生张量交叉验证量化反量化正确性
5. 逐层运行时断点（bias 指针、type 字段打印）

### GGUF 格式踩坑（写 gguf-info.c 时）

- BOOL 类型占 1 字节（非 4 字节）
- 头部：magic(4) + version(4) + n_tensors(8) + n_kv(8)，随后 KV
- 每个 KV：key 长度(8)+key、value type(4)、value
- 张量元数据：name、n_dims(4)、每维(8)、type(4)、offset(8)
- 张量数据区紧跟元数据区，每个张量 offset 为相对基址的 u64，
  不做逐张量填充，但基址需按最大块对齐（本模型 48）
- 数组类型需递归跳过（skipval 递归处理嵌套 array）

## 4. 六项根因与修复

### 4.1 Q5_0 量化格式缺失

现象：加载即丢权重，所有含 Q5_0 的张量行大小算成 0。
定位：gguf-info 对照张量表与 gguf_type_* 分发表。
修复：quant.h 加 block 结构（32/22），quant.c 加
dequantize_row_q5_0（高半字节单取、d 为 fp16），分发表补齐。
验证：tools/q5-check.c 孪生张量（token_embd Q8_0 vs
output.weight Q5_0）全排列点积与 F32 参考误差同量级（约 1e-3）。

### 4.2 QKV bias（72 个 F32 向量）

现象：乱码输出（补 bias 后乱码消失、变连贯英文但质量差）。
定位：GGUF 张量表存在 blk.N.attn_{q,k,v}.bias，前向未消费。
修复：model.h 加字段与指针，model.c 映射张量，前向在
q/k/v 投影后 vec_add；k bias 加在 RoPE 之前（HF 的
apply_rotary_pos_emb 作用于 bias 之后的结果）。
验证：PICOLM_DEBUG_TOKENS=1 打印 24/24 层指针非空、type=0。

### 4.3 GPT-2 字节级分词器（核心修复）

现象：Prompt tokens 与 HF 完全不同，输出乱码。
定位：词表空格前缀为 U+0120（GPT-2 风格），原编码器只处理
0x2581（SentencePiece）；且 Qwen 无 scores，合并依据是
merges 顺序（rank）。
修复（tokenizer.c 重写）：
- 字节渲染：空格-U+0120、LF-U+010A、可打印字节原样、
  其余字节渲染为 U+0100+n
- 解码逆映射：256..323 查表还原，CJK 等原样透传
- merges 哈希表：key=(a+1)<<32|(b+1) 线性探测，rank=顺序
- 风格检测：tokenizer.ggml.pre 匹配 gpt2/qwen2 等，
  否则扫词表前缀计数

### 4.4 特殊 token 字面匹配

现象：模板 prompt 中 im_start 被拆成 6 个碎片
（27,91,318,4906,91,29），模型看不到对话边界，输出回声。
定位：特殊 token 只存在于词表、没有 merges 规则，BPE 拼不出。
修复：加载 tokenizer.ggml.token_type（本模型类型值为 3，
不是常见文档说的 2，收集类型 2/3/4 共 22 个），编码时在
字节渲染之前做最长字面匹配，直接输出 token id；
GPT 风格不加前导 BOS（与 HF apply_chat_template 一致）。
验证：模板 15 tokens 与 HF 完全一致：
[151644 im_start][user][nl]...[151645 im_end][nl][151644][assistant][nl]

### 4.5 NeoX（rotate-half）RoPE —— 最后一处根因

现象：分词/bias 全部正确后，输出仍是回声式文本
（如复述问题后提前结束），首步 logits 不含正确答案。
定位：对照 HF modeling 代码——Llama/Qwen 的 rotary 为
pair(i, i+half) 旋转半式；原实现是交错式 pair(2i, 2i+1)，
两种约定变换不同，造成持续性质量退化。
修复：tensor.c rope() 改为对 q/k 每个头按 (i, i+half) 配对，
角度不变 pos/base^(2i/d)。
验证：Qwen 输出 The capital of France is Paris.（EOS 收尾），
TinyLlama 回归同时通过——该风格对 Llama 系同样正确。

### 4.6 rms eps

原硬编码 1e-5；Qwen 为 1e-6。改为全局 g_rms_eps，
由 layer_norm_rms_epsilon 元数据覆盖（缺键保持 1e-5）。

## 5. 症状与根因对照（排查顺序参考）

| 症状 | 根因 |
|---|---|
| 加载后权重全空 | Q5_0 格式缺失（行大小=0） |
| 输出乱码/非文本 | QKV bias 缺失 |
| 分词与 HF 不一致 | GPT-2 BPE 未支持（merges/pre） |
| 特殊 token 被拆碎 | 无字面匹配（token_type） |
| 回声式输出、答案缺失 | RoPE 风格错误（交错 vs NeoX） |
| 质量轻微退化 | rms eps 不符 |

建议按此顺序逐项排除，每修一项跑一次回归
（TinyLlama + Qwen 模板问答）。

## 6. 性能与已知限制

实测（armv7l 4 核 981MB，-j 4 -t 0）：
- Qwen2.5-0.5B：prefill 0.2-0.6 tok/s，生成 0.5-0.6 tok/s，393MB
- TinyLlama-1.1B：生成约 0.6 tok/s

瓶颈：Q5_0 的 133 个张量没有融合 vec_dot，走
dequantize_row → 通用点积兜底（每行都要临时缓冲）。
优化方向：为 Q5_0 写 vec_dot_q5_0_q8_0（可参照 Q8_0 的
NEON 实现），预期提速 1.5-2 倍；也可在加载时把 Q5_0
预反量化到 F16 缓冲（内存换时间，0.5B 模型约 +1GB 不可行，
大内存平台可选）。

其他限制：
- picolm-server 与 picoclaw 集成时，生成慢导致 HTTP 504，
  需调大上游超时或提速
- 未实现 Q5_K / Q3_K 等更多格式（当前模型未用到）
- KV cache 仅 FP16

## 7. 附录：调试工具与验收

编译工具：
  gcc -O2 -o tools/gguf-info tools/gguf-info.c
  gcc -O2 -o tools/q5-check tools/q5-check.c -lm

gguf-info <model.gguf>   # 打印元数据键与张量表（dims/type/offset）
q5-check [model.gguf]    # Q5_0 vs Q8_0 孪生张量反量化对比

回归验收（两项都通过才算修改完整）：
  # 1. SentencePiece 回归
  ./picolm tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -p "The capital of France is" -n 16 -t 0 -j 4
  # 2. Qwen 模板问答（应输出 Paris 并以 im_end 结束）
  ./picolm qwen2.5-0.5b-instruct-q4_k_m.gguf -p "模板prompt" -n 256 -t 0 -j 4
  # 分词调试
  PICOLM_DEBUG_TOKENS=1 ./picolm <model> -p "hi" -n 1 -t 0
