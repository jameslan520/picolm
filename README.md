# PicoLLM（含 Qwen2.5 GGUF 完整支持）

超轻量级大语言模型推理引擎，纯 C11 实现，单二进制、零第三方依赖
（仅需 libm / pthread），支持 GGUF 量化模型，可运行于：

- x86_64（SSE2，`-march=native` 自动启用）
- ARM 64 位（NEON，`make pi`）
- ARM 32 位（armv6/v7，交叉编译 `make pi-arm32`，实测 Debian armv7l 可用）
- RISC-V 64（`make riscv`）

> 本仓库为**加入 Qwen2.5 支持补丁后的源码存档**，相对上游的主要修改见
> [CHANGELOG.md](CHANGELOG.md)，完整开发过程与技术细节见
> [docs/qwen25-integration.md](docs/qwen25-integration.md)。

## 特性

- `make` 即可构建，无任何外部库
- 多量化格式：F32 / F16 / Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / Q4_K / Q5_K / Q6_K
- 增量解码 + FP16 KV cache + 在线 softmax（Flash Attention 风格）
- **双分词器**：SentencePiece（Llama 系）与 GPT-2 字节级 BPE（Qwen 系）自动识别
- 特殊 token 字面匹配（`<|im_start|>` / `<|im_end|>` 等 22 个控制 token）
- NeoX（rotate-half）RoPE、GQA 分组注意力、SwiGLU FFN、QKV bias
- 语法约束采样（JSON 模式）、temperature / top-p / greedy
- KV cache 落盘复用（`--cache`）
- 调试开关 `PICOLM_DEBUG_TOKENS=1`（详见下方）

## 构建

```bash
make            # 本机原生构建（-march=native）
make debug      # -O0 -g 调试构建
make static     # 静态链接单文件
make pi         # aarch64 + NEON
make clean
```

### 预编译二进制（armhf 系统：Debian armv7 / Raspberry Pi OS 等）

| 文件 | 说明 |
|---|---|
| picolm-armv7 | 通用 ARMv7 硬浮点（vfpv3-d16），任何 armhf 设备可运行，已实测 |
| picolm-armv7-neon | NEON 加速版（Cortex-A7/A9/A53 等），无 NEON 的 CPU（如 Cortex-A5）会非法指令崩溃 |
| tools/gguf-info、tools/q5-check | GGUF 结构检查与 Q5_0 反量化验证工具 |

运行：./picolm-armv7 model.gguf -p "prompt" -n 32 -t 0 -j 4


## 用法

```bash
./picolm <model.gguf> [options]

生成参数：
  -p <prompt>    输入提示词（或通过 stdin 管道传入）
  -n <int>       最大生成 token 数（默认 256）
  -t <float>     温度（默认 0.8，0 = greedy）
  -k <float>     top-p 采样（默认 0.9）
  -s <int>       随机种子（默认 42）
  -c <int>       上下文长度覆盖
  -j <int>       线程数（默认 4）

高级参数：
  --json         语法约束 JSON 输出模式
  --cache <file> KV cache 文件（保存/恢复 prompt 状态）
```

### Qwen2.5 模型示例（推荐使用聊天模板）

```bash
./picolm qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -p '<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
' -n 256 -t 0 -j 4
```

模板格式为 Qwen 标准：`<|im_start|>user\n{内容}<|im_end|>\n<|im_start|>assistant\n`
（GPT 风格分词器不会添加前导 BOS，与 HF `apply_chat_template` 行为一致）

### 调试开关

```bash
PICOLM_DEBUG_TOKENS=1 ./picolm model.gguf -p "..." -n 4
```

输出：prompt 逐步分词结果、模型加载时 bias/分词器状态、
首个生成位置的 top-5 logits（定位生成质量问题时使用）。

## 已验证模型

| 模型 | 分词器 | 结果 |
|---|---|---|
| TinyLlama-1.1B-Chat-v1.0 Q4_K_M | SentencePiece | ✓ 回归通过（"The capital of France is Paris."） |
| Qwen2.5-0.5B-Instruct Q4_K_M | GPT-2 字节级 BPE | ✓ 答案正确、EOS 自然收尾 |

## 性能实测

测试环境：Debian bookworm **armv7l**（4 核 / 981MB RAM），`-j 4 -t 0`：

| 模型 | Prefill | Generation | 内存 |
|---|---|---|---|
| Qwen2.5-0.5B Q4_K_M（13 token 模板 prompt） | 0.5 tok/s | **0.5 tok/s**（121 token 生成） | 393 MB |
| TinyLlama-1.1B Q4_K_M | 0.2 tok/s | 0.6 tok/s | ~400 MB |

> Qwen 偏慢的主因：Q5_0 格式权重（133 个张量）走 `vec_dot` 的
> 通用 dequantize 兜底路径，尚未写专用 NEON 融合内核。见
> [docs/qwen25-integration.md](docs/qwen25-integration.md#性能与已知限制)。

## 仓库结构

```
├── picolm.c / model.c / tensor.c / quant.c / tokenizer.c
├── sampler.c / grammar.c          # 主程序与各模块
├── *.h                            # 对应头文件
├── Makefile                       # 多平台构建
├── CHANGELOG.md                   # 相对上游的修改清单
├── docs/
│   └── qwen25-integration.md      # 开发文档：6 项修复的技术细节与验证过程
└── tools/
    ├── gguf-info.c                # GGUF 元数据/张量结构检查器
    └── q5-check.c                 # Q5_0 反量化正确性对比工具
```

## 许可证

未随附上游项目的 LICENSE 文件，请参照原始 PicoLLM 项目的许可条款。
本仓库的修改部分欢迎在同等条款下使用。

