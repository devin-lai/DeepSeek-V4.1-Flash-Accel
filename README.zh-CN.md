# DeepSeek V4.1 Flash Accel

| **输出吞吐 5.6×** | **逐 token 延迟降低 84.9%** | **8K 预填充吞吐提高 9.5%** | **8K 首 token 延迟降低 8.7%** |
| :--- | :--- | :--- | :--- |
| **6.0 → 33.7 token/s** · 单并发 | **159.52 → 24.04 ms** · 中位 TPOT | **2,424.0 → 2,654.5 total token/s** | **6,289.2 → 5,741.0 ms** |
| CUDA graphs 对比同补丁 eager 模式 | 同一 CUDA graphs 对照实验 | 仅卸载解码器专家，对比默认层顺序 | 同一卸载位置对照实验 |

**测量配置：8× RTX 5090，503 GiB 主机内存。** 记录于 2026-09-14，来自短时合成负载。
CUDA graphs 对照使用 1,024 输入 / 128 输出 token；独立的 eager 卸载位置实验使用
8,192 输入 / 1 输出 token、并发 2。详见[实测性能](#实测性能)。

**使用 vLLM 在 NVIDIA RTX 5090 上部署并加速 DeepSeek-V4.1-Flash。**

[快速开始](#快速开始) · [优化原理](#优化原理) · [基准测试与复现](benchmarks/README.md) · [文档](docs/README.md) · [English](README.md)

本项目在 **8× RTX 5090、503 GiB 主机内存**的机器上运行已发布的模型权重，
提供 vLLM / FlashInfer 补丁、部署预设、CPU 卸载策略、内存规划、故障诊断和研究工具。
这是独立社区项目；模型来自 [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)，
推理依赖 vLLM、FlashInfer 和 DeepGEMM。本项目与 DeepSeek 官方无隶属或背书关系；
目前处于实验阶段，仅有一种硬件配置的测量记录。

TTFT 为首 token 延迟；TPOT 为首 token 之后的平均逐 token 延迟。
吞吐统计整个测试负载中的请求，不能直接等同于单个用户的解码速度。

## 部署条件与支持范围

截至 2026-09-14，仓库内的测量记录覆盖以下配置：

| 项目 | 当前状态 |
| --- | --- |
| GPU | 8× RTX 5090，每张 32 GB，SM120，PCIe Gen5 ×16，无 NVLink |
| 主机内存 | 安装 503 GiB；运行记录中约 452 GiB 被锁页 |
| 模型文件 | 约 476 GiB / 510 GB；还需为环境和缓存预留磁盘空间 |
| 卸载 | Engram 放在 CPU；每个 GPU rank 卸载 12 GiB 解码器专家权重 |
| 文本 | `v41-flash` 可提供服务，CUDA graphs 已启用，需应用仓库补丁 |
| 图像 | `v41-flash-vision` 已通过简单图像探测，更广泛的视觉评测待补充 |
| 上下文 | 预设为 **32,768 token**；本部署尚未验证模型宣称的 1M 上下文 |
| 其他硬件 | 尚需独立验证；以上配置不是经过证明的最低硬件要求 |

固定环境为 vLLM 提交 `8c1d1c2974ee42757ee2e93cc898932edfd9d265` 加本地补丁、
FlashInfer `0.6.18.post1`、PyTorch 2.13 + cu130、CUDA 13.2 toolkit、驱动 595.71.05。
详见[硬件记录](docs/02-hardware-topology.md)和[部署手册](deploy/README.md)。

## 快速开始

在匹配上述配置的 Linux NVIDIA 主机上，准备 Python 3、`pip` 或 `uv`、
`aria2c`、`curl` 和 CUDA toolkit，并确保 `nvcc` 位于 `PATH`。
安装脚本创建 Python 3.11 环境。示例使用可写的 `/data` 目录；如目录布局不同，
请相应调整模型、环境和日志路径。以下命令从仓库根目录执行：

```bash
git clone https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel.git
cd DeepSeek-V4.1-Flash-Accel

# 下载：使用 ModelScope 和 hf-mirror，支持断点续传。
MODEL_DIR=/data/models/DeepSeek-V4.1-Flash bash scripts/download/download_model.sh
python3 scripts/download/verify_shards.py /data/models/DeepSeek-V4.1-Flash

# 安装固定版本环境、插件与补丁，并执行预检。
# 中国大陆用户可将 PYPI 改为 https://pypi.tuna.tsinghua.edu.cn/simple。
PYPI=https://pypi.org/simple \
  VENV=/data/venvs/vllm-dsv41 MODEL=/data/models/DeepSeek-V4.1-Flash \
  bash scripts/env/setup.sh
source /data/venvs/vllm-dsv41/bin/activate

# 前台运行文本服务，仅监听本机。
HOST=127.0.0.1 MODEL=/data/models/DeepSeek-V4.1-Flash \
  PRESET=v41-flash deploy/serve.sh
```

服务就绪后，在仓库根目录打开第二个终端：

```bash
source /data/venvs/vllm-dsv41/bin/activate
deploy/healthcheck.sh
mkdir -p benchmarks/local
python deploy/verify.py --json benchmarks/local/verify.json

curl -sS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","prompt":"The capital of France is","max_tokens":8,"temperature":0}'
```

请在测速前检查探测输出和保存的 JSON；通过这些基础检查不代表完整模型质量已验证。
系统页缓存已热时，记录中的 eager 启动耗时约 285 秒；下载和首次编译另计。
权重校验默认检查文件结构，添加 `--sha256` 可与 ModelScope 的哈希记录比较。
如需图像输入，先停止文本服务，再以 `PRESET=v41-flash-vision` 启动。

[自定义路径、预设与 systemd](deploy/README.md) · [补丁检查与回滚](upstream/README.md)

## 优化原理

### 1. 降低 CUDA 内核启动开销

CUDA graph replay 复用已捕获的执行流程，减少 CPU 逐次启动内核的开销。
vLLM 补丁在捕获后清零空 KV-cache 块，FlashInfer 补丁让被掩码的位置读取有限值，
解决该固定环境中捕获后输出异常的问题；部署仍需要仓库中的其他分派与缓存布局补丁。

![CUDA graphs 复用已捕获的内核序列；补丁在捕获后清零空 KV 块，并为被掩码的位置读取有限值。](docs/assets/cuda-graphs.webp)

**实测：** 单并发输出吞吐 6.0 → 33.7 token/s，中位 TPOT 159.52 → 24.04 ms。
详见[捕获问题调查](benchmarks/results/2026-09-14-v41-cudagraphs.md)。

### 2. 保留编码器专家，仅卸载部分解码器专家

将专家卸载限制在第 20–39 层，使第 0–19 层的编码器专家权重保留在 GPU 上。
部分解码器专家权重和 Engram 表放在主机内存；专家计算仍在 GPU 上执行，
通过 PCIe/UVA 读取主机权重。Engram 查询仍会访问主机内存。

![8 张 RTX 5090 保留编码器专家；Engram 和部分解码器专家权重位于 CPU 内存，通过 PCIe/UVA 被 GPU 读取。](docs/assets/expert-placement.webp)

**实测：** 独立的 eager 卸载位置实验中，8K 预填充总吞吐提高 9.5%，首 token 延迟降低 8.7%。
详见[卸载位置实验](benchmarks/results/2026-09-14-v41-first-serve.md#where-the-offloaded-experts-should-live)。

| 其他优化 | 作用与证据范围 |
| --- | --- |
| 专家并行 | 按完整专家分配，避免张量并行的填充开销。参考布局计算中，专家权重为 33.6 vs 44.8 GiB/rank，减少 25%；这不代表总运行显存减少 25%。[内存分析](docs/03-deployment-design.md#parallelism-expert-parallelism-not-pipeline-parallelism) |
| 单次主机拷贝 | 直接分配锁页缓冲区并复制权重，避免临时的普通主机内存副本。[卸载插件](vllm_dsv41_opt/README.md) |

插图用于解释机制，数值结论以表格和链接的结果文件为准。

## 实测性能

V4.1-Flash，TP8 + 专家并行，Marlin，文本模式，卸载配置如上。
测试使用 `vllm bench serve`、随机 1,024-token 输入 / 128-token 输出、`ignore_eos`。

| 客户端并发请求 | CUDA graphs 输出 token/s | eager 输出 token/s | 加速比 | CUDA graphs 中位 TTFT | CUDA graphs 中位 TPOT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **33.7** | 6.0 | **5.6×** | 692 ms | **24.04 ms** |
| 8 | **92.4** | 38.2 | **2.4×** | 2,167 ms | 72.28 ms |
| 32 | **115.8** | 73.7 | **1.6×** | 19,688 ms | 128.91 ms |

5.6 倍指 `33.7 / 6.0`，比较的是同一部署的 CUDA graphs 与 patched eager 模式，
不是与最新原版 vLLM 或其他推理引擎的比较。输出吞吐包含请求全过程，不能直接用
TPOT 的倒数代替。TTFT 为首 token 延迟，TPOT 为首 token 之后的平均逐 token 延迟。
服务端预设 `MAX_SEQS=16`，因此 32 并发测试包含排队。
三个测试分别只有 4、16、64 个请求，尚无重复试验分布或生产负载评测。

[CUDA graphs 结果](benchmarks/results/2026-09-14-v41/kit-v41-graphs-bench.json) ·
[eager 基线](benchmarks/results/2026-09-14-v41/kit-v41-preset-bench.json) ·
[复现方法](benchmarks/README.md)

**独立的卸载位置实验：** eager 模式，8,192 输入 / 1 输出 token，并发 2，4 个请求。
只改变允许卸载专家的层范围：

| 卸载位置 | 预填充总 token/s | 中位 TTFT |
| --- | ---: | ---: |
| 默认层顺序 | 2,424.0 | 6,289.2 ms |
| 解码器第 20–39 层 | **2,654.5（+9.5%）** | **5,741.0 ms（−8.7%）** |

[卸载位置结果](benchmarks/results/2026-09-14-v41/ladder2-offload-placement.json)。
该收益不能与 CUDA graphs 加速比相乘；total token/s 包含输入和输出 token。

质量证据目前限于基础探测：CUDA graphs 运行记录中，贪心续写匹配 5/6，
一段简短英文文本的困惑度为 2.662，应用题回答为 `$8`。
未匹配的一项是合理续写，但未包含探测所要求的词。
这些结果不能证明完整模型质量保持不变；相同配置的多次运行也曾出现不同续写和困惑度。
详见[探测输出](benchmarks/results/2026-09-14-v41/kit-v41-graphs-verify.json)。

## 参与研究与贡献

欢迎用中文或英文提交问题、复现结果和改进。尤其需要其他 GPU 配置、
长上下文、多图像任务、DSpark 推测解码以及流水线并行的验证。
提交性能结果时请提供模型与代码版本、启动参数、硬件、输出检查和基准文件。
参见[贡献指南](CONTRIBUTING.md)。

[18 类故障记录](docs/05-fault-inventory.md)、[小模型结构复现](tools/tiny/README.md)、
[稀疏 MLA 参考实现](tools/sm120_sparse_mla/README.md)可供研究复用。
[量化研究](docs/06-expert-quantization.md)中的低比特格式尚不是经过端到端验证的部署预设。
补丁报告提供故障复现和修复证据，尚未提交或被上游接受。

[仓库检查](https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel/actions/workflows/check.yml)
覆盖公开文件、语法和文档链接；GPU 性能与输出质量需要在目标硬件上单独验证。

成功部署后，欢迎[分享硬件与测量结果](https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel/issues/new?template=measurement.md)。
引用格式见 [CITATION.cff](CITATION.cff)，请补充实际使用的提交，并引用模型及相关上游项目。
仓库采用 [MIT](LICENSE)，卸载插件采用 [Apache-2.0](vllm_dsv41_opt/LICENSE)，
组件许可与上游归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
