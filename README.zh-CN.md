# DeepSeek V4.1 Flash Accel

**使用 vLLM 在 NVIDIA RTX 5090 上部署并加速 DeepSeek-V4.1-Flash。**

[English](README.md) · [基准测试与复现](benchmarks/README.md) · [文档](docs/README.md) · [参与贡献](CONTRIBUTING.md)

本项目在 **8× RTX 5090 D、503 GiB 主机内存**的机器上运行已发布的模型权重，
提供 vLLM / FlashInfer 补丁、部署预设、CPU 卸载策略、内存规划、故障诊断和研究工具。
这是独立社区项目；模型来自 [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)，
推理依赖 vLLM、FlashInfer 和 DeepGEMM。本项目与 DeepSeek 官方无隶属或背书关系；
目前处于实验阶段，仅有一种硬件配置的测量记录。

**已记录的单并发输出吞吐为 33.7 token/s，是同一套部署在 eager 模式下的 5.6 倍。**
该结果来自单台机器上的短时合成测试；比较基线、硬件条件和结果文件见下文。

## 部署条件与支持范围

截至 2026-09-14，仓库内的测量记录覆盖以下配置：

| 项目 | 当前状态 |
| --- | --- |
| GPU | 8× RTX 5090 D，SM120，PCIe Gen5 ×16，无 NVLink |
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
以下命令从克隆或解压后的仓库根目录执行：

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

# 前台运行文本服务。
MODEL=/data/models/DeepSeek-V4.1-Flash PRESET=v41-flash deploy/serve.sh
```

服务就绪后，在仓库根目录打开第二个终端：

```bash
source /data/venvs/vllm-dsv41/bin/activate
deploy/healthcheck.sh
python deploy/verify.py

curl -sS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","prompt":"The capital of France is","max_tokens":8,"temperature":0}'
```

系统页缓存已热时，记录中的启动耗时约 285 秒；下载和首次编译另计。
权重校验默认检查文件结构，添加 `--sha256` 可与 ModelScope 的哈希记录比较。
如需图像输入，先停止文本服务，再以 `PRESET=v41-flash-vision` 启动。

## 实测性能

V4.1-Flash，TP8 + 专家并行，Marlin，文本模式，卸载配置如上。
测试使用 `vllm bench serve`、随机 1,024-token 输入 / 128-token 输出、`ignore_eos`。

| 客户端并发请求 | CUDA graphs 输出 token/s | eager 输出 token/s | CUDA graphs 中位 TTFT | CUDA graphs 中位 TPOT |
| ---: | ---: | ---: | ---: | ---: |
| 1 | **33.7** | 6.0 | 692 ms | **24.04 ms** |
| 8 | **92.4** | 38.2 | 2,167 ms | 72.28 ms |
| 32 | **115.8** | 73.7 | 19,688 ms | 128.91 ms |

5.6 倍指 `33.7 / 6.0`，比较的是同一部署的 CUDA graphs 与 patched eager 模式，
不是与最新原版 vLLM 或其他推理引擎的比较。输出吞吐包含请求全过程，不能直接用
TPOT 的倒数代替。TTFT 为首 token 延迟，TPOT 为首 token 之后的平均逐 token 延迟。
服务端预设 `MAX_SEQS=16`，因此 32 并发测试包含排队。
三个测试分别只有 4、16、64 个请求，尚无重复试验分布或生产负载评测。

[CUDA graphs 结果](benchmarks/results/2026-09-14-v41/kit-v41-graphs-bench.json) ·
[eager 基线](benchmarks/results/2026-09-14-v41/kit-v41-preset-bench.json) ·
[复现方法](benchmarks/README.md)

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

成功部署后，欢迎[分享硬件与测量结果](https://github.com/devin-lai/DeepSeek-V4.1-Flash-Accel/issues/new?template=measurement.md)。
引用格式见 [CITATION.cff](CITATION.cff)，请补充实际使用的提交，并引用模型及相关上游项目。
仓库采用 [MIT](LICENSE)，卸载插件采用 [Apache-2.0](vllm_dsv41_opt/LICENSE)，
组件许可与上游归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
