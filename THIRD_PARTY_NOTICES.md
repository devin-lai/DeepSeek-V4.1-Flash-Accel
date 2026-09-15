# Licenses and attribution

This is an independent community project. DeepSeek supplies the model; vLLM,
FlashInfer, DeepGEMM, and PyTorch supply the underlying inference stack.

| Material | License and source |
| --- | --- |
| Original deployment scripts, tools, and documentation | [MIT](LICENSE) |
| `vllm_dsv41_opt/` offload plugin | [Apache-2.0](vllm_dsv41_opt/LICENSE), as declared by the package |
| Edits to vLLM in `upstream/vllm/` | Upstream-derived code remains subject to [vLLM's Apache-2.0 license](https://github.com/vllm-project/vllm/blob/main/LICENSE) |
| Edits to FlashInfer in `upstream/flashinfer/` | Upstream-derived code remains subject to [FlashInfer's Apache-2.0 license](https://github.com/flashinfer-ai/flashinfer/blob/main/LICENSE) |

Patch scripts modify separately installed upstream packages. Preserve their
copyright and license notices when applying or redistributing changes. The
root MIT license does not replace component licenses.

Model weights and third-party packages are downloaded separately. Consult
the [DeepSeek-V4.1-Flash model license](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/LICENSE),
[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), and
[PyTorch](https://github.com/pytorch/pytorch) for their own terms and attribution.

The optional GSM8K regression harness uses the separately downloaded test set
from [OpenAI's grade-school-math repository](https://github.com/openai/grade-school-math),
released under its [MIT license](https://github.com/openai/grade-school-math/blob/master/LICENSE)
(Copyright 2021 OpenAI). Saved evaluation artifacts record row IDs, question
hashes, expected numbers and model outputs; they do not bundle the test set.
