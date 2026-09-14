# DeepSeek-V4.1 cannot use pipeline parallelism: MoE routing needs `input_ids`, which no stage past 0 receives

**Component:** `vllm/models/deepseek_v4_1/`, pipeline `IntermediateTensors`
**Version:** vLLM main @ `8c1d1c297`
**Hardware:** 8x RTX 5090, DeepSeek-V4.1-Flash, TP4 x PP2

## Summary

Any `--pipeline-parallel-size > 1` configuration of DeepSeek-V4.1 loads,
profiles, and then dies on the first forward pass:

```
ValueError: DeepSeek V4 vision MoE routing requires input_ids.
```

raised by every stage-1 worker. It reproduces independently of
`--language-model-only`, and at every offload setting tried.

## Root cause

V4.1's MoE gate keeps separate expert-selection biases for text and image
tokens, so the routing function needs to know each token's modality — and the
only thing it is given to derive that from is `input_ids`. Stage 0 has them;
the pipeline boundary carries hidden states and residuals, so stage 1 onward
have no way to recover them, and the gate raises rather than routing everything
as text.

## Why it matters here

On 8x 32 GB cards, `TP4 x PP2` is the natural way to trade interconnect traffic
for per-rank memory: it halves the number of ranks that must each hold a shard
of the 275.7 GiB expert bank, and on a PCIe-only box (no NVLink, GPU P2P
disabled) pipeline stages are much cheaper to cross than tensor-parallel
all-reduces. With this bug, TP8 is the only usable layout, which forces about
10-12 GiB per rank of expert weights into host RAM, and every offloaded gigabyte
costs decode throughput roughly linearly: measured on DeepSeek-V4-Flash, 6 GiB
per rank halves decode and 12 GiB per rank takes about 70 % of it.

So this is not only a correctness gap — it removes the configuration that would
avoid the offload entirely.

## Suggested fixes, cheapest first

1. **Carry the modality mask, not the ids.** The gate does not need vocabulary
   indices, only "is this token an image token". That is one bool per token,
   computable on stage 0 where `input_ids` exist, and it survives a pipeline
   boundary as a small tensor in `IntermediateTensors`.
2. **Let a model declare that it needs `input_ids` across stages**, and have the
   pipeline runner include them when it does. More general, and more bytes on
   the wire.
3. **Fail at configuration time.** Whichever fix lands, `supports_pp` for this
   architecture should be `False` until it does, so the error arrives before the
   weight load rather than after it.

## Reproduction

```bash
vllm serve /data/models/DeepSeek-V4.1-Flash \
  --tokenizer-mode deepseek_v41 --trust-remote-code \
  --tensor-parallel-size 4 --pipeline-parallel-size 2 \
  --engram-config '{"cpu_offload": true}' --cpu-offload-gb 14 \
  --block-size 128 --max-model-len 32768 --language-model-only
```
