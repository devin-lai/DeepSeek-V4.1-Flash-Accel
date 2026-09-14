---
name: Deployment problem
about: Report a failed launch or incorrect model output (English or Chinese)
title: "[Deployment] "
---

## What happened?

Expected behavior, observed behavior, and whether the server became ready.

## Hardware and software

- GPU model, count, VRAM, and topology:
- CPU, host RAM, and available disk:
- OS, driver, CUDA, PyTorch, vLLM revision, FlashInfer version:
- This repository's commit and applied patches:
- Model ID, revision, and weight format:

## Reproduction

Exact launch command, preset, environment overrides, and a minimal request.

```text
Paste commands here.
```

## Evidence

Relevant server log, preflight output, and `tools/faultscan.py` result.
If the server runs, include `deploy/verify.py` output and JSON when available.
State which checks could not run. Remove credentials and private prompts.

```text
Paste output here.
```
