# Contributing

Help make DeepSeek-V4.1-Flash easier to deploy and its performance easier to
reproduce. Deployment reports, documentation, kernel fixes, and research
experiments are welcome. Issues and pull requests may be written in English
or Chinese.

## Pick a contribution

| Contribution | Useful evidence |
| --- | --- |
| Reproduce the reference deployment | Hardware, exact launch command, sanity-probe output, benchmark JSON |
| Try another GPU or memory configuration | Success or failure, memory use, topology, versions, relevant logs |
| Improve a kernel or offload policy | Baseline and changed run on the same setup; reference comparison and output checks |
| Evaluate vision, long context, or DSpark | Workload, quality criteria, settings, and latency/throughput distributions |
| Improve the docs | Which instruction was unclear and the replacement you verified |

The [benchmark protocol](benchmarks/README.md) defines the metadata and
comparison rules for measurements. A failed reproduction is useful too.

## Report a deployment problem

Run commands from the repository root in the installed environment:

```bash
python tools/faultscan.py /path/to/server.log
python deploy/verify.py --json /path/to/verify.json
```

The second command needs a running server. Include its output when the engine
starts but produces empty, repeated, or implausible text. Inspect the generated
JSON as well as the exit code: the current sanity gate can pass with missing
perplexity, and the chat answer does not determine its verdict.

Use the deployment issue template and include the fault ID if one matches.
Remove access tokens and private prompt content from logs before sharing.

## Public files and local work

Publish code, deployment instructions, patch explanations, and reviewed
benchmark evidence. Keep personal data, AI plans, session transcripts, draft
announcements, and machine-specific experiments in `.local/` (ignored).
Runtime logs, weights, caches, credentials, and local presets are also ignored.
Do not force-add them. `.gitignore` does not remove previously tracked files
or data already present in Git history.

Use generic paths such as `/data/models/DeepSeek-V4.1-Flash` in shared examples.
Before sharing measurements, remove usernames, hostnames, private IPs,
credentials, and private prompts. Retain the hardware and configuration
details needed to reproduce the measurement. Only reviewed evidence belongs
in `benchmarks/results/`; place new raw runs in `benchmarks/local/`.

Before committing, stage the intended public files and run:

```bash
python3 scripts/check_public_repo.py
git diff --cached --check
git diff --cached --stat
```

The Python 3.11+ check inspects staged contents for ignored files, common
credential patterns, personal home paths, broken local documentation links,
and syntax errors. GitHub Actions runs the same check on committed files.
This is a basic publication check, not a comprehensive secret scan or GPU test.
Use a GitHub noreply email for commits if you want to keep your email private.

README illustrations are stored in `docs/assets/` as compressed WebP files.
Review image content and metadata before publishing; keep source renders and
draft prompts in `.local/`. The publication check accepts only the reviewed
image hashes listed in `scripts/check_public_repo.py` and checks image links.

## Submit a patch

Describe the observed failure or bottleneck, the resulting behavior, and the
evidence for the change. Keep each fix focused so it can be reviewed and, when
appropriate, proposed upstream independently.

For CUDA changes, run the relevant shape probes and compare numerical output:

```bash
python tools/sm120_sparse_mla/probe_sm120.py --oracle --sweep --json sweep.json
```

This requires an SM120 GPU and the patched FlashInfer installation. The
[tiny model](tools/tiny/README.md) is useful for structural failures; its dummy
weights cannot establish numerical correctness or full-model performance.
Changes affecting actual serving also need real-checkpoint validation. State
clearly which checks you could run and which remain open.

Add new recognizable faults to `faults/inventory.toml`, then regenerate the
reference with `python tools/faultscan.py --markdown > docs/05-fault-inventory.md`.
Update affected documentation and preserve component attribution and license
notices. The existing `vllm-dsv41-opt` package and `DSV41_*` settings remain the
runtime identifiers.

## Research credit

Name the contributors to an experiment in its report and link its saved data.
Cite this project's repository URL and the exact commit or release used;
also credit DeepSeek and the relevant inference and kernel projects. Label
estimated memory savings and expert-level quantization errors separately from
measured end-to-end serving speed and model quality.
