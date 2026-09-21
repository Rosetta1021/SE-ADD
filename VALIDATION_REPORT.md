# Release validation report

Date: 2026-09-21

## Passed

- `compileall` for shared, Qwen2-Audio, MOSS8, and test Python files.
- Qwen2-Audio generation, logits scoring, initial-LoRA, and cumulative-LoRA
  command-line imports in the experiment Qwen environment.
- MOSS8 equivalents with the production MOSS source tree and PEFT overlay on
  `PYTHONPATH`.
- Two input examples, with three non-empty hints per example.
- Mistake-driven builder end to end: two deliberately wrong predictions created
  six SFT records (direct correction, reasoning plus verdict, and cue-conditioned
  correction for each audio).
- Three-message audio/text/assistant conversation schema.
- Deterministic target balance: Real=3 and Fake=3.
- Audit JSON construction.
- Credential and cluster-path scan: no user home/scratch path, account name, or
  API-key variable was found in release files.

## Formal-code integrity

The release copies of the shared selection logic are byte-identical to the
formal experiment versions:

```text
build_mistake_driven_sft.py
  ddcd3cfd7d8eb7b69f2caf9f96d8f0591bdf99bbb2cf51337b002748ddb1c6e2
select_topm_hints.py
  c578f354c0d1599bdcefdc01147691465b0bb25e38a3fdb72453509bf3a49803
```

## Recorded production runtimes

Qwen2-Audio: torch 2.9.0+cu129, transformers 4.57.3, PEFT 0.15.2,
accelerate 1.7.0, librosa 0.11.0, soundfile 0.13.1.

MOSS8 integration: torch 2.9.0+cu129, transformers 4.57.3, PEFT 0.20.0,
accelerate 1.7.0, librosa 0.11.0, soundfile 0.13.1.

`pip check` in the HPC Qwen environment reports only that
`nvidia-cusparselt-cu12 0.7.1 is not supported on this platform`; the model CLI
imports still pass. This is a platform-wheel warning that should be resolved by
installing PyTorch/CUDA packages appropriate to the reproduction host.

## Scope limit

No fresh multi-billion-parameter checkpoint was loaded and no GPU training was
started for this release check. Model weights and licensed audio are external.
The repository therefore passes code, import, schema, and SFT-construction
validation; full numerical reproduction still requires the documented assets
and GPU runtime.
