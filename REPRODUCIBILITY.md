# Reproducibility status

Validated on 2026-09-21 in the original HPC workspace:

- all release Python files compile;
- the Qwen generation, scoring, initial-LoRA, and cumulative-LoRA CLIs import;
- the MOSS8 equivalents import with the production MOSS source/overlay;
- the two-record example schema is valid;
- the unchanged shared builder produces all three correction task types;
- deterministic label balancing and audit output pass assertions;
- no user-specific home/scratch paths or credential names remain in release code.

The smoke test produced six balanced SFT rows from two deliberately
misclassified examples (three correction tasks per example).

Not rerun during release staging:

- pretrained model loading and audio inference;
- GPU LoRA optimization;
- D0--D5 formal experiments;
- paper evaluation datasets.

Those paths were already exercised by the archived experiments, but a fresh
machine must provide model weights, licensed datasets, sufficient GPU memory,
and the correct CUDA PyTorch wheel. Therefore this is a validated code/interface
release, not a claim that full experiments can run without external assets.
