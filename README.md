# SE-ADD: Self-Evolving Audio Deepfake Detection

Research code for reasoning-guided cumulative LoRA adaptation of audio language
models under changing spoofing attacks. Two backends are provided:

- Qwen2-Audio-7B-Instruct
- MOSS-Audio-8B-Instruct

The repository contains code and small schema examples only. It does **not**
redistribute pretrained weights, LoRA checkpoints, ASVspoof audio, HIR-SDD,
In-the-Wild audio, API keys, or cluster logs.

## What one evolution round does

1. The current model generates `K` audio-only forensic cue candidates.
2. The current model is scored with full-sequence Real/Fake log likelihoods.
3. `seadd/build_mistake_driven_sft.py` identifies mistakes using ground truth,
   process-filters the self-generated cues, and constructs corrective/review SFT.
4. Only a LoRA adapter is trained. In later rounds the previous adapter is
   loaded as trainable and saved to a new directory; base weights stay frozen.
5. A development set selects R1 or R2 before the chain advances.

The builder and cue-filter implementation are shared verbatim by both backends.

## Layout

```text
seadd/                         shared cue filtering and SFT construction
backends/qwen2audio/           generation, scoring, initial/cumulative LoRA
backends/moss8/                generation, scoring, initial/cumulative LoRA
examples/manifests/            two synthetic manifest records
examples/outputs/              compatible hint and score records
examples/audio/                placeholders only; no dataset audio
requirements/                  recorded experiment environments
scripts/smoke_test.sh          CPU/schema and Qwen import checks
tests/                         deterministic builder smoke test
```

## Installation

Create separate environments for the two backends. Do not install one set of
PEFT dependencies over the other.

```bash
python3 -m venv .venv-qwen
source .venv-qwen/bin/activate
python -m pip install -r requirements/qwen2audio.txt
```

For MOSS, clone/install the official MOSS-Audio repository and then install the
recorded wrapper dependencies in a separate environment. The CUDA-specific
PyTorch wheel/index must match the host driver.

```bash
git clone https://github.com/OpenMOSS/MOSS-Audio third_party/MOSS-Audio
python3 -m venv .venv-moss
source .venv-moss/bin/activate
python -m pip install -r requirements/moss8.txt
python -m pip install -e third_party/MOSS-Audio
export PYTHONPATH="$PWD/third_party/MOSS-Audio${PYTHONPATH:+:$PYTHONPATH}"
```

The version files record the successful HPC runtime. On a different CUDA
platform, install the matching PyTorch build while preserving the remaining
versions.

## Fast validation

```bash
PYTHON_BIN="$PWD/.venv-qwen/bin/python" ./scripts/smoke_test.sh
```

This validates Python syntax, CLI imports, example schemas, mistake mining,
three-message SFT construction, deterministic label balancing, and audit output.
It intentionally does not download or load a multi-billion-parameter model.

## Minimal round

Use real local audio paths in a manifest shaped like
`examples/manifests/tiny_manifest.jsonl`.

```bash
python backends/qwen2audio/generate_hints.py \
  --model /path/to/Qwen2-Audio-7B-Instruct \
  --dataset_in /path/to/D0_manifest.jsonl \
  --dataset_out outputs/D0_R1_hints.jsonl \
  --audio_root /path/to/audio-root \
  --sample-size 64 --k 3 --seed 2026 \
  --temperature 0.8 --top_p 0.95 --max_tokens 160

python backends/qwen2audio/score_logits.py \
  --model /path/to/Qwen2-Audio-7B-Instruct \
  --evaluators baseline \
  --dataset_in outputs/D0_R1_hints.jsonl \
  --audio_root /path/to/audio-root \
  --output_dir outputs/D0_R1_scores \
  --exp_name D0_R1 --score_direct_baseline

python seadd/build_mistake_driven_sft.py \
  --hints-in outputs/D0_R1_hints.jsonl \
  --scores-in /path/to/the/generated-score.json \
  --sft-out outputs/D0_R1_sft.jsonl \
  --audit-out outputs/D0_R1_audit.json \
  --audio-root /path/to/audio-root \
  --seed 2026 --balance-verdict-labels

python backends/qwen2audio/train_initial_lora.py \
  --model /path/to/Qwen2-Audio-7B-Instruct \
  --dataset_in_lora_jsonl outputs/D0_R1_sft.jsonl \
  --cpt_lora_dir adapters/D0_R1 \
  --lora_rank 8 --max_len 2048 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --num_train_epochs 3 --learning_rate 2e-5

python backends/qwen2audio/train_cumulative_lora.py \
  --model /path/to/Qwen2-Audio-7B-Instruct \
  --input-adapter adapters/D0_R1 \
  --dataset-in outputs/D0_R2_sft.jsonl \
  --output-adapter adapters/D0_R2 \
  --max-len 2048 --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 --num-train-epochs 3 \
  --learning-rate 2e-5
```

MOSS uses the corresponding scripts under `backends/moss8` plus
`--moss-audio-root third_party/MOSS-Audio`. Its generation and scoring modules
must be launched with the MOSS repository on `PYTHONPATH`.

See [DATA.md](DATA.md) for schemas and [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
for what has and has not been validated in this code-only release.

## License

SE-ADD code is released under the Apache License 2.0, Copyright 2026 SE-ADD
Authors. External models, datasets, repositories, and services are not included
or relicensed. See [LICENSE](LICENSE), [NOTICE](NOTICE), and
[THIRD_PARTY.md](THIRD_PARTY.md).
