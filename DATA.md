# Data interface

Full datasets are intentionally excluded. Users obtain each dataset under its
own terms and create JSONL manifests. Relative audio paths are resolved against
`--audio_root`; absolute paths are also accepted.

## Input manifest

Required fields:

- `path` or legacy `context`: audio path
- `label`: `bonafide` or `spoof`

Recommended provenance fields, preserved by hint generation:

- `attack` / `attack_id`
- `utterance_id` or other utterance metadata
- `environment`

## Generated hints

The generated record retains the input metadata and adds `hints`, a list of K
non-empty strings. Cues must describe only observable audio properties and must
not contain a final Real/Fake verdict.

## Scoring output

The JSON result has an `audio` list. Each item contains `audio_index`,
`audio_path`, `ground_truth`, `direct_verdict`, and per-hint scores. The core
score is:

```text
spoof_score = log P("Fake" | input) - log P("Real" | input)
p_fake      = sigmoid(spoof_score)
```

Both candidates are scored as complete token sequences. `spoof_score >= 0`
means Fake.

## SFT output

Every row contains a three-message `conversation`: audio user message, text
user instruction, and text assistant target. `seadd_metadata` records task
type, ground truth, previous prediction/score, selected cues, and balancing
provenance. Keep the separately emitted audit JSON for reproducibility.
