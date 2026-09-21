"""Build one mistake-driven SE-ADD evolution round in MOSS-Audio SFT format.

The builder creates three tasks for each currently misclassified audio:
  1. direct audio -> verdict correction;
  2. audio -> process-constrained observations + correct verdict;
  3. audio + observations -> correct verdict.

Currently correct audios contribute direct-verdict review examples.  Candidate
observations are cleaned cue-by-cue with the process constraints implemented in
``select_topm_hints.py``.  Ground-truth labels anchor every verdict target.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from select_topm_hints import content_jaccard_similarity, process_validation


DIRECT_PROMPT = "Determine whether this audio is real or fake. Answer with Real or Fake."
REASONING_PROMPT = (
    "Analyze this audio and provide concise, audio-only forensic observations. "
    "Then give the final verdict as Real or Fake."
)
CUE_VERDICT_PROMPT = (
    "The following observations were produced for this audio. Use the audio and "
    "the observations together to determine the verdict. Answer with Real or Fake.\n\n"
    "{observations}"
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("audio"), list):
        raise ValueError(f"Expected an evaluation JSON with an audio list: {path}")
    return data


def normalize_label(value: Any) -> str:
    value = str(value).strip().lower()
    if value in {"bonafide", "bona-fide", "real", "human"}:
        return "real"
    if value in {"spoof", "fake", "deepfake"}:
        return "fake"
    raise ValueError(f"Unsupported label: {value!r}")


def resolve_audio_path(row: Dict[str, Any], audio_root: Path | None) -> str:
    value = row.get("path", row.get("context"))
    if not value:
        raise ValueError("Each hint record must contain path or context")
    path = Path(str(value))
    if path.is_absolute() or audio_root is None:
        return str(path)
    return str(audio_root / path)


def sft_entry(
    audio_path: str,
    prompt: str,
    answer: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "conversation": [
            {"role": "user", "message_type": "audio", "content": audio_path},
            {"role": "user", "message_type": "text", "content": prompt},
            {"role": "assistant", "message_type": "text", "content": answer},
        ],
        "seadd_metadata": metadata,
    }


def deduplicate_cues(
    cues: Iterable[str],
    similarity_threshold: float,
    max_cues: int,
) -> List[str]:
    selected: List[str] = []
    for cue in cues:
        if any(
            content_jaccard_similarity(cue, previous) > similarity_threshold
            for previous in selected
        ):
            continue
        selected.append(cue.strip())
        if len(selected) == max_cues:
            break
    return selected


def collect_process_cues(
    hints: List[str],
    audio_index: int,
    args: argparse.Namespace,
) -> tuple[List[str], Dict[str, Any]]:
    rng = random.Random(args.seed + audio_index)
    shuffled_hints = list(enumerate(hints))
    rng.shuffle(shuffled_hints)
    cue_pool: List[str] = []
    filtering_counts = Counter()
    candidate_audit = []

    for hint_index, hint in shuffled_hints:
        validation = process_validation(
            hint,
            min_cues=1,
            max_cues=args.max_candidate_cues,
            max_words=args.max_words,
            intra_cue_similarity=args.intra_cue_similarity,
        )
        filtering_counts.update(validation["rejection_reason_counts"])
        cue_pool.extend(validation["accepted_cues"])
        candidate_audit.append(
            {
                "hint_index": hint_index,
                "accepted_cue_count": len(validation["accepted_cues"]),
                "rejected_cues": validation["rejected_cues"],
            }
        )

    rng.shuffle(cue_pool)
    selected = deduplicate_cues(
        cue_pool,
        args.inter_cue_similarity,
        args.max_selected_cues,
    )
    return selected, {
        "filtering_counts": dict(filtering_counts),
        "candidate_audit": candidate_audit,
        "pooled_cue_count": len(cue_pool),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hints-in", required=True, type=Path)
    parser.add_argument("--scores-in", required=True, type=Path)
    parser.add_argument("--sft-out", required=True, type=Path)
    parser.add_argument("--audit-out", required=True, type=Path)
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--decision-threshold", type=float, default=0.0)
    parser.add_argument("--min-selected-cues", type=int, default=2)
    parser.add_argument("--max-selected-cues", type=int, default=4)
    parser.add_argument("--max-candidate-cues", type=int, default=12)
    parser.add_argument("--max-words", type=int, default=250)
    parser.add_argument("--intra-cue-similarity", type=float, default=0.8)
    parser.add_argument("--inter-cue-similarity", type=float, default=0.8)
    parser.add_argument(
        "--real-review-repeat",
        type=int,
        default=1,
        help="Repeat correct Real direct-verdict reviews to counter class imbalance.",
    )
    parser.add_argument(
        "--balance-verdict-labels",
        action="store_true",
        help=(
            "Deterministically oversample the minority verdict target after task "
            "construction so Real and Fake SFT entries are equally represented."
        ),
    )
    args = parser.parse_args()

    if not 1 <= args.min_selected_cues <= args.max_selected_cues:
        raise ValueError("Require 1 <= min-selected-cues <= max-selected-cues")
    if args.real_review_repeat <= 0:
        raise ValueError("--real-review-repeat must be positive")

    hint_rows = load_jsonl(args.hints_in)
    score_data = load_json(args.scores_in)
    score_rows = score_data["audio"]
    if len(hint_rows) != len(score_rows):
        raise ValueError(
            f"Audio count mismatch: hints={len(hint_rows)}, scores={len(score_rows)}"
        )

    sft_rows: List[Dict[str, Any]] = []
    audit_rows = []
    task_counts = Counter()
    transition_counts = Counter()
    skipped_reasoning = []

    for audio_index, (hint_row, score_row) in enumerate(zip(hint_rows, score_rows)):
        if score_row.get("audio_index") != audio_index:
            raise ValueError(
                f"Audio index mismatch at {audio_index}: {score_row.get('audio_index')}"
            )
        direct = score_row.get("direct_verdict")
        if not isinstance(direct, dict) or "spoof_score" not in direct:
            raise ValueError(
                f"Missing direct_verdict at audio {audio_index}; "
                "rerun scoring with --score_direct_baseline"
            )

        source_label = normalize_label(hint_row.get("label"))
        scored_label = normalize_label(score_row.get("ground_truth"))
        if source_label != scored_label:
            raise ValueError(
                f"Label mismatch at audio {audio_index}: "
                f"source={source_label}, score={scored_label}"
            )

        score = float(direct["spoof_score"])
        prediction = "fake" if score >= args.decision_threshold else "real"
        is_mistake = prediction != source_label
        verdict = source_label.title()
        audio_path = resolve_audio_path(hint_row, args.audio_root)
        base_metadata = {
            "audio_index": audio_index,
            "ground_truth": source_label,
            "previous_prediction": prediction,
            "previous_spoof_score": score,
            "previous_p_fake": direct.get("p_fake"),
            "sample_role": "correction" if is_mistake else "review",
        }
        transition_counts[f"{source_label}->{prediction}"] += 1

        if is_mistake:
            metadata = {**base_metadata, "task_type": "direct_detection"}
            sft_rows.append(sft_entry(audio_path, DIRECT_PROMPT, verdict, metadata))
            task_counts["mistake_direct_detection"] += 1

            hints = hint_row.get("hints")
            if not isinstance(hints, list) or not all(isinstance(h, str) for h in hints):
                raise ValueError(f"Invalid hints list at audio {audio_index}")
            cues, cue_audit = collect_process_cues(hints, audio_index, args)
            if len(cues) >= args.min_selected_cues:
                observations = "\n".join(
                    f"Cue {index}: {cue}" for index, cue in enumerate(cues, start=1)
                )
                reasoning_answer = (
                    f"Forensic observations:\n{observations}\n\nVerdict: {verdict}"
                )
                reasoning_metadata = {
                    **base_metadata,
                    "task_type": "reasoning_and_verdict",
                    "selected_cues": cues,
                }
                sft_rows.append(
                    sft_entry(
                        audio_path,
                        REASONING_PROMPT,
                        reasoning_answer,
                        reasoning_metadata,
                    )
                )
                task_counts["mistake_reasoning_and_verdict"] += 1

                cue_metadata = {
                    **base_metadata,
                    "task_type": "cue_conditioned_detection",
                    "selected_cues": cues,
                }
                sft_rows.append(
                    sft_entry(
                        audio_path,
                        CUE_VERDICT_PROMPT.format(observations=observations),
                        verdict,
                        cue_metadata,
                    )
                )
                task_counts["mistake_cue_conditioned_detection"] += 1
            else:
                skipped_reasoning.append(audio_index)

            audit_rows.append(
                {
                    **base_metadata,
                    "selected_cues": cues,
                    "cue_audit": cue_audit,
                }
            )
        else:
            repeats = args.real_review_repeat if source_label == "real" else 1
            for repeat_index in range(repeats):
                metadata = {
                    **base_metadata,
                    "task_type": "direct_detection_review",
                    "repeat_index": repeat_index,
                }
                sft_rows.append(sft_entry(audio_path, DIRECT_PROMPT, verdict, metadata))
                task_counts[f"review_direct_{source_label}"] += 1
            audit_rows.append(base_metadata)

    target_counts_before_balance = Counter(
        row["seadd_metadata"]["ground_truth"] for row in sft_rows
    )
    balance_counts = Counter()
    if args.balance_verdict_labels:
        target_rows = {
            label: [
                row
                for row in sft_rows
                if row["seadd_metadata"]["ground_truth"] == label
            ]
            for label in ("real", "fake")
        }
        if not target_rows["real"] or not target_rows["fake"]:
            raise ValueError(
                "Cannot balance verdict targets when one label has no SFT entries: "
                f"{target_counts_before_balance}"
            )
        target_size = max(len(target_rows["real"]), len(target_rows["fake"]))
        balance_rng = random.Random(args.seed)
        for label in ("real", "fake"):
            duplicate_index = 0
            while len(target_rows[label]) < target_size:
                duplicate = copy.deepcopy(balance_rng.choice(target_rows[label]))
                duplicate["seadd_metadata"]["is_balance_duplicate"] = True
                duplicate["seadd_metadata"]["balance_duplicate_index"] = duplicate_index
                target_rows[label].append(duplicate)
                sft_rows.append(duplicate)
                balance_counts[f"balance_duplicate_{label}"] += 1
                duplicate_index += 1
        balance_rng.shuffle(sft_rows)

    target_counts_after_balance = Counter(
        row["seadd_metadata"]["ground_truth"] for row in sft_rows
    )

    args.sft_out.parent.mkdir(parents=True, exist_ok=True)
    with args.sft_out.open("w", encoding="utf-8") as handle:
        for row in sft_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "hints_in": str(args.hints_in),
        "scores_in": str(args.scores_in),
        "audio_count": len(hint_rows),
        "sft_count": len(sft_rows),
        "decision_threshold": args.decision_threshold,
        "task_counts": dict(task_counts),
        "balance_verdict_labels": args.balance_verdict_labels,
        "target_counts_before_balance": dict(target_counts_before_balance),
        "balance_counts": dict(balance_counts),
        "target_counts_after_balance": dict(target_counts_after_balance),
        "transition_counts": dict(transition_counts),
        "skipped_reasoning_audio_indices": skipped_reasoning,
        "records": audit_rows,
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"Input audios: {len(hint_rows)}")
    print(f"SFT entries: {len(sft_rows)}")
    print(f"Task counts: {dict(task_counts)}")
    print(f"Target counts before balance: {dict(target_counts_before_balance)}")
    print(f"Balance duplicates: {dict(balance_counts)}")
    print(f"Target counts after balance: {dict(target_counts_after_balance)}")
    print(f"Transitions: {dict(transition_counts)}")
    print(f"Skipped reasoning tasks: {skipped_reasoning}")
    print(f"SFT -> {args.sft_out}")
    print(f"Audit -> {args.audit_out}")


if __name__ == "__main__":
    main()
