"""MOSS-compatible direct and hint-conditioned Qwen2-Audio scoring."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
from pathlib import Path
from typing import Any

import torch

from utils import (
    DEFAULT_MODEL,
    SCORING_METHOD,
    empirical_eer,
    item_ground_truth,
    item_hints,
    load_inference_model,
    load_processor,
    load_records,
    resolve_adapter_dir,
    resolve_audio_path,
    score_verdict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="D0_smoke64")
    parser.add_argument(
        "--evaluators", nargs="+", choices=("baseline", "cpt"), default=["baseline"]
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--cpt_lora_dir", type=Path, default=None)
    parser.add_argument("--dataset_in", required=True, type=Path)
    parser.add_argument(
        "--fixed-hints-in",
        type=Path,
        default=None,
        help="Optional fixed hint JSON/JSONL; paths and order must match dataset_in.",
    )
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Deterministic output path for sharded/DAG execution (one evaluator only).",
    )
    parser.add_argument("--audio_root", type=Path, default=Path("../data"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--fake_text", default="Fake")
    parser.add_argument("--real_text", default="Real")
    parser.add_argument("--hint_context", choices=("neutral", "legacy"), default="neutral")
    parser.add_argument(
        "--score_direct_baseline",
        action="store_true",
        help="Compatibility name: direct score always uses the current evaluator model/adapter.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def attach_fixed_hints(
    records: list[dict[str, Any]], fixed_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(records) != len(fixed_rows):
        raise ValueError(
            f"Fixed hint count mismatch: dataset={len(records)}, hints={len(fixed_rows)}"
        )
    result = []
    for index, (record, fixed) in enumerate(zip(records, fixed_rows)):
        left = str(record.get("path", record.get("context", "")))
        right = str(fixed.get("path", fixed.get("context", "")))
        if left != right:
            raise ValueError(f"Fixed hint path mismatch at {index}: {left!r} != {right!r}")
        updated = dict(record)
        updated["hints"] = item_hints(fixed)
        result.append(updated)
    return result


def evaluate(
    records: list[dict[str, Any]],
    evaluator: str,
    adapter: Path | None,
    processor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model, base_model = load_inference_model(args.model, adapter)
    audio_results = []
    for audio_index, row in enumerate(records):
        audio_path = resolve_audio_path(row, args.audio_root)
        ground_truth = item_ground_truth(row)
        direct = None
        direct_correct_margin = None
        if args.score_direct_baseline:
            direct = score_verdict(
                audio_path,
                None,
                model,
                processor,
                args.fake_text,
                args.real_text,
                args.hint_context,
            )
            direct_correct_margin = (
                direct["spoof_score"] if ground_truth == "fake" else -direct["spoof_score"]
            )

        hint_results = []
        for hint_index, hint in enumerate(item_hints(row)):
            scored = score_verdict(
                audio_path,
                hint,
                model,
                processor,
                args.fake_text,
                args.real_text,
                args.hint_context,
            )
            correct_probability = (
                scored["p_fake"] if ground_truth == "fake" else 1.0 - scored["p_fake"]
            )
            hint_row = {
                "index": hint_index,
                "text": hint,
                "trials": [{**scored, "correct": scored["prediction"] == ground_truth}],
                "accuracy": float(scored["prediction"] == ground_truth),
                "spoof_score": scored["spoof_score"],
                "p_fake": scored["p_fake"],
                "correct_probability": correct_probability,
                "correct_label_probability": correct_probability,
            }
            if direct_correct_margin is not None:
                hint_correct_margin = (
                    scored["spoof_score"] if ground_truth == "fake" else -scored["spoof_score"]
                )
                hint_row["effectiveness_gain"] = hint_correct_margin - direct_correct_margin
            hint_results.append(hint_row)

        mean_score = sum(item["spoof_score"] for item in hint_results) / len(hint_results)
        mean_accuracy = sum(item["accuracy"] for item in hint_results) / len(hint_results)
        best_accuracy = max(item["accuracy"] for item in hint_results)
        audio_results.append(
            {
                "audio_index": audio_index,
                "audio_path": str(audio_path),
                "ground_truth": ground_truth,
                "audio_accuracy": round(mean_accuracy, 4),
                "best_accuracy": round(best_accuracy, 4),
                "spoof_score": mean_score,
                "direct_verdict": direct,
                "best_hints": [item for item in hint_results if item["accuracy"] == best_accuracy],
                "hints": hint_results,
            }
        )
        print(f"[{evaluator}] {audio_index + 1}/{len(records)} accuracy={mean_accuracy:.3f}")

    overall = sum(row["audio_accuracy"] for row in audio_results) / len(audio_results)
    result = {
        "model_type": evaluator,
        "verdict_mode": "logits",
        "n_audios": len(audio_results),
        "eval_times": 1,
        "sampling": {"do_sample": False, "temperature": None, "top_p": None, "seed": None},
        "overall_accuracy": round(overall, 4),
        "hint_context": args.hint_context,
        "scoring": {
            "method": SCORING_METHOD,
            "fake_text": args.fake_text,
            "real_text": args.real_text,
            "fake_token_ids": processor.tokenizer.encode(args.fake_text, add_special_tokens=False),
            "real_token_ids": processor.tokenizer.encode(args.real_text, add_special_tokens=False),
            "spoof_score": "log P(Fake|input) - log P(Real|input)",
            "decision_threshold": 0.0,
        },
        "adapter": str(resolve_adapter_dir(adapter)) if adapter else None,
        "audio": audio_results,
    }
    result["eer"] = empirical_eer(audio_results)
    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def save_result(result: dict[str, Any], args: argparse.Namespace) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_file is not None:
        if len(args.evaluators) != 1:
            raise ValueError("--output-file requires exactly one evaluator")
        path = args.output_file
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = dt.datetime.now().strftime("%m%d_%H%M%S")
        path = args.output_dir / f"{args.exp_name}_{result['model_type']}_{timestamp}.json"
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {path}")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"[{result['model_type']}] results -> {path}")
    return path


def main() -> None:
    args = parse_args()
    if not args.dataset_in.is_file():
        raise FileNotFoundError(args.dataset_in)
    records = load_records(args.dataset_in)
    if args.fixed_hints_in:
        records = attach_fixed_hints(records, load_records(args.fixed_hints_in))
    if args.start < 0:
        raise ValueError("--start cannot be negative")
    records = records[args.start :]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        records = records[: args.limit]
    if not records:
        raise ValueError("No records selected")
    processor = load_processor(args.model)
    for evaluator in args.evaluators:
        if evaluator == "baseline":
            adapter = None
        else:
            adapter = args.adapter or args.cpt_lora_dir
            if adapter is None:
                raise ValueError("cpt evaluation requires --adapter or --cpt_lora_dir")
        save_result(evaluate(records, evaluator, adapter, processor, args), args)
    print("Requested evaluator branches completed.")


if __name__ == "__main__":
    main()
