"""Selectable baseline/CPT/query evaluation with generation or logit scoring."""

import argparse
import datetime as dt
import gc
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from peft import PeftModel

from src.audio_io import load_audio
from src.modeling_moss_audio import MossAudioModel
from src.processing_moss_audio import MossAudioProcessor


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Read either a JSON array or JSONL records."""
    with path.open("r", encoding="utf-8") as handle:
        first = ""
        while True:
            character = handle.read(1)
            if not character:
                return []
            if not character.isspace():
                first = character
                break

    if first == "[":
        with path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return rows

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            rows.append(row)
    return rows


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def item_audio_path(item: Dict[str, Any], audio_root: Path) -> Path:
    value = item.get("path", item.get("context"))
    if not value:
        raise ValueError("Every record must contain 'path' or legacy 'context'")
    path = Path(str(value))
    return path if path.is_absolute() else audio_root / path


def item_ground_truth(item: Dict[str, Any]) -> str:
    label = str(item.get("label", "")).strip().lower()
    if label == "spoof":
        return "fake"
    if label == "bonafide":
        return "real"

    questions = item.get("question")
    if isinstance(questions, list) and questions:
        answer = str(questions[0].get("answer", "")).strip().lower()
        if answer in {"fake", "real"}:
            return answer
    raise ValueError("Cannot derive Fake/Real ground truth from 'label' or 'question'")


def item_hints(item: Dict[str, Any]) -> List[str]:
    hints = item.get("hints")
    if isinstance(hints, str):
        hints = [hints]
    if not isinstance(hints, list) or not hints:
        raise ValueError("Every record must contain at least one hint")
    if any(not isinstance(hint, str) or not hint.strip() for hint in hints):
        raise ValueError("All hints must be non-empty strings")
    return [hint.strip() for hint in hints]


def parse_verdict(response: str) -> str | None:
    matches = set(re.findall(r"\b(fake|real)\b", response.lower()))
    return next(iter(matches)) if len(matches) == 1 else None


def verdict_prompt(hint: str | None, hint_context: str = "neutral") -> str:
    if hint is None:
        return (
            "<|im_start|>user\n"
            "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
            "Is this audio fake or real? Answer with exactly one word: Fake or Real."
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if hint_context == "legacy":
        context = (
            "These are forensic hints that may help with the judgment:\n"
            f"{hint}\n"
        )
    elif hint_context == "neutral":
        context = (
            "The following are neutral observations about the audio. They may "
            "support either class and must not be treated as evidence of fakery "
            "by default. Consider them together with the audio:\n"
            f"{hint}\n"
        )
    else:
        raise ValueError(f"Unsupported hint context: {hint_context}")
    return (
        "<|im_start|>user\n"
        "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
        f"{context}"
        "Is this audio fake or real? Answer with exactly one word: Fake or Real."
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def prepare_verdict_inputs(
    audio_path: Path,
    hint: str | None,
    model,
    processor,
    hint_context: str = "neutral",
):
    audio = load_audio(str(audio_path), sample_rate=processor.config.mel_sr)
    inputs = processor(
        text=verdict_prompt(hint, hint_context),
        audios=[audio],
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    if inputs.get("audio_data") is not None:
        inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
    inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id
    return inputs


def candidate_token_ids(processor, text: str, device) -> torch.Tensor:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise AttributeError("MossAudioProcessor does not expose processor.tokenizer")
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Verdict candidate produced no tokens: {text!r}")
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def candidate_log_likelihood(model, inputs, candidate_ids: torch.Tensor) -> float:
    """Teacher-force one candidate verdict and return its sequence log-likelihood."""
    prefix_ids = inputs["input_ids"]
    prefix_length = prefix_ids.shape[1]
    candidate_length = candidate_ids.shape[1]
    model_inputs = dict(inputs)
    model_inputs["input_ids"] = torch.cat([prefix_ids, candidate_ids], dim=1)

    if inputs.get("attention_mask") is not None:
        extension = torch.ones(
            (inputs["attention_mask"].shape[0], candidate_length),
            dtype=inputs["attention_mask"].dtype,
            device=inputs["attention_mask"].device,
        )
        model_inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], extension], dim=1
        )
    if inputs.get("audio_input_mask") is not None:
        extension = torch.zeros(
            (inputs["audio_input_mask"].shape[0], candidate_length),
            dtype=inputs["audio_input_mask"].dtype,
            device=inputs["audio_input_mask"].device,
        )
        model_inputs["audio_input_mask"] = torch.cat(
            [inputs["audio_input_mask"], extension], dim=1
        )
    # Let the model recompute sequence-dependent fields for the extended input.
    model_inputs.pop("position_ids", None)
    model_inputs.pop("labels", None)

    with torch.no_grad():
        outputs = model(**model_inputs, use_cache=False)
    prediction_logits = outputs.logits[
        0, prefix_length - 1 : prefix_length + candidate_length - 1, :
    ].float()
    token_log_probs = torch.log_softmax(prediction_logits, dim=-1)
    selected = token_log_probs.gather(1, candidate_ids[0].unsqueeze(1)).squeeze(1)
    return float(selected.sum().item())


def judge_from_logits(
    audio_path: Path,
    hint: str | None,
    model,
    processor,
    fake_text: str,
    real_text: str,
    hint_context: str = "neutral",
) -> Dict[str, Any]:
    inputs = prepare_verdict_inputs(
        audio_path, hint, model, processor, hint_context
    )
    fake_ids = candidate_token_ids(processor, fake_text, model.device)
    real_ids = candidate_token_ids(processor, real_text, model.device)
    fake_ll = candidate_log_likelihood(model, inputs, fake_ids)
    real_ll = candidate_log_likelihood(model, inputs, real_ids)
    margin = fake_ll - real_ll
    p_fake = float(torch.sigmoid(torch.tensor(margin)).item())
    return {
        "response": None,
        "prediction": "fake" if margin >= 0 else "real",
        "fake_token_ids": fake_ids[0].tolist(),
        "real_token_ids": real_ids[0].tolist(),
        "fake_log_likelihood": fake_ll,
        "real_log_likelihood": real_ll,
        "spoof_score": margin,
        "p_fake": p_fake,
    }


def judge_once(
    audio_path: Path,
    hint: str,
    model,
    processor,
    max_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: int,
    hint_context: str = "neutral",
) -> Dict[str, Any]:
    inputs = prepare_verdict_inputs(
        audio_path, hint, model, processor, hint_context
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generation_args = {
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "num_beams": 1,
        "use_cache": True,
        "pad_token_id": model.config.eos_token_id,
    }
    if do_sample:
        generation_args["temperature"] = temperature
        generation_args["top_p"] = top_p
    with torch.no_grad():
        generated_ids = model.generate(**inputs, **generation_args)
    input_length = inputs["input_ids"].shape[1]
    response = processor.decode(
        generated_ids[0, input_length:],
        skip_special_tokens=True,
    ).strip()
    return {"response": response, "prediction": parse_verdict(response)}


def evaluate_one_hint(
    audio_path: Path,
    hint: str,
    ground_truth: str,
    model,
    processor,
    max_tokens: int,
    eval_times: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed_base: int,
    verdict_mode: str,
    fake_text: str,
    real_text: str,
    hint_context: str = "neutral",
) -> Dict[str, Any]:
    if verdict_mode == "logits":
        trial = judge_from_logits(
            audio_path, hint, model, processor, fake_text, real_text, hint_context
        )
        trial["correct"] = trial["prediction"] == ground_truth
        correct_probability = (
            trial["p_fake"] if ground_truth == "fake" else 1.0 - trial["p_fake"]
        )
        return {
            "trials": [trial],
            "accuracy": float(trial["correct"]),
            "spoof_score": trial["spoof_score"],
            "p_fake": trial["p_fake"],
            "correct_label_probability": correct_probability,
        }

    trials = []
    for trial_index in range(eval_times):
        trial = judge_once(
            audio_path,
            hint,
            model,
            processor,
            max_tokens,
            do_sample,
            temperature,
            top_p,
            seed_base + trial_index,
            hint_context,
        )
        trial["correct"] = trial["prediction"] == ground_truth
        trials.append(trial)
    accuracy = sum(trial["correct"] for trial in trials) / len(trials)
    return {"trials": trials, "accuracy": round(accuracy, 4)}


def evaluate_dataset(
    records: Sequence[Dict[str, Any]],
    evaluator_name: str,
    model,
    processor,
    audio_root: Path,
    max_tokens: int,
    eval_times: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: int,
    verdict_mode: str,
    fake_text: str,
    real_text: str,
    score_direct_baseline: bool = False,
    hint_context: str = "neutral",
) -> Dict[str, Any]:
    audio_results = []
    for audio_index, item in enumerate(records):
        audio_path = item_audio_path(item, audio_root)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
        ground_truth = item_ground_truth(item)
        direct_result = None
        direct_correct_margin = None
        if verdict_mode == "logits" and score_direct_baseline:
            direct_result = judge_from_logits(
                audio_path, None, model, processor, fake_text, real_text, hint_context
            )
            direct_correct_margin = (
                float(direct_result["spoof_score"])
                if ground_truth == "fake"
                else -float(direct_result["spoof_score"])
            )
        hint_results = []
        for hint_index, hint in enumerate(item_hints(item)):
            result = evaluate_one_hint(
                audio_path,
                hint,
                ground_truth,
                model,
                processor,
                max_tokens,
                eval_times,
                do_sample,
                temperature,
                top_p,
                seed + audio_index * 10_000 + hint_index * 100,
                verdict_mode,
                fake_text,
                real_text,
                hint_context,
            )
            hint_results.append(
                {"index": hint_index, "text": hint, **result}
            )
            if direct_correct_margin is not None:
                hint_correct_margin = (
                    float(result["spoof_score"])
                    if ground_truth == "fake"
                    else -float(result["spoof_score"])
                )
                hint_results[-1]["effectiveness_gain"] = (
                    hint_correct_margin - direct_correct_margin
                )
        best_accuracy = max(entry["accuracy"] for entry in hint_results)
        audio_accuracy = sum(entry["accuracy"] for entry in hint_results) / len(hint_results)
        hint_spoof_scores = [
            entry["spoof_score"] for entry in hint_results if "spoof_score" in entry
        ]
        audio_spoof_score = (
            sum(hint_spoof_scores) / len(hint_spoof_scores)
            if hint_spoof_scores else None
        )
        audio_results.append(
            {
                "audio_index": audio_index,
                "audio_path": str(audio_path),
                "ground_truth": ground_truth,
                "audio_accuracy": round(audio_accuracy, 4),
                "best_accuracy": round(best_accuracy, 4),
                "spoof_score": audio_spoof_score,
                "direct_verdict": direct_result,
                "best_hints": [
                    entry for entry in hint_results if entry["accuracy"] == best_accuracy
                ],
                "hints": hint_results,
            }
        )
        print(
            f"[{evaluator_name}] {audio_index + 1}/{len(records)} "
            f"accuracy={audio_accuracy:.3f}"
        )

    overall = sum(row["audio_accuracy"] for row in audio_results) / len(audio_results)
    result = {
        "model_type": evaluator_name,
        "verdict_mode": verdict_mode,
        "n_audios": len(audio_results),
        "eval_times": eval_times,
        "sampling": {
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None,
            "seed": seed,
        },
        "overall_accuracy": round(overall, 4),
        "hint_context": hint_context,
        "audio": audio_results,
    }
    if verdict_mode == "logits":
        result["eer"] = empirical_eer(audio_results)
        fake_ids = candidate_token_ids(processor, fake_text, model.device)[0].tolist()
        real_ids = candidate_token_ids(processor, real_text, model.device)[0].tolist()
        result["scoring"] = {
            "method": "teacher_forced_full_candidate_sequence_log_likelihood_sum",
            "fake_text": fake_text,
            "real_text": real_text,
            "fake_token_ids": fake_ids,
            "real_token_ids": real_ids,
            "spoof_score": "log P(Fake|input) - log P(Real|input)",
            "decision_threshold": 0.0,
        }
    return result


def empirical_eer(audio_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    labels = [1 if row["ground_truth"] == "fake" else 0 for row in audio_results]
    scores = [float(row["spoof_score"]) for row in audio_results]
    if 0 not in labels or 1 not in labels:
        return {
            "available": False,
            "reason": "EER requires both bonafide and spoof samples",
        }
    thresholds = [float("inf"), *sorted(set(scores), reverse=True), float("-inf")]
    best = None
    for threshold in thresholds:
        predictions = [int(score >= threshold) for score in scores]
        false_positive_rate = sum(
            prediction == 1 for prediction, label in zip(predictions, labels) if label == 0
        ) / labels.count(0)
        false_negative_rate = sum(
            prediction == 0 for prediction, label in zip(predictions, labels) if label == 1
        ) / labels.count(1)
        candidate = {
            "eer": (false_positive_rate + false_negative_rate) / 2.0,
            "threshold": threshold,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "gap": abs(false_positive_rate - false_negative_rate),
        }
        if best is None or candidate["gap"] < best["gap"]:
            best = candidate
    best["available"] = True
    return best


def device_map() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_base_model(model_path: Path):
    model = MossAudioModel.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        dtype="auto",
        device_map=device_map(),
    )
    model.eval()
    return model


def cleanup_model(*models) -> None:
    for model in models:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_lora(
    model_dir: Path,
    data_path: Path,
    output_dir: Path,
    moss_audio_root: Path,
    args: argparse.Namespace,
) -> None:
    root = moss_audio_root.resolve()
    entrypoint = root / "finetune" / "finetune.py"
    if not entrypoint.is_file():
        raise FileNotFoundError(f"Cannot find finetune.py: {entrypoint}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from finetune.finetune import train

    argv = [
        "finetune.py",
        "--model_dir", str(model_dir),
        "--data_path", str(data_path),
        "--output_dir", str(output_dir),
        "--use_lora",
        "--lora_rank", str(args.lora_rank),
        "--bf16",
        "--attn_implementation", args.attn_implementation,
        "--max_len", str(args.max_len),
        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--num_train_epochs", str(args.num_train_epochs),
        "--learning_rate", str(args.learning_rate),
        "--logging_steps", str(args.logging_steps),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        train()
    finally:
        sys.argv = old_argv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def adapter_load_dir(path: Path) -> Path:
    if (path / "adapter_config.json").is_file():
        return path
    configs = list(path.rglob("adapter_config.json"))
    if not configs:
        raise FileNotFoundError(f"No adapter_config.json found under {path}")
    return configs[0].parent


def evaluate_query_adapters(
    records: Sequence[Dict[str, Any]],
    sft_rows: Sequence[Dict[str, Any]],
    processor,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    expected_hints = sum(len(item_hints(item)) for item in records)
    if len(sft_rows) < expected_hints:
        raise ValueError(
            f"Query evaluation needs {expected_hints} SFT rows, found {len(sft_rows)}"
        )

    audio_results = []
    sft_index = 0
    for audio_index, item in enumerate(records):
        audio_path = item_audio_path(item, args.audio_root)
        ground_truth = item_ground_truth(item)
        hint_results = []
        for hint_index, hint in enumerate(item_hints(item)):
            if args.query_lora_dir.exists():
                shutil.rmtree(args.query_lora_dir)
            args.query_lora_dir.parent.mkdir(parents=True, exist_ok=True)

            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".jsonl",
                encoding="utf-8",
                delete=False,
            ) as temporary_handle:
                temporary_path = Path(temporary_handle.name)
                temporary_handle.write(json.dumps(sft_rows[sft_index], ensure_ascii=False) + "\n")
            sft_index += 1

            try:
                train_lora(
                    args.model,
                    temporary_path,
                    args.query_lora_dir,
                    args.moss_audio_root,
                    args,
                )
                base_model = load_base_model(args.model)
                query_model = PeftModel.from_pretrained(
                    base_model,
                    str(adapter_load_dir(args.query_lora_dir)),
                )
                query_model.eval()
                result = evaluate_one_hint(
                    audio_path,
                    hint,
                    ground_truth,
                    query_model,
                    processor,
                    args.max_tokens,
                    args.eval_times,
                    args.do_sample,
                    args.temperature,
                    args.top_p,
                    args.seed + audio_index * 10_000 + hint_index * 100,
                    args.verdict_mode,
                    args.fake_text,
                    args.real_text,
                    args.hint_context,
                )
                hint_results.append({"index": hint_index, "text": hint, **result})
                del query_model
                del base_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            finally:
                temporary_path.unlink(missing_ok=True)
                shutil.rmtree(args.query_lora_dir, ignore_errors=True)

            print(
                f"[query] audio={audio_index + 1}/{len(records)}, "
                f"hint={hint_index + 1}/{len(item_hints(item))}, "
                f"accuracy={hint_results[-1]['accuracy']:.3f}"
            )

        best_accuracy = max(entry["accuracy"] for entry in hint_results)
        audio_accuracy = sum(entry["accuracy"] for entry in hint_results) / len(hint_results)
        hint_spoof_scores = [
            entry["spoof_score"] for entry in hint_results if "spoof_score" in entry
        ]
        audio_spoof_score = (
            sum(hint_spoof_scores) / len(hint_spoof_scores)
            if hint_spoof_scores else None
        )
        audio_results.append(
            {
                "audio_index": audio_index,
                "audio_path": str(audio_path),
                "ground_truth": ground_truth,
                "audio_accuracy": round(audio_accuracy, 4),
                "best_accuracy": round(best_accuracy, 4),
                "spoof_score": audio_spoof_score,
                "best_hints": [entry for entry in hint_results if entry["accuracy"] == best_accuracy],
                "hints": hint_results,
            }
        )

    overall = sum(row["audio_accuracy"] for row in audio_results) / len(audio_results)
    result = {
        "model_type": "query",
        "verdict_mode": args.verdict_mode,
        "n_audios": len(audio_results),
        "eval_times": args.eval_times,
        "sampling": {
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "top_p": args.top_p if args.do_sample else None,
            "seed": args.seed,
        },
        "overall_accuracy": round(overall, 4),
        "audio": audio_results,
    }
    if args.verdict_mode == "logits":
        result["eer"] = empirical_eer(audio_results)
    return result


def save_result(
    result: Dict[str, Any], output_dir: Path, exp_name: str,
    output_file: Path | None = None, overwrite: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%m%d_%H%M%S")
    path = output_file or output_dir / f"{exp_name}_{result['model_type']}_{timestamp}.json"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {path}; pass --overwrite explicitly")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(f"[{result['model_type']}] results -> {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", default="D0_smoke64")
    parser.add_argument(
        "--evaluators",
        nargs="+",
        choices=("baseline", "cpt", "query"),
        default=["baseline"],
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/MOSS-Audio-8B-Instruct"),
    )
    parser.add_argument(
        "--moss_audio_root",
        type=Path,
        default=Path("third_party/MOSS-Audio"),
    )
    parser.add_argument(
        "--dataset_in",
        type=Path,
        default=Path("data/SE-ADD/train/moss/D0_smoke64.jsonl"),
    )
    parser.add_argument(
        "--dataset_in_lora_jsonl",
        type=Path,
        default=Path("data/SE-ADD/train/moss/D0_smoke64_sft.jsonl"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("result/smoke/ttt"))
    parser.add_argument("--audio_root", type=Path, default=Path("data"))
    parser.add_argument("--cpt_lora_dir", type=Path, default=Path("result/smoke/D0_cpt_lora"))
    parser.add_argument("--query_lora_dir", type=Path, default=Path("result/smoke/query_lora_tmp"))
    parser.add_argument("--train_cpt", action="store_true")
    parser.add_argument("--overwrite_cpt", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--eval_times", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=16)
    parser.add_argument(
        "--verdict_mode",
        choices=("generate", "logits"),
        default="generate",
        help="Generate a verdict token or score Fake/Real candidate likelihoods directly.",
    )
    parser.add_argument("--fake_text", default="Fake")
    parser.add_argument("--real_text", default="Real")
    parser.add_argument(
        "--hint_context",
        choices=("neutral", "legacy"),
        default="neutral",
        help=(
            "Frame supplied cues as neutral observations (default), or reproduce "
            "the previous 'forensic hints' wording with legacy."
        ),
    )
    parser.add_argument(
        "--score_direct_baseline",
        action="store_true",
        help=(
            "In logits mode, also score audio without hints and store each "
            "hint's ground-truth-oriented effectiveness_gain."
        ),
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Sample verdicts; required when eval_times is greater than one.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--attn_implementation", default="sdpa")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_times <= 0 or args.max_tokens <= 0:
        raise ValueError("--eval_times and --max_tokens must be positive")
    if not args.do_sample and args.eval_times != 1:
        raise ValueError("Use --do_sample when --eval_times is greater than one")
    if args.verdict_mode == "logits" and (args.do_sample or args.eval_times != 1):
        raise ValueError("Logits mode is deterministic: omit --do_sample and use --eval_times 1")
    if args.temperature <= 0 or not (0 < args.top_p <= 1):
        raise ValueError("--temperature must be positive and --top_p must be in (0, 1]")
    if not args.dataset_in.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {args.dataset_in}")

    records = load_records(args.dataset_in)
    if args.start < 0:
        raise ValueError("--start must be non-negative")
    records = records[args.start:]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        records = records[: args.limit]
    if not records:
        raise ValueError("No records selected")

    print(f"Evaluators: {args.evaluators}; selected audios: {len(records)}")
    processor = MossAudioProcessor.from_pretrained(
        str(args.model),
        trust_remote_code=True,
        enable_time_marker=True,
    )

    if "baseline" in args.evaluators:
        baseline_model = load_base_model(args.model)
        baseline_result = evaluate_dataset(
            records,
            "baseline",
            baseline_model,
            processor,
            args.audio_root,
            args.max_tokens,
            args.eval_times,
            args.do_sample,
            args.temperature,
            args.top_p,
            args.seed,
            args.verdict_mode,
            args.fake_text,
            args.real_text,
            args.score_direct_baseline,
            args.hint_context,
        )
        baseline_result["adapter"] = None
        save_result(baseline_result, args.output_dir, args.exp_name, args.output_file, args.overwrite)
        del baseline_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "cpt" in args.evaluators:
        if args.train_cpt:
            if args.cpt_lora_dir.exists():
                if not args.overwrite_cpt:
                    raise FileExistsError(
                        f"CPT adapter exists: {args.cpt_lora_dir}; use --overwrite_cpt"
                    )
                shutil.rmtree(args.cpt_lora_dir)
            train_lora(
                args.model,
                args.dataset_in_lora_jsonl,
                args.cpt_lora_dir,
                args.moss_audio_root,
                args,
            )
        cpt_base_model = load_base_model(args.model)
        cpt_model = PeftModel.from_pretrained(
            cpt_base_model,
            str(adapter_load_dir(args.cpt_lora_dir)),
        )
        cpt_model.eval()
        cpt_result = evaluate_dataset(
            records,
            "cpt",
            cpt_model,
            processor,
            args.audio_root,
            args.max_tokens,
            args.eval_times,
            args.do_sample,
            args.temperature,
            args.top_p,
            args.seed,
            args.verdict_mode,
            args.fake_text,
            args.real_text,
            args.score_direct_baseline,
            args.hint_context,
        )
        cpt_result["adapter"] = str(adapter_load_dir(args.cpt_lora_dir))
        save_result(cpt_result, args.output_dir, args.exp_name, args.output_file, args.overwrite)
        del cpt_model
        del cpt_base_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "query" in args.evaluators:
        if not args.dataset_in_lora_jsonl.is_file():
            raise FileNotFoundError(f"SFT dataset does not exist: {args.dataset_in_lora_jsonl}")
        sft_rows = load_jsonl(args.dataset_in_lora_jsonl)
        query_result = evaluate_query_adapters(records, sft_rows, processor, args)
        save_result(query_result, args.output_dir, args.exp_name)

    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Requested evaluator branches completed.")


if __name__ == "__main__":
    main()
