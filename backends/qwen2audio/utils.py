"""Shared Qwen2-Audio utilities for the SE-ADD backend."""

from __future__ import annotations

import gc
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import librosa
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2AudioForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


# A Hugging Face model id works when network/cache access is available; users
# can always override it with a local checkpoint path through ``--model``.
DEFAULT_MODEL = Path("Qwen/Qwen2-Audio-7B-Instruct")
DEFAULT_LORA_TARGETS = (
    r"^language_model\.model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(gate_proj|up_proj|down_proj))$"
)
SCORING_METHOD = "teacher_forced_full_candidate_sequence_log_likelihood_sum"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        first = next((character for character in iter(lambda: handle.read(1), "") if not character.isspace()), "")
    if first == "[":
        with path.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError(f"Expected a JSON array of objects: {path}")
        return rows
    return load_jsonl(path)


def resolve_audio_path(row: dict[str, Any], audio_root: Path) -> Path:
    value = row.get("path", row.get("context"))
    if not value:
        raise ValueError("Every record must contain 'path' or legacy 'context'")
    path = Path(str(value))
    return path if path.is_absolute() else audio_root / path


def item_ground_truth(row: dict[str, Any]) -> str:
    label = str(row.get("label", "")).strip().lower()
    if label == "spoof":
        return "fake"
    if label == "bonafide":
        return "real"
    questions = row.get("question")
    if isinstance(questions, list) and questions:
        answer = str(questions[0].get("answer", "")).strip().lower()
        if answer in {"fake", "real"}:
            return answer
    raise ValueError("Cannot derive Fake/Real ground truth")


def item_hints(row: dict[str, Any]) -> list[str]:
    hints = row.get("hints")
    if isinstance(hints, str):
        hints = [hints]
    if not isinstance(hints, list) or not hints:
        raise ValueError("Every record must contain at least one hint")
    if any(not isinstance(hint, str) or not hint.strip() for hint in hints):
        raise ValueError("All hints must be non-empty strings")
    return [hint.strip() for hint in hints]


def largest_remainder_counts(weights: Sequence[float], total: int) -> list[int]:
    exact = [weight * total for weight in weights]
    counts = [int(value) for value in exact]
    order = sorted(range(len(weights)), key=lambda index: exact[index] - counts[index], reverse=True)
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    return counts


def balanced_environment_sample(
    records: Sequence[dict[str, Any]], total: int, seed: int
) -> list[dict[str, Any]]:
    if total <= 0 or total % 2:
        raise ValueError("--sample-size must be a positive even number")
    per_class = total // 2
    bonafide = [row for row in records if row.get("label") == "bonafide"]
    spoof = [row for row in records if row.get("label") == "spoof"]
    if len(bonafide) < per_class or len(spoof) < per_class:
        raise ValueError(
            f"Cannot draw {per_class} per class: bonafide={len(bonafide)}, spoof={len(spoof)}"
        )
    pools: dict[str, list[dict[str, Any]]] = {}
    for row in spoof:
        pools.setdefault(str(row.get("attack_id", "unknown")), []).append(row)
    attack_ids = sorted(pools)
    counts = largest_remainder_counts(
        [len(pools[attack]) / len(spoof) for attack in attack_ids], per_class
    )
    rng = random.Random(seed)
    selected = rng.sample(bonafide, per_class)
    for attack, count in zip(attack_ids, counts):
        selected.extend(rng.sample(pools[attack], count))
    rng.shuffle(selected)
    composition = {
        attack: sum(str(row.get("attack_id")) == attack for row in selected if row.get("label") == "spoof")
        for attack in attack_ids
    }
    print(
        f"Balanced subset: total={len(selected)}, bonafide={per_class}, "
        f"spoof={per_class}, spoof composition={composition}"
    )
    return selected


def audio_conversation(audio_path: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": str(audio_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def load_waveform(audio_path: Path, sampling_rate: int) -> Any:
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    waveform, _ = librosa.load(str(audio_path), sr=sampling_rate, mono=True)
    return waveform


def prepare_audio_inputs(audio_path: Path, prompt: str, processor, device: torch.device):
    conversation = audio_conversation(audio_path, prompt)
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    sampling_rate = processor.feature_extractor.sampling_rate
    waveform = load_waveform(audio_path, sampling_rate)
    inputs = processor(
        text=text,
        audio=[waveform],
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in inputs.items() if torch.is_tensor(value)}


def load_processor(model_path: Path):
    return AutoProcessor.from_pretrained(str(model_path), local_files_only=True)


def load_base_model(model_path: Path, *, training: bool = False):
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "dtype": torch.bfloat16,
    }
    if not training:
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen2-Audio inference requires a CUDA compute node")
        kwargs["device_map"] = "cuda:0"
    model = Qwen2AudioForConditionalGeneration.from_pretrained(str(model_path), **kwargs)
    if training:
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.config.use_cache = False
    else:
        model.eval()
    return model


def resolve_adapter_dir(path: Path) -> Path:
    if (path / "adapter_config.json").is_file():
        return path
    candidates = sorted(path.rglob("adapter_config.json"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one adapter_config.json under {path}; found {candidates}"
        )
    return candidates[0].parent


def load_inference_model(model_path: Path, adapter: Path | None = None):
    base = load_base_model(model_path, training=False)
    if adapter is None:
        return base, base
    adapter_dir = resolve_adapter_dir(adapter)
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    model.eval()
    return model, base


def candidate_ids(processor, text: str, device: torch.device) -> torch.Tensor:
    ids = processor.tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Candidate produced no tokens: {text!r}")
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


def candidate_log_likelihood(model, inputs: dict[str, torch.Tensor], ids: torch.Tensor) -> float:
    prefix_length = inputs["input_ids"].shape[1]
    candidate_length = ids.shape[1]
    model_inputs = dict(inputs)
    model_inputs["input_ids"] = torch.cat([inputs["input_ids"], ids], dim=1)
    extension = torch.ones(
        (inputs["attention_mask"].shape[0], candidate_length),
        dtype=inputs["attention_mask"].dtype,
        device=inputs["attention_mask"].device,
    )
    model_inputs["attention_mask"] = torch.cat([inputs["attention_mask"], extension], dim=1)
    model_inputs.pop("position_ids", None)
    model_inputs.pop("labels", None)
    with torch.inference_mode():
        output = model(**model_inputs, use_cache=False)
    logits = output.logits[0, prefix_length - 1 : prefix_length + candidate_length - 1].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    return float(log_probs.gather(1, ids[0].unsqueeze(1)).sum().item())


def score_verdict(
    audio_path: Path,
    hint: str | None,
    model,
    processor,
    fake_text: str = "Fake",
    real_text: str = "Real",
    hint_context: str = "neutral",
) -> dict[str, Any]:
    if hint is None:
        prompt = "Is this audio fake or real? Answer with exactly one word: Fake or Real."
    elif hint_context == "neutral":
        prompt = (
            "The following are neutral observations about this audio. They may support "
            "either class and must not be treated as evidence of fakery by default. "
            f"Consider them together with the audio:\n{hint}\n"
            "Is this audio fake or real? Answer with exactly one word: Fake or Real."
        )
    elif hint_context == "legacy":
        prompt = (
            f"These are forensic hints that may help with the judgment:\n{hint}\n"
            "Is this audio fake or real? Answer with exactly one word: Fake or Real."
        )
    else:
        raise ValueError(f"Unsupported hint context: {hint_context}")
    device = next(model.parameters()).device
    inputs = prepare_audio_inputs(audio_path, prompt, processor, device)
    fake_ids = candidate_ids(processor, fake_text, device)
    real_ids = candidate_ids(processor, real_text, device)
    fake_ll = candidate_log_likelihood(model, inputs, fake_ids)
    real_ll = candidate_log_likelihood(model, inputs, real_ids)
    score = fake_ll - real_ll
    return {
        "response": None,
        "prediction": "fake" if score >= 0 else "real",
        "fake_token_ids": fake_ids[0].tolist(),
        "real_token_ids": real_ids[0].tolist(),
        "fake_log_likelihood": fake_ll,
        "real_log_likelihood": real_ll,
        "spoof_score": score,
        "p_fake": 1.0 / (1.0 + math.exp(-score)),
        "scoring_method": SCORING_METHOD,
    }


def empirical_eer(audio_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = [1 if row["ground_truth"] == "fake" else 0 for row in audio_rows]
    scores = [float(row["spoof_score"]) for row in audio_rows]
    if 0 not in labels or 1 not in labels:
        return {"available": False, "reason": "EER requires both classes"}
    thresholds = [float("inf"), *sorted(set(scores), reverse=True), float("-inf")]
    best = None
    for threshold in thresholds:
        predictions = [int(score >= threshold) for score in scores]
        fpr = sum(pred == 1 for pred, label in zip(predictions, labels) if label == 0) / labels.count(0)
        fnr = sum(pred == 0 for pred, label in zip(predictions, labels) if label == 1) / labels.count(1)
        candidate = {
            "eer": (fpr + fnr) / 2.0,
            "threshold": threshold,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "gap": abs(fpr - fnr),
        }
        if best is None or candidate["gap"] < best["gap"]:
            best = candidate
    best["available"] = True
    return best


class JsonlConversationDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = load_jsonl(path)
        if not self.rows:
            raise ValueError(f"No SFT rows found: {path}")
        for index, row in enumerate(self.rows):
            conversation = row.get("conversation")
            if not isinstance(conversation, list) or len(conversation) != 3:
                raise ValueError(f"SFT row {index} must contain exactly three messages")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _builder_row_to_qwen(row: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    audio_message, text_message, assistant_message = row["conversation"]
    if audio_message.get("message_type") != "audio":
        raise ValueError("First builder message must be user/audio")
    if text_message.get("message_type") != "text":
        raise ValueError("Second builder message must be user/text")
    if assistant_message.get("role") != "assistant":
        raise ValueError("Third builder message must be assistant/text")
    audio_path = Path(str(audio_message["content"]))
    conversation = audio_conversation(audio_path, str(text_message["content"]))
    conversation.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": str(assistant_message["content"])}],
        }
    )
    return audio_path, conversation


@dataclass
class QwenSftCollator:
    processor: Any
    max_length: int = 2048

    def __post_init__(self) -> None:
        self.assistant_marker = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = []
        waveforms = []
        sampling_rate = self.processor.feature_extractor.sampling_rate
        for row in rows:
            audio_path, conversation = _builder_row_to_qwen(row)
            texts.append(
                self.processor.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=False
                )
            )
            waveforms.append(load_waveform(audio_path, sampling_rate))
        batch = self.processor(
            text=texts,
            audio=waveforms,
            sampling_rate=sampling_rate,
            padding=True,
            return_tensors="pt",
        )
        actual_length = batch["input_ids"].shape[1]
        if actual_length > self.max_length:
            raise ValueError(
                f"Processed multimodal sequence length {actual_length} exceeds "
                f"max_length={self.max_length}; refusing to truncate the audio"
            )
        labels = batch["input_ids"].clone()
        for row_index, input_ids in enumerate(batch["input_ids"]):
            ids = input_ids.tolist()
            marker_start = _find_last_subsequence(ids, self.assistant_marker)
            if marker_start is None:
                raise ValueError("Assistant marker missing after tokenization/truncation")
            response_start = marker_start + len(self.assistant_marker)
            labels[row_index, :response_start] = -100
            labels[row_index, batch["attention_mask"][row_index] == 0] = -100
            if torch.all(labels[row_index] == -100):
                raise ValueError("No assistant target tokens remain after truncation")
        batch["labels"] = labels
        return dict(batch)


def _find_last_subsequence(values: list[int], pattern: list[int]) -> int | None:
    for start in range(len(values) - len(pattern), -1, -1):
        if values[start : start + len(pattern)] == pattern:
            return start
    return None


def create_initial_lora(base_model, rank: int, target_modules: str = DEFAULT_LORA_TARGETS):
    config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    return get_peft_model(base_model, config)


def load_trainable_adapter(base_model, adapter: Path):
    return PeftModel.from_pretrained(
        base_model, str(resolve_adapter_dir(adapter)), is_trainable=True
    )


def trainable_parameter_report(model) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100 * trainable / total:.6f}")
    non_lora = [name for name, parameter in model.named_parameters() if parameter.requires_grad and "lora_" not in name]
    if non_lora:
        raise RuntimeError(f"Non-LoRA parameters are trainable: {non_lora[:20]}")
    if trainable == 0:
        raise RuntimeError("No trainable LoRA parameters")
    return trainable, total


def lora_state_snapshot(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().float().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    }


def lora_change_report(before: dict[str, torch.Tensor], model) -> dict[str, Any]:
    after = {
        name: parameter.detach().cpu().float()
        for name, parameter in model.named_parameters()
        if name in before
    }
    changed = []
    squared_delta = 0.0
    for name, old in before.items():
        if name not in after:
            raise RuntimeError(f"LoRA parameter disappeared during training: {name}")
        delta = after[name] - old
        norm = float(torch.linalg.vector_norm(delta).item())
        squared_delta += norm * norm
        if norm > 0:
            changed.append(name)
    report = {
        "tracked_parameter_tensors": len(before),
        "changed_parameter_tensors": len(changed),
        "global_delta_l2": math.sqrt(squared_delta),
    }
    print(json.dumps(report, indent=2))
    if not changed:
        raise RuntimeError("Training completed but no LoRA parameter changed")
    return report


def cleanup_cuda(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_adapter(
    model,
    processor,
    dataset_path: Path,
    output_dir: Path,
    *,
    max_length: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    epochs: float,
    learning_rate: float,
    logging_steps: int,
    save_steps: int = 50,
    save_total_limit: int = 2,
    resume_from_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train and save PEFT weights only; never merge or save the base model."""
    trainable_parameter_report(model)
    before = lora_state_snapshot(model)
    if not before:
        raise RuntimeError("No LoRA tensors found before training")
    dataset = JsonlConversationDataset(dataset_path)
    collator = QwenSftCollator(processor, max_length=max_length)
    model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        save_only_model=False,
        bf16=True,
        fp16=False,
        optim="adamw_torch_fused",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
    )
    resume_value: bool | str | None
    if resume_from_checkpoint is None:
        resume_value = None
    elif str(resume_from_checkpoint).lower() == "latest":
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        )
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint-* directory under {output_dir}")
        resume_value = str(checkpoints[-1])
    else:
        checkpoint = Path(resume_from_checkpoint)
        if not checkpoint.is_dir():
            raise FileNotFoundError(checkpoint)
        resume_value = str(checkpoint)
    if resume_value is not None:
        print(f"Resuming full Trainer state from: {resume_value}")
    train_result = trainer.train(resume_from_checkpoint=resume_value)
    change_report = lora_change_report(before, model)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    processor.save_pretrained(str(output_dir))
    if not (output_dir / "adapter_config.json").is_file():
        raise RuntimeError("adapter_config.json was not saved")
    if not list(output_dir.glob("adapter_model.*")):
        raise RuntimeError("Adapter weights were not saved")
    report = {
        **change_report,
        "train_samples": len(dataset),
        "train_loss": float(train_result.training_loss),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "max_length": max_length,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "resumed_from_checkpoint": resume_value,
        "base_model_merged": False,
        "saved_artifact": "peft_adapter_only",
    }
    with (output_dir / "seadd_training_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return report
