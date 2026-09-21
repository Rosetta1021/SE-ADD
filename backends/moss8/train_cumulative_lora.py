"""Continue training an existing MOSS-Audio LoRA adapter on a new SE-ADD round.

The base MOSS-Audio checkpoint remains frozen.  Existing LoRA weights are
loaded with ``is_trainable=True`` and updated on the new SFT JSONL.  A new
adapter snapshot is written, preserving the input adapter as the prior round.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import torch
import transformers
from peft import PeftModel


def adapter_load_dir(path: Path) -> Path:
    if (path / "adapter_config.json").is_file():
        return path
    configs = list(path.rglob("adapter_config.json"))
    if len(configs) != 1:
        raise FileNotFoundError(
            f"Expected exactly one adapter_config.json under {path}; found {configs}"
        )
    return configs[0].parent


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict) or "conversation" not in row:
                raise ValueError(f"Missing conversation at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No SFT rows found in {path}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/MOSS-Audio-8B-Instruct"),
    )
    parser.add_argument(
        "--moss-audio-root",
        type=Path,
        default=Path("third_party/MOSS-Audio"),
    )
    parser.add_argument("--input-adapter", required=True, type=Path)
    parser.add_argument("--dataset-in", required=True, type=Path)
    parser.add_argument("--output-adapter", required=True, type=Path)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--skip-reload-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {args.model}")
    if not args.dataset_in.is_file():
        raise FileNotFoundError(f"SFT JSONL does not exist: {args.dataset_in}")
    finetune_file = args.moss_audio_root / "finetune" / "finetune.py"
    if not finetune_file.is_file():
        raise FileNotFoundError(f"MOSS-Audio finetune module not found: {finetune_file}")
    input_adapter = adapter_load_dir(args.input_adapter)
    if args.output_adapter.resolve() == input_adapter.resolve():
        raise ValueError("Input and output adapter directories must differ")
    if args.output_adapter.exists():
        if not args.overwrite_output:
            raise FileExistsError(
                f"Output exists: {args.output_adapter}; pass --overwrite-output to replace it"
            )
        shutil.rmtree(args.output_adapter)
    args.output_adapter.parent.mkdir(parents=True, exist_ok=True)

    moss_root = str(args.moss_audio_root.resolve())
    if moss_root not in sys.path:
        sys.path.insert(0, moss_root)
    from finetune.finetune import MossAudioDataset
    from src.configuration_moss_audio import MossAudioConfig
    from src.modeling_moss_audio import MossAudioModel

    rows = load_jsonl(args.dataset_in)
    config = MossAudioConfig.from_pretrained(str(args.model))
    for cfg in (config, getattr(config, "language_config", None)):
        if cfg is not None:
            cfg._attn_implementation = args.attn_implementation

    print(f"Loading base model: {args.model}")
    base_model = MossAudioModel.from_pretrained(
        str(args.model),
        config=config,
        dtype=torch.bfloat16,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(args.model), trust_remote_code=True
    )
    print(f"Loading trainable prior adapter: {input_adapter}")
    model = PeftModel.from_pretrained(
        base_model,
        str(input_adapter),
        is_trainable=True,
    )
    model.print_trainable_parameters()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable == 0:
        raise RuntimeError("Loaded adapter has no trainable parameters")

    train_dataset = MossAudioDataset(rows, tokenizer, args.max_len, "")
    print(f"Train samples: {len(train_dataset)}")
    training_args = transformers.TrainingArguments(
        output_dir=str(args.output_adapter),
        overwrite_output_dir=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy="no",
        bf16=True,
        optim="adamw_torch_fused",
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.train()
    model.save_pretrained(str(args.output_adapter))
    tokenizer.save_pretrained(str(args.output_adapter))

    config_path = args.output_adapter / "adapter_config.json"
    weights = list(args.output_adapter.glob("adapter_model.*"))
    if not config_path.is_file() or not weights:
        raise RuntimeError(
            f"Cumulative adapter was not saved correctly: config={config_path}, weights={weights}"
        )
    print(f"Saved cumulative adapter -> {args.output_adapter}")

    del trainer, model, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not args.skip_reload_check:
        print("Reloading cumulative adapter for verification...")
        verify_base = MossAudioModel.from_pretrained(
            str(args.model),
            trust_remote_code=True,
            dtype="auto",
            device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        )
        verify_model = PeftModel.from_pretrained(
            verify_base, str(args.output_adapter)
        )
        verify_model.eval()
        print("Cumulative adapter reload succeeded.")
        del verify_model, verify_base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
