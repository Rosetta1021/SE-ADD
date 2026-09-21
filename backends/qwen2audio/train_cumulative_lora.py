"""Continue a prior Qwen2-Audio LoRA without merging it into the base model."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel

from utils import (
    DEFAULT_MODEL,
    load_base_model,
    load_processor,
    load_trainable_adapter,
    resolve_adapter_dir,
    train_adapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input-adapter", required=True, type=Path)
    parser.add_argument("--dataset-in", required=True, type=Path)
    parser.add_argument("--output-adapter", required=True, type=Path)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--skip-reload-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Cumulative LoRA training must run on a CUDA compute node")
    input_adapter = resolve_adapter_dir(args.input_adapter)
    if not args.dataset_in.is_file():
        raise FileNotFoundError(args.dataset_in)
    if args.output_adapter.resolve() == input_adapter.resolve():
        raise ValueError("Input and output adapter directories must differ")
    resuming = args.resume_from_checkpoint is not None
    if args.output_adapter.exists() and not resuming:
        if not args.overwrite_output:
            raise FileExistsError(
                f"Output exists: {args.output_adapter}; use a new directory or --overwrite-output"
            )
        shutil.rmtree(args.output_adapter)
    args.output_adapter.parent.mkdir(parents=True, exist_ok=True)

    processor = load_processor(args.model)
    base_model = load_base_model(args.model, training=True)
    print(f"Loading previous adapter as trainable: {input_adapter}")
    model = load_trainable_adapter(base_model, input_adapter)
    report = train_adapter(
        model,
        processor,
        args.dataset_in,
        args.output_adapter,
        max_length=args.max_len,
        batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    report["input_adapter"] = str(input_adapter)
    with (args.output_adapter / "seadd_training_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Saved cumulative adapter -> {args.output_adapter}")

    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    if not args.skip_reload_check:
        verify_base = load_base_model(args.model, training=False)
        verify_model = PeftModel.from_pretrained(
            verify_base, str(resolve_adapter_dir(args.output_adapter)), is_trainable=False
        )
        verify_model.eval()
        print("Cumulative adapter reload succeeded.")
        del verify_model, verify_base
        gc.collect()
        torch.cuda.empty_cache()
    print("Cumulative LoRA passed (previous adapter updated; base not merged).")


if __name__ == "__main__":
    main()
