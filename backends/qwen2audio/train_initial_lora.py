"""Train an initial Qwen2-Audio LoRA adapter and verify clean reload."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel

from utils import (
    DEFAULT_LORA_TARGETS,
    DEFAULT_MODEL,
    create_initial_lora,
    load_base_model,
    load_processor,
    resolve_adapter_dir,
    train_adapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("cpt-smoke",), default="cpt-smoke")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset_in_lora_jsonl", required=True, type=Path)
    parser.add_argument("--cpt_lora_dir", required=True, type=Path)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora-target-modules", default=DEFAULT_LORA_TARGETS)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--overwrite-adapter", action="store_true")
    parser.add_argument("--skip-reload-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LoRA training must run on a CUDA compute node")
    if not args.model.is_dir() or not args.dataset_in_lora_jsonl.is_file():
        raise FileNotFoundError("Model snapshot or SFT JSONL is missing")
    resuming = args.resume_from_checkpoint is not None
    if args.cpt_lora_dir.exists() and not resuming:
        if not args.overwrite_adapter:
            raise FileExistsError(
                f"Adapter exists: {args.cpt_lora_dir}; use a new directory or --overwrite-adapter"
            )
        shutil.rmtree(args.cpt_lora_dir)
    args.cpt_lora_dir.parent.mkdir(parents=True, exist_ok=True)

    processor = load_processor(args.model)
    base_model = load_base_model(args.model, training=True)
    model = create_initial_lora(base_model, args.lora_rank, args.lora_target_modules)
    print("Initial adapter created; base parameters remain frozen.")
    report = train_adapter(
        model,
        processor,
        args.dataset_in_lora_jsonl,
        args.cpt_lora_dir,
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
    print(f"Saved initial adapter -> {args.cpt_lora_dir}")
    print(json.dumps(report, indent=2))

    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()
    if not args.skip_reload_check:
        verify_base = load_base_model(args.model, training=False)
        verify_model = PeftModel.from_pretrained(
            verify_base, str(resolve_adapter_dir(args.cpt_lora_dir)), is_trainable=False
        )
        verify_model.eval()
        print(
            f"Adapter reload succeeded: model={verify_model.__class__.__name__}, "
            f"device={next(verify_model.parameters()).device}"
        )
        del verify_model, verify_base
        gc.collect()
        torch.cuda.empty_cache()
    print("Qwen initial LoRA smoke training passed (adapter only; base not merged).")


if __name__ == "__main__":
    main()
