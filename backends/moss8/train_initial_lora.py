"""Stage-wise SE-ADD training and evaluation entry point.

The current ``cpt-smoke`` stage verifies the smallest complete LoRA path:
MOSS-formatted JSONL -> one CPT LoRA training run -> adapter reload.
It deliberately skips the expensive per-hint query-adapter loop.
"""

import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel

from src.modeling_moss_audio import MossAudioModel


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if "conversation" not in item:
                raise ValueError(f"Missing 'conversation' at {path}:{line_number}")
            count += 1
    return count


def train_cpt_lora(args: argparse.Namespace) -> None:
    moss_audio_root = args.moss_audio_root.resolve()
    finetune_module = moss_audio_root / "finetune" / "finetune.py"
    if not finetune_module.is_file():
        raise FileNotFoundError(
            f"Cannot find MOSS-Audio finetune entry point: {finetune_module}"
        )
    if str(moss_audio_root) not in sys.path:
        sys.path.insert(0, str(moss_audio_root))

    from finetune.finetune import train

    training_argv = [
        "finetune.py",
        "--model_dir",
        str(args.model),
        "--data_path",
        str(args.dataset_in_lora_jsonl),
        "--output_dir",
        str(args.cpt_lora_dir),
        "--use_lora",
        "--lora_rank",
        str(args.lora_rank),
        "--bf16",
        "--attn_implementation",
        args.attn_implementation,
        "--max_len",
        str(args.max_len),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--learning_rate",
        str(args.learning_rate),
        "--logging_steps",
        str(args.logging_steps),
    ]

    old_argv = sys.argv
    sys.argv = training_argv
    try:
        train()
    finally:
        sys.argv = old_argv
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def find_adapter_weights(adapter_dir: Path) -> list[Path]:
    names = ("adapter_model.safetensors", "adapter_model.bin")
    return [path for name in names for path in adapter_dir.rglob(name)]


def verify_adapter_files(adapter_dir: Path) -> Path:
    configs = list(adapter_dir.rglob("adapter_config.json"))
    weights = find_adapter_weights(adapter_dir)
    if not configs or not weights:
        raise RuntimeError(
            f"LoRA training finished but adapter files were not found under {adapter_dir}. "
            f"configs={configs}, weights={weights}"
        )
    print(f"Adapter config: {configs[0]}")
    print(f"Adapter weights: {weights[0]}")
    load_dir = configs[0].parent
    if not any(path.parent == load_dir for path in weights):
        raise RuntimeError(f"Adapter config and weights are not colocated under {load_dir}")
    return load_dir


def verify_adapter_reload(model_dir: Path, adapter_dir: Path) -> None:
    """Load one base model plus the trained adapter, then release both."""
    if torch.cuda.is_available():
        device_map = "cuda:0"
    elif torch.backends.mps.is_available():
        device_map = "mps"
    else:
        device_map = "cpu"

    print("Reloading one base model to verify the trained adapter...")
    base_model = MossAudioModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        dtype="auto",
        device_map=device_map,
    )
    adapter_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    adapter_model.eval()
    print("Adapter reload succeeded.")

    del adapter_model
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("cpt-smoke",), default="cpt-smoke")
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
    parser.add_argument("--dataset_in_lora_jsonl", required=True, type=Path)
    parser.add_argument(
        "--cpt_lora_dir",
        required=True,
        type=Path,
    )
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument(
        "--overwrite-adapter",
        action="store_true",
        help="Delete an existing smoke adapter directory before training.",
    )
    parser.add_argument(
        "--skip-reload-check",
        action="store_true",
        help="Only verify adapter files; do not load the 4B base model again.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model directory does not exist: {args.model}")
    if not args.dataset_in_lora_jsonl.is_file():
        raise FileNotFoundError(f"SFT JSONL does not exist: {args.dataset_in_lora_jsonl}")

    entries = count_jsonl(args.dataset_in_lora_jsonl)
    if entries == 0:
        raise ValueError(f"No SFT entries found in {args.dataset_in_lora_jsonl}")
    print(f"Validated {entries} SFT entries: {args.dataset_in_lora_jsonl}")

    if args.cpt_lora_dir.exists():
        if not args.overwrite_adapter:
            raise FileExistsError(
                f"Adapter directory already exists: {args.cpt_lora_dir}. "
                "Use a new directory or pass --overwrite-adapter."
            )
        shutil.rmtree(args.cpt_lora_dir)
    args.cpt_lora_dir.parent.mkdir(parents=True, exist_ok=True)

    train_cpt_lora(args)
    adapter_load_dir = verify_adapter_files(args.cpt_lora_dir)
    if not args.skip_reload_check:
        verify_adapter_reload(args.model, adapter_load_dir)

    print(
        f"CPT LoRA smoke test passed: entries={entries}, "
        f"adapter={args.cpt_lora_dir}"
    )


if __name__ == "__main__":
    main()
