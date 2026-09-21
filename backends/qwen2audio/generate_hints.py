"""Generate MOSS-compatible forensic hints with Qwen2-Audio."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch

from utils import (
    DEFAULT_MODEL,
    balanced_environment_sample,
    load_inference_model,
    load_jsonl,
    load_processor,
    prepare_audio_inputs,
    resolve_adapter_dir,
    resolve_audio_path,
)


HINT_PROMPT = """Listen carefully to this audio and list only observable forensic cues relevant to audio deepfake detection.
Provide 2 to 5 concise, non-redundant cues. Each cue must describe one specific phenomenon assessable from audio alone, such as timing, pitch, voice texture, articulation, continuity, background noise, or recording channel properties.
Do not discuss lip-sync, faces, video, or any other visual evidence. Do not use speaker demographics, accent, linguistic content, or duration as evidence. Do not state or imply that the audio is Real, Fake, authentic, synthetic, or a deepfake, and do not give a final verdict.
Use exactly this format:
Cue 1: <observable audio phenomenon>
Cue 2: <observable audio phenomenon>"""

VISUAL_PATTERN = re.compile(r"\b(lip[- ]?sync|facial|face|video|visual|mouth|frame)\b", re.I)
VERDICT_PATTERN = re.compile(
    r"(?im)(?:\bthis (?:audio|clip|recording) (?:is|sounds|appears) "
    r"(?:real|fake|authentic|synthetic|a deepfake)\b|"
    r"\bfinal verdict\b|\bclassified as (?:real|fake|bonafide|spoof)\b|"
    r"^\s*(?:real|fake)\s*[.!]?\s*$)"
)
CUE_PATTERN = re.compile(
    r"(?:\bCue\s+\d+\s*:|^\s*(?:\d+\s*[.)]|[-*])\s*\S)", re.I | re.M
)


def valid_hint(text: str) -> tuple[bool, str | None]:
    text = text.strip()
    if not text:
        return False, "empty"
    if VISUAL_PATTERN.search(text):
        return False, "visual_language"
    if VERDICT_PATTERN.search(text):
        return False, "verdict_leakage"
    if len(CUE_PATTERN.findall(text)) < 2:
        return False, "fewer_than_two_cues"
    return True, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--dataset_in", required=True, type=Path)
    parser.add_argument("--dataset_out", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--n", type=int, default=-1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=160)
    parser.add_argument("--audio_root", type=Path, default=Path("../data"))
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k <= 0 or args.max_tokens <= 0 or args.max_retries < 0:
        raise ValueError("--k and --max_tokens must be positive; --max-retries cannot be negative")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("Invalid sampling parameters")
    if args.dataset_out.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {args.dataset_out}; pass --overwrite to replace it"
        )
    if not args.model.is_dir():
        raise FileNotFoundError(f"Model snapshot does not exist: {args.model}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows = load_jsonl(args.dataset_in)
    if args.sample_size is not None:
        selected = balanced_environment_sample(rows, args.sample_size, args.seed)
    else:
        selected = list(rows)
    end = None if args.n <= 0 else args.start + args.n
    selected = selected[args.start:end]
    if not selected:
        raise ValueError("No records selected")

    processor = load_processor(args.model)
    model, base_model = load_inference_model(args.model, args.adapter)
    adapter_dir = resolve_adapter_dir(args.adapter) if args.adapter else None
    device = next(model.parameters()).device
    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)

    with args.dataset_out.open("w", encoding="utf-8") as output:
        for record_index, row in enumerate(selected):
            audio_path = resolve_audio_path(row, args.audio_root)
            inputs = prepare_audio_inputs(audio_path, HINT_PROMPT, processor, device)
            hints: list[str] = []
            rejection_audit = []
            source_index = args.start + record_index
            for completion_index in range(args.k):
                accepted = None
                for retry in range(args.max_retries + 1):
                    generation_seed = (
                        args.seed + source_index * args.k * (args.max_retries + 1)
                        + completion_index * (args.max_retries + 1) + retry
                    )
                    torch.manual_seed(generation_seed)
                    torch.cuda.manual_seed_all(generation_seed)
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=args.max_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            do_sample=True,
                            num_beams=1,
                            use_cache=True,
                        )
                    input_length = inputs["input_ids"].shape[1]
                    text = processor.decode(
                        generated[0, input_length:], skip_special_tokens=True
                    ).strip()
                    is_valid, reason = valid_hint(text)
                    if is_valid:
                        accepted = text
                        break
                    rejection_audit.append(
                        {
                            "hint_index": completion_index,
                            "retry": retry,
                            "seed": generation_seed,
                            "reason": reason,
                            "response_preview": text[:240],
                        }
                    )
                if accepted is None:
                    raise RuntimeError(
                        f"Could not generate a valid hint for audio={audio_path}, "
                        f"hint_index={completion_index}; audit={rejection_audit[-(args.max_retries + 1):]}"
                    )
                hints.append(accepted)

            output_row = dict(row)
            output_row["hints"] = hints
            output_row["generation_adapter"] = str(adapter_dir) if adapter_dir else None
            output_row["generation_seed"] = args.seed
            output_row["generation_config"] = {
                "k": args.k,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
            }
            output_row["hint_rejection_audit"] = rejection_audit
            output.write(json.dumps(output_row, ensure_ascii=False) + "\n")
            output.flush()
            completed = record_index + 1
            if completed == 1 or completed % args.progress_every == 0 or completed == len(selected):
                print(
                    f"Completed {completed}/{len(selected)}: "
                    f"{row.get('utterance_id', audio_path.name)}"
                )

    print(f"Saved -> {args.dataset_out} ({len(selected)} records, K={args.k})")
    del model, base_model, processor


if __name__ == "__main__":
    main()
