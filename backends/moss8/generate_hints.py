import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from peft import PeftModel

from src.audio_io import load_audio
from src.modeling_moss_audio import MossAudioModel
from src.processing_moss_audio import MossAudioProcessor


MAKE_AUDIO_DATA_TEMPLATES_BASE: dict[str, Dict] = {
    "audio-profile-v3": {
        "type": "text",
        "text": (
            "Listen carefully and create a neutral audio profile for downstream "
            "analysis. Report exactly four concise observations about the requested "
            "audio dimensions. Describe only what is heard; do not assess authenticity, "
            "detect manipulation, infer a class, or explain what any observation means. "
            "Do not mention speaker identity, gender, age, accent, language, semantic "
            "content, visual information, or clip duration. Avoid evaluative words such "
            "as natural, unnatural, synthetic, artificial, robotic, suspicious, real, "
            "fake, authentic, or deepfake.\n"
            "Use exactly this format:\n"
            "Cue 1: <timing, rhythm, pauses, or speaking-rate observation>\n"
            "Cue 2: <pitch, intonation, or voice-texture observation>\n"
            "Cue 3: <articulation, pronunciation-transition, or continuity observation>\n"
            "Cue 4: <recording, background, noise, or channel observation>\n"
        ),
    },
    "forensic-cues-v2": {
        "type": "text",
        "text": (
            "Listen carefully to this audio and report only observable forensic cues "
            "that are relevant to audio deepfake detection.\n"
            "Provide 2 to 5 concise, non-redundant cues. Each cue must describe one "
            "specific phenomenon that can be assessed from audio alone. Do not use "
            "visual evidence, speaker demographics, accent, linguistic content, or "
            "recording duration as evidence. Write neutral observations only: do not "
            "call a cue natural, unnatural, synthetic, artificial, robotic, suspicious, "
            "or typical of any class, and do not explain what verdict a cue supports. "
            "Do not state or imply a final Real/Fake or authentic/deepfake verdict.\n"
            "Use exactly this format:\n"
            "Cue 1: <observable audio phenomenon>\n"
            "Cue 2: <observable audio phenomenon>\n"
        ),
    },
    "hints-instruct": {
        "type": "text",
        "text": (
            "Let's listen to this audio and produce a list of various specific "
            "evidences that help to determine whether it is a deepfake.\n"
        ),
    },
    "hints": {
        "type": "text",
        "text": "Let's listen to this audio and produce a list of hints that helps to tell whether it is a deepfake.",
    },
    "hints-long": {
        "type": "text",
        "text": "Let's listen to this audio and produce a long list of hints that helps to tell whether it is a deepfake.",
    },
    "hints-very-long": {
        "type": "text",
        "text": "Let's listen to this audio and produce a very long list of hints that helps to tell whether it is a deepfake.",
    },
    "hints-chain-of-thought": {
        "type": "text",
        "text": (
            "Let's listen to this audio, think step by step, and then produce a list "
            "of hints that helps to tell whether it is a deepfake. We should first "
            "generate a \"Thought Process\" and then \"Hints\""
        ),
    },
}


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if "path" not in item:
                raise ValueError(f"Missing 'path' field at {path}:{line_number}")
            records.append(item)
    return records


def largest_remainder_counts(weights: Sequence[float], total: int) -> List[int]:
    """Convert proportions to integer counts while preserving the exact total."""
    exact = [weight * total for weight in weights]
    counts = [int(value) for value in exact]
    remaining = total - sum(counts)
    order = sorted(
        range(len(weights)),
        key=lambda index: exact[index] - counts[index],
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def balanced_environment_sample(
    records: Sequence[Dict[str, Any]],
    total: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Sample 1:1 bonafide/spoof while retaining spoof attack composition."""
    if total <= 0 or total % 2:
        raise ValueError("--sample-size must be a positive even number")

    per_class = total // 2
    bonafide = [item for item in records if item.get("label") == "bonafide"]
    spoof = [item for item in records if item.get("label") == "spoof"]
    if len(bonafide) < per_class or len(spoof) < per_class:
        raise ValueError(
            f"Cannot draw {per_class} samples per class: "
            f"bonafide={len(bonafide)}, spoof={len(spoof)}"
        )

    attack_pools: dict[str, List[Dict[str, Any]]] = {}
    for item in spoof:
        attack_id = str(item.get("attack_id", "unknown"))
        attack_pools.setdefault(attack_id, []).append(item)

    attack_ids = sorted(attack_pools)
    attack_weights = [len(attack_pools[a]) / len(spoof) for a in attack_ids]
    attack_counts = largest_remainder_counts(attack_weights, per_class)

    rng = random.Random(seed)
    selected = rng.sample(bonafide, per_class)
    for attack_id, count in zip(attack_ids, attack_counts):
        if count:
            selected.extend(rng.sample(attack_pools[attack_id], count))
    rng.shuffle(selected)

    actual_bona = sum(item.get("label") == "bonafide" for item in selected)
    actual_spoof = sum(item.get("label") == "spoof" for item in selected)
    if actual_bona != per_class or actual_spoof != per_class:
        raise AssertionError(
            f"Sampling failed: bonafide={actual_bona}, spoof={actual_spoof}"
        )

    composition = {
        attack_id: sum(
            item.get("label") == "spoof" and str(item.get("attack_id")) == attack_id
            for item in selected
        )
        for attack_id in attack_ids
    }
    print(
        f"Smoke subset: total={len(selected)}, bonafide={actual_bona}, "
        f"spoof={actual_spoof}, spoof composition={composition}"
    )
    return selected


def resolve_audio_path(item_path: str, audio_root: Path) -> Path:
    path = Path(item_path)
    if path.is_absolute():
        return path
    return audio_root / path


def resolve_adapter_dir(path: Path) -> Path:
    """Accept either an adapter directory or a parent containing one adapter."""
    if (path / "adapter_config.json").is_file():
        return path
    candidates = sorted(path.glob("**/adapter_config.json"))
    if len(candidates) == 1:
        return candidates[0].parent
    if not candidates:
        raise FileNotFoundError(f"No adapter_config.json found under: {path}")
    raise ValueError(
        f"Multiple LoRA adapters found under {path}; pass one exact adapter directory"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/MOSS-Audio-8B-Instruct")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Optional PEFT/LoRA adapter used while generating hints.",
    )
    parser.add_argument(
        "--prompt_key",
        default="audio-profile-v3",
        choices=list(MAKE_AUDIO_DATA_TEMPLATES_BASE),
    )
    parser.add_argument(
        "--dataset_in",
        default="data/asvspoof19_evolving/D0_manifest.jsonl",
    )
    parser.add_argument(
        "--dataset_out",
        default="data/SE-ADD/train/moss/D0_step1.jsonl",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Balanced environment subset size; e.g. 64 means 32 bonafide + 32 spoof.",
    )
    parser.add_argument("--n", type=int, default=-1, help="Process this many records after sampling")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--k", type=int, default=5, help="Completions per audio")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audio_root", type=Path, default=Path("data"))
    parser.add_argument(
        "--preserve-order",
        action="store_true",
        help="Keep manifest order when --sample-size is omitted (for fixed eval shards).",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if torch.cuda.is_available():
        device_map = "cuda:0"
    elif torch.backends.mps.is_available():
        device_map = "mps"
    else:
        device_map = "cpu"

    model = MossAudioModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype="auto",
        device_map=device_map,
    )
    adapter_dir = None
    if args.adapter is not None:
        adapter_dir = resolve_adapter_dir(args.adapter)
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        print(f"Loaded LoRA adapter: {adapter_dir}")
    model.eval()
    processor = MossAudioProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        enable_time_marker=True,
    )

    raw = load_jsonl(args.dataset_in)
    if args.sample_size is not None:
        selected = balanced_environment_sample(raw, args.sample_size, args.seed)
    else:
        selected = list(raw)
        if not args.preserve_order:
            random.Random(args.seed).shuffle(selected)

    start = args.start
    end = None if args.n <= 0 else start + args.n
    subset = selected[start:end]
    print(
        f"Loaded {len(raw)} records; selected {len(selected)}; "
        f"processing selected records {start}:{end} ({len(subset)} records)"
    )

    out_path = Path(args.dataset_out)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {out_path}; pass --overwrite explicitly")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = MAKE_AUDIO_DATA_TEMPLATES_BASE[args.prompt_key]["text"]

    # Write each completed record immediately so a long run leaves usable output.
    with out_path.open("w", encoding="utf-8") as output_handle:
        for record_index, item in enumerate(subset, start=1):
            audio_path = resolve_audio_path(item["path"], args.audio_root)
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

            audio = load_audio(str(audio_path), sample_rate=processor.config.mel_sr)
            inputs = processor(text=prompt, audios=[audio], return_tensors="pt")
            inputs = inputs.to(model.device)
            if inputs.get("audio_data") is not None:
                inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
            inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id

            responses = []
            source_index = start + record_index - 1
            for completion_index in range(args.k):
                # Paired deterministic sampling: the same audio/completion uses
                # the same random seed for Frozen, Random-2, and Top-2 models.
                generation_seed = args.seed + source_index * args.k + completion_index
                torch.manual_seed(generation_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(generation_seed)
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        do_sample=True,
                        num_beams=1,
                        use_cache=True,
                    )
                input_len = inputs["input_ids"].shape[1]
                responses.append(
                    processor.decode(
                        generated_ids[0, input_len:],
                        skip_special_tokens=True,
                    )
                )

            new_item = dict(item)
            new_item["hints"] = responses
            new_item["generation_adapter"] = (
                str(adapter_dir) if adapter_dir is not None else None
            )
            new_item["generation_seed"] = args.seed
            output_handle.write(json.dumps(new_item, ensure_ascii=False) + "\n")
            output_handle.flush()
            print(f"Completed {record_index}/{len(subset)}: {item.get('utterance_id', audio_path.name)}")

    print(f"Saved -> {out_path} ({len(subset)} records)")


if __name__ == "__main__":
    main()
