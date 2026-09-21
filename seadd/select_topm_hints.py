"""Select self-generated hints with outcome- or process-based strategies."""

import argparse
import json
import random
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List


CUE_LINE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*)?Cue\s*\d+(?:\*\*)?\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
LIST_ITEM_RE = re.compile(
    r"(?:^|\n|(?<=\s))\s*(?:[-*]\s+|\d{1,2}[.)]\s+)"
    r"(.+?)"
    r"(?=(?:\n\s*[-*]\s+)|(?:\s+\d{1,2}[.)]\s+)|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
VISUAL_ONLY_PATTERNS = {
    "lip-sync": re.compile(r"\blip[- ]?sync\b", re.I),
    "lip/mouth movement": re.compile(r"\b(?:lip|mouth) movements?\b", re.I),
    "facial evidence": re.compile(r"\b(?:facial expressions?|face movements?)\b", re.I),
    "eye/blink evidence": re.compile(r"\b(?:eye movements?|blink(?:ing)?)\b", re.I),
    "visual appearance": re.compile(
        r"\b(?:skin texture|lighting|video frames?|visual artifacts?|pixels?|image artifacts?)\b",
        re.I,
    ),
}
HARD_VERDICT_LEAKAGE_PATTERNS = {
    "real/fake verdict": re.compile(r"\b(?:real|fake)\b", re.I),
    "deepfake verdict": re.compile(r"\bdeepfakes?\b", re.I),
    "authenticity verdict": re.compile(r"\b(?:authentic|authenticity|genuine)\b", re.I),
}
SOFT_DIRECTIONAL_PATTERNS = {
    "directional characterization": re.compile(
        r"\b(?:natural|unnatural|synthetic|synthesi[sz]ed|artificial|robotic|"
        r"human[- ]like|machine[- ]generated|digital generation|suspicious|unusual|"
        r"perfect(?:ly)?|flawless)\b",
        re.I,
    ),
    "directional inference": re.compile(
        r"\b(?:indicat(?:e|es|ed|ing)|suggest(?:s|ed|ing)?|"
        r"impl(?:y|ies|ied|ying)|a sign of|evidence of|typical of|"
        r"unlikely for|likely to be|possible digital generation)\b",
        re.I,
    ),
}
IRRELEVANT_PATTERNS = {
    "gender": re.compile(r"\b(?:gender|male|female|man|woman)\b", re.I),
    "age": re.compile(r"\b(?:speaker'?s age|young adult|middle[- ]aged|elderly)\b", re.I),
    "accent": re.compile(r"\b(?:accent|American English|British English|Indian English)\b", re.I),
    "linguistic content": re.compile(
        r"\b(?:sentence content|semantic content|what (?:is|was) said|topic of (?:the )?speech)\b",
        re.I,
    ),
    "duration/refusal": re.compile(
        r"\b(?:audio|clip|utterance) (?:is|was) too short\b|"
        r"\b(?:only|contains only) (?:one|a single) sentence\b|"
        r"\bsingle utterance\b|"
        r"\bno (?:speech|voice|pitch|prosodic features?) (?:detected|present|available)\b|"
        r"\bno (?:pitch|prosodic) features? are present\b",
        re.I,
    ),
}

# These lexicons intentionally describe observable audio properties rather than
# class-directional words such as "natural", "synthetic", "real", or "fake".
AUDIO_DIMENSION_PATTERNS = {
    "timing": re.compile(
        r"\b(?:timing|rhythm|pace|rate|tempo|pause|pauses|duration|syllable|"
        r"cadence|hesitation|stress)\w*\b",
        re.I,
    ),
    "voice": re.compile(
        r"\b(?:pitch|intonation|prosody|tone|timbre|voice texture|vocal|"
        r"frequency|harmonic|formant|energy)\w*\b",
        re.I,
    ),
    "articulation": re.compile(
        r"\b(?:articulation|pronunciation|phoneme|consonant|vowel|transition|"
        r"continuity|coarticulation|breath|voicing)\w*\b",
        re.I,
    ),
    "recording": re.compile(
        r"\b(?:recording|background|noise|channel|reverberation|echo|codec|"
        r"compression|bandwidth|clipping|distortion|artifact|signal)\w*\b",
        re.I,
    ),
}
OBSERVATION_PATTERNS = re.compile(
    r"\b(?:stable|variable|consistent|inconsistent|smooth|abrupt|regular|"
    r"irregular|fast|slow|short|long|high|low|rising|falling|flat|clear|"
    r"muffled|sharp|soft|strong|weak|present|absent|audible|prominent|"
    r"limited|frequent|occasional|steady|changing|continuous|discontinuous)\w*\b",
    re.I,
)
GENERIC_INSTRUCTION_PATTERNS = re.compile(
    r"\b(?:analy[sz]e|check|examine|inspect|look for|listen for|determine|"
    r"assess|evaluate|consider)\b",
    re.I,
)
GENERIC_CLASS_TEMPLATE_PATTERNS = re.compile(
    r"\b(?:deepfakes?|synthetic (?:speech|voices?|audio)|human (?:speech|voices?))\b"
    r".{0,35}\b(?:can|could|may|might|often|typically|sometimes|tend to)\b|"
    r"\b(?:can|could|may|might|often|typically|sometimes|tend to)\b"
    r".{0,35}\b(?:deepfakes?|synthetic (?:speech|voices?|audio))\b",
    re.I,
)
CONTENT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "cue", "for", "from",
    "has", "have", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "were", "with", "audio", "speaker", "speech", "voice",
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
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


def load_score_log(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if data.get("verdict_mode") != "logits":
        raise ValueError(
            f"Expected a logits result, got verdict_mode={data.get('verdict_mode')!r}"
        )
    if not isinstance(data.get("audio"), list):
        raise ValueError("Score log does not contain an 'audio' result list")
    return data


def source_hints(row: Dict[str, Any]) -> List[str]:
    hints = row.get("hints")
    if not isinstance(hints, list) or not hints:
        raise ValueError("Each source record must contain a non-empty hints list")
    return hints


def extract_cues(text: str) -> List[str]:
    """Extract cue units without requiring one particular generation format."""
    cues = [match.strip() for match in CUE_LINE_RE.findall(text)]
    if cues:
        return cues

    cues = [re.sub(r"\s+", " ", match).strip() for match in LIST_ITEM_RE.findall(text)]
    cues = [cue for cue in cues if cue]
    if cues:
        return cues

    # Last-resort support for paragraph-style legacy generations. Remove the
    # common task preamble before splitting so mentioning "deepfake" in the
    # instruction restatement does not invalidate otherwise useful observations.
    cleaned = re.sub(
        r"^.*?(?:evidences?|clues?|indicators?)\s*(?:include|are|:)\s*",
        "",
        text.strip(),
        count=1,
        flags=re.I | re.S,
    )
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
        if len(sentence.split()) >= 4
    ]


def observation_core(cue: str) -> str:
    """Remove an explicit class inference tail while preserving its observation."""
    separators = re.compile(
        r"\s*(?:,|;)?\s+(?="
        r"(?:which|that)\s+(?:can|could|may|might|is|are|would)|"
        r"(?:indicating|suggesting|implying)|"
        r"(?:and\s+)?(?:may|might|could)\s+indicate"
        r")",
        re.I,
    )
    parts = separators.split(cue, maxsplit=1)
    if len(parts) == 2:
        tail = parts[1]
        if any(pattern.search(tail) for pattern in SOFT_DIRECTIONAL_PATTERNS.values()) or any(
            pattern.search(tail) for pattern in HARD_VERDICT_LEAKAGE_PATTERNS.values()
        ) or re.search(r"\b(?:human|synthetic|robotic|typical|manipulat)\w*\b", tail, re.I):
            return parts[0].strip(" ,;:-")
    return cue.strip()


def validate_alignment(
    source_row: Dict[str, Any],
    score_row: Dict[str, Any],
    audio_index: int,
) -> None:
    if score_row.get("audio_index") != audio_index:
        raise ValueError(
            f"Audio index mismatch at row {audio_index}: "
            f"score audio_index={score_row.get('audio_index')}"
        )
    source = source_hints(source_row)
    scored = score_row.get("hints")
    if not isinstance(scored, list) or len(source) != len(scored):
        raise ValueError(
            f"Hint count mismatch at audio {audio_index}: "
            f"source={len(source)}, scored={len(scored) if isinstance(scored, list) else None}"
        )
    for hint_index, (source_text, score_entry) in enumerate(zip(source, scored)):
        if score_entry.get("index") != hint_index:
            raise ValueError(f"Hint index mismatch at audio {audio_index}, hint {hint_index}")
        if source_text.strip() != str(score_entry.get("text", "")).strip():
            raise ValueError(f"Hint text mismatch at audio {audio_index}, hint {hint_index}")
        probability = score_entry.get("correct_label_probability")
        if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
            raise ValueError(
                f"Invalid correct_label_probability at audio {audio_index}, hint {hint_index}"
            )


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def content_token_set(text: str) -> set[str]:
    return token_set(text) - CONTENT_STOPWORDS


def jaccard_similarity(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def content_jaccard_similarity(left: str, right: str) -> float:
    left_tokens = content_token_set(left)
    right_tokens = content_token_set(right)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_process_quality(
    candidate: Dict[str, Any],
    valid_candidates: List[Dict[str, Any]],
    max_words: int,
    weights: Dict[str, float],
) -> Dict[str, Any]:
    """Compute transparent, label-free proxies for process quality.

    The score does not claim audio-text grounding. It only ranks candidates by
    observable textual process properties after the hard validity gates.
    """
    text = candidate["text"]
    cues = extract_cues(text)
    dimension_hits = {
        name: sum(bool(pattern.search(cue)) for cue in cues)
        for name, pattern in AUDIO_DIMENSION_PATTERNS.items()
    }
    covered_dimensions = [name for name, count in dimension_hits.items() if count]
    coverage = len(covered_dimensions) / len(AUDIO_DIMENSION_PATTERNS)

    specific_cues = sum(
        any(pattern.search(cue) for pattern in AUDIO_DIMENSION_PATTERNS.values())
        and bool(OBSERVATION_PATTERNS.search(cue))
        and not bool(GENERIC_INSTRUCTION_PATTERNS.search(cue))
        for cue in cues
    )
    specificity = specific_cues / len(cues) if cues else 0.0

    siblings = [other for other in valid_candidates if other is not candidate]
    sibling_cues = [
        sibling_cue
        for sibling in siblings
        for sibling_cue in extract_cues(sibling["text"])
    ]
    # Reward a cue only when another independently sampled candidate describes
    # overlapping content. Exact template words are downweighted by stopwords.
    cue_support = [
        max(
            (content_jaccard_similarity(cue, sibling_cue) for sibling_cue in sibling_cues),
            default=0.0,
        )
        for cue in cues
    ]
    consensus = sum(cue_support) / len(cue_support) if cue_support else 0.0

    word_count = candidate["process"]["word_count"]
    lower_target = 20
    upper_target = min(max_words, 80)
    if word_count < lower_target:
        conciseness = clamp01(word_count / lower_target)
    elif word_count <= upper_target:
        conciseness = 1.0
    else:
        conciseness = clamp01(
            1.0 - (word_count - upper_target) / max(1, max_words - upper_target)
        )

    redundancy = candidate["process"]["max_intra_cue_similarity"]
    non_redundancy = 1.0 - redundancy
    components = {
        "specificity": specificity,
        "coverage": coverage,
        "consensus": consensus,
        "conciseness": conciseness,
        "non_redundancy": non_redundancy,
        "neutrality": candidate["process"]["neutrality"],
    }
    weight_sum = sum(weights.values())
    total = sum(weights[name] * components[name] for name in components) / weight_sum
    return {
        "total": total,
        "components": components,
        "covered_dimensions": covered_dimensions,
        "dimension_hits": dimension_hits,
        "specific_cue_count": specific_cues,
        "cue_support": cue_support,
        "note": "Text-process proxy only; does not verify audio-cue grounding.",
    }


def process_validation(
    text: str,
    min_cues: int,
    max_cues: int,
    max_words: int,
    intra_cue_similarity: float,
) -> Dict[str, Any]:
    extracted_cues = extract_cues(text)
    rejected_cues: List[Dict[str, Any]] = []
    accepted_cues: List[str] = []
    rejection_reason_counts = Counter()

    for cue_index, raw_cue in enumerate(extracted_cues):
        cue = observation_core(raw_cue)
        cue_reasons: List[str] = []
        modality_hits = [
            name for name, pattern in VISUAL_ONLY_PATTERNS.items() if pattern.search(cue)
        ]
        verdict_hits = [
            name for name, pattern in HARD_VERDICT_LEAKAGE_PATTERNS.items()
            if pattern.search(cue)
        ]
        relevance_hits = [
            name for name, pattern in IRRELEVANT_PATTERNS.items() if pattern.search(cue)
        ]
        directional_hits = [
            name for name, pattern in SOFT_DIRECTIONAL_PATTERNS.items()
            if pattern.search(cue)
        ]
        generic_instruction = bool(GENERIC_INSTRUCTION_PATTERNS.search(cue))
        generic_class_template = bool(GENERIC_CLASS_TEMPLATE_PATTERNS.search(cue))
        if modality_hits:
            cue_reasons.append("audio_modality")
        if verdict_hits:
            cue_reasons.append("verdict_leakage")
        if relevance_hits:
            cue_reasons.append("forensic_relevance")
        if directional_hits:
            cue_reasons.append("directional_claim")
        if generic_instruction or generic_class_template:
            cue_reasons.append("generic_template")

        if cue_reasons:
            rejection_reason_counts.update(cue_reasons)
            rejected_cues.append(
                {
                    "index": cue_index,
                    "text": raw_cue,
                    "observation_core": cue,
                    "reasons": cue_reasons,
                    "modality_hits": modality_hits,
                    "verdict_hits": verdict_hits,
                    "relevance_hits": relevance_hits,
                    "directional_hits": directional_hits,
                    "generic_instruction": generic_instruction,
                    "generic_class_template": generic_class_template,
                }
            )
            continue

        if any(
            jaccard_similarity(cue, previous) > intra_cue_similarity
            for previous in accepted_cues
        ):
            rejection_reason_counts["cue_redundancy"] += 1
            rejected_cues.append(
                {"index": cue_index, "text": cue, "reasons": ["cue_redundancy"]}
            )
            continue
        accepted_cues.append(cue)

    # Keep at most max_cues after cue-level cleaning. Extra valid cues are not
    # errors; the cap simply keeps the reconstructed supervision manageable.
    if len(accepted_cues) > max_cues:
        for cue in accepted_cues[max_cues:]:
            rejected_cues.append(
                {"index": None, "text": cue, "reasons": ["cue_cap"]}
            )
            rejection_reason_counts["cue_cap"] += 1
        accepted_cues = accepted_cues[:max_cues]

    cue_text = "\n".join(accepted_cues)
    word_count = len(cue_text.split())
    directional_hits = [
        name for name, pattern in SOFT_DIRECTIONAL_PATTERNS.items()
        if pattern.search(cue_text)
    ]
    neutrality = 1.0 - min(1.0, len(directional_hits) / max(1, len(accepted_cues)))
    max_intra_similarity = max(
        (
            jaccard_similarity(left, right)
            for left, right in combinations(accepted_cues, 2)
        ),
        default=0.0,
    )
    reasons: List[str] = []
    if len(accepted_cues) < min_cues:
        reasons.append("insufficient_valid_cues")
        rejection_reason_counts["insufficient_valid_cues"] += 1

    sanitized_text = "\n".join(
        f"Cue {index}: {cue}" for index, cue in enumerate(accepted_cues, start=1)
    )

    return {
        "valid": not reasons,
        "rejection_reasons": reasons,
        "rejection_reason_counts": dict(rejection_reason_counts),
        "extracted_cue_count": len(extracted_cues),
        "cue_count": len(accepted_cues),
        "accepted_cues": accepted_cues,
        "rejected_cues": rejected_cues,
        "sanitized_text": sanitized_text,
        "word_count": word_count,
        "directional_hits": directional_hits,
        "neutrality": neutrality,
        "max_intra_cue_similarity": max_intra_similarity,
        "over_max_words": word_count > max_words,
    }


def choose_candidates(
    candidates: List[Dict[str, Any]],
    strategy: str,
    top_m: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if strategy == "top-m-correct":
        return sorted(
            candidates,
            key=lambda candidate: (-candidate["correct_label_probability"], candidate["index"]),
        )[:top_m]
    if strategy == "random-m-correct":
        if len(candidates) <= top_m:
            return sorted(candidates, key=lambda candidate: candidate["index"])
        return sorted(rng.sample(candidates, top_m), key=lambda candidate: candidate["index"])
    if strategy in {"process-aware", "process-only", "quality-aware"}:
        if strategy == "process-aware":
            ranked = sorted(
                candidates,
                key=lambda candidate: (
                    -candidate["effectiveness_gain"],
                    candidate["index"],
                ),
            )
        elif strategy == "process-only":
            # A seeded shuffle avoids reintroducing the confidence-selection
            # collapse while keeping the control reproducible.
            ranked = list(candidates)
            rng.shuffle(ranked)
        else:
            ranked = sorted(
                candidates,
                key=lambda candidate: (-candidate["quality"]["total"], candidate["index"]),
            )
        chosen: List[Dict[str, Any]] = []
        for candidate in ranked:
            if all(
                jaccard_similarity(candidate["text"], previous["text"])
                <= candidate["inter_candidate_similarity"]
                for previous in chosen
            ):
                chosen.append(candidate)
            if len(chosen) == top_m:
                break
        return sorted(chosen, key=lambda candidate: candidate["index"])
    raise ValueError(f"Unsupported strategy: {strategy}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-in", required=True, type=Path)
    parser.add_argument(
        "--scores-in",
        type=Path,
        default=None,
        help="Required for outcome-based strategies; omitted for process-only/quality-aware.",
    )
    parser.add_argument("--dataset-out", required=True, type=Path)
    parser.add_argument(
        "--strategy",
        choices=(
            "top-m-correct",
            "random-m-correct",
            "process-aware",
            "process-only",
            "quality-aware",
        ),
        default="top-m-correct",
    )
    parser.add_argument("--top-m", type=int, default=2)
    parser.add_argument("--correctness-gate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--effectiveness-gate", type=float, default=0.0)
    parser.add_argument("--min-cues", type=int, default=2)
    parser.add_argument("--max-cues", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=120)
    parser.add_argument("--intra-cue-similarity", type=float, default=0.8)
    parser.add_argument("--inter-candidate-similarity", type=float, default=0.8)
    parser.add_argument("--quality-specificity-weight", type=float, default=1.0)
    parser.add_argument("--quality-coverage-weight", type=float, default=1.0)
    parser.add_argument("--quality-consensus-weight", type=float, default=1.0)
    parser.add_argument("--quality-conciseness-weight", type=float, default=0.5)
    parser.add_argument("--quality-non-redundancy-weight", type=float, default=0.5)
    parser.add_argument("--quality-neutrality-weight", type=float, default=0.5)
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep records with no accepted hint. By default they are omitted.",
    )
    args = parser.parse_args()

    if args.top_m <= 0:
        raise ValueError("--top-m must be positive")
    if not 0 <= args.correctness_gate < 1:
        raise ValueError("--correctness-gate must be in [0, 1)")
    if args.min_cues <= 0 or args.max_cues < args.min_cues:
        raise ValueError("Require 0 < --min-cues <= --max-cues")
    if args.max_words <= 0:
        raise ValueError("--max-words must be positive")
    if not 0 <= args.intra_cue_similarity <= 1:
        raise ValueError("--intra-cue-similarity must be in [0, 1]")
    if not 0 <= args.inter_candidate_similarity <= 1:
        raise ValueError("--inter-candidate-similarity must be in [0, 1]")
    quality_weights = {
        "specificity": args.quality_specificity_weight,
        "coverage": args.quality_coverage_weight,
        "consensus": args.quality_consensus_weight,
        "conciseness": args.quality_conciseness_weight,
        "non_redundancy": args.quality_non_redundancy_weight,
        "neutrality": args.quality_neutrality_weight,
    }
    if any(weight < 0 for weight in quality_weights.values()):
        raise ValueError("Quality weights must be non-negative")
    if sum(quality_weights.values()) <= 0:
        raise ValueError("At least one quality weight must be positive")

    source_rows = load_jsonl(args.dataset_in)
    if args.strategy in {"process-only", "quality-aware"}:
        score_rows: List[Dict[str, Any] | None] = [None] * len(source_rows)
    else:
        if args.scores_in is None:
            raise ValueError(f"--scores-in is required for strategy {args.strategy}")
        score_log = load_score_log(args.scores_in)
        score_rows = score_log["audio"]
        if len(source_rows) != len(score_rows):
            raise ValueError(
                f"Audio count mismatch: source={len(source_rows)}, scores={len(score_rows)}"
            )

    rng = random.Random(args.seed)
    output_rows = []
    selected_count_distribution = Counter()
    selected_probabilities = []
    selected_effectiveness_gains = []
    dropped_audio_indices = []
    rejection_counts = Counter()
    selected_quality_scores = []

    for audio_index, (source_row, score_row) in enumerate(zip(source_rows, score_rows)):
        if score_row is not None:
            validate_alignment(source_row, score_row, audio_index)
        candidates = []
        score_entries = (
            score_row["hints"]
            if score_row is not None
            else [
                {"index": index, "text": text}
                for index, text in enumerate(source_hints(source_row))
            ]
        )
        for score_entry in score_entries:
            probability = score_entry.get("correct_label_probability")
            candidate = {
                "index": int(score_entry["index"]),
                "text": score_entry["text"],
            }
            if probability is not None:
                probability = float(probability)
                candidate.update(
                    {
                        "correct_label_probability": probability,
                        "spoof_score": float(score_entry["spoof_score"]),
                        "p_fake": float(score_entry["p_fake"]),
                    }
                )
                if probability <= args.correctness_gate:
                    rejection_counts["correctness_gate"] += 1
                    continue

            if args.strategy in {"process-aware", "process-only", "quality-aware"}:
                candidate["process"] = process_validation(
                    candidate["text"],
                    args.min_cues,
                    args.max_cues,
                    args.max_words,
                    args.intra_cue_similarity,
                )
                candidate["inter_candidate_similarity"] = args.inter_candidate_similarity
                rejection_counts.update(
                    candidate["process"]["rejection_reason_counts"]
                )
                if not candidate["process"]["valid"]:
                    continue
                candidate["original_text"] = candidate["text"]
                candidate["text"] = candidate["process"]["sanitized_text"]
                if args.strategy == "process-aware":
                    gain = score_entry.get("effectiveness_gain")
                    if not isinstance(gain, (int, float)):
                        raise ValueError(
                            "process-aware selection requires effectiveness_gain; "
                            "rerun audio_ttt_v2_logits.py with --score_direct_baseline"
                        )
                    candidate["effectiveness_gain"] = float(gain)
                    if candidate["effectiveness_gain"] <= args.effectiveness_gate:
                        rejection_counts["effectiveness_gate"] += 1
                        continue
            candidates.append(candidate)

        if args.strategy == "quality-aware":
            for candidate in candidates:
                candidate["quality"] = score_process_quality(
                    candidate,
                    candidates,
                    args.max_words,
                    quality_weights,
                )

        chosen = choose_candidates(candidates, args.strategy, args.top_m, rng)
        selected_count_distribution[len(chosen)] += 1
        if not chosen:
            dropped_audio_indices.append(audio_index)
            if not args.keep_empty:
                continue

        chosen = sorted(chosen, key=lambda candidate: candidate["index"])
        output_row = dict(source_row)
        output_row["hints"] = [candidate["text"] for candidate in chosen]
        output_row["hint_selection"] = {
            "strategy": args.strategy,
            "correctness_gate": args.correctness_gate,
            "top_m": args.top_m,
            "source_hint_count": len(source_hints(source_row)),
            "selected": chosen,
        }
        output_rows.append(output_row)
        selected_probabilities.extend(
            candidate["correct_label_probability"]
            for candidate in chosen
            if "correct_label_probability" in candidate
        )
        selected_effectiveness_gains.extend(
            candidate["effectiveness_gain"]
            for candidate in chosen
            if "effectiveness_gain" in candidate
        )
        selected_quality_scores.extend(
            candidate["quality"]["total"]
            for candidate in chosen
            if "quality" in candidate
        )

    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
    with args.dataset_out.open("w", encoding="utf-8") as output_handle:
        for row in output_rows:
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    selected_hints = sum(selected_count_distribution.values())
    # The Counter values count audios; compute hints separately.
    selected_hints = sum(
        selected_count * audio_count
        for selected_count, audio_count in selected_count_distribution.items()
    )
    mean_probability = (
        sum(selected_probabilities) / len(selected_probabilities)
        if selected_probabilities else None
    )
    print(f"Strategy: {args.strategy}")
    print(f"Input audios: {len(source_rows)}")
    print(f"Output audios: {len(output_rows)}")
    print(f"Selected hints: {selected_hints}")
    print(f"Selected-count distribution: {dict(sorted(selected_count_distribution.items()))}")
    if mean_probability is None:
        print("Mean selected correct-label probability: N/A (label-free strategy)")
    else:
        print(f"Mean selected correct-label probability: {mean_probability:.4f}")
    if selected_effectiveness_gains:
        print(
            "Mean selected effectiveness gain: "
            f"{sum(selected_effectiveness_gains) / len(selected_effectiveness_gains):.4f}"
        )
    if selected_quality_scores:
        print(
            "Mean selected process-quality score: "
            f"{sum(selected_quality_scores) / len(selected_quality_scores):.4f}"
        )
        print(f"Quality weights: {quality_weights}")
    print(f"Cue filtering counts: {dict(sorted(rejection_counts.items()))}")
    print(f"Dropped audio indices: {dropped_audio_indices}")
    print(f"Saved -> {args.dataset_out}")


if __name__ == "__main__":
    main()
