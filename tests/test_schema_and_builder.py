from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    hints = ROOT / "examples/outputs/tiny_hints.jsonl"
    scores = ROOT / "examples/outputs/tiny_scores.json"
    rows = read_jsonl(hints)
    assert len(rows) == 2
    assert all(len(row["hints"]) == 3 for row in rows)
    with tempfile.TemporaryDirectory(prefix="seadd-smoke-") as work_dir:
        work = Path(work_dir)
        sft = work / "tiny_sft.jsonl"
        audit = work / "tiny_audit.json"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "seadd/build_mistake_driven_sft.py"),
                "--hints-in", str(hints),
                "--scores-in", str(scores),
                "--sft-out", str(sft),
                "--audit-out", str(audit),
                "--seed", "2026",
                "--balance-verdict-labels",
            ],
            check=True,
        )
        output = read_jsonl(sft)
        report = json.loads(audit.read_text(encoding="utf-8"))
    assert output
    assert report["audio_count"] == 2
    assert report["target_counts_after_balance"]["real"] == report["target_counts_after_balance"]["fake"]
    assert all(len(row["conversation"]) == 3 for row in output)
    print(f"PASS: schema valid; builder created {len(output)} balanced SFT rows")


if __name__ == "__main__":
    main()
