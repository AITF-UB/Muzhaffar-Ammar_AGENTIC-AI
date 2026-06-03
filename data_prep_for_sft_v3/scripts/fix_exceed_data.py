"""
Auto-fix exceed generated files.

Rules:
- pilgan    -> must be exactly 10
- pretest   -> must be exactly 10
- flashcard -> must be exactly 5
- essay     -> must be exactly 5
- mindmap   -> ignored

SPECIAL RULE FOR PRETEST:
Distribution must follow:
- LOTS = 4
- MOTS = 3
- HOTS = 3

Behavior:
- If exceed -> trim extra items
- For pretest:
    - trim while preserving 4/3/3 composition
    - remove extra questions from overflowing level groups
- Original file overwritten
- Logs saved into txt file

Run:
    python scripts/fix_exceed_files.py

Optional:
    python scripts/fix_exceed_files.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = ROOT / "output" / "current_experiments"

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / (
    f"fix_exceed_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

EXPECTED_COUNTS = {
    "pilgan": 10,
    "pretest": 10,
    "flashcard": 5,
    "essay": 5,
}

PRETEST_COMPOSITION = {
    "LOTS": 4,
    "MOTS": 3,
    "HOTS": 3,
}


@dataclass
class FixResult:
    path: Path
    task_type: str
    before: int
    after: int
    changed: bool
    note: str


def log(message: str) -> None:
    print(message)

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def iter_json_files(root: Path):
    return sorted(root.rglob("*.json"))


def infer_task_type(data: dict[str, Any], path: Path) -> str | None:
    meta = data.get("meta")

    if isinstance(meta, dict):
        task_type = meta.get("task_type")

        if isinstance(task_type, str):
            return task_type.strip().lower()

    parts = {p.lower() for p in path.parts}

    for t in EXPECTED_COUNTS.keys():
        if t in parts:
            return t

    return None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def normalize_level(level: str | None) -> str:
    if not level:
        return "UNKNOWN"

    level = level.upper().strip()

    if "LOTS" in level:
        return "LOTS"

    if "MOTS" in level:
        return "MOTS"

    if "HOTS" in level:
        return "HOTS"

    return level


def trim_normal_task(
    assistant: list[Any],
    expected: int,
) -> tuple[list[Any], str]:

    trimmed = assistant[:expected]

    removed = len(assistant) - len(trimmed)

    return trimmed, f"removed {removed} exceeding items"


def trim_pretest(
    assistant: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in assistant:
        level = normalize_level(item.get("level"))
        grouped[level].append(item)

    final_items: list[dict[str, Any]] = []

    notes: list[str] = []

    for level, expected_count in PRETEST_COMPOSITION.items():

        current_items = grouped.get(level, [])

        actual_count = len(current_items)

        if actual_count >= expected_count:

            kept = current_items[:expected_count]

            removed = actual_count - expected_count

            if removed > 0:
                notes.append(
                    f"{level}: removed {removed}"
                )

        else:
            kept = current_items

            notes.append(
                f"{level}: insufficient ({actual_count}/{expected_count})"
            )

        final_items.extend(kept)

    total = len(final_items)

    notes.append(f"final_count={total}")

    return final_items, " | ".join(notes)


def process_file(
    path: Path,
    dry_run: bool = False,
) -> FixResult | None:

    try:
        data = load_json(path)

    except Exception as e:
        log(f"[INVALID JSON] {path} :: {e}")
        return None

    if not isinstance(data, dict):
        return None

    assistant = data.get("assistant")

    if not isinstance(assistant, list):
        return None

    task_type = infer_task_type(data, path)

    if task_type not in EXPECTED_COUNTS:
        return None

    expected = EXPECTED_COUNTS[task_type]

    current_count = len(assistant)

    if current_count <= expected:
        return None

    # =====================================================
    # PRETEST SPECIAL HANDLING
    # =====================================================

    if task_type == "pretest":

        fixed_items, note = trim_pretest(assistant)

    else:

        fixed_items, note = trim_normal_task(
            assistant,
            expected,
        )

    new_count = len(fixed_items)

    data["assistant"] = fixed_items

    if not dry_run:
        save_json(path, data)

    return FixResult(
        path=path,
        task_type=task_type,
        before=current_count,
        after=new_count,
        changed=True,
        note=note,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=OUTPUT_DIR,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    log("=" * 80)
    log("FIX EXCEED FILES")
    log("=" * 80)

    log(f"Root      : {args.root}")
    log(f"Dry Run   : {args.dry_run}")
    log(f"Log File  : {LOG_FILE}")
    log("")

    files = list(iter_json_files(args.root))

    total_scanned = 0
    total_fixed = 0

    for path in files:

        total_scanned += 1

        result = process_file(
            path=path,
            dry_run=args.dry_run,
        )

        if result is None:
            continue

        total_fixed += 1

        rel_path = path.relative_to(ROOT)

        log(
            f"[FIXED] "
            f"[{result.task_type.upper()}] "
            f"{rel_path}"
        )

        log(
            f"  before : {result.before}"
        )

        log(
            f"  after  : {result.after}"
        )

        log(
            f"  note   : {result.note}"
        )

        log("")

    log("=" * 80)
    log("SUMMARY")
    log("=" * 80)

    log(f"Scanned Files : {total_scanned}")
    log(f"Fixed Files   : {total_fixed}")
    log(f"Log Saved     : {LOG_FILE}")

    print()

    if args.dry_run:
        log("DRY RUN ENABLED -> no files modified")

    else:
        log("Files successfully updated")


if __name__ == "__main__":
    main()