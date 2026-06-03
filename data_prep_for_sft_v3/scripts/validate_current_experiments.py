"""
Validate generated task outputs under output/current_experiments.

This validator focuses on the constraints you mentioned from the user prompts:
- pilgan: 10 objects (soal)
- pretest: 10 objects (soal)
- flashcard: 5 objects
- essay: 5 objects
- mindmap: depth >= 3 (Root -> Child -> Grandchild)

Run:
    python scripts/validate_current_experiments.py
    python scripts/validate_current_experiments.py --root output/current_experiments

Exit codes:
- 0: all checked files pass
- 1: at least one file fails validation
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable

try:
    from rich.console import Console
    from rich.text import Text
except Exception:  # pragma: no cover
    Console = None  # type: ignore[assignment]
    Text = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parent.parent

# =========================================================
# LOGGING SETUP
# =========================================================

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / (
    f"validate_current_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)


def write_log(message: str) -> None:
    """
    Write logs to txt file.
    """
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def output(message: str, console: Console | None = None) -> None:
    """
    Unified output to terminal + txt log.
    """
    if console:
        console.print(message)
    else:
        print(message)

    write_log(message)


TASK_EXPECTED_LIST_LENGTH: dict[str, int] = {
    "pilgan": 10,
    "pretest": 10,
    "flashcard": 5,
    "essay": 5,
}

SUPPORTED_TASK_TYPES = set(TASK_EXPECTED_LIST_LENGTH.keys()) | {"mindmap"}


@dataclass(frozen=True)
class Finding:
    path: Path
    task_type: str | None
    reason: str
    meta: dict[str, Any] | None = None
    kind: str | None = None
    expected: int | None = None
    got: int | None = None


def _safe_lower(s: Any) -> str:
    return s.lower() if isinstance(s, str) else ""


def infer_task_type(data: dict[str, Any], path: Path) -> str | None:
    meta = data.get("meta")

    if isinstance(meta, dict):
        tt = meta.get("task_type")

        if isinstance(tt, str) and tt.strip():
            return tt.strip()

    return infer_task_type_from_path(path)


def infer_task_type_from_path(path: Path) -> str | None:
    parts = {_safe_lower(p) for p in path.parts}

    for candidate in SUPPORTED_TASK_TYPES:
        if candidate in parts:
            return candidate

    return None


def iter_json_files(root_dir: Path) -> Iterable[Path]:
    return sorted(root_dir.rglob("*.json"))


def _summarize_meta(meta: dict[str, Any] | None) -> str:
    if not isinstance(meta, dict):
        return ""

    def _one(key: str) -> str:
        val = meta.get(key)

        if val is None:
            return ""

        s = str(val).replace("\n", " ").strip()
        return s

    parts: list[str] = []

    mapel = _one("mata_pelajaran")
    bab = _one("bab_judul")
    sub = _one("sub_bab")
    level = _one("level")
    source = _one("source_file")

    if mapel:
        parts.append(f"mapel={mapel}")

    if bab:
        parts.append(f"bab={bab}")

    if sub:
        parts.append(f"sub_bab={sub}")

    if level and level.lower() != "none":
        parts.append(f"level={level}")

    if source:
        parts.append(f"source_file={source}")

    return " | ".join(parts)


def compute_mindmap_depth(node: Any) -> tuple[int, str | None]:
    """
    Return (depth, error).

    Depth=1 means only root node.
    """

    if not isinstance(node, dict):
        return 0, f"mindmap node must be an object, got {type(node).__name__}"

    children = node.get("children")

    if children is None:
        return 0, "mindmap node missing required key 'children'"

    if not isinstance(children, list):
        return 0, f"mindmap 'children' must be a list, got {type(children).__name__}"

    if not children:
        return 1, None

    child_depths: list[int] = []

    for idx, child in enumerate(children):
        d, err = compute_mindmap_depth(child)

        if err is not None:
            return 0, f"mindmap child[{idx}]: {err}"

        child_depths.append(d)

    return 1 + max(child_depths), None


def validate_generated_file(path: Path) -> tuple[str | None, list[Finding]]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))

    except Exception as e:
        task_type = infer_task_type_from_path(path)

        if task_type is None:
            return None, []

        return task_type, [
            Finding(
                path=path,
                task_type=task_type,
                reason=f"invalid JSON: {e}",
                meta=None,
                kind="invalid",
            )
        ]

    wrapper: dict[str, Any] | None
    assistant: Any
    meta: dict[str, Any] | None

    if isinstance(parsed, dict) and "assistant" in parsed:
        wrapper = parsed
        assistant = wrapper.get("assistant")
        meta = wrapper.get("meta") if isinstance(wrapper.get("meta"), dict) else None
        task_type = infer_task_type(wrapper, path)

    else:
        wrapper = None
        assistant = parsed
        meta = None
        task_type = infer_task_type_from_path(path)

    if task_type not in SUPPORTED_TASK_TYPES:
        return None, []

    # =========================================================
    # LIST TASKS
    # =========================================================

    if task_type in TASK_EXPECTED_LIST_LENGTH:
        expected = TASK_EXPECTED_LIST_LENGTH[task_type]

        if not isinstance(assistant, list):
            return task_type, [
                Finding(
                    path=path,
                    task_type=task_type,
                    reason=(
                        f"assistant must be a list of length {expected}, "
                        f"got {type(assistant).__name__}"
                    ),
                    meta=meta,
                    kind="invalid",
                    expected=expected,
                )
            ]

        findings: list[Finding] = []

        if len(assistant) != expected:
            kind = "exceed" if len(assistant) > expected else "missing"

            findings.append(
                Finding(
                    path=path,
                    task_type=task_type,
                    reason=f"expected {expected} items, got {len(assistant)}",
                    meta=meta,
                    kind=kind,
                    expected=expected,
                    got=len(assistant),
                )
            )

        bad_indexes = [
            i
            for i, item in enumerate(assistant)
            if not isinstance(item, dict)
        ]

        if bad_indexes:
            preview = bad_indexes[:10]
            suffix = "..." if len(bad_indexes) > 10 else ""

            findings.append(
                Finding(
                    path=path,
                    task_type=task_type,
                    reason=(
                        f"assistant contains non-object entries "
                        f"at indexes {preview}{suffix}"
                    ),
                    meta=meta,
                    kind="invalid",
                )
            )

        return task_type, findings

    # =========================================================
    # MINDMAP TASK
    # =========================================================

    if task_type == "mindmap":

        if not isinstance(assistant, dict):
            return task_type, [
                Finding(
                    path=path,
                    task_type=task_type,
                    reason=f"assistant must be an object, got {type(assistant).__name__}",
                    meta=meta,
                    kind="invalid",
                )
            ]

        node = (
            assistant.get("root")
            if isinstance(assistant.get("root"), dict)
            else assistant
        )

        depth, err = compute_mindmap_depth(node)

        if err is not None:
            return task_type, [
                Finding(
                    path=path,
                    task_type=task_type,
                    reason=err,
                    meta=meta,
                    kind="invalid",
                )
            ]

        if depth < 3:
            return task_type, [
                Finding(
                    path=path,
                    task_type=task_type,
                    reason=f"mindmap depth too shallow: depth={depth} (need >= 3)",
                    meta=meta,
                    kind="shallow",
                    expected=3,
                    got=depth,
                )
            ]

        return task_type, []

    return task_type, [
        Finding(
            path=path,
            task_type=task_type,
            reason="unsupported task type",
        )
    ]


def _status_priority(kind: str) -> int:
    return {
        "invalid": 4,
        "shallow": 3,
        "exceed": 2,
        "missing": 1,
    }.get(kind, 0)


def _pick_file_kind(findings: list[Finding]) -> str:
    best = "fail"
    best_pri = -1

    for f in findings:
        k = f.kind or "fail"
        pri = _status_priority(k)

        if pri > best_pri:
            best_pri = pri
            best = k

    return best


def _status_label_and_style(kind: str) -> tuple[str, str]:
    kind = kind.lower()

    if kind == "exceed":
        return "EXCEED", "bold red"

    if kind == "missing":
        return "MISSING", "bold yellow"

    if kind == "invalid":
        return "INVALID", "bold magenta"

    if kind == "shallow":
        return "SHALLOW", "bold red"

    if kind == "ok":
        return "OK", "bold green"

    return "FAIL", "bold red"


def _fmt_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")

    except Exception:
        return str(path).replace("\\", "/")


def main(argv: list[str] | None = None) -> int:

    parser = argparse.ArgumentParser(
        description="Validate outputs in output/current_experiments"
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "output" / "current_experiments",
        help="Directory to scan",
    )

    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(sorted(SUPPORTED_TASK_TYPES)),
        help=(
            "Comma-separated task types "
            f"(supported: {', '.join(sorted(SUPPORTED_TASK_TYPES))})"
        ),
    )

    parser.add_argument(
        "--show-ok",
        action="store_true",
        help="Also show passing files",
    )

    args = parser.parse_args(argv)

    console = Console() if Console is not None else None

    start_msg = f"Writing validation logs to: {LOG_FILE}"
    output(start_msg, console)

    scan_root: Path = args.root

    if not scan_root.is_dir():
        output(
            f"ERROR: root directory not found: {_fmt_rel(scan_root)}",
            console,
        )
        return 2

    enabled_tasks = {
        t.strip().lower()
        for t in args.tasks.split(",")
        if t.strip()
    }

    unknown = sorted(enabled_tasks - SUPPORTED_TASK_TYPES)

    if unknown:
        output(
            f"ERROR: unknown task types in --tasks: {', '.join(unknown)}",
            console,
        )
        return 2

    json_files = list(iter_json_files(scan_root))

    checked_files = 0
    skipped_files = 0

    findings: list[Finding] = []

    ok_files: list[tuple[Path, str]] = []

    per_task_checked: dict[str, int] = {
        t: 0
        for t in sorted(SUPPORTED_TASK_TYPES)
    }

    for p in json_files:

        task_type, file_findings = validate_generated_file(p)

        if task_type is None:
            skipped_files += 1
            continue

        if task_type not in enabled_tasks:
            skipped_files += 1
            continue

        checked_files += 1

        per_task_checked[task_type] = (
            per_task_checked.get(task_type, 0) + 1
        )

        findings.extend(file_findings)

        if args.show_ok and not file_findings:
            ok_files.append((p, task_type))

    by_path: dict[Path, list[Finding]] = defaultdict(list)

    for f in findings:
        by_path[f.path].append(f)

    # =========================================================
    # FAILED FILES
    # =========================================================

    if by_path:

        output("\nCACAT FILES (FAILED VALIDATION):", console)

        for p in sorted(by_path.keys(), key=_fmt_rel):

            fs = by_path[p]

            kind = _pick_file_kind(fs)

            label, style = _status_label_and_style(kind)

            task_types = sorted({
                f.task_type or "unknown"
                for f in fs
            })

            tt = "|".join(task_types)

            reasons = "; ".join(sorted({
                f.reason
                for f in fs
            }))

            meta = next(
                (
                    f.meta
                    for f in fs
                    if isinstance(f.meta, dict)
                ),
                None,
            )

            meta_str = _summarize_meta(meta)

            meta_suffix = (
                f" :: {meta_str}"
                if meta_str
                else ""
            )

            len_info = next(
                (
                    f
                    for f in fs
                    if f.expected is not None and f.got is not None
                ),
                None,
            )

            if len_info is not None:
                meta_suffix = (
                    f" (got={len_info.got}, expected={len_info.expected})"
                    + meta_suffix
                )

            plain_msg = (
                f"- [{label}] [{tt}] {_fmt_rel(p)} "
                f":: {reasons}{meta_suffix}"
            )

            if console and Text is not None:

                t = Text()

                t.append("- ")
                t.append(label, style=style)
                t.append(" ")
                t.append(tt, style="cyan")
                t.append(" ")
                t.append(_fmt_rel(p))
                t.append(" :: ")
                t.append(reasons)

                if meta_suffix:
                    t.append(meta_suffix)

                console.print(t)

            else:
                print(plain_msg)

            write_log(plain_msg)

    # =========================================================
    # OK FILES
    # =========================================================

    if args.show_ok and ok_files:

        output("\nOK FILES (PASSED VALIDATION):", console)

        for p, task_type in sorted(
            ok_files,
            key=lambda x: _fmt_rel(x[0]),
        ):

            msg = f"- [OK] [{task_type}] {_fmt_rel(p)}"

            output(msg, console)

    # =========================================================
    # SUMMARY
    # =========================================================

    failing_files = len(by_path)

    ok_count = checked_files - failing_files

    exceed_count = sum(
        1
        for fs in by_path.values()
        if _pick_file_kind(fs) == "exceed"
    )

    missing_count = sum(
        1
        for fs in by_path.values()
        if _pick_file_kind(fs) == "missing"
    )

    invalid_count = sum(
        1
        for fs in by_path.values()
        if _pick_file_kind(fs) == "invalid"
    )

    shallow_count = sum(
        1
        for fs in by_path.values()
        if _pick_file_kind(fs) == "shallow"
    )

    output("\nSUMMARY:", console)

    output(f"- scanned_dir: {_fmt_rel(scan_root)}", console)

    output(f"- json_files_found: {len(json_files)}", console)

    output(f"- checked_files: {checked_files}", console)

    output(f"- skipped_files: {skipped_files}", console)

    per_task_parts = ", ".join(
        [
            f"{t}={per_task_checked.get(t, 0)}"
            for t in sorted(enabled_tasks)
        ]
    )

    output(f"- checked_by_task: {per_task_parts}", console)

    output(f"- ok_files: {ok_count}", console)

    output(f"- exceed_files: {exceed_count}", console)

    output(f"- missing_files: {missing_count}", console)

    output(f"- invalid_files: {invalid_count}", console)

    output(f"- shallow_files: {shallow_count}", console)

    output(f"- failing_files: {failing_files}", console)

    output(f"- failures: {len(findings)}", console)

    output(f"\nLog saved to: {LOG_FILE}", console)

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())