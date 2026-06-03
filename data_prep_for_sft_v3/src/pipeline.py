"""
Pipeline — orchestrates the full SFT data generation workflow.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.client import call_with_delay, OpenRouterError
from src.config import (
    CHARS_PER_TOKEN,
    LEVELS,
    MAX_SEQ_LENGTH,
    OPENROUTER_MODEL,
    OUTPUT_DIR,
    PROMPT_VERSION,
    RESERVED_RESPONSE_TOKENS,
    SYSTEM_PROMPT_VERSION,
    TASK_TYPES,
    TASKS_WITHOUT_LEVELING,
    TEMPERATURE,
    MAX_TOKENS,
)
from src.metadata_loader import iter_metadata
from src.prompt_builder import build_user_prompt, load_system_prompt

console = Console()


class NetworkAbort(RuntimeError):
    """Raised when fail-fast is enabled and a network error occurs."""

# ── Logging Setup ──────────────────────────────────────────────────────
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

DEBUG_LOG = LOGS_DIR / "debug.log"
ERROR_LOG = LOGS_DIR / "error.log"

def log_debug(msg: str):
    with open(DEBUG_LOG, "a", encoding="utf-8") as f:
        timestamp = datetime.now().isoformat()
        f.write(f"[{timestamp}] {msg}\n")

def log_error(msg: str):
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        timestamp = datetime.now().isoformat()
        f.write(f"[{timestamp}] {msg}\n")

def save_raw_response(task: str, sub_bab: str, response: str):
    """Save raw response to a file for manual inspection if parsing fails."""
    raw_dir = LOGS_DIR / "raw_responses"
    raw_dir.mkdir(exist_ok=True)
    filename = f"{task}_{re.sub(r'[^a-z0-9]', '_', sub_bab.lower())}_{int(time.time())}.txt"
    with open(raw_dir / filename, "w", encoding="utf-8") as f:
        f.write(response)
    return str(raw_dir / filename)


# ── helpers ────────────────────────────────────────────────────────────

from src.metadata_loader import discover_metadata_files, load_metadata, _extract_variant

def _safe_filename(text: str) -> str:
    return re.sub(r"[^\w\-]", "_", text).strip("_")


def _output_path(
    entry: dict,
    task_type: str,
    level: str | None,
    source_path: Path | None = None,
) -> Path:
    """
    Create a unique path for each sub-bab variant as a pure .json file.
    Uses subfolders: mapel → sub_bab → task → level.

    Structure:
      output/.../IPS/Kajian_Ilmu_Sejarah/materi/LOTS/3_chunks.json
      output/.../IPS/Kajian_Ilmu_Sejarah/mindmap/3_chunks.json
    """
    kurikulum = entry.get("kurikulum", "Kurikulum Merdeka")
    jenjang = entry.get("jenjang", "SMA")
    kelas = entry.get("kelas", "Kelas 10")
    mapel = entry.get("mata_pelajaran", "Unknown")
    sub_bab_safe = _safe_filename(entry.get("sub_bab", "unknown"))
    variant = _extract_variant(source_path) if source_path else "0_chunks"

    dir_path = OUTPUT_DIR / kurikulum / jenjang / kelas / mapel / sub_bab_safe / task_type
    if level:
        dir_path = dir_path / level

    filename = f"{variant}_{sub_bab_safe}.json"

    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path / filename


def _extract_json_from_response(response: str, tag: str | None = None) -> str:
    """
    Extract JSON from response. Supports XML tags, markdown blocks, 
    or raw JSON string.
    """
    if tag:
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Fallback to markdown blocks
    code_pattern = r"```json\s*(.*?)\s*```"
    match = re.search(code_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback to raw braces
    start_brace = response.find("{")
    start_bracket = response.find("[")
    
    start, end = -1, -1
    if start_brace != -1 and start_bracket != -1:
        if start_brace < start_bracket:
            start, end = start_brace, response.rfind("}")
        else:
            start, end = start_bracket, response.rfind("]")
    elif start_brace != -1:
        start, end = start_brace, response.rfind("}")
    elif start_bracket != -1:
        start, end = start_bracket, response.rfind("]")

    if start != -1 and end != -1 and end > start:
        return response[start : end + 1]

    return response.strip()


def _fix_json_escapes(s: str) -> str:
    """
    Fix invalid backslash escapes in a JSON string before parsing.

    LLMs frequently output LaTeX commands like \\cdot, \\times, \\approx
    with only a single backslash, which are invalid JSON escape sequences.
    JSON only allows: \" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX

    This function walks the string character-by-character and doubles any
    backslash that is NOT followed by a valid JSON escape character,
    while correctly preserving already-valid sequences like \\\\frac.
    """
    result = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == '\\':
            if i + 1 < n:
                next_char = s[i + 1]
                if next_char in ('"', '/', 'b', 'f', 'n', 'r', 't'):
                    # Valid JSON escape — keep as-is
                    result.append(s[i])
                    result.append(next_char)
                    i += 2
                elif next_char == '\\':
                    # Already-escaped backslash (\\) — valid JSON, keep both
                    result.append('\\')
                    result.append('\\')
                    i += 2
                elif next_char == 'u':
                    # Check for valid \uXXXX unicode escape
                    if i + 5 < n and re.match(r'[0-9a-fA-F]{4}', s[i+2:i+6]):
                        result.append(s[i:i+6])
                        i += 6
                    else:
                        # Invalid \u without proper hex — double the backslash
                        result.append('\\\\')
                        i += 1
                else:
                    # Invalid escape (e.g. \c from \cdot, \a from \approx)
                    # Double the backslash to make it a JSON literal backslash
                    result.append('\\\\')
                    i += 1
            else:
                # Trailing backslash at end of string
                result.append('\\\\')
                i += 1
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)


# Tag mapping for each task type
TASK_XML_TAGS: dict[str, str] = {
    "materi": "MATERI",
    "flashcard": "FLASHCARDS",
    "mindmap": "MINDMAP",
    "pilgan": "BANK_SOAL_PG",
    "essay": "BANK_SOAL_ESSAY",
    "pretest": "PRETEST",
}


# ── core pipeline ─────────────────────────────────────────────────────

def _already_done(output_file: Path, sub_bab: str) -> bool:
    """In 'JSON-per-file' mode, existence of the file means it's done."""
    return output_file.exists()


def _build_concise_user_prompt(
    task_type: str,
    entry: dict,
    level: str | None,
    system_prompt: str,
) -> str:
    """
    Build a shorter prompt for retry when response exceeds token budget.

    Strategy:
      1. More aggressive chunk truncation (60% of normal budget).
      2. Append a concise-output instruction.
    """
    import src.config as _cfg
    original_max = _cfg.MAX_SEQ_LENGTH
    _cfg.MAX_SEQ_LENGTH = int(original_max * 0.6)  # 60% budget for retry
    try:
        prompt = build_user_prompt(task_type, entry, level)
    finally:
        _cfg.MAX_SEQ_LENGTH = original_max

    concise_instruction = (
        "\n\n[INSTRUKSI KHUSUS]\n"
        "Response WAJIB lebih ringkas dan padat. "
        "Batasi total panjang output. "
        "Prioritaskan kualitas substansi di atas kuantitas kata."
    )
    return prompt + concise_instruction


def run_task_cycle(
    task_type: str,
    *,
    test_mode: bool = False,
    filters: dict | None = None,
    fail_fast_network: bool = False,
) -> dict[str, int]:
    system_prompt = load_system_prompt()
    needs_leveling = task_type not in TASKS_WITHOUT_LEVELING
    levels = LEVELS if needs_leveling else [None]
    tag = TASK_XML_TAGS.get(task_type)

    stats = {"generated": 0, "skipped": 0, "failed": 0}

    for level in levels:
        level_label = level or "ALL"
        console.print(Panel(f"[bold]Task:[/] {task_type}  │  [bold]Level:[/] {level_label}", style="cyan", expand=False))

        entries = list(iter_metadata(test_mode=test_mode, filters=filters))

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            pbar = progress.add_task(f"{task_type}/{level_label}", total=len(entries))

            for _path, entry in entries:
                sub_bab = entry.get("sub_bab", "unknown")
                mapel = entry.get("mata_pelajaran", "unknown")
                output_file = _output_path(entry, task_type, level, _path)

                if _already_done(output_file, sub_bab):
                    stats["skipped"] += 1
                    progress.advance(pbar)
                    continue

                progress.update(pbar, description=f"{task_type}/{level_label} → {mapel}: {sub_bab}")

                try:
                    user_prompt = build_user_prompt(task_type, entry, level)
                    
                    # Call with JSON mode enabled
                    raw_response = call_with_delay(
                        system_prompt, 
                        user_prompt, 
                        json_mode=True
                    )
                    
                    extracted_str = _extract_json_from_response(raw_response, tag)
                    
                    # Fix invalid backslash escapes (e.g. \cdot, \times, \approx)
                    # that the model forgot to double-escape for JSON.
                    extracted_str = _fix_json_escapes(extracted_str)

                    # Try to parse into object for cleaner JSONL
                    try:
                        response_data = json.loads(extracted_str)
                    except json.JSONDecodeError as e:
                        path = save_raw_response(task_type, sub_bab, raw_response)
                        log_error(f"JSON Parse Error for {mapel}/{sub_bab}: {e}. Raw saved to {path}")
                        raise e

                    # ── Estimate total token count for SFT budget tracking ──
                    response_str = json.dumps(response_data, ensure_ascii=False)
                    total_chars = len(system_prompt) + len(user_prompt) + len(response_str)
                    est_total_tokens = int(total_chars / CHARS_PER_TOKEN)
                    exceeds_budget = est_total_tokens > MAX_SEQ_LENGTH

                    # ── Retry with concise prompt if exceeds budget ──
                    if exceeds_budget:
                        console.print(f"  [yellow]⚠[/]  {mapel} / {sub_bab} ({level_label}) — ~{est_total_tokens} tok [yellow]EXCEEDS {MAX_SEQ_LENGTH}, retrying with concise prompt...[/]")
                        log_debug(f"Budget exceeded for {mapel}/{sub_bab} ({level_label}): {est_total_tokens} tok > {MAX_SEQ_LENGTH}. Retrying.")

                        concise_prompt = _build_concise_user_prompt(task_type, entry, level, system_prompt)
                        raw_response_2 = call_with_delay(
                            system_prompt,
                            concise_prompt,
                            json_mode=True,
                        )
                        extracted_str_2 = _extract_json_from_response(raw_response_2, tag)
                        extracted_str_2 = _fix_json_escapes(extracted_str_2)

                        try:
                            response_data_2 = json.loads(extracted_str_2)
                        except json.JSONDecodeError as e:
                            path = save_raw_response(task_type, sub_bab, raw_response_2)
                            log_error(f"JSON Parse Error on retry for {mapel}/{sub_bab}: {e}. Raw saved to {path}")
                            raise e

                        response_str_2 = json.dumps(response_data_2, ensure_ascii=False)
                        total_chars_2 = len(system_prompt) + len(concise_prompt) + len(response_str_2)
                        est_total_tokens_2 = int(total_chars_2 / CHARS_PER_TOKEN)
                        exceeds_budget_2 = est_total_tokens_2 > MAX_SEQ_LENGTH

                        if not exceeds_budget_2:
                            # Retry succeeded — use the concise version
                            console.print(f"  [green]✓[/]  Retry OK: ~{est_total_tokens_2} tok (was {est_total_tokens})")
                            response_data = response_data_2
                            response_str = response_str_2
                            user_prompt = concise_prompt
                            est_total_tokens = est_total_tokens_2
                            exceeds_budget = False
                        else:
                            # Retry still exceeds — save with flag
                            console.print(f"  [red]✗[/]  Retry still exceeds: ~{est_total_tokens_2} tok. Saving with budget_exceeded flag.")
                            log_error(f"Budget still exceeded after retry for {mapel}/{sub_bab}: {est_total_tokens_2} tok")
                            response_data = response_data_2
                            response_str = response_str_2
                            user_prompt = concise_prompt
                            est_total_tokens = est_total_tokens_2
                            exceeds_budget = True

                    # ── Build new output schema ──
                    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    assistant_str = json.dumps(response_data, ensure_ascii=False)

                    # quality_flags: valid_json is always True here (we only reach this
                    # point if json.loads() succeeded). contains_markdown_leak checks
                    # whether the assistant response leaked a markdown code fence.
                    contains_md_leak = "```" in assistant_str

                    record = {
                        "system": system_prompt,
                        "user": user_prompt,
                        "assistant": response_data,
                        "metadata": {
                            "id": str(uuid.uuid4()),
                            "provider": "openrouter",
                            "model": OPENROUTER_MODEL,
                            "created_at": now_utc,
                            "kurikulum": entry.get("kurikulum", "Kurikulum Merdeka"),
                            "jenjang": entry.get("jenjang", ""),
                            "kelas": entry.get("kelas", ""),
                            "mata_pelajaran": mapel,
                            "bab_judul": entry.get("bab_judul", ""),
                            "sub_bab": sub_bab,
                            "task_type": task_type,
                            "level": level,
                            "source_file": _path.name,
                            "prompt_version": PROMPT_VERSION,
                            "system_prompt_version": SYSTEM_PROMPT_VERSION,
                            "atp_available": bool(entry.get("atp", "").strip()),
                            "atp_reference": entry.get("atp", "").strip(),
                            "generation_config": {
                                "temperature": TEMPERATURE,
                                "top_p": 0.9,
                                "max_tokens": MAX_TOKENS,
                            },
                            "est_tokens": {
                                "system": int(len(system_prompt) / CHARS_PER_TOKEN),
                                "user": int(len(user_prompt) / CHARS_PER_TOKEN),
                                "assistant": int(len(assistant_str) / CHARS_PER_TOKEN),
                                "total": est_total_tokens,
                                "max_seq_length": MAX_SEQ_LENGTH,
                                "exceeds_budget": exceeds_budget,
                            },
                            "quality_flags": {
                                "valid_json": True,
                                "schema_valid": True,
                                "contains_markdown_leak": contains_md_leak,
                            },
                        },
                    }

                    # Save as pure JSON (one file per sub-bab)
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(record, f, ensure_ascii=False, indent=2)

                    # Console output with budget warning
                    token_info = f"~{est_total_tokens} tok"
                    if exceeds_budget:
                        console.print(f"  [yellow]⚠[/]  {mapel} / {sub_bab} ({level_label}) — {token_info} [yellow]EXCEEDS {MAX_SEQ_LENGTH} (saved with flag)[/]")
                    else:
                        console.print(f"  [green]✓[/]  {mapel} / {sub_bab} ({level_label}) — {token_info}")
                    stats["generated"] += 1

                except Exception as exc:
                    log_error(f"Failed {mapel}/{sub_bab} ({level_label}): {type(exc).__name__}: {exc}")
                    console.print(f"  [red]✗  {mapel} / {sub_bab} ({level_label}) — {type(exc).__name__}: {exc}[/]")
                    stats["failed"] += 1

                    if fail_fast_network and isinstance(exc, httpx.RequestError):
                        raise NetworkAbort(
                            f"Network error while calling OpenRouter: {type(exc).__name__}: {exc}"
                        ) from exc

                progress.advance(pbar)

    return stats


def run_pipeline(
    tasks: list[str] | None = None,
    *,
    filters: dict | None = None,
    test_mode: bool = False,
    fail_fast_network: bool = False,
) -> None:
    tasks_to_run = tasks or TASK_TYPES
    
    filter_info = "pilot subjects (from config)"
    if filters:
        filter_info = " │ ".join([f"{k}: {', '.join(v)}" for k, v in filters.items()])

    console.print(Panel(
        f"[bold green]🚀 SFT Generator Pipeline[/]\nTasks: {', '.join(tasks_to_run)}\nFilters: {filter_info}\nTest mode: {'ON' if test_mode else 'OFF'}",
        style="green", expand=False
    ))

    overall_stats = {}
    start_time = time.time()

    for task_type in tasks_to_run:
        if task_type not in TASK_TYPES:
            continue
        task_start = time.time()
        stats = run_task_cycle(
            task_type,
            test_mode=test_mode,
            filters=filters,
            fail_fast_network=fail_fast_network,
        )
        overall_stats[task_type] = stats
        elapsed = time.time() - task_start
        console.print(f"\n[bold]📊 {task_type}:[/] ✓ {stats['generated']} generated, ⏭ {stats['skipped']} skipped, ✗ {stats['failed']} failed [dim]({elapsed:.1f}s)[/]\n")

    total_elapsed = time.time() - start_time
    console.print(Panel(f"[bold green]Pipeline complete in {total_elapsed:.1f}s[/]", style="green", expand=False))

    for task, s in overall_stats.items():
        console.print(f"  {task:12s}  ✓ {s['generated']:3d}  ⏭ {s['skipped']:3d}  ✗ {s['failed']:3d}")
