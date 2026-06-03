"""
Metadata loader — discovers and filters chunk-based curriculum JSON files.

New chunk format (data/chunk/):
[
  {
    "query": { "jenjang", "kelas", "mata_pelajaran", "bab_judul", "sub_bab" },
    "chunks": [ { "id": 1, "content": "..." }, ... ]
  },
  ...
]

Each entry yields both the query metadata AND the chunk content
so the prompt builder can inject the actual learning material.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Generator

from rich.console import Console

# pyrefly: ignore [missing-import]
from src.config import METADATA_DIR, PILOT_SUBJECTS

console = Console()


def discover_metadata_files() -> list[Path]:
    """
    Walk the METADATA_DIR tree and return all .json files,
    sorted by path for deterministic ordering.
    """
    if not METADATA_DIR.exists():
        raise FileNotFoundError(
            f"Metadata directory not found: {METADATA_DIR}\n"
            "Check METADATA_DIR in src/config.py"
        )
    files = sorted(METADATA_DIR.rglob("*.json"))
    console.print(f"[dim]📂 Discovered {len(files)} chunk files[/]")
    return files


def _extract_variant(source_path: Path) -> str:
    """
    Extract chunk variant label from source filename.
    E.g. IPS_3_chunk.json → '3_chunks', Matematika_1_chunk.json → '1_chunks'
    """
    stem = source_path.stem  # e.g. "IPS_3_chunk"
    match = re.search(r'_(\d+)_chunk$', stem)
    return f"{match.group(1)}_chunks" if match else "0_chunks"


def _infer_kurikulum_from_path(path: Path) -> str:
    """
    Infer the kurikulum name from the directory tree.

    Matches directory parts against AVAILABLE_CURRICULA (case-insensitive).
    E.g. .../Kurikulum Merdeka/SMA/... → "Kurikulum Merdeka"
         .../K-13/SMA/...             → "K-13"
         .../KTSP/SMA/...             → "KTSP"
    """
    from src.config import AVAILABLE_CURRICULA

    curricula_lower = {c.lower(): c for c in AVAILABLE_CURRICULA}
    for part in path.parts:
        matched = curricula_lower.get(part.lower())
        if matched:
            return matched
    return "Kurikulum Merdeka"


def load_chunk_file(path: Path) -> list[dict]:
    """
    Load a chunk JSON file and return a flat list of entries.
    
    Each entry is enriched with:
      - 'kurikulum' (inferred from path)
      - 'chunks_text' (all chunk contents joined for context injection)
    
    The original query fields (jenjang, kelas, mata_pelajaran, bab_judul, sub_bab)
    are flattened to top-level for backward compatibility with filters.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")

    kurikulum = _infer_kurikulum_from_path(path)
    entries = []

    for item in data:
        query = item.get("query", {})
        chunks = item.get("chunks", [])

        # Flatten query fields to top-level
        entry = {
            "kurikulum": kurikulum,
            "jenjang": query.get("jenjang", ""),
            "kelas": query.get("kelas", ""),
            "mata_pelajaran": query.get("mata_pelajaran", ""),
            "bab_judul": query.get("bab_judul", ""),
            "sub_bab": query.get("sub_bab", ""),
            "atp": query.get("atp", ""),
            # Join all chunk contents as the learning material context
            "chunks_text": "\n\n".join(c.get("content", "") for c in chunks),
            "chunks_raw": chunks,
        }
        entries.append(entry)

    return entries


def _matches_pilot(entry: dict) -> bool:
    """Check if an entry matches any pilot subject filter."""
    if PILOT_SUBJECTS is None:
        return True  # no filter, include everything
    # Compare on filtering keys including kurikulum
    compare_keys = ["kurikulum", "jenjang", "kelas", "mata_pelajaran"]
    for pilot in PILOT_SUBJECTS:
        if all(
            str(entry.get(k, "")).strip().lower() == str(pilot.get(k, "")).strip().lower()
            for k in compare_keys
        ):
            return True
    return False


def _matches_cli_filters(entry: dict, filters: dict) -> bool:
    """
    Apply CLI-level filters (--subject, --kelas, --jenjang, --kurikulum).

    The filters dict maps filter names to lists of accepted values:
      {"subject": ["IPS"], "kelas": ["Kelas 10"], "jenjang": ["SMA"], "kurikulum": ["K-13"]}

    All specified filters must match (AND logic).
    """
    filter_key_map = {
        "subject": "mata_pelajaran",
        "kelas": "kelas",
        "jenjang": "jenjang",
        "kurikulum": "kurikulum",
    }
    for filter_name, entry_key in filter_key_map.items():
        allowed = filters.get(filter_name)
        if allowed:
            # Case-insensitive check
            entry_val = str(entry.get(entry_key, "")).strip().lower()
            allowed_lower = [str(a).strip().lower() for a in allowed]
            if entry_val not in allowed_lower:
                return False
    return True


def iter_metadata(
    *,
    test_mode: bool = False,
    filters: dict | None = None,
) -> Generator[tuple[Path, dict], None, None]:
    """
    Yield (source_path, entry) for every sub-bab that passes all filters.

    Filters applied in order:
      1. PILOT_SUBJECTS from config (jenjang/kelas/mata_pelajaran)
      2. CLI filters: --subject, --kelas, --jenjang (AND logic)

    If test_mode is True, only the first matching entry per file is yielded.
    """
    files = discover_metadata_files()
    for path in files:
        variant = _extract_variant(path)
        if filters and "variant" in filters:
            allowed_variants = [str(v).strip().lower() for v in filters["variant"]]
            if variant.lower() not in allowed_variants:
                continue

        entries = load_chunk_file(path)
        count = 0
        for entry in entries:
            if not _matches_pilot(entry):
                continue
            if filters and not _matches_cli_filters(entry, filters):
                continue
            yield path, entry
            count += 1
            if test_mode and count >= 1:
                break


# Backward-compatible aliases
load_metadata = load_chunk_file
