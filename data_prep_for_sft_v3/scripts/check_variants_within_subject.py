"""Check consistency of 6 chunk variants within each subject.

Goal (your requirement):
- For a given subject, you have 6 variant files (*_1_chunk.json ... *_6_chunk.json).
- The number of JSON objects (len of the array) should be the same across variants.
- Ideally, the (bab_judul, sub_bab) list should also match (no missing/extra/duplicates).

Outputs:
- output/variant_counts_within_subject.csv
- output/variant_diffs_within_subject.csv

Run:
    python scripts/check_variants_within_subject.py
"""

from __future__ import annotations

from pathlib import Path
import csv
import json
import re
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
CHUNK_DIR = ROOT / "data" / "chunk"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_COUNTS = OUT_DIR / "variant_counts_within_subject.csv"
OUT_DIFFS = OUT_DIR / "variant_diffs_within_subject.csv"

VARIANT_RE = re.compile(r"_(\d+)_chunk\.json$", re.IGNORECASE)


def _extract_query(entry: object) -> dict:
    if isinstance(entry, dict) and "query" in entry and isinstance(entry["query"], dict):
        return entry["query"]
    return entry if isinstance(entry, dict) else {}


def _clean_str(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def _get_variant(path: Path) -> int | None:
    m = VARIANT_RE.search(path.name)
    return int(m.group(1)) if m else None


def _subject_from_entries(entries: list[object], fallback: str) -> str:
    # subject is expected constant inside file; take mode if exists
    c = Counter()
    for e in entries:
        q = _extract_query(e)
        s = _clean_str(q.get("mata_pelajaran") or q.get("mapel") or q.get("subject"))
        if s:
            c[s] += 1
    return c.most_common(1)[0][0] if c else fallback


def _pair_list(entries: list[object]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for e in entries:
        q = _extract_query(e)
        bab = _clean_str(q.get("bab_judul") or q.get("bab") or q.get("chapter"))
        sub = _clean_str(q.get("sub_bab") or q.get("subbab") or q.get("sub_bab_title"))
        pairs.append((bab, sub))
    return pairs


# Load all variant files
files = sorted(CHUNK_DIR.rglob("*_chunk.json"))

# Group by subject
subject_files: dict[str, list[Path]] = defaultdict(list)
file_data: dict[Path, dict] = {}

for p in files:
    try:
        entries = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            raise ValueError("root JSON is not a list")
    except Exception as e:
        print(f"WARN: failed to load {p}: {e}")
        continue

    subject = _subject_from_entries(entries, fallback=p.parent.name)
    variant = _get_variant(p)
    pairs = _pair_list(entries)
    object_count = len(entries)
    pair_counter = Counter(pairs)
    dup_pairs = [(pair, c) for pair, c in pair_counter.items() if c > 1]

    file_data[p] = {
        "subject": subject,
        "variant": variant,
        "object_count": object_count,
        "unique_pair_count": len(pair_counter),
        "dup_pairs": sorted(dup_pairs, key=lambda x: (-x[1], x[0])),
        "pairs": pairs,
        "pair_set": set(pair_counter.keys()),
    }
    subject_files[subject].append(p)

# Write counts report
with OUT_COUNTS.open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "subject",
            "variant",
            "file",
            "object_count",
            "unique_pair_count",
            "duplicate_pair_count",
            "duplicate_pairs",
        ]
    )
    for subject in sorted(subject_files.keys()):
        for p in sorted(subject_files[subject], key=lambda x: (_get_variant(x) or 999, x.name)):
            d = file_data[p]
            dup_str = " || ".join([f"{a} ::: {b} (count={c})" for ((a, b), c) in d["dup_pairs"]])
            w.writerow(
                [
                    subject,
                    d["variant"],
                    str(p.relative_to(ROOT)),
                    d["object_count"],
                    d["unique_pair_count"],
                    len(d["dup_pairs"]),
                    dup_str,
                ]
            )

# Write diffs report (vs reference variant per subject)
with OUT_DIFFS.open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(
        [
            "subject",
            "reference_variant",
            "variant",
            "file",
            "object_count_vs_ref",
            "missing_pairs_count",
            "extra_pairs_count",
            "order_mismatch_count",
            "missing_pairs",
            "extra_pairs",
            "order_mismatch_examples",
        ]
    )

    for subject in sorted(subject_files.keys()):
        paths = sorted(subject_files[subject], key=lambda x: (_get_variant(x) or 999, x.name))
        if not paths:
            continue

        # pick reference: variant 1 if exists, else smallest variant number, else first.
        ref_path = None
        for p in paths:
            if file_data[p]["variant"] == 1:
                ref_path = p
                break
        if ref_path is None:
            ref_path = paths[0]

        ref = file_data[ref_path]
        ref_pairs = ref["pairs"]
        ref_set = ref["pair_set"]
        ref_variant = ref["variant"]

        for p in paths:
            d = file_data[p]
            var_pairs = d["pairs"]
            var_set = d["pair_set"]

            missing = sorted(list(ref_set - var_set))
            extra = sorted(list(var_set - ref_set))

            # order mismatches only meaningful if lengths equal
            order_mismatches = 0
            mismatch_examples: list[str] = []
            if len(ref_pairs) == len(var_pairs):
                for i, (a, b) in enumerate(zip(ref_pairs, var_pairs)):
                    if a != b:
                        order_mismatches += 1
                        if len(mismatch_examples) < 10:
                            mismatch_examples.append(f"idx={i} ref={a[0]}|{a[1]} var={b[0]}|{b[1]}")
            else:
                # if length differs, we still can show first few head mismatches up to min length
                for i, (a, b) in enumerate(zip(ref_pairs, var_pairs)):
                    if a != b:
                        if len(mismatch_examples) < 10:
                            mismatch_examples.append(f"idx={i} ref={a[0]}|{a[1]} var={b[0]}|{b[1]}")

            obj_vs_ref = d["object_count"] - ref["object_count"]
            missing_str = " || ".join([f"{a} ::: {b}" for (a, b) in missing])
            extra_str = " || ".join([f"{a} ::: {b}" for (a, b) in extra])
            mismatch_str = " || ".join(mismatch_examples)

            w.writerow(
                [
                    subject,
                    ref_variant,
                    d["variant"],
                    str(p.relative_to(ROOT)),
                    obj_vs_ref,
                    len(missing),
                    len(extra),
                    order_mismatches,
                    missing_str,
                    extra_str,
                    mismatch_str,
                ]
            )

# Print human summary
print("\n=== Within-subject variant object counts ===")
for subject in sorted(subject_files.keys()):
    paths = sorted(subject_files[subject], key=lambda x: (_get_variant(x) or 999, x.name))
    counts = [file_data[p]["object_count"] for p in paths]
    uniq = sorted(set(counts))
    if len(uniq) == 1:
        print(f"- {subject}: OK (all variants object_count={uniq[0]})")
    else:
        print(f"- {subject}: MISMATCH counts={uniq}")
        for p in paths:
            d = file_data[p]
            print(f"    - var={d['variant']} file={p.name} object_count={d['object_count']} dup_pairs={len(d['dup_pairs'])}")

print("\nSaved:")
print("-", OUT_COUNTS)
print("-", OUT_DIFFS)
