"""Per-variant checker: list missing and duplicated (bab, sub_bab) per chunk file variant.

Outputs:
 - output/chunk_per_variant_report.csv

Run:
    python scripts/check_per_variant.py
"""
from pathlib import Path
import json
import csv
import re
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
CHUNK_DIR = ROOT / 'data' / 'chunk'
OUT_DIR = ROOT / 'output'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / 'chunk_per_variant_report.csv'

# Collect all files
files = sorted(CHUNK_DIR.rglob('*_chunk.json'))

# helper to extract pair
def extract_query(entry):
    if isinstance(entry, dict) and 'query' in entry:
        return entry['query']
    return entry if isinstance(entry, dict) else {}

all_pairs = set()
file_pairs = {}
file_subjects = {}
file_variant = {}

# First pass: collect pairs per file
for p in files:
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:
        print(f"WARN: cannot read {p}: {e}")
        continue
    cnt = Counter()
    subjects = Counter()
    for entry in data:
        q = extract_query(entry)
        subject = (q.get('mata_pelajaran') or q.get('mapel') or q.get('subject') or '').strip()
        bab = (q.get('bab_judul') or q.get('bab') or q.get('chapter') or '').strip()
        sub = (q.get('sub_bab') or q.get('subbab') or q.get('sub_bab_title') or '').strip()
        pair = (bab, sub)
        if bab == '' and sub == '':
            continue
        cnt[pair] += 1
        if subject:
            subjects[subject] += 1
        all_pairs.add(pair)
    file_pairs[p] = cnt
    # pick most common subject for the file (fallback to folder name)
    subject = subjects.most_common(1)[0][0] if subjects else p.parent.name
    file_subjects[p] = subject
    m = re.search(r'_(\d+)_chunk\.json$', p.name)
    file_variant[p] = int(m.group(1)) if m else None

# Second pass: write report
with OUT_CSV.open('w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['file', 'subject', 'variant', 'present_count', 'missing_count', 'duplicates_count', 'missing_pairs', 'duplicate_pairs'])
    rows = []
    for p in sorted(file_pairs.keys()):
        cnt = file_pairs[p]
        present = set(cnt.keys())
        missing = sorted(list(all_pairs - present))
        duplicates = sorted([(pair, c) for pair, c in cnt.items() if c > 1], key=lambda x: -x[1])
        missing_str = ' || '.join([f"{a} ::: {b}" for (a, b) in missing])
        dup_str = ' || '.join([f"{a} ::: {b} (count={c})" for ((a,b), c) in duplicates for a in [a] for b in [b]])
        writer.writerow([
            str(p.relative_to(ROOT)),
            file_subjects.get(p, ''),
            file_variant.get(p, ''),
            len(present),
            len(missing),
            len(duplicates),
            missing_str,
            dup_str
        ])

# Print concise summary
print('\nPer-variant summary:')
summary = []
for p in sorted(file_pairs.keys()):
    cnt = file_pairs[p]
    present = set(cnt.keys())
    missing = all_pairs - present
    duplicates = [pair for pair,count in cnt.items() if count>1]
    summary.append((str(p.relative_to(ROOT)), file_subjects.get(p,''), file_variant.get(p,''), len(present), len(missing), len(duplicates)))

# sort by missing desc
for path, subj, var, present_count, missing_count, dup_count in sorted(summary, key=lambda x: (-x[4], x[0])):
    print(f"- {path}: subject={subj}, variant={var}, present={present_count}, missing={missing_count}, duplicates={dup_count}")

print('\nReport saved to:', OUT_CSV)
