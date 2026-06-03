"""Check chunk files and count unique sub_bab entries per subject.

Usage: from repo root:
    python scripts/check_sub_bab_counts.py

Outputs a CSV report to `output/chunk_sub_bab_counts.csv` and prints a summary.
"""
from pathlib import Path
import json
import csv
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
CHUNK_DIR = ROOT / 'data' / 'chunk'
OUT_DIR = ROOT / 'output'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / 'chunk_sub_bab_counts.csv'

subject_to_pairs = defaultdict(set)
file_index = []

for p in sorted(CHUNK_DIR.rglob('*_chunk.json')):
    try:
        text = p.read_text(encoding='utf-8')
        data = json.loads(text)
    except Exception as e:
        print(f'WARN: failed to load {p}: {e}')
        continue

    for entry in data:
        # support both 'query' wrapper and direct meta
        q = entry.get('query') if isinstance(entry, dict) and 'query' in entry else entry
        if not isinstance(q, dict):
            continue
        subject = q.get('mata_pelajaran') or q.get('mapel') or q.get('subject') or 'UNKNOWN'
        subject = subject.strip() if isinstance(subject, str) else str(subject)
        bab = q.get('bab_judul') or q.get('bab') or q.get('chapter') or 'UNKNOWN_BAB'
        sub_bab = q.get('sub_bab') or q.get('subbab') or q.get('sub_bab_title') or 'UNKNOWN_SUB_BAB'
        bab = bab.strip() if isinstance(bab, str) else str(bab)
        sub_bab = sub_bab.strip() if isinstance(sub_bab, str) else str(sub_bab)

        subject_to_pairs[subject].add((bab, sub_bab))
    file_index.append(str(p.relative_to(ROOT)))

# Build union of all (bab, sub_bab)
all_pairs = set()
for s, pairs in subject_to_pairs.items():
    all_pairs |= pairs

# Write CSV report
with OUT_CSV.open('w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['subject', 'unique_sub_bab_count'])
    for subject, pairs in sorted(subject_to_pairs.items(), key=lambda x: x[0]):
        writer.writerow([subject, len(pairs)])

# Print summary
print('\n=== Sub-bab counts per subject ===')
for subject, pairs in sorted(subject_to_pairs.items(), key=lambda x: (-len(x[1]), x[0])):
    print(f"- {subject}: {len(pairs)} unique (bab, sub_bab) pairs")

# Check equality
counts = {subject: len(pairs) for subject, pairs in subject_to_pairs.items()}
unique_counts = sorted(set(counts.values()))
if len(unique_counts) == 1:
    print('\nAll subjects have the same number of sub_bab:', unique_counts[0])
else:
    print('\nSubjects have different sub_bab counts:')
    for subject, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        missing = all_pairs - subject_to_pairs[subject]
        print(f"  - {subject}: {c} (missing {len(missing)} from union)")
    # Optionally list sample missing items for the subject with smallest count
    min_subject = min(counts, key=lambda s: counts[s])
    missing = sorted(list(all_pairs - subject_to_pairs[min_subject]))
    print(f"\nExample missing for subject '{min_subject}': (showing up to 10)")
    for pair in missing[:10]:
        print('   ', pair)

print('\nReport saved to:', OUT_CSV)
print('Processed files:')
for p in file_index:
    print(' -', p)
