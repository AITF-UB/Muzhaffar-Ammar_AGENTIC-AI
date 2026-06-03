"""Produce detailed lists of missing and duplicated (bab, sub_bab) pairs.

Outputs:
 - output/chunk_sub_bab_missing.csv  (rows: subject,bab_judul,sub_bab)
 - output/chunk_sub_bab_duplicates.csv (rows: subject,bab_judul,sub_bab,count,occurrences)
 - output/chunk_sub_bab_detailed.csv  (rows: subject,bab_judul,sub_bab,count,files)

Run from repo root:
    python scripts/check_sub_bab_details.py
"""
from pathlib import Path
import json
import csv
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent
CHUNK_DIR = ROOT / 'data' / 'chunk'
OUT_DIR = ROOT / 'output'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MISSING_CSV = OUT_DIR / 'chunk_sub_bab_missing.csv'
DUP_CSV = OUT_DIR / 'chunk_sub_bab_duplicates.csv'
DETAIL_CSV = OUT_DIR / 'chunk_sub_bab_detailed.csv'

pairs_per_subject = defaultdict(Counter)
occurrences = defaultdict(lambda: defaultdict(list))
files_processed = []

for p in sorted(CHUNK_DIR.rglob('*_chunk.json')):
    try:
        text = p.read_text(encoding='utf-8')
        data = json.loads(text)
    except Exception as e:
        print(f'WARN: failed to load {p}: {e}')
        continue

    for idx, entry in enumerate(data):
        q = entry.get('query') if isinstance(entry, dict) and 'query' in entry else entry
        if not isinstance(q, dict):
            continue
        subject = q.get('mata_pelajaran') or q.get('mapel') or q.get('subject') or 'UNKNOWN'
        subject = subject.strip() if isinstance(subject, str) else str(subject)
        bab = q.get('bab_judul') or q.get('bab') or q.get('chapter') or 'UNKNOWN_BAB'
        sub_bab = q.get('sub_bab') or q.get('subbab') or q.get('sub_bab_title') or 'UNKNOWN_SUB_BAB'
        bab = bab.strip() if isinstance(bab, str) else str(bab)
        sub_bab = sub_bab.strip() if isinstance(sub_bab, str) else str(sub_bab)

        pair = (bab, sub_bab)
        pairs_per_subject[subject][pair] += 1
        occurrences[subject][pair].append({'file': str(p.relative_to(ROOT)), 'index': idx})

    files_processed.append(str(p.relative_to(ROOT)))

# Build global union of pairs
all_pairs = set()
for subj_pairs in pairs_per_subject.values():
    all_pairs |= set(subj_pairs.keys())

# Write detailed CSV (counts and files)
with DETAIL_CSV.open('w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['subject', 'bab_judul', 'sub_bab', 'count', 'files'])
    for subject in sorted(pairs_per_subject.keys()):
        for (bab, sub_bab), count in sorted(pairs_per_subject[subject].items(), key=lambda x: (-x[1], x[0])):
            files = ['{}:{}'.format(o['file'], o['index']) for o in occurrences[subject][(bab, sub_bab)]]
            writer.writerow([subject, bab, sub_bab, count, ' | '.join(files)])

# Write missing CSV
with MISSING_CSV.open('w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['subject', 'bab_judul', 'sub_bab'])
    for subject in sorted(pairs_per_subject.keys()):
        missing = sorted(all_pairs - set(pairs_per_subject[subject].keys()))
        for bab, sub_bab in missing:
            writer.writerow([subject, bab, sub_bab])

# Write duplicates CSV
with DUP_CSV.open('w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['subject', 'bab_judul', 'sub_bab', 'count', 'occurrences'])
    for subject in sorted(pairs_per_subject.keys()):
        for (bab, sub_bab), count in sorted(pairs_per_subject[subject].items(), key=lambda x: -x[1]):
            if count > 1:
                files = ['{}:{}'.format(o['file'], o['index']) for o in occurrences[subject][(bab, sub_bab)]]
                writer.writerow([subject, bab, sub_bab, count, ' | '.join(files)])

# Print concise summary
print('\n=== Summary per subject ===')
for subject in sorted(pairs_per_subject.keys()):
    unique_count = len(pairs_per_subject[subject])
    missing_count = len(all_pairs - set(pairs_per_subject[subject].keys()))
    dup_count = sum(1 for c in pairs_per_subject[subject].values() if c > 1)
    print(f"- {subject}: {unique_count} unique pairs, {missing_count} missing, {dup_count} duplicated pairs")

# Print top missing examples per subject (up to 10)
print('\n=== Example missing items (up to 10 per subject) ===')
for subject in sorted(pairs_per_subject.keys()):
    missing = sorted(all_pairs - set(pairs_per_subject[subject].keys()))
    if not missing:
        print(f"- {subject}: none")
        continue
    print(f"- {subject} missing {len(missing)} items; examples:")
    for bab, sub_bab in missing[:10]:
        print(f"    - {bab} | {sub_bab}")

# Print duplicated items (if any)
print('\n=== Duplicated (bab, sub_bab) entries per subject ===')
for subject in sorted(pairs_per_subject.keys()):
    dup_items = [(pair, count) for pair, count in pairs_per_subject[subject].items() if count > 1]
    if dup_items:
        print(f"- {subject} has {len(dup_items)} duplicated pairs:")
        for (bab, sub_bab), count in dup_items:
            occ = occurrences[subject][(bab, sub_bab)]
            files = ', '.join([f"{o['file']}:{o['index']}" for o in occ])
            print(f"    - {bab} | {sub_bab}  (count={count})  files: {files}")

print('\nFiles processed:', len(files_processed))
print('Detailed CSV:', DETAIL_CSV)
print('Missing CSV:', MISSING_CSV)
print('Duplicates CSV:', DUP_CSV)
