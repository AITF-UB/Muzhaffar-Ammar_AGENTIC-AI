"""Analyze actual combined chunks_text sizes per sub_bab across all variants."""
import json
from pathlib import Path

CHARS_PER_TOKEN = 3.2
PROJECT = Path(__file__).resolve().parent.parent
chunk_dir = PROJECT / "data" / "chunk"

# Collect: {variant_name: [(sub_bab, n_chunks, combined_chars, combined_tokens)]}
results = []

for f in sorted(chunk_dir.rglob("*.json")):
    try:
        raw = f.read_text(encoding="utf-8").strip()
        data = json.loads(raw)
    except (json.JSONDecodeError, Exception):
        continue

    variant = f.stem  # e.g. "Matematika_Umum_4_chunk"
    for item in data:
        sub_bab = item.get("query", {}).get("sub_bab", "unknown")
        chunks = item.get("chunks", [])
        combined = "\n\n".join(c.get("content", "") for c in chunks)
        n = len(chunks)
        chars = len(combined)
        tokens = int(chars / CHARS_PER_TOKEN)
        results.append((variant, sub_bab, n, chars, tokens))

# Sort by tokens descending to find worst cases
results.sort(key=lambda x: x[4], reverse=True)

print("=== TOP 15 LARGEST combined chunks_text ===")
print(f"{'Variant':<35} {'Sub Bab':<35} {'#C':>3} {'Chars':>6} {'~Tok':>6}")
print("-" * 90)
for variant, sub_bab, n, chars, tokens in results[:15]:
    print(f"{variant:<35} {sub_bab[:34]:<35} {n:>3} {chars:>6} {tokens:>6}")

# Stats
all_tokens = [r[4] for r in results]
print(f"\n=== OVERALL STATS ({len(results)} sub_bab entries) ===")
print(f"  Min combined: {min(all_tokens)} tok")
print(f"  Avg combined: {sum(all_tokens)//len(all_tokens)} tok")
print(f"  P75 combined: {sorted(all_tokens)[int(len(all_tokens)*0.75)]} tok")
print(f"  P90 combined: {sorted(all_tokens)[int(len(all_tokens)*0.90)]} tok")
print(f"  P95 combined: {sorted(all_tokens)[int(len(all_tokens)*0.95)]} tok")
print(f"  Max combined: {max(all_tokens)} tok")

# Fixed parts cost
system_prompt = (PROJECT / "instruction" / "system_prompt.md").read_text(encoding="utf-8")
template = (PROJECT / "instruction" / "materi_task.md").read_text(encoding="utf-8")
fixed_chars = len(system_prompt) + len(template) + 320
fixed_tokens = int(fixed_chars / CHARS_PER_TOKEN)

print(f"\n=== BUDGET TABLE (fixed ~{fixed_tokens} tok, reserved_response=2500 tok) ===")
print(f"{'max_seq':>10} {'chunk_budget':>14} {'fits_all':>10} {'truncated':>10} {'pct_ok':>8}")
for seq in [4096, 6144, 8192]:
    budget = seq - 2500 - fixed_tokens
    fits = sum(1 for t in all_tokens if t <= budget)
    trunc = len(all_tokens) - fits
    pct = fits / len(all_tokens) * 100
    print(f"{seq:>10} {budget:>10} tok {fits:>10} {trunc:>10} {pct:>7.1f}%")
