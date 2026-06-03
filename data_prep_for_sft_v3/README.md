# SFT Data Generator — Sekolah Rakyat

Generate Supervised Fine-Tuning (SFT) data for educational content using OpenRouter LLM API.

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Add your API key
#    Edit .env and replace the placeholder with your OpenRouter key
#    Get one at: https://openrouter.ai/keys

# 3. Choose your model
#    Edit src/models.py to select which model to use

# 4. Preview what will be generated
uv run main.py --dry-run

# 5. Run in test mode (1 sub_bab per subject — cheap & fast)
uv run main.py --test

# 6. Run full production
uv run main.py
```

## CLI Flags

| Flag | Description |
|------|-------------|
| `--task <name>` | Generate only specific task(s). Repeatable. Options: `materi`, `flashcard`, `mindmap`, `pilgan`, `essay`, `pretest` |
| `--variant <name>` | Filter by specific chunk variant (e.g. `1_chunks`). Repeatable. |
| `--kurikulum <name>` | Filter by curriculum. Repeatable. Options: `"Kurikulum Merdeka"`, `"K-13"`, `KTSP` |
| `--subject <name>` | Filter by mata_pelajaran. Repeatable. |
| `--kelas <name>` | Filter by kelas. Repeatable. |
| `--jenjang <name>` | Filter by jenjang. Repeatable. |
| `--test` | Test mode — only 1 sub_bab per subject |
| `--dry-run` | Preview the plan without API calls |
| `--list-subjects` | List all available subjects across all curricula |
| `--fail-fast-network` | Stop early when OpenRouter is unreachable (DNS/network/timeout) after retry/backoff |

### Examples

```bash
# Only generate materi
uv run main.py --task materi

# Generate materi + flashcard in test mode
uv run main.py --task materi --task flashcard --test

# Only K-13 curriculum
uv run main.py --kurikulum "K-13" --test

# KTSP + specific subject
uv run main.py --kurikulum KTSP --subject IPS --test

# Full pipeline, all tasks
uv run main.py
```

## Project Structure

```
├── .env                     # ← Your API key (gitignored)
├── main.py                  # CLI entry point
├── src/
│   ├── config.py            # Paths, pilot subjects, rate-limits
│   ├── models.py            # ← Model selection (edit this!)
│   ├── client.py            # OpenRouter HTTP client + retry
│   ├── prompt_builder.py    # Loads prompts from instruction/*.md
│   ├── metadata_loader.py   # Discovers & filters curriculum JSONs
│   └── pipeline.py          # Orchestration engine
├── instruction/             # Prompt templates (edit these!)
│   ├── system_prompt.md
│   ├── leveling_criteria.md
│   ├── materi_task.md
│   ├── flash_card_task.md
│   ├── mind_map_task.md
│   ├── pilgan_task.md
│   └── essay_task.md
├── data/                    # Input metadata & chunk text
│   └── chunk/
│       ├── Kurikulum Merdeka/SMA/Kelas {10,11,12}/*_chunk.json
│       ├── K-13/SMA/Kelas {10,11,12}/*_chunk.json
│       └── KTSP/SMA/Kelas {10,11,12}/*_chunk.json
└── output/                  # Generated SFT data (auto-created)
    └── Kurikulum Merdeka/SMA/Kelas 10/{Subject}/
        └── {Sub_Bab}/
            ├── materi/
            │   ├── LOTS/
            │   │   ├── 1_chunks.json
            │   │   └── 2_chunks.json
            │   └── MOTS/
            ├── flashcard/
            └── mindmap/
                └── 1_chunks.json  # no leveling
```

## Key Config Files

| File | What to edit |
|------|-------------|
| `.env` | `OPENROUTER_API_KEY` |
| `src/models.py` | Model name, temperature, max_tokens |
| `src/config.py` | Pilot subjects, rate-limits, paths |
| `instruction/*.md` | Prompt templates — loaded at runtime |

## How It Works

1. **Data** is read from `data/chunk/` — supports multiple curricula (Kurikulum Merdeka, K-13, KTSP).
2. **Prompts** are loaded from `instruction/*.md` files and filled with chunk content, target cognitive level criteria, and **dynamic stimulus rules** from `instruction/stimulus.md` (matched by subject).
3. **Tasks** run in strict order: materi → flashcard → mindmap → pilgan → essay → pretest.
4. **Leveling**: Each task (except mindmap/pretest) runs 3× per entry: LOTS, MOTS, HOTS.
5. **Output** is saved in **TRL messages format** (`{"meta": ..., "messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`). Re-running skips existing files (resume-safe).
6. **Token Budget Retry**: If total estimated tokens exceed 8192, the pipeline retries with a compressed prompt (60% chunk budget + concise instruction). If still over budget, saves with a `budget_exceeded` flag.

## Output Format (TRL Messages)

Each JSON file uses the HuggingFace TRL `messages` format for SFT training:

```json
{
  "meta": {
    "kurikulum": "K-13",
    "jenjang": "SMA",
    "kelas": "Kelas 10",
    "mata_pelajaran": "IPS",
    "bab_judul": "Sejarah dan Geografi",
    "sub_bab": "Kajian Ilmu Sejarah",
    "task_type": "materi",
    "level": "LOTS",
    "source_file": "IPS_1_chunk.json",
    "est_tokens": {
      "system": 734,
      "user": 1461,
      "assistant": 2500,
      "total": 4695,
      "max_seq_length": 8192,
      "exceeds_budget": false
    }
  },
  "messages": [
    {
      "role": "system",
      "content": "[IDENTITY]\nYou are an expert Content Specialist..."
    },
    {
      "role": "user",
      "content": "[KRITERIA_PEMBELAJARAN_MENDALAM]\n..."
    },
    {
      "role": "assistant",
      "content": "{\"judul_utama\": \"...\", \"konten_markdown\": \"...\"}"
    }
  ]
}
```

> **Note**: The `meta` block is for traceability/filtering only — strip it before training. The `messages` array is directly compatible with HuggingFace `SFTTrainer`.
```bash
# Filter by curriculum
uv run main.py --kurikulum "Kurikulum Merdeka" --test

# Filter by subject only
uv run main.py --subject Sosiologi --test

# Filter by kelas
uv run main.py --kelas "Kelas 12" --test

# Filter by jenjang
uv run main.py --jenjang SMA --test

# Filter by chunk variant
uv run main.py --variant 1_chunks --test

# Combine filters (AND logic — all must match)
uv run main.py --kurikulum "Kurikulum Merdeka" --subject "Matematika Umum" --kelas "Kelas 10" --jenjang SMA --variant 1_chunks --test

# Multiple values per filter (OR within same filter)
uv run main.py --subject Sosiologi --subject "Bahasa Indonesia" --kelas "Kelas 12"

# List everything available
uv run main.py --list-subjects

# Preview plan
uv run main.py --subject Sosiologi --dry-run
```


# Pembagian

Ilham = Matematika

command:

uv run main.py --kurikulum "Kurikulum Merdeka" --subject "Matematika Umum" --kelas "Kelas 10" --jenjang SMA --variant 1_chunks

uv run main.py --kurikulum "Kurikulum Merdeka" --subject "Matematika Umum" --kelas "Kelas 10" --jenjang SMA --variant 2_chunks 

uv run main.py --kurikulum "Kurikulum Merdeka" --subject "Matematika Umum" --kelas "Kelas 10" --jenjang SMA --variant 3_chunks 

uv run main.py --kurikulum "Kurikulum Merdeka" --subject "Matematika Umum" --kelas "Kelas 10" --jenjang SMA --variant 4_chunks 

uv run main.py --kurikulum "Kurikulum Merdeka" --subject "Matematika Umum" --kelas "Kelas 10" --jenjang SMA --variant 5_chunks 

uv run main.py --kurikulum "Kurikulum Merdeka" --subject "Matematika Umum" --kelas "Kelas 10" --jenjang SMA --variant 6_chunks 


Akbar = IPS

command: 

uv run main.py --kurikulum "Kurikulum Merdeka" --subject "IPS" --kelas "Kelas 10" --jenjang SMA --variant 1_chunks

uv run main.py --kurikulum "Kurikulum Merdeka" --subject "IPS" --kelas "Kelas 10" --jenjang SMA --variant 2_chunks 

uv run main.py --kurikulum "Kurikulum Merdeka" --subject "IPS" --kelas "Kelas 10" --jenjang SMA --variant 3_chunks 

uv run main.py --kurikulum "Kurikulum Merdeka" --subject "IPS" --kelas "Kelas 10" --jenjang SMA --variant 4_chunks 

Zara = Bahasa Indonesia

uv run main.py --kurikulum "Kurikulum Merdeka" --subject "Bahasa Indonesia" --kelas "Kelas 10" --jenjang SMA --variant 1_chunks

uv run main.py --kurikulum "Kurikulum Merdeka" --subject "Bahasa Indonesia" --kelas "Kelas 10" --jenjang SMA --variant 2_chunks 

uv run main.py --kurikulum "Kurikulum Merdeka" --subject "Bahasa Indonesia" --kelas "Kelas 10" --jenjang SMA --variant 3_chunks


Note: Jika beberapa task gagal, silahkan retry nanti otomatis bakal continues task yang gagal. Bisa dilihat di folder log atau ouput terminal.

Tip: Kalau yang gagal adalah error network (mis. `getaddrinfo failed`, `network unreachable`, atau timeout), jalankan dengan `--fail-fast-network` supaya pipeline berhenti lebih cepat dan bisa di-resume ketika koneksi sudah normal.

