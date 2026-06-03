"""
Configuration — single source of truth for all tunables.

Edit this file to change:
  - which subjects / kelas to pilot
  - which OpenRouter model to use
  - rate-limits, retries, temperature
  - paths to metadata & prompts
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where the curriculum metadata JSONs live
METADATA_DIR = PROJECT_ROOT / "data" / "chunk"

# Where the prompt template files live (loaded at runtime, easy to edit)
PROMPTS_DIR = PROJECT_ROOT / "instruction"

# Where generated SFT output is written
OUTPUT_DIR = PROJECT_ROOT / "output" / "current_experiments"

# ── OpenRouter ─────────────────────────────────────────────────────────
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1/chat/completions"
# Model, temperature, max_tokens — edit src/models.py to change
# pyrefly: ignore [missing-import]
from src.models import MODEL as OPENROUTER_MODEL  
# pyrefly: ignore [missing-import]
from src.models import MAX_TOKENS, TEMPERATURE  

# ── SFT Token Budget ──────────────────────────────────────────────────
# Target max_seq_length for SFT training (system + user + assistant).
# Determines how aggressively chunks are truncated before sending to LLM.
#
# Presets (uncomment one):
#   4096  → QLoRA 24GB safe, but truncates ~50% samples with 3+ chunks
#   6144  → QLoRA 24GB + gradient checkpointing, fits 100% of current data
#   8192  → QLoRA 24GB tight / LoRA 48GB+, headroom for future chunk growth
# MAX_SEQ_LENGTH: int = 4096
# MAX_SEQ_LENGTH: int = 6144
MAX_SEQ_LENGTH: int = 8192

# Tokens reserved for the assistant response.
# Materi tasks need ~2000-2500 tok; pilgan/essay need ~1500-2000 tok.
RESERVED_RESPONSE_TOKENS: int = 3000

# Rough token-to-character ratio for Indonesian text.
# Indonesian averages ~3.5 chars per token on Qwen tokenizers.
# We use 3.2 as a conservative estimate (slightly over-counts tokens).
CHARS_PER_TOKEN: float = 3.2

# ── Rate-limit & Retry ────────────────────────────────────────────────
MAX_RETRIES: int = 5
RETRY_WAIT_MIN: float = 2.0    # seconds
RETRY_WAIT_MAX: float = 30.0   # seconds
REQUEST_DELAY: float = 1.5     # seconds between requests (be nice to the API)

# ── Task ordering ─────────────────────────────────────────────────────
# Tasks are processed sequentially: finish ALL materi, then flashcard, etc.
TASK_TYPES: list[str] = [
    "materi",
    "flashcard",
    "mindmap",
    "pilgan",
    "essay",
    "pretest",
]

# Map task names → prompt file basenames (inside PROMPTS_DIR)
TASK_PROMPT_FILES: dict[str, str] = {
    "materi":    "materi_task.md",
    "flashcard": "flash_card_task.md",
    "mindmap":   "mindmap_task.md",
    "pilgan":    "pilgan_task.md",
    "essay":     "essay_task.md",
    "pretest":   "pretest_task.md",
}

# System prompt file (inside PROMPTS_DIR)
SYSTEM_PROMPT_FILE: str = "system_prompt.md"

# Leveling criteria file (inside PROMPTS_DIR)
LEVELING_CRITERIA_FILE: str = "leveling_criteria.md"

# Prompt versioning — bump when instruction files change
PROMPT_VERSION: str = "v3"
SYSTEM_PROMPT_VERSION: str = "v2"

# ── Cognitive levels ──────────────────────────────────────────────────
# mindmap is exempt from leveling (per orchestration spec)
LEVELS: list[str] = ["LOTS", "MOTS", "HOTS"]
TASKS_WITHOUT_LEVELING: set[str] = {"mindmap", "pretest"}

# ── Available curricula ──────────────────────────────────────────────
AVAILABLE_CURRICULA: list[str] = ["Kurikulum Merdeka", "K-13", "KTSP"]

# ── Pilot subjects ───────────────────────────────────────────────────
# To run the full dataset, set PILOT_SUBJECTS to None
PILOT_SUBJECTS: list[dict] | None = [
    # Kurikulum Merdeka
    {"kurikulum": "Kurikulum Merdeka", "jenjang": "SMA", "kelas": "Kelas 10",
     "mata_pelajaran": "IPS"},
    {"kurikulum": "Kurikulum Merdeka", "jenjang": "SMA", "kelas": "Kelas 10",
     "mata_pelajaran": "Bahasa Indonesia"},
    {"kurikulum": "Kurikulum Merdeka", "jenjang": "SMA", "kelas": "Kelas 10",
     "mata_pelajaran": "Matematika Umum"},
    # K-13
    {"kurikulum": "K-13", "jenjang": "SMA", "kelas": "Kelas 10",
     "mata_pelajaran": "IPS"},
    {"kurikulum": "K-13", "jenjang": "SMA", "kelas": "Kelas 10",
     "mata_pelajaran": "Bahasa Indonesia"},
    {"kurikulum": "K-13", "jenjang": "SMA", "kelas": "Kelas 10",
     "mata_pelajaran": "Matematika Umum"},
    # KTSP
    {"kurikulum": "KTSP", "jenjang": "SMA", "kelas": "Kelas 10",
     "mata_pelajaran": "IPS"},
    {"kurikulum": "KTSP", "jenjang": "SMA", "kelas": "Kelas 10",
     "mata_pelajaran": "Bahasa Indonesia"},
    {"kurikulum": "KTSP", "jenjang": "SMA", "kelas": "Kelas 10",
     "mata_pelajaran": "Matematika Umum"},
]

# ── Test mode ─────────────────────────────────────────────────────────
# In test mode, only the first sub_bab of each subject is processed.
TEST_MODE: bool = False
