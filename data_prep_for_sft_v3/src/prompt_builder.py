"""
Prompt builder — loads prompts from disk and fills placeholders.

Each prompt template lives in its own .md file under `instruction/`.
This module reads them at runtime so you can edit prompts without touching code.

Token-optimized:
  - Injects ONLY the target level criteria (not all 3)
  - Context block uses chunk content directly (no redundant identity fields)
  - Standardized {LEVELING_CRITERIA} placeholder across all task files
  - Chunk truncation to fit within SFT max_seq_length budget
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path

# pyrefly: ignore [missing-import]
from src.config import (
    CHARS_PER_TOKEN,
    LEVELING_CRITERIA_FILE,
    MAX_SEQ_LENGTH,
    PROMPTS_DIR,
    RESERVED_RESPONSE_TOKENS,
    SYSTEM_PROMPT_FILE,
    TASK_PROMPT_FILES,
)

STIMULUS_FILE: str = "stimulus.md"

log = logging.getLogger(__name__)


def _read(filename: str) -> str:
    """Read a prompt file from PROMPTS_DIR, strip trailing whitespace."""
    path = PROMPTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_system_prompt() -> str:
    """Return the full system prompt text."""
    return _read(SYSTEM_PROMPT_FILE)


def _load_leveling_dict() -> dict:
    """Parse the leveling criteria file as a Python dict literal."""
    raw = _read(LEVELING_CRITERIA_FILE)
    try:
        # Handle "VAR_NAME = {...}" format — strip everything before first "{"
        brace_idx = raw.find("{")
        if brace_idx > 0:
            raw = raw[brace_idx:]
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {}


def format_all_level_criteria() -> str:
    """Return a combined criteria string for ALL levels (used for pretest)."""
    criteria = _load_leveling_dict()
    if not criteria:
        return "LEVELS: LOTS, MOTS, HOTS"
    
    parts = []
    for lvl, entry in criteria.items():
        # Support both old keys (fokus/kedalaman) and new keys (cognitive_focus/reasoning_depth)
        fokus = entry.get("cognitive_focus") or entry.get("fokus", [])
        kedalaman = entry.get("reasoning_depth") or entry.get("kedalaman", "")
        parts.append(
            f"LEVEL: {lvl} ({entry.get('label', lvl)})\n"
            f"Taxonomy: {entry.get('taxonomy', {})}\n"
            f"Fokus: {fokus}\n"
            f"Kedalaman: {kedalaman}"
        )
    return "\n\n".join(parts)


def format_level_criteria(level: str) -> str:
    """
    Return a compact, human-readable criteria string for ONE level only.
    This avoids injecting all 3 levels into every prompt.
    Supports both old keys (fokus/kedalaman) and new keys (cognitive_focus/reasoning_depth).
    """
    criteria = _load_leveling_dict()
    entry = criteria.get(level)
    if not entry:
        return f"LEVEL: {level}"

    # Support both old and new key names
    fokus = entry.get("cognitive_focus") or entry.get("fokus", [])
    kedalaman = entry.get("reasoning_depth") or entry.get("kedalaman", "")

    return (
        f"LEVEL: {level} ({entry.get('label', level)})\n"
        f"Taxonomy: {entry.get('taxonomy', {})}\n"
        f"Fokus: {fokus}\n"
        f"Kedalaman: {kedalaman}"
    )


# ── Dynamic Stimulus ─────────────────────────────────────────────────

def _load_registry_raw(filename: str) -> str:
    """
    Read a registry file and return the full raw text (variable assignment stripped).
    Used as a fallback / for pretest which needs all levels.
    """
    path = PROMPTS_DIR / filename
    if not path.exists():
        log.warning("Registry file not found: %s", path)
        return "{}"
    raw = path.read_text(encoding="utf-8").strip()
    # Strip "VARIABLE_NAME = " prefix if present
    brace_idx = raw.find("{")
    if brace_idx > 0:
        raw = raw[brace_idx:]
    return raw


# Task-specific leveling registry compilation rules [Orch Rules 1, 2, 6]
LEVEL_FIELDS_BY_TASK = {
    "pilgan": ["dominant_operations", "question_behavior", "preferred_question_patterns", "avoid_patterns", "reasoning_priority"],
    "pretest": ["dominant_operations", "question_behavior", "preferred_question_patterns", "avoid_patterns", "reasoning_priority"],
    "materi": ["dominant_operations", "reasoning_priority", "expected_reasoning", "instruction_style"],
    "mindmap": ["dominant_operations", "reasoning_priority"],
    "essay": ["dominant_operations", "question_behavior", "preferred_question_patterns", "avoid_patterns", "reasoning_priority", "expected_reasoning"],
}

# Task-specific subject registry compilation rules [Orch Rules 1, 2, 6]
SUBJECT_FIELDS_BY_TASK = {
    "pilgan": ["required_elements", "reasoning_space", "writing_rules"],
    "pretest": ["required_elements", "reasoning_space", "writing_rules"],
    "materi": ["allowed_contexts", "reasoning_space", "writing_rules"],
    "mindmap": ["required_elements", "reasoning_space", "writing_rules"],
    "essay": ["required_elements", "reasoning_space", "writing_rules"],
}


def compile_leveling_registry(task_type: str, level: str | None = None) -> str:
    """
    Dynamically compile leveling registry based on the active task to save tokens.
    Extracts only "generation_behavior" and drops pedagogical/taxonomy metadata.
    Implements [RULE_1_LEVEL_COMPILATION] and [RULE_6_TASK_DEPENDENT_COMPILATION].
    """
    if task_type == "flashcard":
        return "{}"

    registry = _load_leveling_dict()
    if not registry:
        return "{}"

    # If a specific level is requested, filter just for that level
    levels_to_process = [level] if level else list(registry.keys())
    compiled_registry = {}

    # Get fields to keep for this task
    keep_fields = LEVEL_FIELDS_BY_TASK.get(task_type)

    for lvl in levels_to_process:
        entry = registry.get(lvl)
        if not entry:
            continue

        compiled_entry = {}
        # Keep label if present (needed to know label mapping, e.g. "label": "Memahami")
        if "label" in entry:
            compiled_entry["label"] = entry["label"]

        # Only extract the generation_behavior dict
        gen_behavior = entry.get("generation_behavior", {})
        if gen_behavior:
            if keep_fields:
                filtered_gen = {k: v for k, v in gen_behavior.items() if k in keep_fields}
                compiled_entry["generation_behavior"] = filtered_gen
            else:
                compiled_entry["generation_behavior"] = gen_behavior
        else:
            # Fallback if structure is flat (old style)
            fallback_fields = ["reasoning_depth", "cognitive_focus", "question_behavior"]
            for field in (keep_fields or fallback_fields):
                if field in entry:
                    compiled_entry[field] = entry[field]

        compiled_registry[lvl] = compiled_entry

    return json.dumps(compiled_registry, ensure_ascii=False, indent=2)


def compile_subject_registry(task_type: str, mata_pelajaran: str) -> str:
    """
    Dynamically compile subject configuration based on the active task and subject to save tokens.
    Implements [RULE_2_SUBJECT_COMPILATION] and [RULE_6_TASK_DEPENDENT_COMPILATION].
    """
    rules = _load_stimulus_rules()
    if not rules:
        return "{}"

    full_key = _normalize_subject_key(mata_pelajaran)

    # Lookup target key
    target_key = None
    if full_key in rules:
        target_key = full_key
    else:
        first_word = full_key.split("_")[0]
        if first_word in rules:
            target_key = first_word

    if not target_key:
        log.warning(
            "No stimulus entry for '%s' (tried: '%s', '%s'). Using empty.",
            mata_pelajaran, full_key, full_key.split("_")[0] if full_key else "",
        )
        return "{}"

    subject_entry = rules[target_key]
    if not isinstance(subject_entry, dict):
        # Fallback if it is not a dictionary
        return json.dumps({target_key: subject_entry}, ensure_ascii=False, indent=2)

    compiled_entry = {}

    # Get fields to keep for this task
    keep_fields = SUBJECT_FIELDS_BY_TASK.get(
        task_type, ["required_elements", "reasoning_space", "writing_rules"]
    )

    for field in keep_fields:
        if field in subject_entry:
            compiled_entry[field] = subject_entry[field]

    return json.dumps({target_key: compiled_entry}, ensure_ascii=False, indent=2)


def _load_stimulus_rules() -> dict:
    """Load stimulus.md as a Python dict (via ast.literal_eval) for subject lookups."""
    path = PROMPTS_DIR / STIMULUS_FILE
    if not path.exists():
        log.warning("stimulus.md not found at %s", path)
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    try:
        brace_idx = raw.find("{")
        if brace_idx > 0:
            raw = raw[brace_idx:]
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        log.warning("Failed to parse stimulus.md as Python dict")
        return {}


def _normalize_subject_key(mata_pelajaran: str) -> str:
    """
    Normalize mata_pelajaran to a stimulus key.

    Examples:
      'Matematika Umum' → 'matematika'
      'Bahasa Indonesia' → 'bahasa_indonesia'
      'IPS' → 'ips'
      'Matematika' → 'matematika'
    """
    key = mata_pelajaran.lower().strip()
    # Try full form first: "bahasa indonesia" → "bahasa_indonesia"
    full_key = key.replace(" ", "_")
    return full_key


def resolve_stimulus(mata_pelajaran: str) -> str:
    """
    Return the stimulus rules for a given subject from stimulus.md as a string.

    Lookup strategy:
      1. Exact match on full normalized key (e.g. 'bahasa_indonesia')
      2. First-word match (e.g. 'matematika_umum' → 'matematika')
      3. Generic fallback

    Always returns a str — if the registry entry is a dict, it is serialized to
    a JSON string so it can be safely injected into the legacy {stimulus} placeholder.
    """
    rules = _load_stimulus_rules()
    if not rules:
        return (
            "- Wajib menyajikan stimulus berupa fenomena atau data riil Indonesia.\n"
            "- Konten harus memicu proses kognitif terarah.\n"
            "- Teks wajib berbasis contoh kasus kontekstual."
        )

    full_key = _normalize_subject_key(mata_pelajaran)

    # 1. Exact match
    if full_key in rules:
        result = rules[full_key]
        return json.dumps(result, ensure_ascii=False, indent=2) if isinstance(result, dict) else str(result)

    # 2. First-word match (e.g. 'matematika_umum' → 'matematika')
    first_word = full_key.split("_")[0]
    if first_word in rules:
        result = rules[first_word]
        return json.dumps(result, ensure_ascii=False, indent=2) if isinstance(result, dict) else str(result)

    # 3. Generic fallback
    log.warning(
        "No stimulus rule for '%s' (tried keys: '%s', '%s'). Using generic.",
        mata_pelajaran, full_key, first_word,
    )
    return (
        "- Wajib menyajikan stimulus berupa fenomena atau data riil Indonesia.\n"
        "- Konten harus memicu proses kognitif terarah.\n"
        "- Teks wajib berbasis contoh kasus kontekstual."
    )


# ── Token Budget ──────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Estimate token count from character length using CHARS_PER_TOKEN ratio."""
    return int(len(text) / CHARS_PER_TOKEN)


def _truncate_to_budget(
    chunks_text: str,
    fixed_chars: int,
    sub_bab: str = "",
) -> str:
    """
    Truncate chunks_text so the total SFT sample fits within MAX_SEQ_LENGTH.

    Token budget breakdown:
      MAX_SEQ_LENGTH = system_prompt + user_prompt + assistant_response
                     = fixed_parts + chunks_text + RESERVED_RESPONSE_TOKENS

    If chunks_text fits within the remaining budget, it is returned unchanged.
    If it exceeds the budget, it is truncated at the last sentence boundary
    (period followed by space) and a [TRUNCATED] marker is appended.

    Parameters
    ----------
    chunks_text : str
        The raw chunk content to potentially truncate.
    fixed_chars : int
        Character count of all non-chunk parts (system + template + header + criteria).
    sub_bab : str
        Used for logging when truncation happens.
    """
    if not chunks_text:
        return chunks_text

    # Calculate remaining character budget for chunks
    fixed_tokens = int(fixed_chars / CHARS_PER_TOKEN)
    remaining_tokens = MAX_SEQ_LENGTH - RESERVED_RESPONSE_TOKENS - fixed_tokens

    if remaining_tokens <= 0:
        log.warning(
            "[TokenBudget] No room for chunks in '%s'. "
            "Fixed parts already use %d tokens (budget: %d - %d reserved).",
            sub_bab, fixed_tokens, MAX_SEQ_LENGTH, RESERVED_RESPONSE_TOKENS,
        )
        return ""

    max_chunk_chars = int(remaining_tokens * CHARS_PER_TOKEN)

    if len(chunks_text) <= max_chunk_chars:
        return chunks_text

    # Truncate at last sentence boundary
    truncated = chunks_text[:max_chunk_chars]
    last_period = truncated.rfind(". ")
    if last_period > max_chunk_chars * 0.5:  # only cut at sentence if >50% kept
        truncated = truncated[: last_period + 1]

    original_tok = _estimate_tokens(chunks_text)
    truncated_tok = _estimate_tokens(truncated)
    log.warning(
        "[TokenBudget] Truncated chunks for '%s': %d → %d tokens (-%d tok, budget: %d tok)",
        sub_bab, original_tok, truncated_tok,
        original_tok - truncated_tok, remaining_tokens,
    )

    return truncated + "\n[TRUNCATED]"


# ── Prompt Assembly ───────────────────────────────────────────────────

def build_user_prompt(
    task_type: str,
    metadata: dict,
    level: str | None = None,
) -> str:
    """
    Build a complete user prompt for a given task.

    1. Load the raw prompt template from disk.
    2. Format leveling criteria for the specific target level only.
    3. Inject the chunk content as [KONTEKS_PEMBELAJARAN].
    4. Fill in identity placeholders.
    5. Truncate chunks if total exceeds SFT token budget.

    Parameters
    ----------
    task_type : str
        One of: materi, flashcard, mindmap, pilgan, essay
    metadata : dict
        A single sub_bab entry from the chunk JSON (with chunks_text).
    level : str | None
        LOTS / MOTS / HOTS. None for mindmap (no leveling).
    """
    filename = TASK_PROMPT_FILES.get(task_type)
    if not filename:
        raise ValueError(f"Unknown task type: {task_type}")

    template = _read(filename)
    system_prompt_text = load_system_prompt()

    # ── Build leveling criteria (only target level) ──
    if level:
        criteria_text = format_level_criteria(level)
    else:
        if task_type == "pretest":
            criteria_text = format_all_level_criteria()
        else:
            criteria_text = ""

    # ── Minimal identity header ──
    header = (
        f"Mapel: {metadata.get('mata_pelajaran', '')} | "
        f"Bab: {metadata.get('bab_judul', '')} | "
        f"Sub Bab: {metadata.get('sub_bab', '')}"
    )
    if level:
        header += f" | Target: {level}"

    # ── Calculate fixed-part character count for token budget ──
    # Fixed parts = system prompt + template (without {context}) + header + criteria
    fixed_chars = len(system_prompt_text) + len(template) + len(header) + len(criteria_text)

    # ── Truncate chunks to fit within SFT token budget ──
    chunks_text = metadata.get("chunks_text", "")
    chunks_text = _truncate_to_budget(
        chunks_text,
        fixed_chars,
        sub_bab=metadata.get("sub_bab", ""),
    )

    # ── Assemble context block ──
    context_parts = [header]

    if chunks_text:
        context_parts.append(f"\n--- MATERI ---\n{chunks_text}")

    context_block = "\n".join(context_parts)

    # ── Perform placeholder substitutions ──
    prompt = template

    # Replace identity fields
    prompt = prompt.replace("{kurikulum_asal}", metadata.get("kurikulum", "Kurikulum Merdeka"))
    prompt = prompt.replace("{kelas}", metadata.get("kelas", ""))
    prompt = prompt.replace("{mapel}", metadata.get("mata_pelajaran", ""))
    prompt = prompt.replace("{sub_bab}", metadata.get("sub_bab", ""))

    # ── Registry placeholders: inject ONLY the relevant level + subject slice ──
    # This avoids injecting all 3 levels and all subjects into every prompt,
    # saving ~1000–2000 tokens per sample.
    #
    # Exception: pretest uses all levels (LOTS/MOTS/HOTS distribution).
    mata_pelajaran = metadata.get("mata_pelajaran", "")

    # ── Registry placeholders: dynamically compile only task-relevant fields ──
    # Saves substantial tokens while maintaining pedagogical alignment (Orch Rules 1, 2, 6)
    leveling_registry_raw = compile_leveling_registry(task_type, level)
    stimulus_registry_raw = compile_subject_registry(task_type, mata_pelajaran)

    prompt = prompt.replace("{LEVELING_REGISTRY}", leveling_registry_raw)
    prompt = prompt.replace("{COMPILED_LEVEL_CONFIGURATION}", leveling_registry_raw)
    prompt = prompt.replace("{STIMULUS_REGISTRY}", stimulus_registry_raw)
    prompt = prompt.replace("{COMPILED_SUBJECT_CONFIGURATION}", stimulus_registry_raw)

    # ── Legacy-style placeholders (kept for backward compat) ──
    # Inject dynamic stimulus rules based on subject
    stimulus_text = resolve_stimulus(metadata.get("mata_pelajaran", ""))
    prompt = prompt.replace("{stimulus}", stimulus_text)

    # Replace level placeholder
    if level:
        prompt = prompt.replace("{LOTS/MOTS/HOTS}", level)
        prompt = prompt.replace("{Target Leveling}", level)
    else:
        prompt = prompt.replace("{LOTS/MOTS/HOTS}", "")
        prompt = prompt.replace("{Target Leveling}", "")

    # Inject leveling criteria — single standardized placeholder (legacy)
    prompt = prompt.replace("{LEVELING_CRITERIA}", criteria_text)

    # ── ATP injection — standalone {atp} placeholder ──
    # Used by [ATP_ALIGNMENT] sections in task files.
    # Falls back to a clear "not available" note so the LLM ignores it gracefully.
    atp_text = metadata.get("atp", "").strip()
    atp_block = atp_text if atp_text else "ATP tidak tersedia untuk sub-bab ini."
    prompt = prompt.replace("{atp}", atp_block)

    # Inject context
    prompt = prompt.replace("{context}", context_block)

    return prompt
