# Chunk Grounding Rule — Suggestions

## The Problem

Currently the RAG chunks from `{chunks}` are injected as `--- MATERI ---` inside `[KONTEKS_PEMBELAJARAN]`. Without an explicit grounding rule, the LLM will either:

1. **Copy-paste** the textbook content almost verbatim (lazy/low quality), or
2. **Hallucinate** freely and ignore the chunk entirely (inaccurate)

You need a rule that sits in the middle: **use the chunk as the factual spine, but rewrite with pedagogical value-add**.

---

## Suggested Rules (Pick One or Combine)

### Option A — "Anchor & Enrich" (Recommended)

```
- Source Fidelity: Treat {chunks} as your SINGLE factual anchor.
  Every core claim, data point, and terminology MUST originate from {chunks}.
  DO NOT invent facts, statistics, or expert quotes absent from the source.
  However, NEVER copy sentences verbatim — rewrite all content with:
  (1) cause-effect reasoning the source doesn't explicitly state,
  (2) real Indonesian contextual examples that extend the source material,
  (3) vocabulary and sentence complexity aligned to the target cognitive level.
```

**Why this works:**
- "SINGLE factual anchor" — forces the LLM to stay on-topic and prevents hallucinated facts
- "NEVER copy verbatim" — hard prohibition on lazy parroting
- The 3 sub-points define *how* to improve: reasoning, context, and level-appropriate language
- Aligns with your existing `[BEHAVORIAL_PILLARS]` (Meaningful = cause-effect, Depth = level alignment)

---

### Option B — "Transform, Don't Transfer"

```
- Source Grounding: {chunks} is your factual boundary — all content MUST be
  traceable to the provided material. PROHIBITED: fabricating data, names, or
  events not present in {chunks}.
  Transform the source by: simplifying OR deepening explanations to match
  [KRITERIA_PEMBELAJARAN_MENDALAM], adding cause-effect chains, and connecting
  concepts to real Indonesian issues. The output must feel like a mentor
  explaining the topic, NOT a textbook page being recited.
```

**Why this works:**
- "Factual boundary" is a strong mental model — think of it as a fence
- "Traceable" is audit-friendly language (important for SFT data quality)
- "Mentor explaining, NOT textbook recited" gives a clear persona contrast
- Leverages existing `[STYLE_TONE]` persona directive

---

### Option C — "Minimal & Strict" (if you want fewer tokens)

```
- Chunk Rule: Ground ALL facts in {chunks}. Zero fabrication.
  Rewrite with added reasoning, local context, and level-appropriate depth.
  Verbatim copying = failure.
```

**Why this works:**
- Ultra-compact (saves tokens in system prompt)
- Still covers the three critical dimensions: fidelity, enrichment, anti-copy

---

## Where to Place It

Add it inside `[RULES]` in [system_prompt.md](file:///c:/Local%20D/Galeri%20Belajar/Project/SR_02/data_prep_for_sft_v3/instruction/system_prompt.md), after the existing rules. Example placement:

```diff
 [RULES]
 - Zero Noise: Output ONLY raw JSON. No intros/outros.
 - LaTeX Tech: Double backslash MANDATORY (\\\\times, \\\\frac). Use $ (inline) or $$ (display).
 - JSON Safety: Use \\n for new lines and \\" for internal quotes.
 - Zero Labeling: DO NOT mention framework labels (LOTS, HOTS, SCQ, etc) in narrative text.
+- Source Fidelity: Treat {chunks} as your SINGLE factual anchor. Every core claim, data point, and terminology MUST originate from {chunks}. DO NOT invent facts, statistics, or expert quotes absent from the source. However, NEVER copy sentences verbatim — rewrite all content with: (1) cause-effect reasoning the source doesn't explicitly state, (2) real Indonesian contextual examples that extend the source material, (3) vocabulary and sentence complexity aligned to the target cognitive level.
```

> [!TIP]
> I recommend **Option A** — it's the best balance of clarity, token efficiency, and alignment with your existing Behavioral Pillars. Option B is more verbose but might be clearer for weaker models. Option C is good if you're optimizing for input token cost.

## What Each Option Prevents

| Failure Mode | Option A | Option B | Option C |
|---|---|---|---|
| Verbatim copy-paste | ✅ "NEVER copy verbatim" | ✅ "NOT textbook recited" | ✅ "Verbatim = failure" |
| Hallucinated facts | ✅ "SINGLE factual anchor" | ✅ "Factual boundary" | ✅ "Zero fabrication" |
| Flat/textbook tone | ✅ Cause-effect + examples | ✅ "Mentor explaining" | ⚠️ Implied only |
| Level mismatch | ✅ "Aligned to target level" | ✅ "Match KRITERIA" | ✅ "Level-appropriate" |
