[TASK]
Generate exactly 5 flashcards in JSON array format.

[OUTPUT_SCHEMA]
[
  {
    "level": "Must be exactly 'LOTS', 'MOTS', or 'HOTS'. Do not use label names like 'Memahami', 'Mengaplikasi', or 'Merefleksi'.",
    "front": "",
    "back": ""
  }
]

[FLASHCARD_LEVEL_CONFIGURATION]
{
  "LOTS": {
    "focus": [
      "definisi",
      "rumus",
      "fakta dasar",
      "konsep inti"
    ],
    "style": "direct recall"
  },

  "MOTS": {
    "focus": [
      "hubungan konsep",
      "aplikasi sederhana",
      "contoh penggunaan"
    ],
    "style": "applied recall"
  },

  "HOTS": {
    "focus": [
      "kesimpulan",
      "strategi",
      "dampak",
      "evaluasi singkat"
    ],
    "style": "evaluative takeaway"
  }
}

[LEARNING_CONTEXT]
{context}

[FLASHCARD_REQUIREMENTS]
- Generate flashcards aligned with the configured flashcard level.
- Keep answers concise, useful, and easy to recall.
- Focus on essential knowledge compression.
- Avoid long analytical explanations.
- Use natural Indonesian educational language.

[LEVEL_ALIGNMENT]

- LOTS:
  Focus on direct recall of concepts, formulas, definitions, or basic facts.
- MOTS:
  Focus on simple application, conceptual relationships, or practical usage.
- HOTS:
  Focus on concise evaluative conclusions, impacts, strategies, or insights.

[FRONT_RULES]
- Maximum 20 characters including spaces.
- Use concise prompts, terms, formulas, or key ideas.
- Avoid ambiguous wording.
- Avoid full sentence questions when unnecessary.

[BACK_RULES]
- Maximum 50 characters including spaces.
- Must contain definitive answers, formulas, concepts, impacts, or conclusions.
- Avoid rhetorical or open-ended responses.
- Avoid filler words.
- STEM subjects:
  - may include concise LaTeX formulas
  - prioritize formulas or technical meaning
- Theory subjects:
  - prioritize impacts, relationships, or concise takeaways

[CONTENT_GUIDELINES]
Allowed content:
- formulas
- definitions
- key concepts
- practical applications
- impacts
- relationships
- concise strategies
- evaluative takeaways

[VARIATION_RULES]
Distribute flashcards naturally across:
- conceptual recall
- applied understanding
- evaluative takeaway

Avoid generating cards with repetitive wording patterns.

[STRICT_OUTPUT_RULES]
- Return raw JSON only.
- No markdown code block.
- No additional commentary.
- Escape newline characters.
- Escape internal quotes.
- The "level" field in the output JSON MUST be exactly the key name ("LOTS", "MOTS", or "HOTS"), NOT the label name (like "Memahami", "Mengaplikasi", or "Merefleksi").