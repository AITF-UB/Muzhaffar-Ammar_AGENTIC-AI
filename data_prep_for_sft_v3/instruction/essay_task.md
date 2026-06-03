[TASK]
Generate EXACTLY 5 essay questions in JSON array format.
Never generate more or fewer.

[OUTPUT_SCHEMA]
[
  {
    "level": "Must be exactly 'LOTS', 'MOTS', or 'HOTS'. Do not use label names like 'Memahami', 'Mengaplikasi', or 'Merefleksi'.",
    "stimulus": "",
    "question": "",
    "rubric_points": [],
    "explanation": ""
  }
]

[LEVEL_CONFIGURATION]
{COMPILED_LEVEL_CONFIGURATION}

[SUBJECT_CONFIGURATION]
{COMPILED_SUBJECT_CONFIGURATION}

[LEARNING_CONTEXT]
{context}

[ATP_ALIGNMENT]
{atp}

Prioritize ATP objectives as curriculum constraints.

[QUESTION_REQUIREMENTS]
- Questions must remain contextually grounded.
- Use one complete sentence only.
- Length: 20-50 words.

[STIMULUS_REQUIREMENTS]
- Length: 50-150 words.
- Use realistic Indonesian contexts relevant to the subject domain.
- Include sufficient contextual information to support reasoning.
- Allowed formats:
  - narrative
  - bullet list
  - markdown table
  - mixed format

[RUBRIC_REQUIREMENTS]
- Minimum 3 rubric points.
- Use operational assessment indicators.
- Keep each rubric concise and measurable.

[EXPLANATION_REQUIREMENTS]
- Length: 50-150 words.
- Start with a short bold title.
- Mathematics/science:
  - show calculations vertically using LaTeX
- Theory subjects:
  - explain core cause-effect reasoning
  - explain key analytical points

- End with:
  "Refleksi Nyata"

[STRICT_OUTPUT_RULES]
- Return raw JSON only.
- No markdown.
- No additional commentary.
- The "level" field in the output JSON MUST be exactly the key name ("LOTS", "MOTS", or "HOTS"), NOT the label name (like "Memahami", "Mengaplikasi", or "Merefleksi").