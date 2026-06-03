[TASK]
Generate learning material in JSON format.

[OUTPUT_SCHEMA]
{
  "judul_utama": "",
  "konten_markdown": ""
}

[LEVEL_CONFIGURATION]
{COMPILED_LEVEL_CONFIGURATION}

[SUBJECT_CONFIGURATION]
{COMPILED_SUBJECT_CONFIGURATION}

[LEARNING_CONTEXT]
{context}

[ATP_ALIGNMENT]
{atp}

Prioritize ATP objectives as curriculum constraints.

[CONTENT_REQUIREMENTS]
- Generate a minimum of 500 words.
- Use natural Indonesian educational language suitable for SMA students.
- Maintain strong contextual relevance to Indonesian realities.
- Build clear cause-effect explanations.
- Prioritize practical understanding before abstraction.

[CONTENT_STRUCTURE]
The material should flow naturally through these sections:

- contextual opening
- concept explanation
- technical explanation or formulation
- real Indonesian case study
- reflective closing

Use markdown subheadings with:
### Heading

Do not use:
- numbering
- rigid section labels
- colon symbols in headings

[OPENING_REQUIREMENTS]
- Start with realistic Indonesian situations, problems, or numerical/social phenomena.
- Build curiosity naturally through contextual tension.
- Avoid rhetorical greetings or motivational filler.

[CONCEPT_EXPLANATION]
- Explain concepts gradually and logically.
- Connect concepts with Indonesian daily life situations.
- Avoid abstract academic wording.

[TECHNICAL_REQUIREMENTS]
- STEM subjects:
  - include formulas using LaTeX
  - explain variables clearly
  - use escaped LaTeX:
    \\\\frac
    \\\\times
    \\\\sqrt

- Theory subjects:
  - explain conceptual relationships
  - strengthen causal understanding

[CASE_STUDY_REQUIREMENTS]
- Use realistic Indonesian cases:
  - UMKM
  - pertanian
  - distribusi
  - sekolah
  - lingkungan sosial
  - ekonomi keluarga
  - industri lokal

- Demonstrate step-by-step reasoning or analysis.
- Show practical application of concepts.

[REFLECTION_REQUIREMENTS]
- End with reflective guidance related to learning awareness or daily application.
- Reflection must remain grounded and realistic.
- Avoid overly philosophical narration.

[TITLE_REQUIREMENTS]
- Combine the main concept and Indonesian contextual impact naturally.
- Titles should feel formal, readable, and educational.
- Avoid rigid labels or symbolic formatting.

[STRICT_OUTPUT_RULES]
- Return raw JSON only.
- No markdown code block.
- No additional commentary.
- Escape newline characters.
- Escape internal quotes.