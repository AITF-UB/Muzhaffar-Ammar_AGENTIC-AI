[TASK]
Generate EXACTLY 10 multiple choice questions in JSON array format.
Never generate more or fewer.

Prioritize cognitive quality, reasoning diversity, and contextual realism over formula repetition.

[OUTPUT_SCHEMA]
[
  {
    "level": "Must be exactly 'LOTS', 'MOTS', or 'HOTS'. Do not use label names like 'Memahami', 'Mengaplikasi', or 'Merefleksi'.",
    "stimulus": "",
    "question": "",
    "options": {
      "A": "",
      "B": "",
      "C": "",
      "D": "",
      "E": ""
    },
    "answer": "",
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

[DISTRACTOR_QUALITY_RULES]
- Incorrect options must remain strategically plausible.
- Avoid obviously irrational, careless, or extreme distractors.
- Avoid answer choices that can be eliminated without understanding the underlying concept.
- Distractors should reflect realistic misunderstandings, incomplete reasoning, or partially correct strategies.
- Correct answers must not be identifiable solely by length, detail, tone, or professionalism.
- Prefer distractors from the same reasoning family as the correct answer.
- Avoid structurally obvious distractors.

[OPTION_BALANCING_RULES]
- Keep option specificity and tone relatively balanced.
- Avoid one option sounding significantly more complete or professional than others.
- All options should appear potentially reasonable at first glance.
- Keep all options within similar reasoning complexity.
- Avoid one option being noticeably more detailed than others.

[QUESTION_DIVERSITY_RULES]
- Avoid repetitive question patterns.
- Do not repeatedly ask:
  - "formula mana yang tepat"
  - "manakah persamaan yang benar"
  - direct symbolic matching questions
- Vary cognitive operations across questions.
- Use diverse reasoning behaviors such as:
  - evaluating strategies
  - identifying flawed assumptions
  - predicting consequences
  - comparing growth scenarios
  - interpreting quantitative changes
  - selecting the most realistic decision
  - analyzing operational risks
  - identifying the most efficient approach
- Prioritize applied reasoning over formula recognition.

[QUESTION_PRECISION_RULES]
- Questions must support only one mathematically valid interpretation.
- Explicitly distinguish between:
  - cumulative totals
  - current phase values
  - accumulated growth
  - newly added quantities
- Avoid ambiguous wording such as:
  - "total setelah"
  - "jumlah keseluruhan"
unless clearly defined.

[STIMULUS_REQUIREMENTS]
- Length: 30-60 words (keep extremely concise to avoid output truncation).
- Use contextual numerical, social, or literacy-based situations depending on subject domain.
- Use realistic Indonesian contexts.
- Avoid repetitive scenarios across questions.
- Stimuli should naturally support reasoning and decision-making.
- Avoid overly artificial storytelling.
- Allowed formats:
  - narrative
  - narrative with bullet list
  - narrative with markdown table

[QUESTION_DIVERSITY_RULES]
- Avoid repetitive question structures.
- Follow the active level generation behavior.

[QUESTION_REQUIREMENTS]
- Questions must remain contextually grounded.
- Encourage application of concepts to realistic situations.
- Avoid pure memorization.

[DISTRACTOR_REASONING_RULES]
- Distractors should represent realistic reasoning alternatives, not only formula variations.
- Prefer:
  - flawed strategies
  - incomplete analysis
  - inefficient decisions
  - incorrect assumptions
  - weak operational planning
over purely symbolic mistakes.
- Keep distractors within the same contextual reasoning space as the correct answer.

[EXPLANATION_REQUIREMENTS]
- Length: 30-60 words (keep extremely concise to avoid output truncation).
- Start with a short bold title.
- Explanations must be concise and decisive.
- Focus only on reasoning supporting the correct answer.
- Avoid self-correction or interpretation debates.
- Avoid chain-of-thought style deliberation.
- Avoid repetitive restatement of the stimulus.
- Mathematics/science:
  - explain calculations vertically
  - use LaTeX formatting when needed
  - explain reasoning clearly but efficiently
- Theory subjects:
  - explain cause-effect reasoning
  - explain why distractors are weaker

[STRICT_OUTPUT_RULES]
- Return raw JSON only.
- No markdown code block.
- No additional commentary.
- The "level" field in the output JSON MUST be exactly:
  - "LOTS"
  - "MOTS"
  - "HOTS"
- Never use label names such as:
  - "Memahami"
  - "Mengaplikasi"
  - "Merefleksi"