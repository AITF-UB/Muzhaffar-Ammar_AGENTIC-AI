[TASK]
Generate a recursive nested tree mindmap in JSON format.

[OUTPUT_SCHEMA]
{
  "root": {
    "name": "",
    "description": "",
    "children": []
  }
}

[LEVEL_CONFIGURATION]
{COMPILED_LEVEL_CONFIGURATION}

[LEARNING_CONTEXT]
{context}

[SOURCE_MATERIAL]
{chunks}

[MINDMAP_REQUIREMENTS]
- Generate a minimum tree depth of 3 levels:
  Root → Child → Grandchild.
- Leaf nodes must end with:
  "children": []

- Organize concepts hierarchically.
- Preserve logical relationships between concepts.
- Keep descriptions concise and information-dense.

[DESCRIPTION_RULES]
- Maximum 15 words per description.
- Avoid unnecessary filler words.
- Do not use second-person language.
- Avoid rigid numbering or label formatting.

[STRICT_OUTPUT_RULES]
- Return raw JSON only.
- No markdown code block.
- No additional commentary.
- Escape newline characters.
- Escape internal quotes.