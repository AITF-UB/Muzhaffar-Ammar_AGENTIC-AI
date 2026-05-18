[KONTEKS_PEMBELAJARAN]
{context}

[TASK]
Generate a recursive nested tree Mindmap JSON from {chunks} matching this schema:
{
  "tipe": "mindmap", "mata_pelajaran": "{mapel}", "materi": "{sub_bab}",
  "nodes": [{
    "konsep_utama": "", "deskripsi": "",
    "children": [{ "sub_konsep": "", "penjelasan": "", "children": [] }]
  }]
}

[CONSTRAINTS]
- Inputs: Target={target_level}, ATP={atp}, Context={context}.
- Depth: Min 3 recursive levels (Root -> Child -> Grandchild). Terminate leaf nodes with [].
- Length: Max 15 words for 'deskripsi' and 'penjelasan'. Be punchy and conversational.
- Grounding: Map names and facts strictly from {chunks}. Use internal knowledge only for local analogies.