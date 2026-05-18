[KRITERIA_PEMBELAJARAN_MENDALAM]
{LEVELING_CRITERIA}

[KONTEKS_PEMBELAJARAN]
{context}

[TASK]
Hasilkan 5 unit Flashcards Tradisional dalam format JSON. Isi konten (back) harus bervariasi antara definisi, istilah, rumus, analogi, atau fun-fact yang relevan dengan Target Level.

[RULES]
- Character Limit: Panjang teks pada field "back" WAJIB berada di rentang 50 - 150 karakter.
- Content Variety: Sesuaikan isi back dengan level kognitif. Gunakan analogi atau fun-fact untuk MOTS/HOTS agar tetap bermakna meski formatnya tradisional.


[STRUCTURE]
JSON
[
  {
    "front": "[Stimulus/Pertanyaan singkat]",
    "back": "[Isi: Definisi/Istilah/Rumus/Analogi/Fun-fact]"
  }
]