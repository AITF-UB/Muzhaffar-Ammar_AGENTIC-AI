[KRITERIA_PEMBELAJARAN_MENDALAM]
{LEVELING_CRITERIA}

[KONTEKS_PEMBELAJARAN]
{context}

[TASK]
Hasilkan recursive nested tree JSON mindmap dari {chunks}. TANPA teks pembuka/penutup atau blok markdown (```json).
*Escape Requirement*: Escape baris baru menjadi "\\n" dan tanda kutip ganda internal menjadi "\\\"".

[STRUCTURE]
{
  "root": {
    "name": "[Nama Topik Utama]",
    "description": "[Deskripsi esensial, maks 15 kata]",
    "children": [
      {
        "name": "[Nama Sub-Konsep]",
        "description": "[Penjelasan hubungan konsep, maks 15 kata]",
        "children": []
      }
    ]
  }
}

[CONSTRAINTS]
- Kedalaman Pohon (Tree Depth): Minimal 3 tingkat kedalaman rekursif (Root -> Child -> Grandchild). Akhiri ujung cabang (leaf nodes) dengan "children": [].
- Batas Ketat Kata: Setiap field "description" WAJIB sangat padat, komunikatif, langsung pada inti masalah, dan MAKSIMAL 15 kata. DILARANG menggunakan kata "kamu".
- Format Bebas Tanda Baca: DILARANG keras menggunakan tanda titik dua (:) atau penomoran kaku (1, 2, A, B) pada field "name" maupun "description".
- Target Level Alignment ({target_level}): Isi "description" wajib fokus pada: LOTS jika mengidentifikasi pola dasar/fakta; MOTS jika menjabarkan sebab-akibat/analogi; HOTS jika menyajikan pertanyaan reflektif/metakognisi.