[KRITERIA_PEMBELAJARAN_MENDALAM]
{LEVELING_CRITERIA}

[KONTEKS_PEMBELAJARAN]
{context}

[TASK]
Hasilkan tepat 5 flashcards tradisional dalam JSON murni tanpa pembuka/penutup atau blok markdown (```json).
Wajib selaraskan kedalaman "back" dengan {LEVELING_CRITERIA}.

Format:
[
  { "front": "[Pertanyaan]", "back": "[Jawaban]" }
]

[CONSTRAINTS]
- "front": MAX 20 karakter dengan spasi.
- "back": MAX 50 karakter dengan spasi. Wajib hitung ketat!
- ANTI-RETORIS: Field "back" WAJIB berupa jawaban definitif, solusi konkret, kesimpulan, atau rumus. DILARANG keras memakai kalimat tanya, kalimat menggantung, atau perintah refleksi terbuka.

[CONTENT]
Komposisikan isi "back" dari 5 kartu secara dinamis menggunakan variasi elemen berikut:
- Konteks Lokal RI (Analogi budaya, isu sosial nyata, atau tren fenomena riil Indonesia).
- Teoretis (STEM: Definisi teknis OR Rumus utama wajib LaTeX $...$ | NON-STEM: Istilah kunci OR Dampak peristiwa).