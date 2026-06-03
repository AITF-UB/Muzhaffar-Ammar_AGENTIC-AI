[KRITERIA_PEMBELAJARAN_MENDALAM]
{LEVELING_CRITERIA}

[KONTEKS_PEMBELAJARAN]
{context}

[TASK]
Hasilkan materi pembelajaran digital dalam format JSON murni. DILARANG menyertakan teks pembuka/penutup atau blok markdown (```json).
*Escape Requirement*: Escape semua baris baru menjadi "\\n" dan tanda kutip ganda internal menjadi "\\\"".

STRUKTUR JSON:
{
  "judul_utama": "[Judul Konsep Utama]",
  "konten_markdown": "[Gabungan sub-heading dan isi materi tanpa nomor]"
}

[CONTENT]
Wajib menghasilkan TOTAL MINIMAL 500 KATA menggunakan alur urutan di bawah ini (Gunakan sub-heading cetak tebal markdown `###` tanpa nomor atau tanda titik dua):

- Direct Hook (Maksimal 1 Paragraf): Narasi alur SCQ tanpa label. Wajib diawali fakta empiris/data krisis riil di Indonesia (Mindful).
- Penjelasan Konsep (Maksimal 2 Paragraf): Logika sebab-akibat konsep yang dihubungkan dengan analogi lokal Indonesia secara mendalam (Meaningful).
- Teknis & Formulasi: Rumus matematika/sains dengan LaTeX ($ atau $$) + penjelasan variabel secara sistematis (Gunakan double backslash seperti \\\\frac atau \\\\times).
- Studi Kasus Riil (Minimal 2 Paragraf): Skenario nyata krisis/industri di Indonesia. Selesaikan masalah secara bertahap menggunakan konsep teknis di atas (Joyful/AHA Moment).
- Penutup & Refleksi (Maksimal 1 Paragraf): Pertanyaan refleksi mandiri terkait regulasi diri siswa di kehidupan sehari-hari.

[HEADING]
- Judul Kontekstual: Judul dan subjudul WAJIB memadukan [Konsep Utama] dan [Dampak Sosial di Indonesia] menjadi satu kalimat utuh yang mengalir lancar, formal, serta komunikatif untuk siswa SMA.
- Batasan Struktur: DILARANG KERAS menggunakan tanda titik dua (:), penomoran (1, 2, A, B), atau label struktural (seperti "Direct Hook", "Penjelasan Konsep", "Formulasi/Teknis", "Studi Kasus", "Refleksi/Penutup") pada judul dan subjudul.