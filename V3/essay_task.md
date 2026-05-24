[KRITERIA_PEMBELAJARAN_MENDALAM]
{LEVELING_CRITERIA}

[KONTEKS_PEMBELAJARAN]
{context}

[TASK]
Hasilkan tepat 5 unit soal Esai HOTS dalam format JSON murni berbentuk Array. DILARANG menyertakan teks pembuka/penutup atau blok markdown (```json).
*Target Level Alignment*: Semua komponen wajib patuh pada kedalaman tingkat {target_level}.

STRUKTUR JSON:
[
  {
    "level": "{target_level}",
    "stimulus": "[Teks naratif, kombinasi list, atau tabel data krisis riil Indonesia]",
    "question": "[Pertanyaan analisis krisis/sebab-akibat/metakognitif]",
    "rubric_points": [
      "[Indikator 1: Komponen teknis/argumen kunci materi wajib ada]",
      "[Indikator 2: Analisis kausalitas/hubungan krisis nyata wajib tertulis]",
      "[Indikator 3: Bentuk refleksi/solusi/metakognisi kontekstual siswa]"
    ],
    "explanation": "[Uraian solusi ilmiah atau inti jawaban utuh dan mengalir]"
  }
]

[RULES & CONSTRAINTS]
- Aturan Stimulus Pilgan (50 - 150 kata): Wajib menyusun teks pengantar soal yang kritikal berdasarkan empat poin aturan mutlak berikut:
  * Keaslian Konteks: Teks wajib bersumber dari data krisis atau fakta empiris riil di Indonesia (DILARANG pakai skenario imajiner/dongeng).
  * Sifat Kritikal: Soal hanya bisa dijawab jika siswa menganalisis data/teks di dalam stimulus (DILARANG membuat soal yang bisa dijawab pakai hafalan luar).
  * Adaptasi Rumpun Mapel: 
    1. Mapel STEM: Wajib memuat tren angka kuantitatif, notasi matematis, atau tabel data teknis untuk memicu proses komputasi.
    2. Mapel Non-STEM: Wajib memuat dinamika sosial, kronologi sejarah, atau argumen kebijakan riil di masyarakat.
  * Variasi Format Visual: Format penyajian teks wajib bervariasi. Dibebaskan penuh menggunakan sekumpulan paragraf narasi murni, kombinasi paragraf + poin list, kombinasi paragraf + tabel Markdown, atau gabungan dari ketiga elemen tersebut sekaligus sesuai kebutuhan kedalaman materi.
- Question (20 - 50 kata): Wajib berupa SATU kalimat tanya utuh. DILARANG keras membuat pertanyaan bercabang, memakai kata hubung "dan" untuk memisahkan pertanyaan konsep dengan refleksi, atau menanyakan hafalan/definisi kaku.
- "rubric_points" (Max 50 Kata) WAJIB berupa array minimal 3 poin checklist operasional guru.
- Explanation (50 - 150 kata): Langsung uraikan logika jawaban tanpa kalimat pembuka. Wajib ditulis ke bawah pakai daftar poin (-) ATAU tabel Markdown demi keterbacaan:
  1. Mapel Hitungan (Matematika, Fisika, Kimia): Wajib diawali judul pendek bercetak tebal (contoh: **Langkah Hitungan**). Operasi hitungan, substitusi, atau hasil akhir WAJIB ditulis berbaris vertikal ke bawah pakai LaTeX terpisah (lingkungan array).
  2. Mapel Teori (Sosial, Bahasa, dll.): Wajib diawali judul pendek bercetak tebal (contoh: **Sebab-Akibat Masalah**). Bagian awal wajib membongkar hubungan sebab-akibat utama dari stimulus.
- Bagian terakhir WAJIB mencantumkan **Refleksi Nyata:** berupa satu blok kalimat naratif/renungan langsung dampak materi dalam aksi sehari-hari (DILARANG pakai kalimat tanya atau memuat kata "organik"/"kontekstual").