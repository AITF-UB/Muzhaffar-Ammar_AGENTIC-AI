[KRITERIA_PEMBELAJARAN_MENDALAM]
{LEVELING_CRITERIA}

[KONTEKS_PEMBELAJARAN]
{context}

[TASK]
Hasilkan 5 soal Pilihan Ganda dalam format **JSON** dengan ketentuan:

[STRUCTURE]
{"level": "", "stimulus": "", "question": "", "options": {"A":"", "B":"", "C":"", "D":"", "E":""}, "answer": "", "explanation": ""}

[RULES]
- Aturan Stimulus Pilgan (50 - 150 kata): 
{stimulus}
- Variasi Format Visual: Format penyajian teks wajib bervariasi menggunakan sekumpulan paragraf narasi murni, kombinasi paragraf + poin list, kombinasi paragraf + tabel Markdown, atau gabungan dari ketiga elemen tersebut sekaligus sesuai kebutuhan kedalaman materi.
- Question (20 - 50 kata): Wajib berupa SATU kalimat tanya utuh. DILARANG keras membuat pertanyaan bercabang, memakai kata hubung "dan" untuk memisahkan pertanyaan konsep dengan refleksi, atau menanyakan hafalan/definisi kaku.
- Variasi Pilihan Jawaban (10-20 kata per pilihan): Wajib mematuhi empat sub-aturan mutlak di bawah ini demi kerapian visual aplikasi:
  * Substansi Berbeda Total: Setiap opsi (A–E) WAJIB memiliki isi logika hitungan atau konsep yang berbeda total. DILARANG KERAS menuliskan rumus atau hasil akhir yang sama lalu hanya memutar-mutar susunan tata bahasanya (anti-kalimat kosmetik).
  * Pengecoh Berbobot Tinggi: DILARANG membuat opsi salah berupa kesalahan teknis hitung (salah pencet kalkulator/salah angka). Semua opsi (A–E) wajib memiliki argumen kebenaran ilmiahnya sendiri secara mandiri. Namun, hanya ada SATU jawaban yang paling benar karena paling tepat, utuh, dan sesuai dengan batasan masalah yang diminta oleh data stimulus.
  * Sinonim Kata Setara: Gunakan kata-kata yang bervariasi (sinonim) namun jenis katanya tetap setara.
  * Keseimbangan Panjang Kata: Jumlah panjang kata antar-opsi dibuat hampir sama.
- Explanation (50 - 150 kata): Langsung uraikan logika jawaban tanpa kalimat pembuka atau abjad kunci. Wajib ditulis ke bawah pakai daftar poin (-) ATAU tabel Markdown demi keterbacaan:
  1. Mapel Hitungan (Matematika, Fisika, Kimia): Wajib diawali judul pendek bercetak tebal (contoh: **Langkah Hitungan**). Operasi hitungan bertingkat atau eliminasi WAJIB ditulis berbaris vertikal ke bawah pakai LaTeX terpisah (lingkungan array). DILARANG menulis hitungan mendatar dalam satu kalimat.
  2. Mapel Teori (Sosial, Bahasa, dll.): Wajib diawali judul pendek bercetak tebal (contoh: **Alasan Benar**). Poin awal wajib membongkar hubungan sebab-akibat nyata dari jawaban benar, bagian berikutnya menyanggah secara logis mengapa opsi pengecoh salah berdasarkan stimulus.