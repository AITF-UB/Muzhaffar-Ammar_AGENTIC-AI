import json
from langchain_core.tools import tool

# 1. Retriever Tool (Simulasi RAG dari database sekolah)
@tool
def retriever_tool(query: str) -> str:
    """
    Mencari dokumen materi mentah dari database sekolah (Vector DB) berdasarkan query.
    Selalu gunakan tool ini PERTAMA KALI untuk mendapatkan fakta historis atau pedagogik sebelum menulis materi.
    """
    # Di Fase 3, ini akan diganti dengan koneksi ke Vector Database sungguhan.
    return f"[FAKTA RETRIEVER]: Materi dasar tentang '{query}'. (Sumber: Buku Kemendikbud Hal 45)"

# 2. Curriculum Mapper/Validator Tool
@tool
def curriculum_mapper_tool(topik: str, tingkat: str) -> str:
    """
    Memvalidasi apakah topik yang dibahas sesuai dengan tingkat pendidikan siswa (SD/SMP/SMA).
    Gunakan ini untuk memastikan materi tidak keluar dari batas kurikulum yang ditetapkan.
    """
    return f"[VALIDASI KURIKULUM]: Topik '{topik}' tervalidasi dan sangat sesuai untuk kurikulum tingkat {tingkat}."

# 3. Content Generator Tool (Simulasi LLM Tim 2)
@tool
def content_generator_tool(fakta_mentah: str, instruksi_format: str) -> str:
    """
    Mengirimkan fakta mentah ke LLM Ahli Konten (Tim 2) untuk diubah menjadi narasi pembelajaran yang imersif.
    Gunakan ini setelah mendapatkan data dari retriever dan memastikan validasi kurikulum.
    """
    return f"[DRAFT KONTEN DARI TIM 2]: Menggunakan fakta '{fakta_mentah}', berikut narasi pembelajarannya: 'Mari kita pelajari konsep ini dengan saksama...'"

# 4. Emotion Adapter Tool (Menggunakan data Tim 1)
@tool
def emotion_adapter_tool(draft_konten: str, status_emosi: str) -> str:
    """
    Menyesuaikan gaya bahasa draft konten berdasarkan status emosi siswa saat ini (data dari Tim 1).
    Wajib digunakan untuk memodifikasi teks jika emosi siswa terdeteksi negatif (misal: frustrasi, bingung).
    """
    if status_emosi.lower() in ["frustrasi", "bingung", "sedih", "bosan"]:
        return f"[KONTEN ADAPTIF (EMOSI: {status_emosi})]: Ayo kita pelan-pelan ya. Konsep ini mirip seperti... (Teks disederhanakan dan ditambah analogi)"
    return f"[KONTEN ADAPTIF (EMOSI: {status_emosi})]: Bagus sekali, mari kita lanjutkan dengan tantangan berikutnya! (Teks normal)"

# 5. Citation Builder Tool
@tool
def citation_builder_tool(konten: str, sumber: str) -> str:
    """
    Menambahkan kutipan atau sitasi yang valid di akhir materi untuk menjaga integritas akademik.
    """
    return f"{konten}\n\n[SITASI]: {sumber}"

# 6. Difficulty Adjuster Tool
@tool
def difficulty_adjuster_tool(konten: str, riwayat_nilai_rata_rata: int) -> str:
    """
    Menyesuaikan tingkat kesulitan materi atau latihan soal berdasarkan riwayat nilai historis siswa.
    Jika nilai siswa di atas 80, naikkan level kesulitannya.
    """
    level = "HOTS (Tingkat Tinggi)" if riwayat_nilai_rata_rata > 80 else "Dasar"
    return f"{konten}\n[PENYESUAIAN LEVEL]: Latihan soal disesuaikan ke tingkat {level}."

# 7. Content Structurer Tool (Wajib untuk UI Tim 6)
@tool
def content_structurer_tool(materi_final: str, topik: str, tingkat: str, saran_guru: str) -> str:
    """
    Membungkus seluruh materi akhir menjadi format JSON strict yang diminta oleh infrastruktur MVP Tim 6.
    Ini HARUS menjadi langkah aksi TERAKHIR sebelum agen memberikan Final Answer.
    """
    payload = {
        "topik": topik,
        "tingkat": tingkat,
        "struktur_materi": [
            {
                "judul_bagian": f"Pengantar {topik}",
                "tipe": "teks_imersif",
                "konten": materi_final,
            },
            {
                "judul_bagian": f"Ringkasan {topik}",
                "tipe": "slide",
                "konten": "1. Poin Pertama\n2. Poin Kedua"
            }
        ],
        "rekomendasi_guru": saran_guru
    }
    return json.dumps(payload)

# Menggabungkan semua tool ke dalam satu list untuk di-bind ke LLM nanti
tools_list = [
    retriever_tool,
    curriculum_mapper_tool,
    content_generator_tool,
    emotion_adapter_tool,
    citation_builder_tool,
    difficulty_adjuster_tool,
    content_structurer_tool
]