import json
import re
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ================================================================
# 1. Knowledge Base (RAG) Setup - Seperti tes_1.ipynb
# ================================================================

class DatabaseSekolah:
    DOCUMENTS = [
        Document(
            page_content="""
            Pernyataan Kondisional (If-Else) dalam pemrograman digunakan
            untuk mengambil keputusan. Jika kondisi BENAR (True), blok kode dieksekusi.
            Jika SALAH (False), eksekusi berpindah ke blok Else.

            Contoh:
            if nilai > 75:
                print("Lulus")
            else:
                print("Remedial")

            Kondisional sangat penting dalam logika algoritma karena
            membuat program menjadi dinamis.
            """,
            metadata={"subject": "Informatika", "topic": "Pernyataan Kondisional"}
        ),
        Document(
            page_content="""
            Variabel dalam pemrograman adalah tempat menyimpan data.
            Tipe data umum meliputi Integer (angka), String (teks), 
            dan Boolean (True/False).

            Nama variabel tidak boleh diawali dengan angka.
            """,
            metadata={"subject": "Informatika", "topic": "Variabel"}
        ),
        Document(
            page_content="""
            Teorema Pythagoras menyatakan bahwa dalam segitiga siku-siku,
            kuadrat sisi miring (hipotenusa) sama dengan jumlah kuadrat
            kedua sisi lainnya. Rumusnya: c² = a² + b²

            Dimana:
            - c = hipotenusa (sisi terpanjang)
            - a dan b = kedua sisi siku-siku

            Contoh: Jika a = 3 cm dan b = 4 cm, maka:
            c² = 3² + 4² = 9 + 16 = 25, jadi c = √25 = 5 cm

            Triple Pythagoras yang umum: (3,4,5), (5,12,13), (8,15,17)

            Penerapan: mengukur jarak diagonal, arsitektur, navigasi.
            """,
            metadata={"subject": "matematika", "topic": "pythagoras"}
        ),
        Document(
            page_content="""
            Persamaan Kuadrat: ax² + bx + c = 0, dimana a ≠ 0

            Cara menyelesaikan:
            1. Faktorisasi: cari (x - p)(x - q) = 0
            2. Melengkapkan kuadrat sempurna
            3. Rumus ABC: x = (-b ± √(b² - 4ac)) / 2a

            Diskriminan (D) = b² - 4ac:
            - D > 0 → dua akar real berbeda
            - D = 0 → dua akar real sama
            - D < 0 → tidak ada akar real

            Contoh: x² - 5x + 6 = 0
            Faktorisasi: (x-2)(x-3) = 0 → x = 2 atau x = 3
            """,
            metadata={"subject": "matematika", "topic": "persamaan_kuadrat"}
        ),
        Document(
            page_content="""
            Hukum Newton tentang Gerak:

            Hukum I (Inersia): Benda tetap diam atau bergerak lurus
            beraturan jika tidak ada gaya luar.
            Contoh: penumpang terdorong ke depan saat mobil rem mendadak.

            Hukum II (F = ma): Percepatan berbanding lurus dengan gaya
            dan berbanding terbalik dengan massa.
            F = gaya (Newton), m = massa (kg), a = percepatan (m/s²)

            Hukum III (Aksi-Reaksi): Setiap aksi menimbulkan reaksi
            yang sama besar tapi berlawanan arah.
            Contoh: roket meluncur karena gas terdorong ke bawah.
            """,
            metadata={"subject": "fisika", "topic": "hukum_newton"}
        ),
        Document(
            page_content="""
            Fotosintesis: proses tumbuhan membuat makanan menggunakan
            cahaya matahari.

            Rumus: 6CO₂ + 6H₂O + Cahaya → C₆H₁₂O₆ + 6O₂
            (Karbondioksida + Air + Cahaya → Glukosa + Oksigen)

            Terjadi di KLOROPLAS yang mengandung KLOROFIL (pigmen hijau).

            Tahapan:
            1. Reaksi Terang (membran tilakoid): menyerap cahaya,
               memecah air, menghasilkan ATP & NADPH
            2. Reaksi Gelap/Siklus Calvin (stroma): mengikat CO₂,
               menghasilkan glukosa

            Faktor: intensitas cahaya, CO₂, suhu, ketersediaan air.
            """,
            metadata={"subject": "biologi", "topic": "fotosintesis"}
        ),
        Document(
            page_content="""
            Proklamasi Kemerdekaan Indonesia: 17 Agustus 1945

            Dibacakan oleh Ir. Soekarno didampingi Drs. Mohammad Hatta
            di Jalan Pegangsaan Timur No. 56, Jakarta.

            Kronologi:
            - 6 Agustus: Bom atom Hiroshima
            - 9 Agustus: Bom atom Nagasaki
            - 14 Agustus: Jepang menyerah
            - 16 Agustus: Peristiwa Rengasdengklok
            - 17 Agustus: Proklamasi dibacakan

            Teks proklamasi diketik Sayuti Melik.
            Bendera Merah Putih dijahit Ibu Fatmawati.
            """,
            metadata={"subject": "sejarah", "topic": "proklamasi"}
        ),
        Document(
            page_content="""
            Sistem Tata Surya: Matahari sebagai pusat, 8 planet:
            1. Merkurius - terkecil, terdekat Matahari
            2. Venus - terpanas, disebut Bintang Kejora
            3. Bumi - satu-satunya yang ada kehidupan
            4. Mars - planet merah, gunung Olympus Mons
            5. Jupiter - terbesar, Bintik Merah Besar
            6. Saturnus - terkenal dengan cincinnya
            7. Uranus - rotasi miring 90 derajat
            8. Neptunus - terjauh, berwarna biru

            Juga ada: asteroid, komet, planet kerdil (Pluto).
            Bulan = satelit alami Bumi.
            """,
            metadata={"subject": "ipa", "topic": "tata_surya"}
        ),
    ]

    def __init__(self):
        print("📚 Membangun Database Sekolah untuk Router Agent...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100
        )
        docs = splitter.split_documents(self.DOCUMENTS)
        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            collection_name="sekolah_db_router"
        )
        print(f"   ✅ {len(docs)} chunks KB siap!")

    def search(self, query: str, k: int = 2) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=k)

# Inisialisasi KB secara global untuk digunakan node
kb_sekolah = DatabaseSekolah()

# ================================================================
# 2. Node Utility Functions & JSON Cleaners
# ================================================================

def clean_json_from_llm(raw_text: str) -> dict:
    """Fallback JSON parser tangguh untuk mengekstrak string JSON kotor dari LLM"""
    # 1. Hapus Markdown code blocks (```json ... ```)
    clean_text = re.sub(r'```(?:json)?', '', raw_text).strip()
    
    # 2. Cari kurung kurawal pembuka dan penutup terakhir
    start_idx = clean_text.find('{')
    end_idx = clean_text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = clean_text[start_idx:end_idx+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
            
    # Cari kurung siku (untuk JSON array yg di-wrap)
    start_idx = clean_text.find('[')
    end_idx = clean_text.rfind(']')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = clean_text[start_idx:end_idx+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
            
    # Jika semua gagal, return format error aman
    return {"error": "Gagal parsing JSON dari LLM", "raw": raw_text[:100]}

import uuid
import json as _json

# ----------------------------------------------------------------
# Helper: filter N bab terburuk sebelum masuk LLM 
# ----------------------------------------------------------------
def _ambil_prioritas_belajar(riwayat: list, top_n: int = 5) -> list:
    """Ambil N bab paling lemah dari riwayat progress, sort by pemahaman lalu skor."""
    urutan_pemahaman = {"Belum Paham": 0, "Paham Dasar": 1, "Paham Mendalam": 2}
    valid = [r for r in riwayat if isinstance(r, dict)]
    sorted_riwayat = sorted(
        valid,
        key=lambda x: (
            urutan_pemahaman.get(x.get("tingkat_pemahaman", "Belum Paham"), 0),
            x.get("skor_terakhir", 100)
        )
    )
    return sorted_riwayat[:top_n]

# ----------------------------------------------------------------
# Recommender — formatter (first_time + returning)
# ----------------------------------------------------------------
def util_format_recommender(student_id: str, first_time: bool, pesan: str, rekomendasi: list) -> dict:
    """Formatter utama recommender — v2 support first_time dan returning."""
    return {
        "tipe": "rekomendasi_topik",
        "student_id": student_id,
        "first_time": first_time,
        "pesan_empatik": pesan,
        "rekomendasi": rekomendasi
    }

def util_format_flashcard(topik: str, flashcards: list) -> dict:
    return {
        "tipe": "flashcard_set",
        "topik": topik,
        "jumlah_kartu": len(flashcards) if isinstance(flashcards, list) else 0,
        "kartu": flashcards
    }
    
def util_format_mindmap(topik: str, nodes: list) -> dict:
    return {
        "tipe": "mindmap_hierarki",
        "topik": topik,
        "nodes": nodes
    }

# ----------------------------------------------------------------
# Helper: generate soal_id unik (Python-side, bukan LLM)
# ----------------------------------------------------------------
def generate_soal_id(tipe: str, topik: str) -> str:
    """Generate ID unik per soal: {tipe}-{topik_slug}-{4hex}"""
    slug = topik.lower().replace(" ", "_")[:20]
    return f"{tipe}-{slug}-{uuid.uuid4().hex[:4]}"

# ----------------------------------------------------------------
# Quiz PG — generate formatter (tambah soal_id per soal)
# ----------------------------------------------------------------
def util_format_quiz(topik: str, soal: list) -> dict:
    """Inject soal_id ke tiap soal PG, lalu wrap payload."""
    for item in soal:
        if isinstance(item, dict) and "soal_id" not in item:
            item["soal_id"] = generate_soal_id("pg", topik)
    return {
        "tipe": "quiz_soal",
        "topik": topik,
        "jumlah_soal": len(soal) if isinstance(soal, list) else 0,
        "soal": soal
    }

# ----------------------------------------------------------------
# Quiz Uraian — generate formatter
# ----------------------------------------------------------------
def util_format_quiz_uraian(topik: str, soal: list) -> dict:
    """Inject soal_id ke tiap soal uraian, lalu wrap payload."""
    for item in soal:
        if isinstance(item, dict) and "soal_id" not in item:
            item["soal_id"] = generate_soal_id("uraian", topik)
    return {
        "tipe": "quiz_uraian",
        "topik": topik,
        "jumlah_soal": len(soal) if isinstance(soal, list) else 0,
        "soal": soal
    }

# ----------------------------------------------------------------
# Evaluasi Quiz PG — formatter (deterministik)
# ----------------------------------------------------------------
def util_format_evaluasi_quiz(topik: str, detail: list) -> dict:
    total_benar = sum(1 for d in detail if d.get("benar", False))
    total_skor  = sum(d.get("skor", 0) for d in detail)
    skor_maks   = sum(d.get("skor_maksimal", 10) for d in detail)
    return {
        "tipe": "hasil_evaluasi_quiz",
        "topik": topik,
        "total_benar": total_benar,
        "total_soal": len(detail),
        "total_skor": total_skor,
        "skor_maksimal": skor_maks,
        "detail": detail
    }

# ----------------------------------------------------------------
# Evaluasi Uraian — formatter (LLM)
# ----------------------------------------------------------------
def util_format_evaluasi_uraian(topik: str, detail: list, pemahaman: str, ringkasan: str) -> dict:
    total_skor = sum(d.get("skor", 0) for d in detail)
    skor_maks  = sum(d.get("skor_maksimal", 20) for d in detail)
    return {
        "tipe": "hasil_evaluasi_uraian",
        "topik": topik,
        "total_skor": total_skor,
        "skor_maksimal": skor_maks,
        "tingkat_pemahaman": pemahaman,
        "ringkasan_feedback": ringkasan,
        "detail": detail
    }

# ----------------------------------------------------------------
# Konten Belajar — formatter (konten panjang terstruktur)
# ----------------------------------------------------------------
def util_format_konten_belajar(topik: str, bab: str, judul: str, konten_list: list, sumber: list) -> dict:
    return {
        "tipe": "konten_belajar",
        "topik": topik,
        "bab": bab,
        "judul_konten": judul,
        "jumlah_sub_bab": len(konten_list),
        "konten": konten_list,
        "sumber": sumber
    }

# ----------------------------------------------------------------
# RAG Query — formatter (pure RAG, no LLM)
# ----------------------------------------------------------------
def util_format_rag_query(query: str, topik: str, chunks: list) -> dict:
    return {
        "tipe": "rag_konteks",
        "query": query,
        "topik": topik,
        "jumlah_chunk": len(chunks),
        "konteks": chunks,
        "petunjuk_untuk_chatbot": (
            "Gunakan konteks di atas untuk menjawab pertanyaan siswa. "
            "Jangan mengarang di luar konteks yang diberikan."
        )
    }
