import json
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Kita HAPUS dekorator @tool dari file ini karena fungsi-fungsi ini
# akan dipanggil MANUALLY oleh masing-masing Node di LangGraph (DAG)
# bukan secara otonom oleh LLM yang menggunakan tool binding.

# ================================================================
# 1. Knowledge Base (RAG) Setup - Seperti tes_1.ipynb
# ================================================================

class DatabaseSekolah:
    # Dokumen di-hardcode seperti pada tes_1.ipynb untuk demo
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
        print("📚 Membangun Database Sekolah...")
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
            collection_name="sekolah_db"
        )
        print(f"   ✅ {len(docs)} chunks KB siap!")

    def search(self, query: str, k: int = 2) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=k)

# Inisialisasi KB secara global untuk digunakan node
kb_sekolah = DatabaseSekolah()

# ================================================================
# 2. Node Utility Functions
# ================================================================

def util_mapper_kurikulum(topik: str, tingkat: str) -> str:
    """Utility tambahan (opsional) untuk validasi"""
    return f"[VALIDASI]: '{topik}' sesuai kurikulum {tingkat}."

def util_adapt_emotion(draft_konten: str, status_emosi: str) -> str:
    """Sistem adaptif yang disuntikkan oleh Node Emosi"""
    if status_emosi.lower() in ["frustrasi", "bingung", "sedih", "bosan"]:
        return f"💡 *Hei, jangan menyerah! Mari kita bedah pelan-pelan ya. Konsep ini mirip seperti saat kita...*\n{draft_konten}"
    return f"🚀 *Kamu sudah melangkah sejauh ini! Mari gaskan materi berikutnya!*\n{draft_konten}"

def util_build_citation(konten: str, sumber: str) -> str:
    return f"{konten}\n\n*Sumber*: {sumber}"

def util_difficulty_adjuster(riwayat_nilai: int) -> tuple[str, str]:
    """Mengembalikan 2 nilai: (Instruksi Prompt, Label Cetak)"""
    if riwayat_nilai > 80:
        return "Gunakan pertanyaan tingkat tinggi (HOTS) pada instruksimu.", "HOTS (Tingkat Tinggi)"
    return "Gunakan analogi dasar yang sangat mudah diresapi.", "Dasar"

def util_structure_json(materi_final: str, topik: str, tingkat: str, saran_guru: str) -> dict:
    """
    Format wajib untuk Frontend Tim 6.
    Langsung memulangkan dictionary Python yang valid.
    """
    return {
        "topik": topik,
        "tingkat": tingkat,
        "struktur_materi": [
            {
                "judul_bagian": f"Pengantar {topik}",
                "tipe": "teks_imersif",
                "konten": materi_final,
            }
        ],
        "rekomendasi_guru": saran_guru
    }