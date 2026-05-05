import os
import json
import re
import uuid
from langchain_core.documents import Document

# ================================================================
# Qdrant
# ================================================================

# --- [QDRANT — DISABLED] ---
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

# ================================================================
# 1. Knowledge Base (RAG) Setup
# ================================================================


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        passages = [f"passage: {text.strip()}" for text in texts]
        return self.model.encode(
            passages, batch_size=32, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).tolist()

    def embed_query(self, text):
        query = f"query: {text.strip()}"
        return self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True,
        )[0].tolist()


class DatabaseSekolah:
    def __init__(self):
        _embed_model = os.getenv("EMBED_MODEL",       "microsoft/harrier-oss-v1-0.6b")
        _qdrant_host = os.getenv("QDRANT_HOST",       "76.13.195.1")
        _qdrant_port = int(os.getenv("QDRANT_PORT",   "6333"))
        _collection  = os.getenv("QDRANT_COLLECTION", "semantic_chunks")

        print(f"📚 Membangun Database Sekolah (Qdrant)...")
        print(f"   Model Embedding : {_embed_model}")
        print(f"   Qdrant Host     : {_qdrant_host}:{_qdrant_port}")
        print(f"   Collection      : {_collection}")

        self.embeddings      = SentenceTransformerEmbeddings(_embed_model)
        self.client          = QdrantClient(host=_qdrant_host, port=_qdrant_port)
        self.collection_name = _collection

        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

        try:
            count = self.client.count(self.collection_name).count
            print(f"   ✅ Terhubung ke Qdrant. Jumlah chunks: {count}")
        except Exception as e:
            print(f"   ❌ Gagal terhubung ke Qdrant: {e}")

    def search(self, query: str, k: int = 5) -> list[Document]:
        import ast
        docs = self.vectorstore.similarity_search(query, k=k)
        for d in docs:
            raw_backup = d.page_content
            d.metadata["raw_sebelum_regex"]  = raw_backup
            d.metadata["apakah_kena_regex"]  = False
            match = re.search(r"splits=\[(.*?)\]\s+is_triggered", d.page_content, re.DOTALL)
            if match:
                d.metadata["apakah_kena_regex"] = True
                try:
                    splits_str  = "[" + match.group(1) + "]"
                    splits_list = ast.literal_eval(splits_str)
                    if isinstance(splits_list, list):
                        d.page_content = " ".join(splits_list)
                except Exception:
                    pass
        return docs

kb_sekolah = DatabaseSekolah() 


# =================================================================
# [DUMMY] — In-memory ChromaDB sementara selama Qdrant error
# =================================================================
# try:
#     import chromadb
#     from chromadb.config import Settings as ChromaSettings
#     _USE_CHROMA = True
# except ImportError:
#     _USE_CHROMA = False
#     print("⚠️  chromadb tidak terinstall, fallback ke dummy statis.")


# # Konten dummy per topik — akan di-match keyword sederhana
# _DUMMY_CORPUS = [
#     Document(
#         page_content=(
#             "Bilangan berpangkat adalah perkalian berulang suatu bilangan. "
#             "Notasi: aⁿ = a × a × ... × a (sebanyak n kali). "
#             "Sifat-sifat: a⁰ = 1, a¹ = a, aᵐ × aⁿ = aᵐ⁺ⁿ, aᵐ / aⁿ = aᵐ⁻ⁿ, (aᵐ)ⁿ = aᵐⁿ. "
#             "Bilangan berpangkat negatif: a⁻ⁿ = 1/aⁿ."
#         ),
#         metadata={"source": "matematika_bilangan_berpangkat.md", "apakah_kena_regex": False, "raw_sebelum_regex": ""}
#     ),
#     Document(
#         page_content=(
#             "Hukum Newton menjelaskan hubungan antara gaya dan gerak benda. "
#             "Hukum I Newton (Inersia): Benda diam tetap diam dan benda bergerak tetap bergerak lurus beraturan "
#             "jika tidak ada resultan gaya yang bekerja padanya. "
#             "Hukum II Newton: ΣF = m × a, gaya berbanding lurus dengan percepatan. "
#             "Hukum III Newton (Aksi-Reaksi): Setiap aksi menimbulkan reaksi yang sama besar dan berlawanan arah."
#         ),
#         metadata={"source": "fisika_hukum_newton.md", "apakah_kena_regex": False, "raw_sebelum_regex": ""}
#     ),
#     Document(
#         page_content=(
#             "Fotosintesis adalah proses biokimia tumbuhan untuk menghasilkan glukosa dari CO₂ dan H₂O "
#             "menggunakan energi cahaya matahari. "
#             "Persamaan reaksi: 6CO₂ + 6H₂O + cahaya → C₆H₁₂O₆ + 6O₂. "
#             "Reaksi Terang terjadi di membran tilakoid menghasilkan ATP dan NADPH. "
#             "Reaksi Gelap (Siklus Calvin) terjadi di stroma menggunakan ATP dan NADPH untuk fiksasi CO₂."
#         ),
#         metadata={"source": "biologi_fotosintesis.md", "apakah_kena_regex": False, "raw_sebelum_regex": ""}
#     ),
#     Document(
#         page_content=(
#             "Sistem persamaan linear dua variabel (SPLDV) adalah kumpulan dua persamaan linear "
#             "dengan dua variabel x dan y. "
#             "Metode penyelesaian: substitusi, eliminasi, dan grafik. "
#             "Contoh: 2x + y = 5 dan x - y = 1, solusi: x = 2, y = 1."
#         ),
#         metadata={"source": "matematika_spldv.md", "apakah_kena_regex": False, "raw_sebelum_regex": ""}
#     ),
#     Document(
#         page_content=(
#             "Gerak Lurus Beraturan (GLB) adalah gerak dengan kecepatan konstan (percepatan = 0). "
#             "Rumus: s = v × t. "
#             "Gerak Lurus Berubah Beraturan (GLBB) adalah gerak dengan percepatan konstan. "
#             "Rumus: vt = v₀ + a×t, s = v₀t + ½at², vt² = v₀² + 2as."
#         ),
#         metadata={"source": "fisika_glb_glbb.md", "apakah_kena_regex": False, "raw_sebelum_regex": ""}
#     ),
#     Document(
#         page_content=(
#             "Sel adalah unit struktural dan fungsional terkecil makhluk hidup. "
#             "Komponen sel: membran sel, sitoplasma, inti sel (nukleus), mitokondria, ribosom, retikulum endoplasma. "
#             "Sel prokariot tidak memiliki membran inti, sel eukariot memiliki membran inti."
#         ),
#         metadata={"source": "biologi_sel.md", "apakah_kena_regex": False, "raw_sebelum_regex": ""}
#     ),
#     Document(
#         page_content=(
#             "Teks narasi adalah teks yang menceritakan suatu peristiwa secara kronologis. "
#             "Struktur teks narasi: orientasi, komplikasi, resolusi, dan koda. "
#             "Ciri-ciri: menggunakan kata kerja aktif, alur waktu, tokoh, dan latar."
#         ),
#         metadata={"source": "bindo_teks_narasi.md", "apakah_kena_regex": False, "raw_sebelum_regex": ""}
#     ),
#     Document(
#         page_content=(
#             "Materi umum: konsep dasar ilmu pengetahuan alam dan sosial. "
#             "Sains mempelajari alam semesta secara sistematis melalui observasi, hipotesis, eksperimen, dan analisis. "
#             "Metode ilmiah: perumusan masalah, studi literatur, pengumpulan data, analisis, kesimpulan."
#         ),
#         metadata={"source": "umum_sains_dasar.md", "apakah_kena_regex": False, "raw_sebelum_regex": ""}
#     ),
# ]


# class DatabaseSekolahDummy:
#     """
#     Dummy Knowledge Base — in-memory, tanpa Qdrant.
#     Pakai keyword matching sederhana untuk simulate RAG retrieval.
#     [TEMPORARY — hapus dan pakai DatabaseSekolah asli kalau Qdrant sudah oke]
#     """
#     def __init__(self):
#         print("📚 [DUMMY MODE] Menggunakan in-memory dummy corpus (Qdrant offline).")
#         print(f"   Jumlah dokumen dummy: {len(_DUMMY_CORPUS)}")

#     def search(self, query: str, k: int = 5) -> list[Document]:
#         """Keyword-based matching sederhana."""
#         query_lower = query.lower()
#         scored = []
#         for doc in _DUMMY_CORPUS:
#             # Hitung overlap kata
#             doc_words  = set(re.split(r'\W+', doc.page_content.lower()))
#             query_words = set(re.split(r'\W+', query_lower))
#             score = len(doc_words & query_words)
#             scored.append((score, doc))

#         # Sort descending, ambil top-k
#         scored.sort(key=lambda x: x[0], reverse=True)
#         results = [doc for _, doc in scored[:k] if _ > 0]

#         # Kalau tidak ada yang match, return semua dummy (fallback)
#         if not results:
#             results = [doc for _, doc in scored[:k]]

#         return results


# # Inisialisasi KB global — pakai DUMMY sementara
# kb_sekolah = DatabaseSekolahDummy()


# ================================================================
# 2. Level Difficulty Constants
# ================================================================

LEVEL_LABELS = {
    "LOTS": "Low Order Thinking Skills (Mengingat & Memahami)",
    "MOTS": "Mid Order Thinking Skills (Menerapkan & Menganalisis)",
    "HOTS": "High Order Thinking Skills (Mengevaluasi & Mencipta)",
}

LEVEL_INSTRUKSI = {
    "LOTS": (
        "Tingkat kesulitan: LOTS — Low Order Thinking Skills.\n"
        "Fokus pada MENGINGAT dan MEMAHAMI. Konten/soal bersifat faktual dan definitif. "
        "Siswa hanya perlu mengingat, mendefinisikan, atau menjelaskan ulang konsep dasar "
        "secara langsung berdasarkan materi. Gunakan kalimat yang sederhana dan lugas."
    ),
    "MOTS": (
        "Tingkat kesulitan: MOTS — Mid Order Thinking Skills.\n"
        "Fokus pada MENERAPKAN dan MENGANALISIS. Konten/soal mengharuskan siswa "
        "menerapkan rumus/konsep dalam situasi baru, membandingkan antar konsep, "
        "atau menjelaskan hubungan sebab-akibat. Gunakan konteks yang memerlukan pemikiran lebih."
    ),
    "HOTS": (
        "Tingkat kesulitan: HOTS — High Order Thinking Skills.\n"
        "Fokus pada MENGEVALUASI dan MENCIPTA. Konten/soal membutuhkan penalaran kritis, "
        "penilaian situasi kompleks, sintesis dari berbagai konsep, atau pembuatan solusi orisinal. "
        "Hindari soal/konten faktual — semua harus membutuhkan pemikiran tingkat tinggi."
    ),
}


# ================================================================
# 3. Utility Functions
# ================================================================

def clean_json_from_llm(raw_text: str) -> dict:
    """Fallback JSON parser tangguh untuk mengekstrak string JSON kotor dari LLM."""
    clean_text = re.sub(r'```(?:json)?', '', raw_text).strip()

    start_idx = clean_text.find('{')
    end_idx   = clean_text.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(clean_text[start_idx:end_idx+1])
        except json.JSONDecodeError:
            pass

    start_idx = clean_text.find('[')
    end_idx   = clean_text.rfind(']')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(clean_text[start_idx:end_idx+1])
        except json.JSONDecodeError:
            pass

    return {"error": "Gagal parsing JSON dari LLM", "raw": raw_text[:200]}


def generate_soal_id(tipe: str, level: str, topik_slug: str) -> str:
    """Generate ID unik per soal: {tipe}-{level}-{topik_slug}-{4hex}"""
    slug  = topik_slug.lower().replace(" ", "_")[:20]
    lvl   = level.lower()
    return f"{tipe}-{lvl}-{slug}-{uuid.uuid4().hex[:4]}"


def _get_sumber_from_docs(docs: list) -> str:
    """Ambil nama sumber file dari docs RAG."""
    for d in docs:
        if d.metadata and "source" in d.metadata:
            return os.path.basename(d.metadata["source"]).replace(".md", "").replace("_", " ").title()
    return "Materi Sekolah"


# ================================================================
# 4. Formatters — Multi-Level Output
# ================================================================

def _wrap_level(label: str, data: list | dict) -> dict:
    """Helper wrap satu level dengan label-nya."""
    if isinstance(data, list):
        return {"label": label, "jumlah": len(data), "data": data}
    return {"label": label, "data": data}


def util_format_bacaan_multi(
    jenjang: str, kelas: str, matpel: str, elemen: str, materi: str,
    lots: dict, mots: dict, hots: dict
) -> dict:
    return {
        "tipe":          "bacaan",
        "jenjang":       jenjang,
        "kelas":         kelas,
        "mata_pelajaran": matpel,
        "elemen":        elemen,
        "materi":        materi,
        "bacaan_per_level": {
            "LOTS": {
                "label":       LEVEL_LABELS["LOTS"],
                "markdown":    lots.get("markdown", ""),
                "sumber":      lots.get("sumber", []),
            },
            "MOTS": {
                "label":       LEVEL_LABELS["MOTS"],
                "markdown":    mots.get("markdown", ""),
                "sumber":      mots.get("sumber", []),
            },
            "HOTS": {
                "label":       LEVEL_LABELS["HOTS"],
                "markdown":    hots.get("markdown", ""),
                "sumber":      hots.get("sumber", []),
            },
        },
    }


def util_format_flashcard_multi(
    jenjang: str, kelas: str, matpel: str, elemen: str, materi: str,
    lots: list, mots: list, hots: list
) -> dict:
    return {
        "tipe":           "flashcard_set",
        "jenjang":        jenjang,
        "kelas":          kelas,
        "mata_pelajaran": matpel,
        "elemen":         elemen,
        "materi":         materi,
        "kartu_per_level": {
            "LOTS": {"label": LEVEL_LABELS["LOTS"], "jumlah_kartu": len(lots), "kartu": lots},
            "MOTS": {"label": LEVEL_LABELS["MOTS"], "jumlah_kartu": len(mots), "kartu": mots},
            "HOTS": {"label": LEVEL_LABELS["HOTS"], "jumlah_kartu": len(hots), "kartu": hots},
        },
    }


def util_format_quiz_multi(
    jenjang: str, kelas: str, matpel: str, elemen: str, materi: str,
    lots: list, mots: list, hots: list
) -> dict:
    return {
        "tipe":           "quiz_soal",
        "jenjang":        jenjang,
        "kelas":          kelas,
        "mata_pelajaran": matpel,
        "elemen":         elemen,
        "materi":         materi,
        "soal_per_level": {
            "LOTS": {"label": LEVEL_LABELS["LOTS"], "jumlah_soal": len(lots), "soal": lots},
            "MOTS": {"label": LEVEL_LABELS["MOTS"], "jumlah_soal": len(mots), "soal": mots},
            "HOTS": {"label": LEVEL_LABELS["HOTS"], "jumlah_soal": len(hots), "soal": hots},
        },
    }


def util_format_quiz_uraian_multi(
    jenjang: str, kelas: str, matpel: str, elemen: str, materi: str,
    lots: list, mots: list, hots: list
) -> dict:
    return {
        "tipe":           "quiz_uraian",
        "jenjang":        jenjang,
        "kelas":          kelas,
        "mata_pelajaran": matpel,
        "elemen":         elemen,
        "materi":         materi,
        "soal_per_level": {
            "LOTS": {"label": LEVEL_LABELS["LOTS"], "jumlah_soal": len(lots), "soal": lots},
            "MOTS": {"label": LEVEL_LABELS["MOTS"], "jumlah_soal": len(mots), "soal": mots},
            "HOTS": {"label": LEVEL_LABELS["HOTS"], "jumlah_soal": len(hots), "soal": hots},
        },
    }


def util_format_mindmap(matpel: str, materi: str, nodes: list) -> dict:
    return {
        "tipe":           "mindmap_hierarki",
        "mata_pelajaran": matpel,
        "materi":         materi,
        "nodes":          nodes,
    }


# ================================================================
# 5. Formatters — Evaluasi (tidak berubah substansi)
# ================================================================



def util_format_evaluasi_uraian(matpel: str, bab: str, detail: list, overall: dict) -> dict:
    return {
        "tipe":              "hasil_evaluasi_uraian",
        "mata_pelajaran":    matpel,
        "materi":            bab,
        "total_skor":        overall.get("skor_total",        sum(d.get("skor", 0) for d in detail)),
        "skor_maksimal":     overall.get("skor_maksimal",     sum(d.get("skor_maksimal", 20) for d in detail)),
        "tingkat_pemahaman": overall.get("tingkat_pemahaman", "Tidak diketahui"),
        "ringkasan_feedback": overall.get("catatan", ""),
        "nomor_terlemah":    overall.get("nomor_terlemah", None),
        "nomor_terkuat":     overall.get("nomor_terkuat",  None),
        "detail":            detail,
    }


# ================================================================
# 6. Formatter — RAG Query
# ================================================================

def util_format_rag_query(query: str, matpel: str, materi: str, chunks: list) -> dict:
    return {
        "tipe":    "rag_konteks",
        "query":   query,
        "mata_pelajaran": matpel,
        "materi":  materi,
        "jumlah_chunk": len(chunks),
        "konteks": chunks,
        "petunjuk_untuk_chatbot": (
            "Gunakan konteks di atas untuk menjawab pertanyaan siswa. "
            "Jangan mengarang di luar konteks yang diberikan."
        ),
    }


# ================================================================
# 7. Helper — Rekomendasi (tetap ada, tanpa emotion)
# ================================================================

def _ambil_prioritas_belajar(riwayat: list, top_n: int = 5) -> list:
    """Ambil N bab paling lemah dari riwayat progress."""
    urutan = {"Belum Paham": 0, "Paham Dasar": 1, "Paham": 2, "Paham Mendalam": 3}
    valid  = [r for r in riwayat if isinstance(r, dict)]
    return sorted(
        valid,
        key=lambda x: (
            urutan.get(x.get("tingkat_pemahaman", "Belum Paham"), 0),
            x.get("skor_terakhir", 100)
        )
    )[:top_n]


def util_format_recommender(student_id: str, first_time: bool, pesan: str, rekomendasi: list) -> dict:
    return {
        "tipe":          "rekomendasi_topik",
        "student_id":    student_id,
        "first_time":    first_time,
        "pesan_empatik": pesan,
        "rekomendasi":   rekomendasi,
    }
