import os
import json
import re
import uuid
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

# ================================================================
# 1. Knowledge Base (RAG) Setup — Qdrant (configurable via env)
# ================================================================

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        passages = [f"passage: {text.strip()}" for text in texts]
        return self.model.encode(
            passages,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text):
        query = f"query: {text.strip()}"
        return self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0].tolist()


class DatabaseSekolah:
    def __init__(self):
        _embed_model    = os.getenv("EMBED_MODEL",       "LazarusNLP/all-indo-e5-small-v4")
        _qdrant_host    = os.getenv("QDRANT_HOST",       "10.243.18.109")
        _qdrant_port    = int(os.getenv("QDRANT_PORT",   "6333"))
        _collection     = os.getenv("QDRANT_COLLECTION", "embed_pdf_chunks_v1")

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
            txt = d.page_content
            d.metadata["raw_sebelum_regex"]      = raw_backup
            d.metadata["apakah_kena_regex"]       = False

            match = re.search(r"splits=\[(.*?)\]\s+is_triggered", txt, re.DOTALL)
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


# Inisialisasi KB global
kb_sekolah = DatabaseSekolah()


# ================================================================
# 2. Level Difficulty Constants
# ================================================================

LEVEL_LABELS = {
    "LOTS": "Low Order Thinking Skills (C1–C2: Mengingat & Memahami)",
    "MOTS": "Mid Order Thinking Skills (C3–C4: Menerapkan & Menganalisis)",
    "HOTS": "High Order Thinking Skills (C5–C6: Mengevaluasi & Mencipta)",
}

LEVEL_INSTRUKSI = {
    "LOTS": (
        "Tingkat kesulitan: LOTS — Low Order Thinking Skills (Bloom C1–C2).\n"
        "Fokus pada MENGINGAT dan MEMAHAMI. Konten/soal bersifat faktual dan definitif. "
        "Siswa hanya perlu mengingat, mendefinisikan, atau menjelaskan ulang konsep dasar "
        "secara langsung berdasarkan materi. Gunakan kalimat yang sederhana dan lugas."
    ),
    "MOTS": (
        "Tingkat kesulitan: MOTS — Mid Order Thinking Skills (Bloom C3–C4).\n"
        "Fokus pada MENERAPKAN dan MENGANALISIS. Konten/soal mengharuskan siswa "
        "menerapkan rumus/konsep dalam situasi baru, membandingkan antar konsep, "
        "atau menjelaskan hubungan sebab-akibat. Gunakan konteks yang memerlukan pemikiran lebih."
    ),
    "HOTS": (
        "Tingkat kesulitan: HOTS — High Order Thinking Skills (Bloom C5–C6).\n"
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
                "judul":       lots.get("judul_konten", materi),
                "jumlah_sub_bab": len(lots.get("konten", [])),
                "konten":      lots.get("konten", []),
                "sumber":      lots.get("sumber", []),
            },
            "MOTS": {
                "label":       LEVEL_LABELS["MOTS"],
                "judul":       mots.get("judul_konten", materi),
                "jumlah_sub_bab": len(mots.get("konten", [])),
                "konten":      mots.get("konten", []),
                "sumber":      mots.get("sumber", []),
            },
            "HOTS": {
                "label":       LEVEL_LABELS["HOTS"],
                "judul":       hots.get("judul_konten", materi),
                "jumlah_sub_bab": len(hots.get("konten", [])),
                "konten":      hots.get("konten", []),
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

def util_format_evaluasi_quiz(matpel: str, bab: str, detail: list) -> dict:
    total_benar = sum(1 for d in detail if d.get("benar", False))
    total_skor  = sum(d.get("skor", 0) for d in detail)
    skor_maks   = sum(d.get("skor_maksimal", 10) for d in detail)
    return {
        "tipe":         "hasil_evaluasi_quiz",
        "mata_pelajaran": matpel,
        "materi":       bab,
        "total_benar":  total_benar,
        "total_soal":   len(detail),
        "total_skor":   total_skor,
        "skor_maksimal": skor_maks,
        "detail":       detail,
    }


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
