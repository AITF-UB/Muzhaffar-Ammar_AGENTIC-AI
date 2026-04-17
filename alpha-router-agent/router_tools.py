import json
import re
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

# ================================================================
# 1. Knowledge Base (RAG) Setup - Qdrant (LazarusNLP all-indo-e5-small-v4)
# ================================================================

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "LazarusNLP/all-indo-e5-small-v4"):
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
        print("📚 Membangun Database Sekolah (Qdrant) untuk Router Agent...")
        self.embeddings = SentenceTransformerEmbeddings("microsoft/harrier-oss-v1-0.6b")
        
        self.client = QdrantClient(host="10.243.18.109", port=6333)
        self.collection_name = "embed_pdf_chunks_v1"
        
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        
        try:
            count = self.client.count(self.collection_name).count
            print(f"   ✅ Terhubung ke Qdrant. Jumlah chunks di {self.collection_name}: {count}")
        except Exception as e:
            print(f"   ❌ Gagal terhubung ke Qdrant: {e}")

    def search(self, query: str, k: int = 2) -> list[Document]:
        import re, ast
        # Menggunakan pure Similarity Search agar relevansi paling utuh didapat (bukan MMR yang bisa menyimpang ke footer/daftar pustaka)
        docs = self.vectorstore.similarity_search(
            query,
            k=k
        )
        
        # Bersihkan page_content jika bentuk stringnya berisi repr object (splits=...)
        for d in docs:
            raw_text_backup = d.page_content
            txt = d.page_content
            
            d.metadata["raw_sebelum_regex"] = raw_text_backup
            d.metadata["apakah_kena_regex_di_router"] = False

            # Cek jika formatnya seperti: splits=['kalimat 1', 'kalimat 2'] is_triggered=...
            match = re.search(r"splits=\[(.*?)\]\s+is_triggered", txt, re.DOTALL)
            if match:
                d.metadata["apakah_kena_regex_di_router"] = True
                try:
                    splits_str = "[" + match.group(1) + "]"
                    splits_list = ast.literal_eval(splits_str)
                    if isinstance(splits_list, list):
                        d.page_content = " ".join(splits_list)
                except Exception:
                    pass
                    
        return docs

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

def util_format_flashcard(matpel: str, bab: str, flashcards: list) -> dict:
    return {
        "tipe": "flashcard_set",
        "matpel": matpel,
        "bab": bab,
        "jumlah_kartu": len(flashcards) if isinstance(flashcards, list) else 0,
        "kartu": flashcards
    }
    
def util_format_mindmap(matpel: str, bab: str, nodes: list) -> dict:
    return {
        "tipe": "mindmap_hierarki",
        "matpel": matpel,
        "bab": bab,
        "nodes": nodes
    }

# ----------------------------------------------------------------
# Helper: generate soal_id unik (Python-side, bukan LLM)
# ----------------------------------------------------------------
def generate_soal_id(tipe: str, topik_slug: str) -> str:
    """Generate ID unik per soal: {tipe}-{topik_slug}-{4hex}"""
    slug = topik_slug.lower().replace(" ", "_")[:20]
    return f"{tipe}-{slug}-{uuid.uuid4().hex[:4]}"

# ----------------------------------------------------------------
# Quiz PG — generate formatter (tambah soal_id per soal)
# ----------------------------------------------------------------
def util_format_quiz(matpel: str, bab: str, soal: list) -> dict:
    """Inject soal_id ke tiap soal PG, lalu wrap payload."""
    for item in soal:
        if isinstance(item, dict) and "soal_id" not in item:
            item["soal_id"] = generate_soal_id("pg", bab)
    return {
        "tipe": "quiz_soal",
        "matpel": matpel,
        "bab": bab,
        "jumlah_soal": len(soal) if isinstance(soal, list) else 0,
        "soal": soal
    }

# ----------------------------------------------------------------
# Quiz Uraian — generate formatter
# ----------------------------------------------------------------
def util_format_quiz_uraian(matpel: str, bab: str, soal: list) -> dict:
    """Inject soal_id ke tiap soal uraian, lalu wrap payload."""
    for item in soal:
        if isinstance(item, dict) and "soal_id" not in item:
            item["soal_id"] = generate_soal_id("uraian", bab)
    return {
        "tipe": "quiz_uraian",
        "matpel": matpel,
        "bab": bab,
        "jumlah_soal": len(soal) if isinstance(soal, list) else 0,
        "soal": soal
    }

# ----------------------------------------------------------------
# Evaluasi Quiz PG — formatter (deterministik)
# ----------------------------------------------------------------
def util_format_evaluasi_quiz(matpel: str, bab: str, detail: list) -> dict:
    total_benar = sum(1 for d in detail if d.get("benar", False))
    total_skor  = sum(d.get("skor", 0) for d in detail)
    skor_maks   = sum(d.get("skor_maksimal", 10) for d in detail)
    return {
        "tipe": "hasil_evaluasi_quiz",
        "matpel": matpel,
        "bab": bab,
        "total_benar": total_benar,
        "total_soal": len(detail),
        "total_skor": total_skor,
        "skor_maksimal": skor_maks,
        "detail": detail
    }

# ----------------------------------------------------------------
# Evaluasi Uraian — formatter (LLM)
# ----------------------------------------------------------------
def util_format_evaluasi_uraian(matpel: str, bab: str, detail: list, overall: dict) -> dict:
    return {
        "tipe": "hasil_evaluasi_uraian",
        "matpel": matpel,
        "bab": bab,
        "total_skor": overall.get("skor_total", sum(d.get("skor", 0) for d in detail)),
        "skor_maksimal": overall.get("skor_maksimal", sum(d.get("skor_maksimal", 20) for d in detail)),
        "tingkat_pemahaman": overall.get("tingkat_pemahaman", "Tidak diketahui"),
        "ringkasan_feedback": overall.get("catatan", ""),
        "nomor_terlemah": overall.get("nomor_terlemah", None),
        "nomor_terkuat": overall.get("nomor_terkuat", None),
        "detail": detail
    }

# ----------------------------------------------------------------
# Konten Belajar — formatter (konten panjang terstruktur)
# ----------------------------------------------------------------
def util_format_konten_belajar(matpel: str, bab: str, judul: str, konten_list: list, sumber: list) -> dict:
    return {
        "tipe": "konten_belajar",
        "matpel": matpel,
        "bab": bab,
        "judul_konten": judul,
        "jumlah_sub_bab": len(konten_list),
        "konten": konten_list,
        "sumber": sumber
    }

# ----------------------------------------------------------------
# RAG Query — formatter (pure RAG, no LLM)
# ----------------------------------------------------------------
def util_format_rag_query(query: str, matpel: str, bab: str, chunks: list) -> dict:
    return {
        "tipe": "rag_konteks",
        "query": query,
        "matpel": matpel,
        "bab": bab,
        "jumlah_chunk": len(chunks),
        "konteks": chunks,
        "petunjuk_untuk_chatbot": (
            "Gunakan konteks di atas untuk menjawab pertanyaan siswa. "
            "Jangan mengarang di luar konteks yang diberikan."
        )
    }
