"""
Router Agent — FastAPI Microservice
====================================
Jalankan dengan:
    uvicorn router_api:app --reload --port 8000

Swagger UI tersedia di:
    http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Optional
import json

from router_agent import router_agent_app
from router_state import AgentState

# ================================================================
# 1. APP INITIALIZATION
# ================================================================

app = FastAPI(
    title="Router Agent API",
    description=(
        "Microservice untuk Agentic AI Pipeline berbasis LangGraph.\n\n"
        "**6 Mekanik Generate:**\n"
        "- **rekomendasi** — v2: first_time + returning (filter 5 bab terlemah)\n"
        "- **konten_belajar** — Materi panjang terstruktur 5 sub-bab via RAG+LLM\n"
        "- **rag_query** — Pure RAG retriever (no LLM) untuk T&A chatbot Tim 5\n"
        "- **flashcard** — NotebookLM-style flashcard dengan kutipan sumber\n"
        "- **mindmap** — Peta konsep hierarki (parent-child)\n"
        "- **quiz** — Soal pilihan ganda (PG) grounded RAG + soal_id\n"
        "- **quiz_uraian** — Soal esai/uraian grounded RAG + kunci + soal_id\n\n"
        "**2 Mekanik Evaluasi:**\n"
        "- **evaluasi_quiz** — Nilai jawaban PG (deterministik, no LLM)\n"
        "- **evaluasi_uraian** — Nilai jawaban uraian (LLM, feedback per soal)\n\n"
        "**Try-Out (langsung execute):**\n"
        "- `GET /tryout/evaluasi_quiz` — 5 soal Hukum Newton\n"
        "- `GET /tryout/evaluasi_uraian` — 5 soal Fotosintesis\n"
    ),
    version="3.0.0",
)

# CORS — izinkan semua origin untuk development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# 2. PYDANTIC REQUEST / RESPONSE MODELS
# ================================================================

class EmotionInput(BaseModel):
    emosi: str = Field(
        default="netral",
        description="Kondisi emosi siswa saat ini",
        examples=["sedih", "fokus", "penasaran", "semangat", "netral"]
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence level dari deteksi emosi (0.0 - 1.0)"
    )


class AgentRequest(BaseModel):
    task: str = Field(
        description="Jenis task yang diminta",
        examples=["rekomendasi", "flashcard", "mindmap", "quiz", "quiz_uraian", "evaluasi_quiz", "evaluasi_uraian"]
    )
    request_params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Parameter spesifik sesuai task:\n"
            "- rekomendasi: {nilai_pretest: int, nilai_ujian: int}\n"
            "- flashcard: {topik: str}\n"
            "- mindmap: {topik: str}\n"
            "- quiz: {topik: str, jumlah_soal: int}\n"
            "- quiz_uraian: {topik: str, jumlah_soal: int}\n"
            "- evaluasi_quiz: {topik: str, skor_per_soal: int, soal_pg: list, jawaban_siswa: list}\n"
            "- evaluasi_uraian: {topik: str, soal_uraian: list, jawaban_siswa: list}\n"
        ),
        examples=[{"topik": "Hukum Newton", "jumlah_soal": 5}]
    )
    emotion: EmotionInput = Field(default_factory=EmotionInput)


class AgentResponse(BaseModel):
    status: str
    task: str
    nodes_executed: List[str]
    output: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    message: str
    available_tasks: List[str]


# ---- Model spesifik untuk Rekomendasi v2 ----

class HasilPretestItem(BaseModel):
    matpel: str
    skor: int
    topik_lemah: List[str] = Field(default_factory=list)


class RiwayatProgressItem(BaseModel):
    matpel: str
    bab: str
    skor_terakhir: int
    jumlah_evaluasi: int = 1
    tingkat_pemahaman: str = "Belum Paham"
    emosi_dominan: str = "netral"
    jumlah_prompt: int = 0


class RekomendasiRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "student_id": "siswa-042",
            "first_time": False,
            "matpel_dipilih": ["Matematika", "Fisika"],
            "riwayat_progress": [
                {"matpel": "Matematika", "bab": "Teorema Pythagoras", "skor_terakhir": 20, "tingkat_pemahaman": "Belum Paham", "emosi_dominan": "bingung", "jumlah_prompt": 22},
                {"matpel": "Fisika", "bab": "Hukum Newton", "skor_terakhir": 35, "tingkat_pemahaman": "Belum Paham", "emosi_dominan": "frustrasi", "jumlah_prompt": 18},
                {"matpel": "Matematika", "bab": "Persamaan Kuadrat", "skor_terakhir": 50, "tingkat_pemahaman": "Paham Dasar", "emosi_dominan": "netral", "jumlah_prompt": 10}
            ],
            "emotion": {"emosi": "semangat", "confidence": 0.85}
        }
    })
    student_id: str = "unknown"
    first_time: bool = False
    matpel_dipilih: List[str] = Field(default_factory=list)
    hasil_pretest: Optional[List[HasilPretestItem]] = Field(
        default=None,
        description="Diisi jika first_time=true"
    )
    riwayat_progress: Optional[List[RiwayatProgressItem]] = Field(
        default=None,
        description="Diisi jika first_time=false — Tim 6 kirim semua riwayat, kita filter 5 terlemah"
    )
    emotion: EmotionInput = Field(default_factory=EmotionInput)


# ---- Model spesifik untuk Konten Belajar ----

class KontenBelajarRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "matpel": "Fisika",
            "bab": "Hukum Newton I dan Inersia",
            "level": "SMA",
            "emotion": {"emosi": "semangat", "confidence": 0.85}
        }
    })
    matpel: str = Field(default="Fisika", description="Nama mata pelajaran (Matematika, Fisika, Biologi, dst)")
    bab: str = Field(default="Hukum Newton I dan Inersia", description="Nama bab/chapter yang akan dibuat materinya")
    level: str = Field(default="SMA", description="Jenjang sekolah: SD, SMP, SMA")
    emotion: EmotionInput = Field(default_factory=EmotionInput)


# ---- Model spesifik untuk RAG Query ----

class RagQueryRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "apa yang dimaksud dengan inersia dan bagaimana contohnya?",
            "matpel": "Fisika",
            "k": 3
        }
    })
    query: str = Field(description="Pertanyaan siswa dalam bahasa natural")
    matpel: str = Field(default="", description="Mata pelajaran sebagai konteks pencarian (opsional)")
    k: int = Field(default=3, ge=1, le=10, description="Jumlah dokumen relevan yang dikembalikan")




class SoalPGEval(BaseModel):
    soal_id: str = Field(example="pg-hukum_newton-a1b2")
    nomor: int = Field(example=1)
    jawaban_benar: str = Field(example="A")
    pembahasan: str = Field(default="", example="Hukum I Newton menyatakan benda mempertahankan keadaannya.")


class JawabanSiswaPG(BaseModel):
    soal_id: str = Field(example="pg-hukum_newton-a1b2")
    jawaban: str = Field(example="A")


class EvaluasiQuizRequest(BaseModel):
    matpel: str = Field(default="Fisika", example="Fisika")
    bab: str = Field(default="Hukum Newton", example="Hukum Newton")
    skor_per_soal: int = Field(default=10, example=10)
    soal_pg: List[SoalPGEval] = Field(
        default_factory=list,
        example=[
            {"soal_id": "pg-hukum_newton-a1b2", "nomor": 1, "jawaban_benar": "A", "pembahasan": "Hukum I Newton: benda mempertahankan keadaannya."},
            {"soal_id": "pg-hukum_newton-c3d4", "nomor": 2, "jawaban_benar": "B", "pembahasan": "F = m x a adalah rumus Hukum Newton II."},
        ]
    )
    jawaban_siswa: List[JawabanSiswaPG] = Field(
        default_factory=list,
        example=[
            {"soal_id": "pg-hukum_newton-a1b2", "jawaban": "A"},
            {"soal_id": "pg-hukum_newton-c3d4", "jawaban": "C"},
        ]
    )
    emotion: EmotionInput = Field(default_factory=EmotionInput)


# ---- Model spesifik untuk Evaluasi Uraian ----

class SoalUraianEval(BaseModel):
    soal_id: str = Field(example="uraian-fotosintesis-e5f6")
    nomor: int = Field(example=1)
    pertanyaan: str = Field(example="Jelaskan apa yang dimaksud dengan fotosintesis!")
    kunci_jawaban: str = Field(example="Fotosintesis adalah proses tumbuhan membuat makanan menggunakan cahaya matahari. Rumus: 6CO₂ + 6H₂O + Cahaya → C₆H₁₂O₆ + 6O₂.")
    skor_maksimal: int = Field(default=20, example=20)


class JawabanSiswaUraian(BaseModel):
    soal_id: str = Field(example="uraian-fotosintesis-e5f6")
    jawaban: str = Field(example="Fotosintesis adalah proses pembuatan makanan oleh tumbuhan dengan bantuan cahaya matahari.")


class EvaluasiUraianRequest(BaseModel):
    matpel: str = Field(default="Biologi", example="Biologi")
    bab: str = Field(default="Fotosintesis", example="Fotosintesis")
    soal_uraian: List[SoalUraianEval] = Field(
        default_factory=list,
        example=[
            {
                "soal_id": "uraian-fotosintesis-e5f6",
                "nomor": 1,
                "pertanyaan": "Jelaskan apa yang dimaksud dengan fotosintesis!",
                "kunci_jawaban": "Fotosintesis adalah proses tumbuhan membuat makanan menggunakan cahaya matahari, menghasilkan glukosa dan oksigen.",
                "skor_maksimal": 20
            }
        ]
    )
    jawaban_siswa: List[JawabanSiswaUraian] = Field(
        default_factory=list,
        example=[
            {"soal_id": "uraian-fotosintesis-e5f6", "jawaban": "Fotosintesis adalah proses pembuatan makanan oleh tumbuhan dengan bantuan cahaya matahari."}
        ]
    )
    emotion: EmotionInput = Field(default_factory=EmotionInput)


# ---- Model untuk Flashcard ----

class FlashcardRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "matpel": "Matematika",
            "bab": "Logaritma",
            "nilai_siswa": 45,
            "emotion": {"emosi": "fokus", "confidence": 0.8}
        }
    })
    matpel: str = Field(default="Matematika", description="Nama mata pelajaran (Matematika, Fisika, Biologi, dst)")
    bab: str = Field(default="Logaritma", description="Nama bab/chapter yang akan dibuatkan flashcard")
    nilai_siswa: Optional[int] = Field(
        default=None,
        ge=0, le=100,
        description="Nilai pretest atau nilai quiz sebelumnya di bab ini (0-100). "
                    "None=Dasar, 0-40=Dasar, 41-70=Menengah, 71-100=HOTS"
    )
    emotion: EmotionInput = Field(default_factory=EmotionInput)


# ---- Model untuk Mindmap ----

class MindmapRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "matpel": "Biologi",
            "bab": "Fotosintesis",
            "nilai_siswa": 45,
            "emotion": {"emosi": "penasaran", "confidence": 0.8}
        }
    })
    matpel: str = Field(default="Biologi", description="Nama mata pelajaran (Matematika, Fisika, Biologi, dst)")
    bab: str = Field(default="Fotosintesis", description="Nama bab/chapter yang akan dipetakan konsepnya")
    nilai_siswa: Optional[int] = Field(
        default=None,
        ge=0, le=100,
        description="Nilai pretest atau nilai quiz sebelumnya di bab ini (0-100). "
                    "Struktur mindmap selalu komprehensif — hanya gaya bahasa yang menyesuaikan. "
                    "None/≤70=bahasa sederhana, >70=bahasa teknis"
    )
    emotion: EmotionInput = Field(default_factory=EmotionInput)


# ---- Model untuk Quiz PG ----

class QuizRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "matpel": "Matematika",
            "bab": "Teorema Pythagoras",
            "nilai_siswa": 45,
            "emotion": {"emosi": "semangat", "confidence": 0.9}
        }
    })
    matpel: str = Field(default="Matematika", description="Nama mata pelajaran")
    bab: str = Field(default="Teorema Pythagoras", description="Nama bab yang akan dibuatkan soal PG")
    nilai_siswa: Optional[int] = Field(
        default=None,
        ge=0, le=100,
        description="Nilai pretest atau nilai quiz sebelumnya di bab ini (0-100). "
                    "None=Dasar, 0-40=Dasar, 41-70=Menengah, 71-100=HOTS"
    )
    emotion: EmotionInput = Field(default_factory=EmotionInput)


# ---- Model untuk Quiz Uraian ----

class QuizUraianRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "matpel": "Fisika",
            "bab": "Hukum Newton",
            "nilai_siswa": 45,
            "emotion": {"emosi": "fokus", "confidence": 0.8}
        }
    })
    matpel: str = Field(default="Fisika", description="Nama mata pelajaran")
    bab: str = Field(default="Hukum Newton", description="Nama bab yang akan dibuatkan soal uraian/esai")
    nilai_siswa: Optional[int] = Field(
        default=None,
        ge=0, le=100,
        description="Nilai pretest atau nilai quiz sebelumnya di bab ini (0-100). "
                    "None=Dasar, 0-40=Dasar, 41-70=Menengah, 71-100=HOTS"
    )
    emotion: EmotionInput = Field(default_factory=EmotionInput)


# ================================================================
# 3. HELPER — Jalankan graph dan collect node execution trace
# ================================================================

def _run_graph(task: str, request_params: dict, emotion: dict) -> tuple[dict, list[str]]:
    """Eksekusi LangGraph pipeline, return (final_payload, nodes_executed)"""
    initial_state: AgentState = {
        "task": task,
        "request_params": request_params,
        "emotion_input": emotion,
        "top_recommendations": "",
        "flashcards_data": "",
        "mindmap_data": "",
        "quiz_data": "",
        "quiz_uraian_data": "",
        "evaluasi_quiz_result": "",
        "evaluasi_uraian_result": "",
        "konten_belajar_data": "",
        "rag_query_result": "",
        "final_payload": {},
    }

    final_payload = {}
    nodes_executed = []

    for step in router_agent_app.stream(initial_state, stream_mode="updates"):
        for node_name, node_data in step.items():
            nodes_executed.append(node_name)
            if node_name == "structurer":
                final_payload = node_data.get("final_payload", {})

    return final_payload, nodes_executed


# ================================================================
# 4. DATA TRYOUT — Contoh soal hardcoded untuk eksekusi langsung
# ================================================================

# Contoh 5 soal PG Hukum Newton (jawaban_benar sudah di-set)
_TRYOUT_SOAL_PG = [
    {
        "soal_id": "pg-hukum_newton-tryout-0001",
        "nomor": 1,
        "pertanyaan": "Hukum Newton I menyatakan bahwa sebuah benda akan...",
        "pilihan": {
            "A": "Tetap diam atau bergerak lurus beraturan jika tidak ada gaya luar",
            "B": "Bergerak dengan percepatan jika gaya bekerja",
            "C": "Menghasilkan reaksi yang sama dengan aksinya",
            "D": "Bergerak melingkar jika gaya gravitasi bekerja"
        },
        "jawaban_benar": "A",
        "pembahasan": "Hukum I Newton (Inersia): benda mempertahankan keadaannya jika tidak ada gaya luar yang bekerja.",
        "sumber": "Fisika — Hukum Newton"
    },
    {
        "soal_id": "pg-hukum_newton-tryout-0002",
        "nomor": 2,
        "pertanyaan": "Rumus matematis Hukum Newton II adalah...",
        "pilihan": {
            "A": "F = m/a",
            "B": "F = m × a",
            "C": "F = a/m",
            "D": "F = m + a"
        },
        "jawaban_benar": "B",
        "pembahasan": "Hukum II Newton: F = m × a, gaya berbanding lurus dengan massa dan percepatan.",
        "sumber": "Fisika — Hukum Newton"
    },
    {
        "soal_id": "pg-hukum_newton-tryout-0003",
        "nomor": 3,
        "pertanyaan": "Contoh penerapan Hukum Newton III dalam kehidupan sehari-hari adalah...",
        "pilihan": {
            "A": "Penumpang terdorong ke depan saat mobil rem mendadak",
            "B": "Roket meluncur karena gas terdorong ke bawah",
            "C": "Benda jatuh karena gaya gravitasi",
            "D": "Bola menggelinding karena tidak ada gaya gesek"
        },
        "jawaban_benar": "B",
        "pembahasan": "Hukum III Newton (Aksi-Reaksi): roket meluncur ke atas karena gas terdorong ke bawah.",
        "sumber": "Fisika — Hukum Newton"
    },
    {
        "soal_id": "pg-hukum_newton-tryout-0004",
        "nomor": 4,
        "pertanyaan": "Penumpang terdorong ke depan saat mobil rem mendadak adalah contoh dari...",
        "pilihan": {
            "A": "Hukum Newton II",
            "B": "Hukum Newton III",
            "C": "Hukum Newton I",
            "D": "Hukum Gravitasi Newton"
        },
        "jawaban_benar": "C",
        "pembahasan": "Ini adalah contoh inersia (Hukum Newton I): tubuh penumpang cenderung mempertahankan gerak ke depan.",
        "sumber": "Fisika — Hukum Newton"
    },
    {
        "soal_id": "pg-hukum_newton-tryout-0005",
        "nomor": 5,
        "pertanyaan": "Jika massa benda 5 kg dan percepatan 3 m/s², maka gaya yang bekerja adalah...",
        "pilihan": {
            "A": "8 Newton",
            "B": "1.67 Newton",
            "C": "15 Newton",
            "D": "2 Newton"
        },
        "jawaban_benar": "C",
        "pembahasan": "F = m × a = 5 × 3 = 15 Newton.",
        "sumber": "Fisika — Hukum Newton"
    },
]

# Jawaban siswa tryout PG (sengaja campuran benar dan salah)
_TRYOUT_JAWABAN_PG = [
    {"soal_id": "pg-hukum_newton-tryout-0001", "jawaban": "A"},  # benar
    {"soal_id": "pg-hukum_newton-tryout-0002", "jawaban": "A"},  # salah (B)
    {"soal_id": "pg-hukum_newton-tryout-0003", "jawaban": "B"},  # benar
    {"soal_id": "pg-hukum_newton-tryout-0004", "jawaban": "C"},  # benar
    {"soal_id": "pg-hukum_newton-tryout-0005", "jawaban": "D"},  # salah (C)
]

# Contoh 5 soal uraian Fotosintesis
_TRYOUT_SOAL_URAIAN = [
    {
        "soal_id": "uraian-fotosintesis-tryout-0001",
        "nomor": 1,
        "pertanyaan": "Jelaskan apa yang dimaksud dengan fotosintesis dan tuliskan rumus kimianya!",
        "kunci_jawaban": "Fotosintesis adalah proses tumbuhan membuat makanan menggunakan cahaya matahari. Rumus: 6CO₂ + 6H₂O + Cahaya → C₆H₁₂O₆ + 6O₂ (Karbondioksida + Air + Cahaya → Glukosa + Oksigen).",
        "skor_maksimal": 20,
        "sumber": "Biologi — Fotosintesis"
    },
    {
        "soal_id": "uraian-fotosintesis-tryout-0002",
        "nomor": 2,
        "pertanyaan": "Di manakah fotosintesis terjadi dan apa peran klorofil dalam proses tersebut?",
        "kunci_jawaban": "Fotosintesis terjadi di kloroplas. Klorofil adalah pigmen hijau dalam kloroplas yang menyerap cahaya matahari, merupakan komponen utama yang memungkinkan proses fotosintesis berlangsung.",
        "skor_maksimal": 20,
        "sumber": "Biologi — Fotosintesis"
    },
    {
        "soal_id": "uraian-fotosintesis-tryout-0003",
        "nomor": 3,
        "pertanyaan": "Jelaskan perbedaan antara Reaksi Terang dan Reaksi Gelap (Siklus Calvin) dalam fotosintesis!",
        "kunci_jawaban": "Reaksi Terang terjadi di membran tilakoid: menyerap cahaya, memecah air (fotolisis), menghasilkan ATP dan NADPH. Reaksi Gelap/Siklus Calvin terjadi di stroma: mengikat CO₂ menggunakan ATP dan NADPH, menghasilkan glukosa.",
        "skor_maksimal": 20,
        "sumber": "Biologi — Fotosintesis"
    },
    {
        "soal_id": "uraian-fotosintesis-tryout-0004",
        "nomor": 4,
        "pertanyaan": "Sebutkan dan jelaskan 3 faktor yang mempengaruhi laju fotosintesis!",
        "kunci_jawaban": "1. Intensitas cahaya: semakin tinggi cahaya, laju fotosintesis meningkat hingga titik jenuh. 2. Konsentrasi CO₂: semakin banyak CO₂, laju fotosintesis meningkat. 3. Suhu: enzim bekerja optimal pada suhu tertentu, terlalu panas atau dingin menghambat fotosintesis. 4. Ketersediaan air: air dibutuhkan sebagai bahan baku reaksi terang.",
        "skor_maksimal": 20,
        "sumber": "Biologi — Fotosintesis"
    },
    {
        "soal_id": "uraian-fotosintesis-tryout-0005",
        "nomor": 5,
        "pertanyaan": "Apa yang dihasilkan oleh fotosintesis dan bagaimana hasilnya bermanfaat bagi kehidupan?",
        "kunci_jawaban": "Fotosintesis menghasilkan glukosa (C₆H₁₂O₆) sebagai sumber energi dan oksigen (O₂). Glukosa digunakan tumbuhan untuk pertumbuhan dan metabolisme. Oksigen yang dilepas sangat penting bagi pernapasan makhluk hidup di bumi.",
        "skor_maksimal": 20,
        "sumber": "Biologi — Fotosintesis"
    },
]

# Jawaban siswa tryout uraian (sebagian lengkap, sebagian singkat)
_TRYOUT_JAWABAN_URAIAN = [
    {
        "soal_id": "uraian-fotosintesis-tryout-0001",
        "jawaban": "Fotosintesis adalah proses pembuatan makanan oleh tumbuhan dengan bantuan cahaya matahari. Rumusnya: CO₂ + H₂O + cahaya menghasilkan glukosa dan O₂."
    },
    {
        "soal_id": "uraian-fotosintesis-tryout-0002",
        "jawaban": "Fotosintesis terjadi di kloroplas. Klorofil menyerap cahaya."
    },
    {
        "soal_id": "uraian-fotosintesis-tryout-0003",
        "jawaban": "Reaksi terang butuh cahaya dan reaksi gelap tidak butuh cahaya."
    },
    {
        "soal_id": "uraian-fotosintesis-tryout-0004",
        "jawaban": "Faktor yang mempengaruhi fotosintesis adalah cahaya, suhu, dan CO₂. Cahaya dibutuhkan untuk mengaktifkan klorofil. Suhu mempengaruhi enzim. CO₂ adalah bahan baku."
    },
    {
        "soal_id": "uraian-fotosintesis-tryout-0005",
        "jawaban": "Fotosintesis menghasilkan glukosa dan oksigen. Oksigen berguna untuk bernapas."
    },
]


# ================================================================
# 5. ENDPOINTS
# ================================================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Cek status service dan daftar task yang tersedia."""
    return HealthResponse(
        status="ok",
        message="Router Agent API berjalan normal 🚀",
        available_tasks=[
            "rekomendasi", "konten_belajar", "rag_query",
            "flashcard", "mindmap", "quiz", "quiz_uraian",
            "evaluasi_quiz", "evaluasi_uraian"
        ],
    )


@app.post("/agent/run", response_model=AgentResponse, tags=["Agent"])
def run_agent(request: AgentRequest):
    """
    **Endpoint utama (generic)** — jalankan agent pipeline sesuai `task`.

    Endpoint ini adalah single entry-point yang menerima semua jenis task via JSON body.
    Cocok digunakan Tim 6 untuk integrasi penuh tanpa perlu menghafal endpoint per-task.

    ### Contoh per task:

    **quiz:**
    ```json
    {
      "task": "quiz",
      "request_params": {"topik": "Teorema Pythagoras", "jumlah_soal": 5},
      "emotion": {"emosi": "semangat", "confidence": 0.9}
    }
    ```

    **evaluasi_quiz:**
    ```json
    {
      "task": "evaluasi_quiz",
      "request_params": {
        "topik": "Hukum Newton",
        "skor_per_soal": 10,
        "soal_pg": [{"soal_id": "pg-hukum_newton-xxx", "nomor": 1, "jawaban_benar": "A", "pembahasan": "..."}],
        "jawaban_siswa": [{"soal_id": "pg-hukum_newton-xxx", "jawaban": "A"}]
      }
    }
    ```

    **evaluasi_uraian:**
    ```json
    {
      "task": "evaluasi_uraian",
      "request_params": {
        "topik": "Fotosintesis",
        "soal_uraian": [{"soal_id": "uraian-xxx", "nomor": 1, "pertanyaan": "...", "kunci_jawaban": "...", "skor_maksimal": 20}],
        "jawaban_siswa": [{"soal_id": "uraian-xxx", "jawaban": "Jawaban siswa..."}]
      }
    }
    ```
    """
    VALID_TASKS = {
        "rekomendasi", "konten_belajar", "rag_query",
        "flashcard", "mindmap", "quiz", "quiz_uraian",
        "evaluasi_quiz", "evaluasi_uraian"
    }
    if request.task.lower() not in VALID_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Task '{request.task}' tidak dikenal. Pilihan valid: {sorted(VALID_TASKS)}"
        )

    try:
        final_payload, nodes_executed = _run_graph(
            task=request.task.lower(),
            request_params=request.request_params,
            emotion=request.emotion.model_dump(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )

    return AgentResponse(
        status="success",
        task=request.task.lower(),
        nodes_executed=nodes_executed,
        output=final_payload,
    )


# ================================================================
# 6. CONVENIENCE SHORTCUT ENDPOINTS
# ================================================================

@app.post("/agent/rekomendasi", response_model=AgentResponse, tags=["Agent - Shortcuts"])
def run_rekomendasi(request: RekomendasiRequest):
    """
    Rekomendasikan 3 bab prioritas berdasarkan data progress siswa.

    - **first_time=true** — kirim `hasil_pretest` dari 3 matpel yang dipilih
    - **first_time=false** — kirim `riwayat_progress` (semua bab boleh, kita otomatis filter 5 terlemah)
    """
    final_payload, nodes_executed = _run_graph(
        task="rekomendasi",
        request_params=request.model_dump(exclude={"emotion"}),
        emotion=request.emotion.model_dump(),
    )
    return AgentResponse(status="success", task="rekomendasi", nodes_executed=nodes_executed, output=final_payload)


@app.post("/agent/flashcard", response_model=AgentResponse, tags=["Agent - Shortcuts"])
def run_flashcard(request: FlashcardRequest):
    """Generate **5 flashcard** NotebookLM-style + kutipan sumber, tingkat kesulitan adaptif berdasarkan `nilai_siswa`."""
    final_payload, nodes_executed = _run_graph(
        task="flashcard",
        request_params={"matpel": request.matpel, "bab": request.bab, "nilai_siswa": request.nilai_siswa},
        emotion=request.emotion.model_dump(),
    )
    return AgentResponse(status="success", task="flashcard", nodes_executed=nodes_executed, output=final_payload)


@app.post("/agent/mindmap", response_model=AgentResponse, tags=["Agent - Shortcuts"])
def run_mindmap(request: MindmapRequest):
    """Generate **mindmap hierarki konsep** komprehensif. Struktur selalu lengkap, gaya bahasa menyesuaikan `nilai_siswa`."""
    final_payload, nodes_executed = _run_graph(
        task="mindmap",
        request_params={"matpel": request.matpel, "bab": request.bab, "nilai_siswa": request.nilai_siswa},
        emotion=request.emotion.model_dump(),
    )
    return AgentResponse(status="success", task="mindmap", nodes_executed=nodes_executed, output=final_payload)


@app.post("/agent/quiz", response_model=AgentResponse, tags=["Agent - Shortcuts"])
def run_quiz(request: QuizRequest):
    """
    Generate **5 soal pilihan ganda** (fixed) dengan tingkat kesulitan adaptif.

    | `nilai_siswa` | Level Soal |
    |---|---|
    | Tidak ada | Dasar |
    | 0 – 40 | Dasar |
    | 41 – 70 | Menengah |
    | 71 – 100 | HOTS |
    """
    final_payload, nodes_executed = _run_graph(
        task="quiz",
        request_params={"matpel": request.matpel, "bab": request.bab, "nilai_siswa": request.nilai_siswa},
        emotion=request.emotion.model_dump(),
    )
    return AgentResponse(status="success", task="quiz", nodes_executed=nodes_executed, output=final_payload)


@app.post("/agent/quiz_uraian", response_model=AgentResponse, tags=["Agent - Shortcuts"])
def run_quiz_uraian(request: QuizUraianRequest):
    """
    Generate **5 soal uraian/esai** (fixed) dengan tingkat kesulitan adaptif.

    | `nilai_siswa` | Level Soal |
    |---|---|
    | Tidak ada | Dasar |
    | 0 – 40 | Dasar |
    | 41 – 70 | Menengah |
    | 71 – 100 | HOTS |
    """
    final_payload, nodes_executed = _run_graph(
        task="quiz_uraian",
        request_params={"matpel": request.matpel, "bab": request.bab, "nilai_siswa": request.nilai_siswa},
        emotion=request.emotion.model_dump(),
    )
    return AgentResponse(status="success", task="quiz_uraian", nodes_executed=nodes_executed, output=final_payload)


@app.post("/agent/evaluasi_quiz", response_model=AgentResponse, tags=["Agent - Evaluasi"])
def run_evaluasi_quiz(request: EvaluasiQuizRequest):
    """
    **Evaluasi jawaban Quiz PG** — deterministik, tanpa LLM.

    Kirim soal (dari output `/agent/quiz`) beserta jawaban siswa.
    Sistem langsung hitung skor berdasarkan `soal_id` — tidak ada LLM call sama sekali.
    """
    final_payload, nodes_executed = _run_graph(
        task="evaluasi_quiz",
        request_params={
            "matpel": request.matpel,
            "bab": request.bab,
            "skor_per_soal": request.skor_per_soal,
            "soal_pg": [s.model_dump() for s in request.soal_pg],
            "jawaban_siswa": [j.model_dump() for j in request.jawaban_siswa],
        },
        emotion=request.emotion.model_dump(),
    )
    return AgentResponse(status="success", task="evaluasi_quiz", nodes_executed=nodes_executed, output=final_payload)


@app.post("/agent/evaluasi_uraian", response_model=AgentResponse, tags=["Agent - Evaluasi"])
def run_evaluasi_uraian(request: EvaluasiUraianRequest):
    """
    **Evaluasi jawaban Quiz Uraian** — LLM menilai kemiripan + pemahaman + feedback per soal.

    Kirim soal + kunci (dari output `/agent/quiz_uraian`) beserta jawaban siswa.
    LLM akan menilai setiap soal secara independen lalu memberikan assessment keseluruhan.
    """
    final_payload, nodes_executed = _run_graph(
        task="evaluasi_uraian",
        request_params={
            "matpel": request.matpel,
            "bab": request.bab,
            "soal_uraian": [s.model_dump() for s in request.soal_uraian],
            "jawaban_siswa": [j.model_dump() for j in request.jawaban_siswa],
        },
        emotion=request.emotion.model_dump(),
    )
    return AgentResponse(status="success", task="evaluasi_uraian", nodes_executed=nodes_executed, output=final_payload)


# ================================================================
# 8. SHORTCUT ENDPOINTS — Konten Belajar & RAG Query
# ================================================================

@app.post("/agent/konten_belajar", response_model=AgentResponse, tags=["Agent - Shortcuts"])
def run_konten_belajar(request: KontenBelajarRequest):
    """Generate materi pembelajaran lengkap dan terstruktur (5 sub-bab) untuk satu bab."""
    final_payload, nodes_executed = _run_graph(
        task="konten_belajar",
        request_params={"matpel": request.matpel, "bab": request.bab, "level": request.level},
        emotion=request.emotion.model_dump(),
    )
    return AgentResponse(status="success", task="konten_belajar", nodes_executed=nodes_executed, output=final_payload)


@app.post("/agent/rag_query", response_model=AgentResponse, tags=["Agent - Shortcuts"])
def run_rag_query(request: RagQueryRequest):
    """Ambil konteks relevan dari Knowledge Base untuk menjawab pertanyaan siswa (tanpa LLM)."""
    final_payload, nodes_executed = _run_graph(
        task="rag_query",
        request_params={"query": request.query, "matpel": request.matpel, "k": request.k},
        emotion={"emosi": "netral", "confidence": 0.8},
    )
    return AgentResponse(status="success", task="rag_query", nodes_executed=nodes_executed, output=final_payload)

import os
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "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"
