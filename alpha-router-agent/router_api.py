"""
Router Agent — FastAPI Microservice (v4.0 — Teacher-Centric)
=============================================================
Jalankan dengan:
    uvicorn router_api:app --reload --port 8000

Swagger UI: http://127.0.0.1:8000/docs
"""

import os
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
        "Microservice Agentic AI Pipeline — Teacher-Centric Flow\n\n"
        "**Generate Konten (input dari guru):**\n"
        "- **bacaan** — Teks bacaan / materi terstruktur (LOTS + MOTS + HOTS)\n"
        "- **flashcard** — Flashcard bergaya NotebookLM (LOTS + MOTS + HOTS)\n"
        "- **quiz** — Soal pilihan ganda / PG (LOTS + MOTS + HOTS)\n"
        "- **quiz_uraian** — Soal esai / uraian (LOTS + MOTS + HOTS)\n"
        "- **mindmap** — Peta konsep hierarki lengkap (1 versi)\n\n"
        "**Evaluasi Jawaban Siswa:**\n"
        "- **evaluasi_quiz** — Nilai jawaban PG (deterministik, no LLM)\n"
        "- **evaluasi_uraian** — Nilai jawaban uraian (LLM, feedback per soal)\n\n"
        "**Lainnya:**\n"
        "- **rag_query** — Pure RAG retriever (no LLM) untuk chatbot Tim 5\n"
        "- **rekomendasi** — Rekomendasikan topik belajar berdasarkan progress siswa\n"
    ),
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================================================
# 2. SHARED BASE MODEL — Teacher Input
# ================================================================

class TeacherContentRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "jenjang": "10",
            "kelas": "10A",
            "mata_pelajaran": "Matematika",
            "elemen": "Bilangan",
            "materi": "Bilangan Berpangkat",
            "atp": [
                "Peserta didik dapat memahami konsep bilangan berpangkat",
                "Peserta didik dapat menyelesaikan operasi bilangan berpangkat",
                "Peserta didik dapat menerapkan sifat-sifat bilangan berpangkat"
            ]
        }
    })
    jenjang: str = Field(
        description="Tingkat kelas: '10', '11', atau '12'",
        examples=["10", "11", "12"]
    )
    kelas: str = Field(
        description="Kelas spesifik (untuk identifikasi unik penyimpanan)",
        examples=["10A", "10B", "11C"]
    )
    mata_pelajaran: str = Field(
        description="Nama mata pelajaran",
        examples=["Matematika", "Fisika", "Biologi", "Bahasa Indonesia"]
    )
    elemen: str = Field(
        description="Elemen Capaian Pembelajaran",
        examples=["Bilangan", "Aljabar", "Geometri", "Sistem Kehidupan"]
    )
    materi: str = Field(
        description="Sub-materi / topik spesifik yang akan di-generate",
        examples=["Bilangan Berpangkat", "Hukum Newton", "Fotosintesis"]
    )
    atp: List[str] = Field(
        description="Alur Tujuan Pembelajaran — daftar tujuan pembelajaran secara berurutan",
        default_factory=list,
        examples=[["Memahami konsep bilangan berpangkat", "Menerapkan operasi bilangan berpangkat"]]
    )


# ================================================================
# 3. PYDANTIC MODELS — Evaluasi & RAG
# ================================================================


class SoalUraianEval(BaseModel):
    soal_id: str
    nomor: int
    level: str = Field(default="", examples=["LOTS", "MOTS", "HOTS"])
    pertanyaan: str
    kunci_jawaban: str
    skor_maksimal: int = Field(default=20)

class JawabanSiswaUraian(BaseModel):
    soal_id: str
    jawaban: str

class EvaluasiUraianRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "mata_pelajaran": "Matematika",
            "materi": "Bilangan Berpangkat",
            "soal_uraian": [
                {"soal_id": "uraian-lots-bilangan_berpangkat-c3d4", "nomor": 1, "level": "LOTS",
                 "pertanyaan": "Jelaskan apa yang dimaksud bilangan berpangkat!",
                 "kunci_jawaban": "Bilangan berpangkat adalah...", "skor_maksimal": 20}
            ],
            "jawaban_siswa": [
                {"soal_id": "uraian-lots-bilangan_berpangkat-c3d4", "jawaban": "Bilangan berpangkat adalah perkalian berulang..."}
            ]
        }
    })
    mata_pelajaran: str = Field(default="Matematika")
    materi: str = Field(default="")
    soal_uraian: List[SoalUraianEval] = Field(default_factory=list)
    jawaban_siswa: List[JawabanSiswaUraian] = Field(default_factory=list)


class RagQueryRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {"query": "apa yang dimaksud inersia?", "matpel": "Fisika", "k": 3}
    })
    query: str = Field(description="Pertanyaan siswa dalam bahasa natural")
    matpel: str = Field(default="", description="Mata pelajaran sebagai konteks (opsional)")
    k: int = Field(default=3, ge=1, le=10)


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
                {"matpel": "Matematika", "bab": "Teorema Pythagoras", "skor_terakhir": 20,
                 "tingkat_pemahaman": "Belum Paham", "jumlah_prompt": 22}
            ]
        }
    })
    student_id: str = "unknown"
    first_time: bool = False
    matpel_dipilih: List[str] = Field(default_factory=list)
    hasil_pretest: Optional[List[HasilPretestItem]] = Field(default=None)
    riwayat_progress: Optional[List[RiwayatProgressItem]] = Field(default=None)


# ================================================================
# 4. RESPONSE MODEL
# ================================================================

class AgentResponse(BaseModel):
    status: str
    task: str
    nodes_executed: List[str]
    output: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    message: str
    available_tasks: List[str]


# ================================================================
# 5. HELPER — Run Graph
# ================================================================

def _build_initial_state(task: str, request_params: dict) -> AgentState:
    return {
        "task": task,
        "request_params": request_params,
        "bacaan_lots_data": "",
        "bacaan_mots_data": "",
        "bacaan_hots_data": "",
        "flashcard_lots_data": "",
        "flashcard_mots_data": "",
        "flashcard_hots_data": "",
        "quiz_lots_data": "",
        "quiz_mots_data": "",
        "quiz_hots_data": "",
        "quiz_uraian_lots_data": "",
        "quiz_uraian_mots_data": "",
        "quiz_uraian_hots_data": "",
        "mindmap_data": "",
        "top_recommendations": "",
        "evaluasi_quiz_result": "",
        "evaluasi_uraian_result": "",
        "rag_query_result": "",
        "final_payload": {},
    }


def _run_graph(task: str, request_params: dict) -> tuple[dict, list[str]]:
    initial_state = _build_initial_state(task, request_params)
    final_payload  = {}
    nodes_executed = []

    for step in router_agent_app.stream(initial_state, stream_mode="updates"):
        for node_name, node_data in step.items():
            nodes_executed.append(node_name)
            if node_name == "structurer":
                final_payload = node_data.get("final_payload", {})

    return final_payload, nodes_executed


# ================================================================
# 6. HEALTH CHECK
# ================================================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    return HealthResponse(
        status="ok",
        message="Router Agent API v4.0 — Teacher-Centric 🚀",
        available_tasks=[
            "bacaan", "flashcard", "mindmap", "quiz", "quiz_uraian",
            "evaluasi_quiz", "evaluasi_uraian", "rag_query", "rekomendasi"
        ],
    )


# ================================================================
# 7. GENERATE ENDPOINTS (Teacher Input)
# ================================================================

@app.post("/agent/bacaan", response_model=AgentResponse, tags=["Generate — Guru"])
def run_bacaan(request: TeacherContentRequest):
    """
    Generate **teks bacaan / materi** dalam **3 level** (LOTS, MOTS, HOTS) sekaligus.

    Setiap level menghasilkan 5 sub-bab terstruktur dengan gaya dan kedalaman berbeda.
    """
    payload, nodes = _run_graph("bacaan", request.model_dump())
    return AgentResponse(status="success", task="bacaan", nodes_executed=nodes, output=payload)


@app.post("/agent/flashcard", response_model=AgentResponse, tags=["Generate — Guru"])
def run_flashcard(request: TeacherContentRequest):
    """
    Generate **flashcard** dalam **3 level** (LOTS, MOTS, HOTS) sekaligus.

    Setiap level menghasilkan 10 kartu (front & back) bergaya NotebookLM
    dengan kesulitan berbeda.
    """
    payload, nodes = _run_graph("flashcard", request.model_dump())
    return AgentResponse(status="success", task="flashcard", nodes_executed=nodes, output=payload)


@app.post("/agent/mindmap", response_model=AgentResponse, tags=["Generate — Guru"])
def run_mindmap(request: TeacherContentRequest):
    """
    Generate **mindmap hierarki konsep** — 1 versi lengkap dan detail.

    Mindmap tidak memiliki level kesulitan; selalu komprehensif mencakup
    semua relasi penting antar konsep dalam materi.
    """
    payload, nodes = _run_graph("mindmap", request.model_dump())
    return AgentResponse(status="success", task="mindmap", nodes_executed=nodes, output=payload)


@app.post("/agent/quiz", response_model=AgentResponse, tags=["Generate — Guru"])
def run_quiz(request: TeacherContentRequest):
    """
    Generate **soal pilihan ganda (PG)** dalam **3 level** (LOTS, MOTS, HOTS) sekaligus.

    Setiap level menghasilkan 5 soal dengan tingkat kognitif berbeda.
    Setiap soal sudah dilengkapi `soal_id` unik untuk keperluan evaluasi.

    | Level | Karakteristik |
    |---|---|---|
    | LOTS | Mengingat & Memahami |
    | MOTS | Menerapkan & Menganalisis |
    | HOTS | Mengevaluasi & Mencipta |
    """
    payload, nodes = _run_graph("quiz", request.model_dump())
    return AgentResponse(status="success", task="quiz", nodes_executed=nodes, output=payload)


@app.post("/agent/quiz_uraian", response_model=AgentResponse, tags=["Generate — Guru"])
def run_quiz_uraian(request: TeacherContentRequest):
    """
    Generate **soal uraian / esai** dalam **3 level** (LOTS, MOTS, HOTS) sekaligus.

    Setiap level menghasilkan 5 soal esai dengan kunci jawaban ideal. Setiap soal sudah dilengkapi `soal_id` unik.
    """
    payload, nodes = _run_graph("quiz_uraian", request.model_dump())
    return AgentResponse(status="success", task="quiz_uraian", nodes_executed=nodes, output=payload)


# ================================================================
# 8. EVALUASI ENDPOINTS (Student-Side)
# ================================================================


@app.post("/agent/evaluasi_uraian", response_model=AgentResponse, tags=["Evaluasi — Siswa"])
def run_evaluasi_uraian(request: EvaluasiUraianRequest):
    """
    **Evaluasi jawaban Quiz Uraian** — LLM menilai setiap soal + feedback konstruktif.

    Kirim soal + kunci (dari output `/agent/quiz_uraian`) beserta jawaban siswa.
    """
    payload, nodes = _run_graph(
        "evaluasi_uraian",
        {
            "mata_pelajaran": request.mata_pelajaran,
            "materi": request.materi,
            "soal_uraian": [s.model_dump() for s in request.soal_uraian],
            "jawaban_siswa": [j.model_dump() for j in request.jawaban_siswa],
        }
    )
    return AgentResponse(status="success", task="evaluasi_uraian", nodes_executed=nodes, output=payload)


# ================================================================
# 9. RAG & REKOMENDASI ENDPOINTS
# ================================================================

@app.post("/agent/rag_query", response_model=AgentResponse, tags=["RAG & Rekomendasi"])
def run_rag_query(request: RagQueryRequest):
    """Ambil konteks relevan dari Knowledge Base untuk menjawab pertanyaan siswa (tanpa LLM)."""
    payload, nodes = _run_graph(
        "rag_query",
        {"query": request.query, "matpel": request.matpel, "k": request.k}
    )
    return AgentResponse(status="success", task="rag_query", nodes_executed=nodes, output=payload)


@app.post("/agent/rekomendasi", response_model=AgentResponse, tags=["RAG & Rekomendasi"])
def run_rekomendasi(request: RekomendasiRequest):
    """
    Rekomendasikan 3 bab prioritas belajar untuk siswa.

    - **first_time=true** → kirim `hasil_pretest`
    - **first_time=false** → kirim `riwayat_progress` (semua bab, otomatis filter 5 terlemah)
    """
    payload, nodes = _run_graph("rekomendasi", request.model_dump())
    return AgentResponse(status="success", task="rekomendasi", nodes_executed=nodes, output=payload)


# ================================================================
# 10. LANGSMITH TRACING (dari env)
# ================================================================
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
