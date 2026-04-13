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
from pydantic import BaseModel, Field
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
        "**4 Mekanik Generate:**\n"
        "- **rekomendasi** — Analisis nilai & emosi → 3 topik prioritas\n"
        "- **flashcard** — NotebookLM-style flashcard dengan kutipan sumber\n"
        "- **mindmap** — Peta konsep hierarki (parent-child)\n"
        "- **quiz** — Soal pilihan ganda (PG) grounded RAG + soal_id\n"
        "- **quiz_uraian** — Soal esai/uraian grounded RAG + kunci + soal_id\n\n"
        "**2 Mekanik Grading:**\n"
        "- **grade_quiz** — Nilai jawaban PG (deterministik, no LLM)\n"
        "- **grade_uraian** — Nilai jawaban uraian (LLM, feedback per soal)\n"
    ),
    version="2.0.0",
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
        examples=["rekomendasi", "flashcard", "mindmap", "quiz"]
    )
    request_params: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Parameter spesifik sesuai task:\n"
            "- rekomendasi: {nilai_pretest: int, nilai_ujian: int}\n"
            "- flashcard: {topik: str}\n"
            "- mindmap: {topik: str}\n"
            "- quiz: {topik: str, jumlah_soal: int}\n"
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
        "grade_quiz_result": "",
        "grade_uraian_result": "",
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
# 4. ENDPOINTS
# ================================================================

@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Cek status service dan daftar task yang tersedia."""
    return HealthResponse(
        status="ok",
        message="Router Agent API berjalan normal 🚀",
        available_tasks=["rekomendasi", "flashcard", "mindmap", "quiz", "quiz_uraian", "grade_quiz", "grade_uraian"],
    )


@app.post("/agent/run", response_model=AgentResponse, tags=["Agent"])
def run_agent(request: AgentRequest):
    """
    **Endpoint utama** — jalankan agent pipeline sesuai `task`.

    ### Contoh request per task:

    **rekomendasi:**
    ```json
    {
      "task": "rekomendasi",
      "request_params": {"nilai_pretest": 60, "nilai_ujian": 45},
      "emotion": {"emosi": "sedih", "confidence": 0.9}
    }
    ```

    **flashcard:**
    ```json
    {
      "task": "flashcard",
      "request_params": {"topik": "Hukum Newton"},
      "emotion": {"emosi": "fokus", "confidence": 0.8}
    }
    ```

    **mindmap:**
    ```json
    {
      "task": "mindmap",
      "request_params": {"topik": "Fotosintesis"},
      "emotion": {"emosi": "penasaran", "confidence": 0.85}
    }
    ```

    **quiz:**
    ```json
    {
      "task": "quiz",
      "request_params": {"topik": "Teorema Pythagoras", "jumlah_soal": 3},
      "emotion": {"emosi": "semangat", "confidence": 0.9}
    }
    ```
    """
    VALID_TASKS = {"rekomendasi", "flashcard", "mindmap", "quiz", "quiz_uraian", "grade_quiz", "grade_uraian"}
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
# 5. CONVENIENCE SHORTCUT ENDPOINTS (opsional, tapi bikin DX bagus)
# ================================================================

@app.post("/agent/rekomendasi", response_model=AgentResponse, tags=["Agent - Shortcuts"])
def run_rekomendasi(
    nilai_pretest: int = 70,
    nilai_ujian: int = 55,
    emosi: str = "netral",
    confidence: float = 0.8,
):
    """Shortcut endpoint untuk rekomendasi topik berdasarkan nilai & emosi siswa."""
    final_payload, nodes_executed = _run_graph(
        task="rekomendasi",
        request_params={"nilai_pretest": nilai_pretest, "nilai_ujian": nilai_ujian},
        emotion={"emosi": emosi, "confidence": confidence},
    )
    return AgentResponse(
        status="success",
        task="rekomendasi",
        nodes_executed=nodes_executed,
        output=final_payload,
    )


@app.post("/agent/flashcard", response_model=AgentResponse, tags=["Agent - Shortcuts"])
def run_flashcard(topik: str = "Hukum Newton", emosi: str = "fokus", confidence: float = 0.8):
    """Shortcut endpoint untuk generate flashcard NotebookLM-style."""
    final_payload, nodes_executed = _run_graph(
        task="flashcard",
        request_params={"topik": topik},
        emotion={"emosi": emosi, "confidence": confidence},
    )
    return AgentResponse(
        status="success",
        task="flashcard",
        nodes_executed=nodes_executed,
        output=final_payload,
    )


@app.post("/agent/mindmap", response_model=AgentResponse, tags=["Agent - Shortcuts"])
def run_mindmap(topik: str = "Fotosintesis", emosi: str = "penasaran", confidence: float = 0.8):
    """Shortcut endpoint untuk generate mindmap hierarki konsep."""
    final_payload, nodes_executed = _run_graph(
        task="mindmap",
        request_params={"topik": topik},
        emotion={"emosi": emosi, "confidence": confidence},
    )
    return AgentResponse(
        status="success",
        task="mindmap",
        nodes_executed=nodes_executed,
        output=final_payload,
    )


@app.post("/agent/quiz", response_model=AgentResponse, tags=["Agent - Shortcuts"])
def run_quiz(
    topik: str = "Teorema Pythagoras",
    jumlah_soal: int = 5,
    emosi: str = "semangat",
    confidence: float = 0.8,
):
    """Shortcut endpoint untuk generate quiz pilihan ganda NotebookLM-style (+ soal_id)."""
    final_payload, nodes_executed = _run_graph(
        task="quiz",
        request_params={"topik": topik, "jumlah_soal": jumlah_soal},
        emotion={"emosi": emosi, "confidence": confidence},
    )
    return AgentResponse(
        status="success",
        task="quiz",
        nodes_executed=nodes_executed,
        output=final_payload,
    )


@app.post("/agent/quiz_uraian", response_model=AgentResponse, tags=["Agent - Shortcuts"])
def run_quiz_uraian(
    topik: str = "Hukum Newton",
    jumlah_soal: int = 5,
    emosi: str = "fokus",
    confidence: float = 0.8,
):
    """Shortcut endpoint untuk generate soal uraian/esai (+ kunci jawaban + soal_id)."""
    final_payload, nodes_executed = _run_graph(
        task="quiz_uraian",
        request_params={"topik": topik, "jumlah_soal": jumlah_soal},
        emotion={"emosi": emosi, "confidence": confidence},
    )
    return AgentResponse(
        status="success",
        task="quiz_uraian",
        nodes_executed=nodes_executed,
        output=final_payload,
    )


@app.post("/agent/grade_quiz", response_model=AgentResponse, tags=["Agent - Grading"])
def run_grade_quiz(request: AgentRequest):
    """
    **Grade jawaban Quiz PG** — deterministik, tanpa LLM.

    Contoh request:
    ```json
    {
      "task": "grade_quiz",
      "request_params": {
        "topik": "Hukum Newton",
        "skor_per_soal": 10,
        "soal_pg": [
          {"soal_id": "pg-hukum_newton-a3f2", "nomor": 1, "jawaban_benar": "A", "pembahasan": "..."}
        ],
        "jawaban_siswa": [
          {"soal_id": "pg-hukum_newton-a3f2", "jawaban": "A"}
        ]
      },
      "emotion": {"emosi": "netral", "confidence": 0.8}
    }
    ```
    """
    final_payload, nodes_executed = _run_graph(
        task="grade_quiz",
        request_params=request.request_params,
        emotion=request.emotion.model_dump(),
    )
    return AgentResponse(
        status="success",
        task="grade_quiz",
        nodes_executed=nodes_executed,
        output=final_payload,
    )


@app.post("/agent/grade_uraian", response_model=AgentResponse, tags=["Agent - Grading"])
def run_grade_uraian(request: AgentRequest):
    """
    **Grade jawaban Quiz Uraian** — LLM menilai kemiripan + pemahaman + feedback.

    Contoh request:
    ```json
    {
      "task": "grade_uraian",
      "request_params": {
        "topik": "Hukum Newton",
        "soal_uraian": [
          {
            "soal_id": "uraian-hukum_newton-9c1b",
            "nomor": 1,
            "pertanyaan": "Jelaskan Hukum Newton I!",
            "kunci_jawaban": "Benda tetap diam atau bergerak lurus...",
            "skor_maksimal": 20
          }
        ],
        "jawaban_siswa": [
          {"soal_id": "uraian-hukum_newton-9c1b", "jawaban": "Hukum Newton I adalah..."}
        ]
      },
      "emotion": {"emosi": "netral", "confidence": 0.8}
    }
    ```
    """
    final_payload, nodes_executed = _run_graph(
        task="grade_uraian",
        request_params=request.request_params,
        emotion=request.emotion.model_dump(),
    )
    return AgentResponse(
        status="success",
        task="grade_uraian",
        nodes_executed=nodes_executed,
        output=final_payload,
    )
