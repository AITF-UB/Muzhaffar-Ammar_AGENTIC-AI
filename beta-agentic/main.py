import uvicorn
import os

# Mematikan handler Ctrl+C bawaan Fortran (MKL/Sentence-Transformer)
# agar Uvicorn --reload tidak crash saat file berubah.
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from jinja2 import Environment, FileSystemLoader

from api_models import (
    StandardResponse, GenerateRequest,
    SesiSummaryRequest, EssaySubmitRequest, RekomendasiRequest, InsightRequest
)
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

from graph import beta_graph
from llm import get_llm
from graph import beta_graph
from llm import get_llm
from tools import clean_json_from_llm, get_sentence_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload model ke memory saat startup agar request pertama tidak lambat
    print("Pre-loading Sentence Transformer model...")
    get_sentence_model()
    yield
    print("Shutting down...")

app = FastAPI(title="Beta Agentic SR API", version="3.6", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "templates")))
llm = get_llm()

def load_prompt(template_name: str, **kwargs) -> str:
    template = env.get_template(template_name)
    return template.render(**kwargs)

# ---------------------------------------------------------
# KONTEN ENDPOINTS
# ---------------------------------------------------------
@app.post("/konten/generate", response_model=StandardResponse)
def generate_konten(req: GenerateRequest):
    try:
        # Menyiapkan State Awal untuk Graf
        initial_state = {
            "request_params": req.model_dump(),
            "tipe": req.tipe,
            "level": req.level,
            "revision_count": 0,
            "instruksi_revisi": req.instruksi_revisi
        }
        
        # Mengeksekusi State Machine
        final_state = beta_graph.invoke(initial_state)
        final_payload = final_state["final_payload"]
        
        # Pertahankan konten_id jika diberikan dari klien (untuk regen)
        if req.konten_id:
            final_payload["konten_id"] = req.konten_id
            
        return StandardResponse(data=final_payload)
        
    except Exception as e:
        return StandardResponse(error={"code": "GEN_ERROR", "message": str(e)})


# ---------------------------------------------------------
# SUMMARY SESI (Tim 3 RAG)
# ---------------------------------------------------------
@app.post("/sesi/{id}/summary", response_model=StandardResponse)
def generate_summary(id: str, req: SesiSummaryRequest):
    try:
        prompt = load_prompt(
            "summary.j2",
            req=req.model_dump()
        )
        sys_msg = SystemMessage(content="Kamu adalah AI yang merangkum hasil belajar siswa selama satu sesi menjadi JSON.")
        res = llm.invoke([sys_msg, HumanMessage(content=prompt)])
        content = clean_json_from_llm(res.content)
        
        now = datetime.utcnow()
        berlaku = now + timedelta(days=1)
        
        return StandardResponse(data={
            "teks": content.get("teks", "Gagal menghasilkan summary."),
            "dibuat_at": now.isoformat() + "Z",
            "berlaku_hingga": berlaku.isoformat() + "Z"
        })
    except Exception as e:
        return StandardResponse(error={"code": "SUMMARY_ERR", "message": str(e)})

# ---------------------------------------------------------
# QUIZ EVALUATION ENDPOINTS (Dipanggil BE)
# ---------------------------------------------------------
@app.post("/siswa/{id}/quiz/essay", response_model=StandardResponse)
def submit_essay(id: str, req: EssaySubmitRequest):
    try:
        # Kita evaluasi setiap jawaban secara individual
        evaluasi_hasil = {}
        sys_msg = SystemMessage(content="Kamu adalah Guru Penilai Esai JSON.")
        
        for e_id, jwb_siswa in req.jawaban.items():
            soal_teks = req.soal.get(e_id, "")
            rubrik_teks = req.rubrik.get(e_id, "")
            
            usr_prompt = load_prompt(
                "essay_evaluation.j2",
                soal=soal_teks,
                rubrik=rubrik_teks,
                jawaban_siswa=jwb_siswa
            )
            res = llm.invoke([sys_msg, HumanMessage(content=usr_prompt)])
            evaluasi_hasil[e_id] = clean_json_from_llm(res.content)
            
        return StandardResponse(data={"siswa_id": id, "evaluasi": evaluasi_hasil})
    except Exception as e:
        return StandardResponse(error={"code": "EVAL_ERR", "message": str(e)})


# ---------------------------------------------------------
# RAG SERVICES ENDPOINTS
# ---------------------------------------------------------
@app.post("/rag/rekomendasi", response_model=StandardResponse)
def rekomendasi(req: RekomendasiRequest):
    try:
        prompt = load_prompt(
            "rekomendasi.j2",
            siswa_id=req.siswa_id,
            sudah_selesai=req.sudah_selesai_ids,
            sedang_dipelajari=req.sedang_dipelajari_ids,
            levels=req.levels
        )
        sys_msg = SystemMessage(content="Kamu adalah AI Recommender JSON.")
        res = llm.invoke([sys_msg, HumanMessage(content=prompt)])
        content = clean_json_from_llm(res.content)
        return StandardResponse(data=content)
    except Exception as e:
        return StandardResponse(error={"code": "REKOM_ERR", "message": str(e)})

@app.post("/rag/insight", response_model=StandardResponse)
def insight(req: InsightRequest):
    try:
        prompt = load_prompt(
            "insight.j2",
            nama=req.nama,
            streak=req.streak,
            total_topik=req.total_topik,
            poin=req.total_poin_kuiz,
            durasi=req.total_durasi
        )
        sys_msg = SystemMessage(content="Kamu adalah Penyedia Motivasi Pendek JSON.")
        usr_msg = HumanMessage(content=prompt)
        res = llm.invoke([sys_msg, usr_msg])
        content = clean_json_from_llm(res.content)
        return StandardResponse(data=content)
    except Exception as e:
        return StandardResponse(error={"code": "INSIGHT_ERR", "message": str(e)})

# Serve extraction folder for images
EXTRACTION_BASE_DIR = Path(__file__).resolve().parent.parent / "extraction"
if EXTRACTION_BASE_DIR.exists():
    app.mount("/extraction", StaticFiles(directory=str(EXTRACTION_BASE_DIR)), name="extraction")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "beta-agentic" 