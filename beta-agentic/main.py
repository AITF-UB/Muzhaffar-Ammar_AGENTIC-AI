import uvicorn
import os
from typing import List

# Mematikan handler Ctrl+C bawaan Fortran (MKL/Sentence-Transformer)
# agar Uvicorn --reload tidak crash saat file berubah.
os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from jinja2 import Environment, FileSystemLoader

from api_models import (
    GenerateRequest,
    SesiSummaryRequest, EssayEvalItem, RekomendasiRequest, InsightRequest
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
@app.post("/konten/generate")
async def generate_konten(req: GenerateRequest):
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
        final_state = await beta_graph.ainvoke(initial_state)
        final_payload = final_state["final_payload"]
        
        # Pertahankan konten_id jika diberikan dari klien (kecuali untuk quiz dan essay)
        if req.konten_id and req.tipe not in ["quiz_pg", "quiz_essay", "pretest"]:
            final_payload["konten_id"] = req.konten_id
            
        return final_payload
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GEN_ERROR: {str(e)}")


# ---------------------------------------------------------
# SUMMARY SESI (Tim 3 RAG)
# ---------------------------------------------------------
@app.post("/sesi/summary")
def generate_summary(req: SesiSummaryRequest):
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
        
        return {
            "teks": content.get("teks", "Gagal menghasilkan summary."),
            "dibuat_at": now.isoformat() + "Z",
            "berlaku_hingga": berlaku.isoformat() + "Z"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SUMMARY_ERR: {str(e)}")

# ---------------------------------------------------------
# QUIZ EVALUATION ENDPOINTS (Dipanggil BE)
# ---------------------------------------------------------
@app.post("/siswa/quiz/essay")
def submit_essay(req: List[EssayEvalItem]):
    try:
        evaluasi_hasil = []
        total_skor = 0
        sys_msg = SystemMessage(content="Kamu adalah Guru Penilai Esai JSON.")
        
        for item in req:
            usr_prompt = load_prompt(
                "essay_evaluation.j2",
                soal=item.soal,
                rubrik=item.rubrik,
                jawaban_siswa=item.jawaban_siswa,
                stimulus=item.stimulus,
                penjelasan=item.penjelasan
            )
            res = llm.invoke([sys_msg, HumanMessage(content=usr_prompt)])
            hasil = clean_json_from_llm(res.content)
            
            skor = hasil.get("skor", 0)
            total_skor += skor
            evaluasi_hasil.append(hasil)
            
        return {"evaluasi": evaluasi_hasil, "total_skor": total_skor}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"EVAL_ERR: {str(e)}")


# ---------------------------------------------------------
# RAG SERVICES ENDPOINTS
# ---------------------------------------------------------
@app.post("/rag/rekomendasi")
def rekomendasi(req: RekomendasiRequest):
    try:
        prompt = load_prompt(
            "rekomendasi.j2",
            siswa_id=req.siswa_id,
            available_ids=req.available_ids,
            sudah_selesai=req.sudah_selesai_ids,
            sedang_dipelajari=req.sedang_dipelajari_ids
        )
        sys_msg = SystemMessage(content="Kamu adalah AI Recommender JSON.")
        res = llm.invoke([sys_msg, HumanMessage(content=prompt)])
        content = clean_json_from_llm(res.content)
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"REKOM_ERR: {str(e)}")

@app.post("/rag/insight")
def insight(req: InsightRequest):
    try:
        prompt = load_prompt(
            "insight.j2",
            nama=req.nama,
            streak=req.streak,
            total_topik=req.total_topik,
            poin=req.total_poin_kuiz,
            durasi=req.total_durasi_menit
        )
        sys_msg = SystemMessage(content="Kamu adalah Penyedia Motivasi Pendek JSON.")
        usr_msg = HumanMessage(content=prompt)
        res = llm.invoke([sys_msg, usr_msg])
        content = clean_json_from_llm(res.content)
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"INSIGHT_ERR: {str(e)}")

# Serve extraction folder for images
EXTRACTION_BASE_DIR = Path(__file__).resolve().parent / "extraction"
if EXTRACTION_BASE_DIR.exists():
    app.mount("/extraction", StaticFiles(directory=str(EXTRACTION_BASE_DIR)), name="extraction")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_PROJECT"] = "beta-agentic" 