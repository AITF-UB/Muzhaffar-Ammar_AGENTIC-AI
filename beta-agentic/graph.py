import os
import json
import time
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

from state import AgentState
from tools import RAGEngine, clean_json_from_llm, extract_source, generate_konten_id
from llm import get_llm

env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "templates")))
llm = get_llm()

def load_prompt(template_name: str, **kwargs) -> str:
    template = env.get_template(template_name)
    return template.render(**kwargs)

# ================================================================
# 1. NODES
# ================================================================
def retrieve_node(state: AgentState) -> dict:
    """Melakukan pencarian ke Qdrant menggunakan RAGEngine."""
    tipe = state["tipe"]
    req = state["request_params"]
    query = f"{req['elemen_label']} {req.get('materi', '')}".strip()
    
    rag_results = RAGEngine.unified_search(query, tipe)
    
    # Format texts
    text_ctx = "\n---\n".join([f"{t['text']}" for t in rag_results["text"]])
    sumber = extract_source(rag_results["text"])
    
    # Format images (multimodal constraint)
    img_ctx_list = []
    for img_path in rag_results["images"]:
        img_ctx_list.append({
            "image_path": img_path,
            "filename": os.path.basename(img_path)
        })
        
    return {
        "rag_context": text_ctx if text_ctx else "Tidak ada dokumen relevan di database.",
        "sumber_text": sumber,
        "image_context": img_ctx_list
    }

def generate_node(state: AgentState) -> dict:
    """Men-generate konten sesuai tipe menggunakan Jinja Template."""
    tipe = state["tipe"]
    req = state["request_params"]
    lvl = state["level"]
    
    # 1. Pretest Logic (hidden generation if first run)
    pretest_data = None
    # We simulate checking if it's the first run. 
    # For MVP beta-agentic, we always generate it if pretest is missing and it's 'bacaan' (as a trigger).
    if tipe == "bacaan" and not state.get("pretest_data"):
        pretest_sys = load_prompt("system.j2", matpel=req["mapel_id"], materi=req.get("materi", ""), level="Campuran")
        pretest_usr = load_prompt("pretest.j2", matpel=req["mapel_id"], elemen=req["elemen_id"], rag_context=state["rag_context"])
        pt_res = llm.invoke([SystemMessage(content=pretest_sys), HumanMessage(content=pretest_usr)])
        pretest_data = clean_json_from_llm(pt_res.content)
    
    # 2. Main Generation
    sys_prompt = load_prompt("system.j2", matpel=req["mapel_id"], materi=req.get("materi", ""), level=lvl)
    
    if tipe == "bacaan":
        usr_prompt = load_prompt("bacaan.j2", jenjang=req["jenjang"], kelas=req.get("kelas_id", ""), atp=req.get("atp", ""), rag_context=state["rag_context"], level=lvl)
    elif tipe == "quiz_pg":
        usr_prompt = load_prompt("quiz_pg.j2", jenjang=req["jenjang"], kelas=req.get("kelas_id", ""), atp=req.get("atp", ""), rag_context=state["rag_context"], image_context=state["image_context"], level=lvl)
    elif tipe == "quiz_essay":
        usr_prompt = load_prompt("quiz_essay.j2", jenjang=req["jenjang"], kelas=req.get("kelas_id", ""), atp=req.get("atp", ""), rag_context=state["rag_context"], image_context=state["image_context"], level=lvl)
    elif tipe == "flashcard":
        usr_prompt = load_prompt("flashcard.j2", jenjang=req["jenjang"], kelas=req.get("kelas_id", ""), rag_context=state["rag_context"], level=lvl)
    elif tipe == "mindmap":
        usr_prompt = load_prompt("mindmap.j2", matpel=req["mapel_id"], materi=req.get("materi", ""), rag_context=state["rag_context"])
    else:
        raise ValueError(f"Tipe {tipe} tidak dikenali.")

    if state.get("instruksi_revisi"):
        usr_prompt += f"\n\n[INSTRUKSI REVISI DARI GURU]:\n{state['instruksi_revisi']}\nSesuaikan dan perbaiki hasil generasimu berdasarkan instruksi ini!"

    if state.get("evaluator_result") and state["revision_count"] > 0:
        usr_prompt += f"\n\n[FEEDBACK REVISI SEBELUMNYA]:\n{state['evaluator_result'].get('poin_revisi')}\nPerbaiki JSON-mu!"

    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=usr_prompt)])
    content_dict = clean_json_from_llm(response.content)
    
    return {
        "generated_content": content_dict,
        "pretest_data": pretest_data if pretest_data else state.get("pretest_data")
    }

def evaluator_node(state: AgentState) -> dict:
    """Mengevaluasi output generator."""
    if state["revision_count"] >= 2:
        return {"evaluator_result": {"skor": 100, "poin_revisi": []}}

    req = state["request_params"]
    sys_prompt = "Kamu adalah Evaluator JSON dan Konten Pendidikan yang sangat ketat (Killer Grader)."
    usr_prompt = load_prompt(
        "evaluator.j2",
        materi=req.get("materi", ""),
        atp=req.get("atp", ""),
        level=state["level"],
        tipe=state["tipe"],
        rag_context=state.get("rag_context", ""),
        generated_content=json.dumps(state["generated_content"], indent=2)
    )
    
    response = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=usr_prompt)])
    eval_dict = clean_json_from_llm(response.content)
    
    # Fallback if evaluation is weird
    if not isinstance(eval_dict, dict) or "skor" not in eval_dict:
        eval_dict = {"skor": 0, "poin_revisi": ["JSON output sebelumnya terpotong atau rusak. Buat ulang JSON dengan valid dan lengkap."]}
        
    return {
        "evaluator_result": eval_dict,
        "revision_count": state["revision_count"] + 1
    }

def structurer_node(state: AgentState) -> dict:
    """Membungkus hasil akhir sesuai API Contract SR LATEST (V3.6)"""
    tipe = state["tipe"]
    req = state["request_params"]
    content = state["generated_content"]
    
    # Auto-inject source for bacaan and flashcard if LLM didn't (or if we strictly override)
    if tipe == "bacaan" or tipe == "flashcard":
        if isinstance(content, dict):
            content["source"] = state["sumber_text"]
            
    konten_id = generate_konten_id(tipe, state["level"], req.get("materi_id", "materi"))
    
    # Envelope "data" internal
    payload_data = {
        "konten_id": konten_id,
        "tipe": tipe,
        "level": (state["level"] or "").lower(),
        "content": content,
        "dibuat_at": datetime.utcnow().isoformat() + "Z"
    }
    
    # [Webhook/Pretest Logic]
    # Jika ada pretest_data, kita selipkan di payload menggunakan hidden key agar API BE bisa memprosesnya.
    if state.get("pretest_data"):
        payload_data["_pretest_data"] = state["pretest_data"]
        
    return {"final_payload": payload_data}

# ================================================================
# 2. EDGES & GRAPH
# ================================================================
def should_revise(state: AgentState) -> str:
    eval_res = state.get("evaluator_result", {})
    skor = eval_res.get("skor", 100)
    status = eval_res.get("status", "layak")
    
    if (skor < 80 or status == "tidak_layak") and state["revision_count"] < 2:
        return "revise"
    return "pass"

builder = StateGraph(AgentState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.add_node("evaluate", evaluator_node)
builder.add_node("structure", structurer_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", "evaluate")
builder.add_conditional_edges("evaluate", should_revise, {"revise": "generate", "pass": "structure"})
builder.add_edge("structure", END)

beta_graph = builder.compile()
