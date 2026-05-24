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
    
    # Fokuskan query RAG HANYA pada elemen dan materi.
    query = f"{req.get('materi', '')}".strip()
    
    # Mapping sederhana dari mapel_id ke string di metadata Qdrant
    mapel_mapping = {
        "bahasa_indonesia": "Bahasa Indonesia",
        "bindo": "Bahasa Indonesia",
        "matematika_umum": "Matematika",
        "mat": "Matematika",
        "matematika": "Matematika",
        "mtk": "Matematika",
        "ips": "IPS"
    }
    raw_mapel = req.get("mapel_id", "")
    mapel_key = raw_mapel.lower().replace(" ", "_")
    mapel_str = mapel_mapping.get(mapel_key, raw_mapel)
    
    # Ekstrak angka kelas dari jenjang (mendukung angka & romawi)
    jenjang_str = str(req.get("jenjang", "")).lower().strip()
    kelas_int = None
    
    roman_map = {"X": 10, "xi": 11, "xii": 12}
    if jenjang_str in roman_map:
        kelas_int = roman_map[jenjang_str]
    else:
        digits = ''.join(filter(str.isdigit, jenjang_str))
        if digits:
            kelas_int = int(digits)
    
    rag_results = RAGEngine.unified_search(query, tipe, mapel=mapel_str, kelas=kelas_int)
    
    # Format texts
    text_ctx_parts = []
    for t in rag_results["text"]:
        part = t["text"]
        vis = t.get("visual_context", [])
        if isinstance(vis, str):
            vis = [vis]
        if vis:
            vis_str = ", ".join([os.path.basename(v) for v in vis])
            part = f"[Referensi File Gambar: {vis_str}]\n" + part
        text_ctx_parts.append(part)
        
    text_ctx = "\n---\n".join(text_ctx_parts)
    sumber = extract_source(rag_results["text"])
    
    # Format images (multimodal constraint)
    # Build formatted image context string
    img_ctx_str = ""
    if rag_results["images"]:
        for idx, img_path in enumerate(rag_results["images"]):
            img_ctx_str += f"Gambar {idx+1}:\n"
            img_ctx_str += f"- filename: {os.path.basename(img_path)}\n"
            img_ctx_str += f"- image_path: {img_path}\n\n"
        
    return {
        "rag_context": text_ctx if text_ctx else "Tidak ada dokumen relevan di database.",
        "sumber_text": sumber,
        "image_context": img_ctx_str.strip()
    }

def generate_node(state: AgentState) -> dict:
    """Men-generate konten sesuai tipe menggunakan Jinja Template."""
    tipe = state["tipe"]
    req = state["request_params"]
    lvl = state["level"]
    

    # 2. Main Generation
    sys_prompt = load_prompt("system.j2", matpel=req["mapel_id"], materi=req.get("materi", ""), level=lvl)
    
    if tipe == "bacaan":
        usr_prompt = load_prompt("bacaan.j2", jenjang=req["jenjang"], kelas=req.get("kelas_id", ""), atp=req.get("atp", ""), rag_context=state["rag_context"], image_context=state["image_context"], level=lvl)
    elif tipe == "pretest":
        usr_prompt = load_prompt("pretest.j2", jenjang=req["jenjang"], kelas=req.get("kelas_id", ""), atp=req.get("atp", ""), rag_context=state["rag_context"], level=lvl)
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
        "generated_content": content_dict
    }

def evaluator_node(state: AgentState) -> dict:
    """Mengevaluasi output generator."""
    if state["revision_count"] >= 2:
        return {"evaluator_result": {"skor": 100, "poin_revisi": []}}

    req = state["request_params"]
    sys_prompt = "Kamu adalah Evaluator JSON dan Konten Pendidikan."
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
    
    # Mapping format JSON dari V3 (judul_utama & konten_markdown) menjadi 'text' untuk frontend
    if tipe == "bacaan":
        if isinstance(content, dict):
            if "judul_utama" in content and "konten_markdown" in content:
                content["text"] = f"# {content['judul_utama']}\n\n{content['konten_markdown']}"
                content.pop("judul_utama", None)
                content.pop("konten_markdown", None)
            content["source"] = state["sumber_text"]
            
    if tipe == "flashcard":
        if isinstance(content, dict):
            content["source"] = state["sumber_text"]
            
    konten_id = generate_konten_id(tipe, state["level"], req.get("materi_id", "materi"), req.get("kelas_id", "all"))
    
    # Envelope "data" internal
    payload_data = {
        "konten_id": konten_id,
        "tipe": tipe,
        "level": (state["level"] or "").lower(),
        "content": content,
        "dibuat_at": datetime.utcnow().isoformat() + "Z"
    }
    
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
