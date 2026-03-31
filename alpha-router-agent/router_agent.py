import os
import re
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# Import State DAG dan Tool Utility
from router_state import AgentState
from router_tools import (
    kb_sekolah,
    clean_json_from_llm,
    util_format_recommender,
    util_format_flashcard,
    util_format_mindmap,
)

# ================================================================
# 1. SETUP HUGGINGFACE LLM GRATIS
# ================================================================

# Gunakan model dan library yang sama, tapi karena environment conda sudah disesuaikan, 
# kita asumsikan env HF_TOKEN sudah ada atau tidak diperlukan untuk lokal tertentu

from huggingface_hub import InferenceClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration

HF_TOKEN = os.getenv("HF_TOKEN", "")

class HFChatModel(BaseChatModel):
    client: InferenceClient = None
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.3
    max_tokens: int = 2000 # Kita perbesar sedikit

    class Config:
        arbitrary_types_allowed = True
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = InferenceClient(
            model=self.model_id,
            token=HF_TOKEN
        )

    @property
    def _llm_type(self) -> str:
        return "hf-chat"

    def _generate(self, messages, stop=None, **kwargs) -> ChatResult:
        hf_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                hf_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                hf_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                hf_messages.append({"role": "assistant", "content": msg.content})
            else:
                hf_messages.append({"role": "user", "content": str(msg.content)})

        response = self.client.chat_completion(
            messages=hf_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=stop or [],
        )

        content = response.choices[0].message.content
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

# Instantiate The Brain
llm = HFChatModel()

def _chat(system: str, user: str) -> str:
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    return response.content.strip()

# ================================================================
# 2. DEFINISI NODE SPESIALIS (SINGLE-AGENTS)
# ================================================================

def router_task(state: AgentState) -> str:
    """Kondisional Edge: Mengarahkan graph ke node spesialis sesuai parameter 'task' """
    task = state.get("task", "").lower()
    if "rekomendasi" in task:
        return "to_recommender"
    elif "flashcard" in task:
        return "to_flashcard"
    elif "mindmap" in task:
        return "to_mindmap"
    else:
        return "to_structurer" # Fallback jika task tidak dikenali


def recommender_node(state: AgentState) -> dict:
    """Spesialis 1: Menganalisa nilai dan emosi -> Rekomendasi 3 Topik"""
    params = state["request_params"]
    emosi = state["emotion_input"].get("emosi", "netral")
    
    pretest = params.get("nilai_pretest", 0)
    ujian = params.get("nilai_ujian", 0)
    
    prompt_sys = f"""Kamu adalah asisten pengajar yang berempati.
TUGAS: Analisis data nilai dan emosi siswa, berikan saran empatik pendek, lalu pilihkan HANYA 3 TOPIK prioritas.
ATURAN MUTLAK:
1. Keluarkan output dalam format JSON murni.
2. JANGAN tambahkan teks pengantar apapun selain XML Tag.
3. Bungkus jawabanmu DALAM TAG <REKOMENDASI> ... </REKOMENDASI>"""

    prompt_usr = f"""Data Siswa:
- Nilai Pretest: {pretest}
- Nilai Ujian Terakhir: {ujian}
- Status Emosi Saat Ini: {emosi}

Hasilkan JSON ini tepat dalam tag <REKOMENDASI>:
{{
  "saran_empatik": "string",
  "topik_prioritas": [
       {{"topik": "Nama Topik", "alasan": "Alasan singkat"}}
  ]
}}"""

    result = _chat(system=prompt_sys, user=prompt_usr)
    
    # Ekstraksi dengan Regex
    match = re.search(r"<REKOMENDASI>(.*?)</REKOMENDASI>", result, re.DOTALL | re.IGNORECASE)
    extracted = match.group(1).strip() if match else result
    
    return {"top_recommendations": extracted}


def flashcard_node(state: AgentState) -> dict:
    """Spesialis 2: RAG + NotebookLM style Flashcards dengan kutipan sumber"""
    topik = state["request_params"].get("topik", "Sains")
    
    # 1. RAG Retrieve
    docs = kb_sekolah.search(topik, k=3)
    context = "\n---\n".join([d.page_content.strip() for d in docs])
    
    prompt_sys = f"""Kamu adalah spesialis pembuat soal Flashcard berstandar tinggi.
TUGAS: Buat 5 pasang pertanyaan (Front) dan jawaban (Back) mengenai topik yang diminta HANYA berdasarkan referensi berikut.
ATURAN NOTEBOOKLM (WAJIB):
Pada bagian "Back", WAJIB sertakan field "kutipan_sumber" berisi salinan tepat 1 kalimat dari teks referensi yang membuktikan kebenaran jawaban tersebut.
ATURAN MUTLAK KEDUA:
Keluarkan output dalam format JSON Array murni. Bungkus dalam tag <FLASHCARD> ... </FLASHCARD>."""

    prompt_usr = f"""Topik Diminta: {topik}

Referensi:
{context}

Format yang diminta (DALAM TAG <FLASHCARD>):
[
  {{
    "front": "Pertanyaan?",
    "back": "Jawaban dan penjelasan.",
    "kutipan_sumber": "Salinan 1 kalimat persis dari referensi"
  }}
]"""

    result = _chat(system=prompt_sys, user=prompt_usr)
    
    # Ekstraksi Regex
    match = re.search(r"<FLASHCARD>(.*?)</FLASHCARD>", result, re.DOTALL | re.IGNORECASE)
    extracted = match.group(1).strip() if match else result
    
    return {"flashcards_data": extracted}


def mindmap_node(state: AgentState) -> dict:
    """Spesialis 3: RAG + Node/Edge Hierarki Generator"""
    topik = state["request_params"].get("topik", "Sains")
    
    # 1. RAG Retrieve
    docs = kb_sekolah.search(topik, k=2)
    context = "\n---\n".join([d.page_content.strip() for d in docs])
    
    prompt_sys = f"""Kamu adalah ahli pembuat Peta Konsep (Mindmap).
TUGAS: Buat struktur hierarki (parent-child) untuk konsep dari teks referensi bersangkutan.
Setiap node dapat memiliki child_nodes (nested).
ATURAN MUTLAK:
Keluarkan output berformat JSON murni. Bungkus dalam tag <MINDMAP> ... </MINDMAP>"""

    prompt_usr = f"""Topik: {topik}
Referensi:
{context}

Format yang diminta (DALAM TAG <MINDMAP>):
{{
  "konsep_utama": "Nama Konsep",
  "deskripsi": "Deskripsi singkat 1 kalimat dengan mengutip referensi",
  "children": [
    {{
      "sub_konsep": "Nama",
      "penjelasan": "Deskripsi",
      "children": []
    }}
  ]
}}"""

    result = _chat(system=prompt_sys, user=prompt_usr)
    
    match = re.search(r"<MINDMAP>(.*?)</MINDMAP>", result, re.DOTALL | re.IGNORECASE)
    extracted = match.group(1).strip() if match else result
    
    return {"mindmap_data": extracted}


def structurer_node(state: AgentState) -> dict:
    """Muara Akhir: Merapikan Payload menjadi Strict JSON Dictionary untuk UI"""
    task = state.get("task", "")
    
    # Tentukan data mana yang berisi string untuk diolah
    raw_data = ""
    formatter_func = None
    args_for_formatter = {}
    
    if task == "rekomendasi":
        raw_data = state.get("top_recommendations", "")
        parsed_json = clean_json_from_llm(raw_data)
        
        emosi = "Siswa merasa " + state["emotion_input"].get("emosi", "netral")
        if isinstance(parsed_json, dict) and "saran_empatik" in parsed_json:
            emosi = parsed_json["saran_empatik"]
            topics = parsed_json.get("topik_prioritas", [])
        else:
            topics = [{"error": "Data rusak dari spesialis"}]
            
        final_payload = util_format_recommender(topics, emosi)
        
    elif task == "flashcard":
        raw_data = state.get("flashcards_data", "[]")
        parsed_json = clean_json_from_llm(raw_data)
        
        topik = state["request_params"].get("topik", "Tidak diketahui")
        if not isinstance(parsed_json, list):
            # Coba cari dalam field keys kalau bentuknya dict aneh
            if isinstance(parsed_json, dict):
                for k, v in parsed_json.items():
                    if isinstance(v, list):
                        parsed_json = v
                        break
            if not isinstance(parsed_json, list):
                parsed_json = [{"error": "Data rusak"}]
                
        final_payload = util_format_flashcard(topik, parsed_json)
        
    elif task == "mindmap":
        raw_data = state.get("mindmap_data", "{}")
        parsed_json = clean_json_from_llm(raw_data)
        
        topik = state["request_params"].get("topik", "Tidak diketahui")
        if not isinstance(parsed_json, dict) or "error" in parsed_json:
             parsed_json = {"error": "Format mindmap rusak"}
             
        # Bungkus ke wrapper format UI menggunakan dict keys yang aman array
        nodes_arr = [parsed_json] if isinstance(parsed_json, dict) else parsed_json
        final_payload = util_format_mindmap(topik, nodes_arr)
        
    else:
        final_payload = {"error": f"Task '{task}' tidak dikenal"}

    return {"final_payload": final_payload}


# ================================================================
# 3. ROUTING LOGIC & GRAPH COMPILATION
# ================================================================

workflow = StateGraph(AgentState)

# Daftar Node (Masing-masing Agen Spesialis berdiri sendiri)
workflow.add_node("recommender", recommender_node)
workflow.add_node("flashcard", flashcard_node)
workflow.add_node("mindmap", mindmap_node)
workflow.add_node("structurer", structurer_node)

# Konfigurasi Percabangan Multi-Single-Agent
workflow.add_conditional_edges(
    START,
    router_task,
    {
        "to_recommender": "recommender",
        "to_flashcard": "flashcard",
        "to_mindmap": "mindmap",
        "to_structurer": "structurer" # Fallthrough bypass
    }
)

# Semua jalur bermuara di mesin Structurer
workflow.add_edge("recommender", "structurer")
workflow.add_edge("flashcard", "structurer")
workflow.add_edge("mindmap", "structurer")

# Structurer mengakhiri pipeline
workflow.add_edge("structurer", END)

# Kompilasi
router_agent_app = workflow.compile()
