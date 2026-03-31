import os
import re
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# Import State DAG dan Tool Utility dari file yang direfactor
from multi_state import AgentState
from multi_tools import (
    kb_sekolah,
    util_mapper_kurikulum,
    util_adapt_emotion,
    util_build_citation,
    util_difficulty_adjuster,
    util_structure_json
)

# ================================================================
# 1. SETUP HUGGINGFACE LLM GRATIS (SESUAI TES_1.IPYNB)
# ================================================================

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN", "")

from huggingface_hub import InferenceClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration

# Wrapper Custom HF Chat Mencegah TypeError langhcain-huggingface yang umum terjadi
class HFChatModel(BaseChatModel):
    client: InferenceClient = None
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.3
    max_tokens: int = 1500

    class Config:
        arbitrary_types_allowed = True
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = InferenceClient(
            model=self.model_id,
            token=HF_TOKEN,
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
# 2. DEFINISI NODE (AGEN PABRIK PERAKITAN / DAG PIPELINE)
# ================================================================

def retriever_node(state: AgentState) -> dict:
    """Node 1: Cari dokumen dari VectorDB (Tim 2/Knowledge)"""
    topik = state["request_params"].get("topik", "")
    
    docs = kb_sekolah.search(topik, k=2)
    texts = [d.page_content.strip() for d in docs]
    
    # Simpan di state layaknya memori papan tulis bersama
    return {
        "retrieved_documents": texts,
        "messages": [AIMessage(content=f"[Retrieved {len(texts)} document chunks about {topik}]")]
    }


def grader_node(state: AgentState) -> dict:
    """Node 2: Evaluasi Relevansi RAG (Self-Reflection Level 1)"""
    topik = state["request_params"].get("topik", "")
    docs = state.get("retrieved_documents", [])
    
    if not docs:
        return {"documents_relevant": False}
        
    combined = "\\n---\\n".join(docs)
    
    result = _chat(
        system="Tentukan apakah dokumen RELEVAN untuk topik ini. Jawab HANYA: 'ya' atau 'tidak'.",
        user=f"Topik: {topik}\\n\\nDokumen:\\n{combined}"
    )
    
    relevant = any(w in result.lower() for w in ["ya", "yes", "relevan"])
    
    return {
        "documents_relevant": relevant,
        "messages": [AIMessage(content=f"[RAG Relevance Evaluated: {relevant}]")]
    }


def content_generator_node(state: AgentState) -> dict:
    """Node 3: Menulis Narasi Berbasis RAG (Tim 2)"""
    params = state["request_params"]
    topik = params.get("topik", "")
    tingkat = params.get("tingkat", "SMA")
    riwayat = params.get("riwayat_nilai_rata_rata", 70)
    
    docs = state.get("retrieved_documents", [])
    relevant = state.get("documents_relevant", False)
    
    # 1. Persiapan Pipeline Instruksi
    context = "\n".join(docs) if relevant and docs else "Gunakan pengetahuan umummu karena referensi tidak ditemukan."
    mapper_msg = util_mapper_kurikulum(topik, tingkat)
    
    # Ambil instruksi prompt dan label level dari utility
    diff_msg, level_label = util_difficulty_adjuster(riwayat)
    
    # 2. Generasi Berbasis Fakta
    draft = _chat(
        system=f"Kamu guru {tingkat} yang kreatif. Wajib baca instruksi kurikulum: {mapper_msg}\nJuga ikuti pedoman kesulitan: {diff_msg}\nTulislah penjelasan 1-3 paragraf saja yang padat.",
        user=f"Referensi: {context}\n\nJelaskan topik '{topik}' berdasarkan referensi di atas:"
    )
    
    # 3. Validasi Akademik (Citation Builder & Penyesuaian Level)
    if relevant:
        draft = util_build_citation(draft, "[Tervalidasi: Modul Sekolah Dasar/Menengah]")
        
    # Memaksa cap level tercetak di materi akhir tanpa campur tangan AI
    draft += f"\n\n[PENYESUAIAN LEVEL]: Materi disesuaikan ke tingkat {level_label} berdasarkan riwayat nilai."
        
    return {
        "draft_content": draft,
        "messages": [AIMessage(content="[Draft Content Generated with Citations and Difficulty Adjusted]")]
    }


def emotion_adapter_node(state: AgentState) -> dict:
    """Node 4: Adaptasi Emosi Pelajar (Tim 1)"""
    emosi = state["emotion_input"].get("emosi", "netral")
    draft = state.get("draft_content", "")
    
    # Menyuntikkan helper emosional dari tools
    adapted = util_adapt_emotion(draft, emosi)
    
    return {
        "adapted_content": adapted,
        "messages": [AIMessage(content=f"[Emotion Adapted for '{emosi}']")]
    }
    

def quality_checker_node(state: AgentState) -> dict:
    """Node 5: QC Self-Reflection (Skor 1-10)"""
    adapted = state.get("adapted_content", "")
    tingkat = state["request_params"].get("tingkat", "")
    
    result = _chat(
        system="Kamu quality assurance. Evaluasi kesesuaian materi ini dengan tingkat siswa.\\nBerikan SKOR: [1-10]\\nFEEDBACK: [alasan]",
        user=f"Tingkat Siswa: {tingkat}\\nMateri:\\n{adapted}"
    )
    
    score = 7 # Default toleransi
    try:
        for line in result.split('\\n'):
            if 'SKOR' in line.upper():
                digits = ''.join(c for c in line if c.isdigit())
                if digits:
                    score = min(max(int(digits[:2]), 1), 10)
                break
    except:
        pass
        
    rev = state.get("revision_count", 0) + 1
    
    return {
        "quality_score": score,
        "quality_feedback": result,
        "revision_count": rev,
        "messages": [AIMessage(content=f"[QC Evaluated - Score: {score}/10 | Revision: {rev}]")]
    }


def revision_node(state: AgentState) -> dict:
    """Node 5B: Revisi Konten Jika QC Menolak"""
    adapted = state.get("adapted_content", "")
    feedback = state.get("quality_feedback", "")
    
    # 1. Pagari LLM dengan instruksi batas yang mutlak
    revised_raw = _chat(
        system="Kamu adalah editor materi pendidikan. Perbaiki materi berdasarkan feedback. HUKUM MUTLAK: Kamu HANYA BOLEH memberikan teks materi perbaikannya saja. JANGAN menambahkan kata pengantar. JANGAN mengulang feedback. WAJIB membungkus hasil materi perbaikanmu dengan tag <MATERI>teks_disini</MATERI>.",
        user=f"FEEDBACK REVISI:\n{feedback}\n\nMATERI SAAT INI:\n{adapted}\n\nKeluarkan teks perbaikan di dalam tag <MATERI> sekarang:"
    )
    
    # 2. Parsing Paksa: Ekstrak hanya teks yang ada di dalam batas tag
    match = re.search(r"<MATERI>(.*?)</MATERI>", revised_raw, re.DOTALL | re.IGNORECASE)
    
    if match:
        clean_revised = match.group(1).strip()
    else:
        # Fallback pertahanan terakhir jika AI masih keras kepala membuang tag
        clean_revised = revised_raw.split("MATERI SETELAH PERBAIKAN:")[-1].strip()
        clean_revised = clean_revised.split("FEEDBACK:")[-1].strip()
        # Buang sisa markdown json jika masih terbawa
        clean_revised = clean_revised.replace("```json", "").replace("```", "").strip()
    
    return {
        "adapted_content": clean_revised,
        "messages": [AIMessage(content=f"[Content Revised and Boundaries Clipped Strictly]")]
    }

def structurer_node(state: AgentState) -> dict:
    """Node 6: Finalisasi ke Bentuk JSON Strict (Tim 6)"""
    params = state["request_params"]
    topik = params.get("topik", "")
    tingkat = params.get("tingkat", "")
    riwayat = params.get("riwayat_nilai_rata_rata", 70)
    emosi = state["emotion_input"].get("emosi", "netral")
    final_text = state.get("adapted_content", "")
    
    # Membangun analitik saran guru secara terstruktur
    if emosi in ["frustrasi", "bingung", "sedih"]:
        saran_guru = f"Siswa terdeteksi {emosi} (Nilai rata-rata: {riwayat}). Pendekatan sistem saat ini menggunakan banyak analogi dasar. Mohon pantau pemahaman konsep fundamental siswa."
    else:
        saran_guru = f"Siswa dalam kondisi {emosi} (Nilai rata-rata: {riwayat}). Siswa siap menerima tantangan materi sesuai standar kurikulum."
    
    # Masukkan saran_guru ke dalam util_structure_json
    final_json = util_structure_json(final_text, topik, tingkat, saran_guru)
    
    return {
        "final_payload": final_json,
        "messages": [AIMessage(content=f"[Final JSON Payload Constructed]")]
    }


# ================================================================
# 3. ROUTING LOGIC & GRAPH COMPILATION
# ================================================================

def should_revise(state: AgentState) -> str:
    """Conditional Edge: QC Loop"""
    score = state.get("quality_score", 7)
    revisions = state.get("revision_count", 0)
    max_rev = state.get("max_revisions", 2)
    
    if score >= 7 or revisions >= max_rev:
        return "accept"
    return "revise"

# Merakit DAG
workflow = StateGraph(AgentState)

# Daftarkan Nodes
workflow.add_node("retriever", retriever_node)
workflow.add_node("grader", grader_node)
workflow.add_node("content_generator", content_generator_node)
workflow.add_node("emotion_adapter", emotion_adapter_node)
workflow.add_node("quality_checker", quality_checker_node)
workflow.add_node("revision", revision_node)
workflow.add_node("structurer", structurer_node)

# Tentukan Alur Sekuensial Pabrik Perakitan
workflow.add_edge(START, "retriever")
workflow.add_edge("retriever", "grader")
workflow.add_edge("grader", "content_generator")
workflow.add_edge("content_generator", "emotion_adapter")
workflow.add_edge("emotion_adapter", "quality_checker")

# Tentukan Alur Kondisional QC Reflection
workflow.add_conditional_edges(
    "quality_checker",
    should_revise,
    {
        "accept": "structurer",
        "revise": "revision"
    }
)

# Loop back QC
workflow.add_edge("revision", "quality_checker")

# Final Edge JSON
workflow.add_edge("structurer", END)

# Kompilasi graph layaknya aplikasi
alpha_agent_app = workflow.compile()