import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# Impor struktur memori dan daftar alat yang sudah kita buat
from state import AgentState
from tools import tools_list

# Muat variabel lingkungan (API Key) secara aman
os.environ["GOOGLE_API_KEY"] = ""

# Inisialisasi Model LLM
# Pastikan GEMINI_API_KEY sudah ada di file .env
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0) # Temperature 0 agar responnya deterministik dan tidak berhalusinasi

# Ikat alat (tools) ke otak AI
llm_with_tools = llm.bind_tools(tools_list)

# PROMPT SISTEM: Ini adalah "hukum mutlak" yang mengatur cara agen berpikir
system_prompt = """Kamu adalah Agentic AI utama untuk Tim 3 di proyek Sekolah Rakyat.
Tujuanmu adalah menyusun materi pembelajaran terstruktur berbasis data.
Kamu BUKAN sekadar chatbot. Kamu adalah orkestrator alat (tools).

Aturan Mutlak:
1. Kamu WAJIB membaca Konteks Sistem (Parameter dan Emosi) sebelum bertindak.
2. Kamu WAJIB menggunakan 'retriever_tool' terlebih dahulu untuk mencari fakta dasar. Jangan mengarang materi sendiri.
3. Gunakan 'curriculum_mapper_tool' untuk memastikan materi sesuai tingkat pendidikan.
4. Gunakan 'emotion_adapter_tool' untuk menyesuaikan gaya bahasa jika emosi siswa terdeteksi negatif.
5. Gunakan 'content_structurer_tool' SEBAGAI LANGKAH TERAKHIR untuk memformat hasil akhir ke dalam JSON strict.
6. Jawaban akhirmu kepada pengguna HARUS berupa JSON dari 'content_structurer_tool', bukan teks narasi biasa.
"""

# ---------------------------------------------------------------------------
# DEFINISI NODE (PEKERJA)
# ---------------------------------------------------------------------------

# Node 1: Otak AI (Berpikir dan Memilih Alat)
def call_model(state: AgentState):
    messages = state["messages"]
    
    # Jika ini adalah langkah pertama, suntikkan Hukum Mutlak dan Konteks (Emosi & Parameter) dari Tim 1 & 6
    if len(messages) == 1:
        # Mengambil variabel spesifik dari AgentState
        params = state.get('request_params', {})
        emosi = state.get('emotion_input', {})
        
        context_msg = f"""Konteks Sistem Saat Ini:
        - Parameter Request Siswa (Tim 6): {params}
        - Status Emosi Siswa (Tim 1): {emosi}
        """
        # Gabungkan System Prompt, Konteks, dan pesan User
        messages = [
            SystemMessage(content=system_prompt), 
            SystemMessage(content=context_msg)
        ] + messages

    # Panggil LLM untuk mengevaluasi pesan dan memutuskan tindakan
    response = llm_with_tools.invoke(messages)
    
    # Simpan hasil pemikiran LLM ke dalam memori
    return {"messages": [response]}

# Node 2: Tangan AI (Mengeksekusi Alat)
# ToolNode secara otomatis akan mencari tool mana yang diminta oleh LLM di dalam tools_list
tool_node = ToolNode(tools_list)

# ---------------------------------------------------------------------------
# DEFINISI EDGE & ROUTING (LOGIKA PERGERAKAN)
# ---------------------------------------------------------------------------

def should_continue(state: AgentState):
    """Router untuk mengecek apakah AI meminta penggunaan tool atau sudah selesai."""
    last_message = state["messages"][-1]
    
    # Jika AI memanggil tool (misal: mencari di database)
    if last_message.tool_calls:
        return "ke_tools"
    # Jika tidak ada panggilan tool, berarti AI sudah memberikan jawaban akhir (Final Answer)
    return "selesai"

# ---------------------------------------------------------------------------
# MERAKIT GRAF LANGGRAPH
# ---------------------------------------------------------------------------
workflow = StateGraph(AgentState)

# Daftarkan Node
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Tentukan Alur: Mulai -> Agent
workflow.add_edge(START, "agent")

# Tentukan Alur Kondisional dari Agent
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "ke_tools": "tools",  # Jika router bilang "ke_tools", pergi ke Node "tools"
        "selesai": END        # Jika router bilang "selesai", akhiri graf
    }
)

# Tentukan Alur: Setelah alat selesai dijalankan, KEMBALI ke Agent untuk evaluasi
workflow.add_edge("tools", "agent")

# Kompilasi graf menjadi aplikasi yang siap dijalankan
alpha_agent_app = workflow.compile()