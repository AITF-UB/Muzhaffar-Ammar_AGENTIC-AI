# state
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
import os

# Mendefinisikan struktur memori Agen
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


from langchain_core.tools import tool

# Ini adalah "Vector DB" tiruan kita
buku_referensi = {
    "pecahan": "Pecahan adalah bilangan yang memiliki pembilang dan penyebut, contohnya 1/2.",
    "fotosintesis": "Fotosintesis adalah proses tumbuhan mengubah cahaya matahari menjadi energi.",
    "kemerdekaan": "Indonesia memproklamasikan kemerdekaan pada 17 Agustus 1945."
}

@tool
def cari_materi(topik: str) -> str:
    """
    Gunakan tool ini HANYA untuk mencari definisi atau materi pelajaran.
    Masukkan kata kunci (topik) yang relevan secara spesifik, misalnya 'pecahan'.
    """
    kata_kunci = topik.lower()
    # Mengembalikan isi buku jika ada, atau pesan error jika tidak ada
    return buku_referensi.get(kata_kunci, "Maaf, materi tersebut tidak ditemukan di buku referensi.")

# Kita simpan tool ini dalam sebuah list, karena agen bisa punya lebih dari 1 tool nanti
tools_untuk_agen = [cari_materi] 

os.environ["GOOGLE_API_KEY"] = ""

# Inisialisasi model Gemini 1.5 Flash
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

# Kita "ikat" tool pencari materi ke LLM
llm_dengan_tools = llm.bind_tools(tools_untuk_agen)

# Node 1: Otak AI (LLM)
def jalankan_llm(state: AgentState):
    # 1. Ambil riwayat percakapan dari memori
    riwayat_pesan = state["messages"]
    
    # 2. Suapkan riwayat tersebut ke LLM yang sudah punya tools
    respon_ai = llm_dengan_tools.invoke(riwayat_pesan)
    
    # 3. Kembalikan respon AI untuk ditambahkan ke dalam memori
    return {"messages": [respon_ai]}

# Node 2: Tangan AI (Pengeksekusi Tool)
node_tools = ToolNode(tools_untuk_agen)

# Fungsi Penentu Rute (Router)
def cek_rute(state: AgentState):
    # Ambil pesan paling terakhir dari memori
    pesan_terakhir = state["messages"][-1]
    
    # Cek apakah AI meminta pemanggilan tool
    if pesan_terakhir.tool_calls:
        return "ke_tools" # Arahkan ke Node Tool
    else:
        return "selesai"  # Akhiri proses
    
# 1. Buat kerangka graf dengan memori AgentState yang kita buat di Langkah 1
workflow = StateGraph(AgentState)

# 2. Masukkan node-node pekerja kita ke dalam graf
workflow.add_node("otak_ai", jalankan_llm)
workflow.add_node("tangan_ai", node_tools)

# 3. Hubungkan START langsung ke otak agar AI membaca pertanyaan dulu
workflow.add_edge(START, "otak_ai")

# 4. Tambahkan percabangan (conditional edges) setelah otak bekerja
workflow.add_conditional_edges(
    "otak_ai", 
    cek_rute, 
    {
        "ke_tools": "tangan_ai", # Jika butuh tool, pergi ke tangan
        "selesai": END         # Jika sudah punya jawaban, akhiri
    }
)

# 5. Hubungkan kembali tangan ke otak agar siklusnya tertutup
workflow.add_edge("tangan_ai", "otak_ai")

# 6. Kompilasi graf menjadi agen yang bisa dijalankan
agen = workflow.compile()

# --- PENGUJIAN AGEN ---
pertanyaan_user = "Tolong jelaskan secara singkat apa itu pecahan?"

# Kita siapkan memori awal yang hanya berisi pertanyaan user
input_state = {"messages": [("user", pertanyaan_user)]}

print("Memulai proses agen...\n")

# Kita gunakan .stream() untuk melihat setiap langkah yang dilakukan agen
for output in agen.stream(input_state):
    for nama_node, hasil_node in output.items():
        print(f"🤖 Agen sedang berada di: {nama_node}")
        print(f"Isi pesan yang dihasilkan: {hasil_node['messages'][0].content}")
        print("-" * 30)