import os
import re
import json
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

# Import State DAG dan Tool Utility
from router_state import AgentState
from router_tools import (
    kb_sekolah,
    clean_json_from_llm,
    generate_soal_id,
    util_format_recommender,
    util_format_flashcard,
    util_format_mindmap,
    util_format_quiz,
    util_format_quiz_uraian,
    util_format_grade_quiz,
    util_format_grade_uraian,
)

# ================================================================
# 1. SETUP HUGGINGFACE LLM
# ================================================================

from huggingface_hub import InferenceClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration

HF_TOKEN = os.getenv("HF_TOKEN", "")

class HFChatModel(BaseChatModel):
    client: InferenceClient = None
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.3
    max_tokens: int = 2000

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
# 2. ROUTER — Rule-based (NO LLM)
# ================================================================

def router_task(state: AgentState) -> str:
    """Kondisional Edge: rule-based routing berdasarkan field 'task'"""
    task = state.get("task", "").lower()
    if "rekomendasi" in task:
        return "to_recommender"
    elif "flashcard" in task:
        return "to_flashcard"
    elif "mindmap" in task:
        return "to_mindmap"
    elif "grade_uraian" in task:     # cek grade_uraian SEBELUM quiz_uraian
        return "to_grade_uraian"
    elif "grade_quiz" in task:
        return "to_grade_quiz"
    elif "quiz_uraian" in task:      # cek quiz_uraian SEBELUM quiz
        return "to_quiz_uraian"
    elif "quiz" in task:
        return "to_quiz"
    else:
        return "to_structurer"       # Fallback


# ================================================================
# 3. NODE SPESIALIS
# ================================================================

# ---- Mekanik 1: Rekomendasi ----

def recommender_node(state: AgentState) -> dict:
    """Spesialis 1: Menganalisa nilai dan emosi -> Rekomendasi 3 Topik"""
    params  = state["request_params"]
    emosi   = state["emotion_input"].get("emosi", "netral")
    pretest = params.get("nilai_pretest", 0)
    ujian   = params.get("nilai_ujian", 0)

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
    match = re.search(r"<REKOMENDASI>(.*?)</REKOMENDASI>", result, re.DOTALL | re.IGNORECASE)
    extracted = match.group(1).strip() if match else result
    return {"top_recommendations": extracted}


# ---- Mekanik 2: Flashcard ----

def flashcard_node(state: AgentState) -> dict:
    """Spesialis 2: RAG + NotebookLM style Flashcards dengan kutipan sumber"""
    topik = state["request_params"].get("topik", "Sains")
    docs  = kb_sekolah.search(topik, k=3)
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
    match = re.search(r"<FLASHCARD>(.*?)</FLASHCARD>", result, re.DOTALL | re.IGNORECASE)
    extracted = match.group(1).strip() if match else result
    return {"flashcards_data": extracted}


# ---- Mekanik 3: Mindmap ----

def mindmap_node(state: AgentState) -> dict:
    """Spesialis 3: RAG + Node/Edge Hierarki Generator"""
    topik = state["request_params"].get("topik", "Sains")
    docs  = kb_sekolah.search(topik, k=2)
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


# ---- Mekanik 4a: Quiz PG — Generate ----

def quiz_node(state: AgentState) -> dict:
    """Spesialis 4a: NotebookLM-style Quiz PG — grounded RAG, soal_id di-inject Python-side"""
    topik  = state["request_params"].get("topik", "Sains")
    jumlah = state["request_params"].get("jumlah_soal", 5)

    docs = kb_sekolah.search(topik, k=3)
    context = "\n---\n".join([d.page_content.strip() for d in docs])
    sumber_list = list({
        f"{d.metadata.get('subject', 'Umum').title()} — {d.metadata.get('topic', '?').replace('_', ' ').title()}"
        for d in docs
    })

    prompt_sys = f"""Kamu adalah pembuat soal ujian profesional bergaya NotebookLM.
TUGAS: Buat tepat {jumlah} soal pilihan ganda KHUSUS tentang topik "{topik}".
ATURAN WAJIB:
1. Soal HARUS berhubungan langsung dengan "{topik}". Abaikan bagian referensi yang tidak relevan.
2. Setiap soal memiliki 4 pilihan (A, B, C, D) dan SATU jawaban benar.
3. Field "pembahasan" berisi penjelasan singkat mengapa jawaban benar.
4. Field "sumber" cukup diisi nama mata pelajaran dan topik referensinya saja (sudah disediakan).
5. DILARANG membuat soal di luar topik "{topik}".
6. Output: JSON Array murni dalam tag <QUIZ> ... </QUIZ>."""

    prompt_usr = f"""Topik: {topik}
Jumlah Soal: {jumlah}
Sumber referensi yang tersedia: {', '.join(sumber_list)}

Referensi:
{context}

Format yang diminta (DALAM TAG <QUIZ>):
[
  {{
    "nomor": 1,
    "pertanyaan": "Pertanyaan tentang {topik}?",
    "pilihan": {{
      "A": "Pilihan A",
      "B": "Pilihan B",
      "C": "Pilihan C",
      "D": "Pilihan D"
    }},
    "jawaban_benar": "A",
    "pembahasan": "Penjelasan singkat mengapa A benar.",
    "sumber": "Nama Mata Pelajaran — Nama Topik"
  }}
]"""

    result = _chat(system=prompt_sys, user=prompt_usr)
    match = re.search(r"<QUIZ>(.*?)</QUIZ>", result, re.DOTALL | re.IGNORECASE)
    extracted = match.group(1).strip() if match else result
    return {"quiz_data": extracted}


# ---- Mekanik 4b: Quiz PG — Grade (deterministik, NO LLM) ----

def grade_quiz_node(state: AgentState) -> dict:
    """Spesialis 4b: Grade Quiz PG — deterministik, tanpa LLM, match by soal_id"""
    params        = state["request_params"]
    soal_pg       = params.get("soal_pg", [])
    jawaban_siswa = params.get("jawaban_siswa", [])
    skor_per_soal = params.get("skor_per_soal", 10)

    lookup = {
        s["soal_id"]: s
        for s in soal_pg
        if isinstance(s, dict) and "soal_id" in s
    }

    detail = []
    for jawaban in jawaban_siswa:
        if not isinstance(jawaban, dict):
            continue
        sid       = jawaban.get("soal_id", "")
        ans       = jawaban.get("jawaban", "").strip().upper()
        soal      = lookup.get(sid, {})
        benar_ans = soal.get("jawaban_benar", "").strip().upper()
        benar     = (ans == benar_ans) if benar_ans else False

        detail.append({
            "soal_id":       sid,
            "nomor":         soal.get("nomor", "-"),
            "benar":         benar,
            "jawaban_siswa": ans,
            "jawaban_benar": benar_ans,
            "skor":          skor_per_soal if benar else 0,
            "skor_maksimal": skor_per_soal,
            "pembahasan":    soal.get("pembahasan", "")
        })

    return {"grade_quiz_result": json.dumps(detail, ensure_ascii=False)}


# ---- Mekanik 4c: Quiz Uraian — Generate ----

def quiz_uraian_node(state: AgentState) -> dict:
    """Spesialis 4c: Quiz Uraian — RAG + LLM generate soal esai + kunci jawaban"""
    topik  = state["request_params"].get("topik", "Sains")
    jumlah = state["request_params"].get("jumlah_soal", 5)

    docs = kb_sekolah.search(topik, k=3)
    context = "\n---\n".join([d.page_content.strip() for d in docs])
    sumber_list = list({
        f"{d.metadata.get('subject', 'Umum').title()} — {d.metadata.get('topic', '?').replace('_', ' ').title()}"
        for d in docs
    })

    prompt_sys = f"""Kamu adalah pembuat soal ujian esai profesional bergaya NotebookLM.
TUGAS: Buat tepat {jumlah} soal uraian/esai KHUSUS tentang topik "{topik}".
ATURAN WAJIB:
1. Soal HARUS berhubungan langsung dengan "{topik}". Abaikan referensi yang tidak relevan.
2. Setiap soal bersifat terbuka (uraian), bukan pilihan ganda.
3. Field "kunci_jawaban" berisi jawaban ideal yang lengkap dan terstruktur.
4. Field "skor_maksimal" diisi angka 20 untuk semua soal.
5. Field "sumber" cukup diisi nama mata pelajaran dan topik referensinya.
6. DILARANG membuat soal di luar topik "{topik}".
7. Output: JSON Array murni dalam tag <QUIZ_URAIAN> ... </QUIZ_URAIAN>."""

    prompt_usr = f"""Topik: {topik}
Jumlah Soal: {jumlah}
Sumber referensi yang tersedia: {', '.join(sumber_list)}

Referensi:
{context}

Format yang diminta (DALAM TAG <QUIZ_URAIAN>):
[
  {{
    "nomor": 1,
    "pertanyaan": "Jelaskan tentang {topik}...",
    "kunci_jawaban": "Jawaban ideal yang lengkap dan terstruktur berdasarkan referensi.",
    "skor_maksimal": 20,
    "sumber": "Nama Mata Pelajaran — Nama Topik"
  }}
]"""

    result = _chat(system=prompt_sys, user=prompt_usr)
    match = re.search(r"<QUIZ_URAIAN>(.*?)</QUIZ_URAIAN>", result, re.DOTALL | re.IGNORECASE)
    extracted = match.group(1).strip() if match else result
    return {"quiz_uraian_data": extracted}


# ---- Mekanik 4d: Quiz Uraian — Grade (LLM) ----

def grade_uraian_node(state: AgentState) -> dict:
    """Spesialis 4d: Grade Uraian — LLM nilai kemiripan + pemahaman + feedback per soal"""
    params        = state["request_params"]
    topik         = params.get("topik", "Tidak diketahui")
    soal_uraian   = params.get("soal_uraian", [])
    jawaban_siswa = params.get("jawaban_siswa", [])

    lookup = {
        s["soal_id"]: s
        for s in soal_uraian
        if isinstance(s, dict) and "soal_id" in s
    }

    # LLM call per soal
    detail = []
    for jawaban in jawaban_siswa:
        if not isinstance(jawaban, dict):
            continue
        sid       = jawaban.get("soal_id", "")
        ans_siswa = jawaban.get("jawaban", "").strip()
        soal      = lookup.get(sid, {})
        pertanyaan = soal.get("pertanyaan", "")
        kunci      = soal.get("kunci_jawaban", "")
        skor_maks  = soal.get("skor_maksimal", 20)

        if not pertanyaan or not kunci:
            detail.append({
                "soal_id": sid, "nomor": soal.get("nomor", "-"),
                "skor": 0, "skor_maksimal": skor_maks,
                "feedback": "Data soal tidak lengkap."
            })
            continue

        prompt_sys = f"""Kamu adalah penilai jawaban esai siswa yang objektif dan konstruktif.
TUGAS: Nilai jawaban siswa berdasarkan kunci jawaban yang tersedia.
OUTPUT: JSON murni satu objek, bungkus dalam tag <NILAI> ... </NILAI>.
Skala skor: 0 sampai {skor_maks}."""

        prompt_usr = f"""Pertanyaan: {pertanyaan}
Kunci Jawaban: {kunci}
Jawaban Siswa: {ans_siswa if ans_siswa else "(tidak menjawab)"}
Skor Maksimal: {skor_maks}

Hasilkan JSON dalam tag <NILAI>:
{{
  "skor": <angka 0-{skor_maks}>,
  "feedback": "Umpan balik spesifik dan konstruktif untuk jawaban siswa ini."
}}"""

        llm_result = _chat(system=prompt_sys, user=prompt_usr)
        m = re.search(r"<NILAI>(.*?)</NILAI>", llm_result, re.DOTALL | re.IGNORECASE)
        nilai_raw = m.group(1).strip() if m else llm_result
        nilai_obj = clean_json_from_llm(nilai_raw)

        skor = int(nilai_obj.get("skor", 0)) if isinstance(nilai_obj, dict) else 0
        skor = max(0, min(skor, skor_maks))

        detail.append({
            "soal_id":       sid,
            "nomor":         soal.get("nomor", "-"),
            "skor":          skor,
            "skor_maksimal": skor_maks,
            "feedback":      nilai_obj.get("feedback", "Tidak ada feedback.") if isinstance(nilai_obj, dict) else "Error parsing."
        })

    # 1 LLM call overall
    total = sum(d.get("skor", 0) for d in detail)
    maks  = sum(d.get("skor_maksimal", 20) for d in detail) or 1
    pct   = (total / maks) * 100

    prompt_overall_sys = "Kamu adalah guru yang memberikan penilaian holistik tingkat pemahaman siswa."
    prompt_overall_usr = f"""Topik: {topik}
Total skor siswa: {total} dari {maks} ({pct:.1f}%)
Detail per soal:
{json.dumps(detail, ensure_ascii=False, indent=2)}

Hasilkan JSON dalam tag <OVERALL>:
{{
  "tingkat_pemahaman": "<Belum Paham | Paham Dasar | Paham Mendalam>",
  "ringkasan_feedback": "Narasi singkat 2-3 kalimat tentang pemahaman siswa secara keseluruhan."
}}"""

    overall_raw = _chat(system=prompt_overall_sys, user=prompt_overall_usr)
    m2 = re.search(r"<OVERALL>(.*?)</OVERALL>", overall_raw, re.DOTALL | re.IGNORECASE)
    overall_obj = clean_json_from_llm(m2.group(1).strip() if m2 else overall_raw)

    pemahaman = overall_obj.get("tingkat_pemahaman", "Tidak diketahui") if isinstance(overall_obj, dict) else "Tidak diketahui"
    ringkasan = overall_obj.get("ringkasan_feedback", "") if isinstance(overall_obj, dict) else ""

    return {"grade_uraian_result": json.dumps({
        "detail": detail,
        "tingkat_pemahaman": pemahaman,
        "ringkasan_feedback": ringkasan
    }, ensure_ascii=False)}


# ================================================================
# 4. STRUCTURER — Muara semua jalur
# ================================================================

def structurer_node(state: AgentState) -> dict:
    """Muara Akhir: Merapikan Payload menjadi Strict JSON Dictionary untuk UI"""
    task = state.get("task", "")

    if task == "rekomendasi":
        raw_data    = state.get("top_recommendations", "")
        parsed_json = clean_json_from_llm(raw_data)
        emosi = "Siswa merasa " + state["emotion_input"].get("emosi", "netral")
        if isinstance(parsed_json, dict) and "saran_empatik" in parsed_json:
            emosi  = parsed_json["saran_empatik"]
            topics = parsed_json.get("topik_prioritas", [])
        else:
            topics = [{"error": "Data rusak dari spesialis"}]
        final_payload = util_format_recommender(topics, emosi)

    elif task == "flashcard":
        raw_data    = state.get("flashcards_data", "[]")
        parsed_json = clean_json_from_llm(raw_data)
        topik = state["request_params"].get("topik", "Tidak diketahui")
        if not isinstance(parsed_json, list):
            if isinstance(parsed_json, dict):
                for k, v in parsed_json.items():
                    if isinstance(v, list): parsed_json = v; break
            if not isinstance(parsed_json, list):
                parsed_json = [{"error": "Data rusak"}]
        final_payload = util_format_flashcard(topik, parsed_json)

    elif task == "mindmap":
        raw_data    = state.get("mindmap_data", "{}")
        parsed_json = clean_json_from_llm(raw_data)
        topik = state["request_params"].get("topik", "Tidak diketahui")
        if not isinstance(parsed_json, dict) or "error" in parsed_json:
            parsed_json = {"error": "Format mindmap rusak"}
        nodes_arr = [parsed_json] if isinstance(parsed_json, dict) else parsed_json
        final_payload = util_format_mindmap(topik, nodes_arr)

    elif task == "quiz":
        raw_data    = state.get("quiz_data", "[]")
        parsed_json = clean_json_from_llm(raw_data)
        topik = state["request_params"].get("topik", "Tidak diketahui")
        if not isinstance(parsed_json, list):
            if isinstance(parsed_json, dict):
                for k, v in parsed_json.items():
                    if isinstance(v, list): parsed_json = v; break
            if not isinstance(parsed_json, list):
                parsed_json = [{"error": "Data rusak"}]
        final_payload = util_format_quiz(topik, parsed_json)

    elif task == "quiz_uraian":
        raw_data    = state.get("quiz_uraian_data", "[]")
        parsed_json = clean_json_from_llm(raw_data)
        topik = state["request_params"].get("topik", "Tidak diketahui")
        if not isinstance(parsed_json, list):
            if isinstance(parsed_json, dict):
                for k, v in parsed_json.items():
                    if isinstance(v, list): parsed_json = v; break
            if not isinstance(parsed_json, list):
                parsed_json = [{"error": "Data rusak"}]
        final_payload = util_format_quiz_uraian(topik, parsed_json)

    elif task == "grade_quiz":
        raw_data = state.get("grade_quiz_result", "[]")
        topik    = state["request_params"].get("topik", "Tidak diketahui")
        detail   = json.loads(raw_data) if isinstance(raw_data, str) and raw_data else []
        if not isinstance(detail, list):
            detail = [{"error": "Data grade rusak"}]
        final_payload = util_format_grade_quiz(topik, detail)

    elif task == "grade_uraian":
        raw_data = state.get("grade_uraian_result", "{}")
        topik    = state["request_params"].get("topik", "Tidak diketahui")
        parsed   = json.loads(raw_data) if isinstance(raw_data, str) and raw_data else {}
        if not isinstance(parsed, dict): parsed = {}
        detail    = parsed.get("detail", [])
        pemahaman = parsed.get("tingkat_pemahaman", "Tidak diketahui")
        ringkasan = parsed.get("ringkasan_feedback", "")
        final_payload = util_format_grade_uraian(topik, detail, pemahaman, ringkasan)

    else:
        final_payload = {"error": f"Task '{task}' tidak dikenal"}

    return {"final_payload": final_payload}


# ================================================================
# 5. GRAPH COMPILATION
# ================================================================

workflow = StateGraph(AgentState)

workflow.add_node("recommender",  recommender_node)
workflow.add_node("flashcard",    flashcard_node)
workflow.add_node("mindmap",      mindmap_node)
workflow.add_node("quiz",         quiz_node)
workflow.add_node("quiz_uraian",  quiz_uraian_node)
workflow.add_node("grade_quiz",   grade_quiz_node)
workflow.add_node("grade_uraian", grade_uraian_node)
workflow.add_node("structurer",   structurer_node)

workflow.add_conditional_edges(
    START,
    router_task,
    {
        "to_recommender":  "recommender",
        "to_flashcard":    "flashcard",
        "to_mindmap":      "mindmap",
        "to_quiz":         "quiz",
        "to_quiz_uraian":  "quiz_uraian",
        "to_grade_quiz":   "grade_quiz",
        "to_grade_uraian": "grade_uraian",
        "to_structurer":   "structurer",
    }
)

for node in ["recommender", "flashcard", "mindmap", "quiz", "quiz_uraian", "grade_quiz", "grade_uraian"]:
    workflow.add_edge(node, "structurer")

workflow.add_edge("structurer", END)

router_agent_app = workflow.compile()
