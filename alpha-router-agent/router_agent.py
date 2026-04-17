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
    _ambil_prioritas_belajar,
    util_format_recommender,
    util_format_flashcard,
    util_format_mindmap,
    util_format_quiz,
    util_format_quiz_uraian,
    util_format_evaluasi_quiz,
    util_format_evaluasi_uraian,
    util_format_konten_belajar,
    util_format_rag_query,
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

# Standard LLM
llm = HFChatModel()
# LLM untuk konten panjang (token lebih banyak)
llm_long = HFChatModel(max_tokens=3000)

def _chat(system: str, user: str) -> str:
    return llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content.strip()

def _chat_long(system: str, user: str) -> str:
    return llm_long.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content.strip()

# ================================================================
# 2. ROUTER — Rule-based (NO LLM)
# ================================================================

def router_task(state: AgentState) -> str:
    """Kondisional Edge: rule-based routing berdasarkan field 'task'"""
    task = state.get("task", "").lower()
    if "rekomendasi" in task:
        return "to_recommender"
    elif "konten_belajar" in task:
        return "to_konten_belajar"
    elif "rag_query" in task:
        return "to_rag_query"
    elif "flashcard" in task:
        return "to_flashcard"
    elif "mindmap" in task:
        return "to_mindmap"
    elif "evaluasi_uraian" in task:   # cek SEBELUM quiz_uraian
        return "to_evaluasi_uraian"
    elif "evaluasi_quiz" in task:
        return "to_evaluasi_quiz"
    elif "quiz_uraian" in task:       # cek SEBELUM quiz
        return "to_quiz_uraian"
    elif "quiz" in task:
        return "to_quiz"
    else:
        return "to_structurer"        # Fallback


# ================================================================
# 3. NODE SPESIALIS
# ================================================================

# ---- Mekanik 1: Rekomendasi v2 ----

def recommender_node(state: AgentState) -> dict:
    """Spesialis 1: Recommender v2 — support first_time dan returning dengan riwayat progress."""
    params     = state["request_params"]
    emosi      = state["emotion_input"].get("emosi", "netral")
    student_id = params.get("student_id", "unknown")
    first_time = params.get("first_time", True)

    if first_time:
        # ── Alur pertama kali: berdasarkan hasil pretest ──
        hasil_pretest  = params.get("hasil_pretest", [])
        matpel_dipilih = params.get("matpel_dipilih", [])
        context_str    = json.dumps(hasil_pretest, ensure_ascii=False, indent=2)

        prompt_sys = """Kamu adalah sistem rekomendasi belajar yang cerdas dan empatik.
TUGAS: Analisis hasil pretest siswa dan rekomendasikan 3 bab yang paling perlu dipelajari.
ATURAN WAJIB:
1. Output JSON murni. Bungkus dalam tag <REKOMENDASI> ... </REKOMENDASI>.
2. Rekomendasi diurutkan dari yang paling mendesak (paling lemah).
3. Sertakan alasan spesifik dan saran aksi yang konkret untuk tiap rekomendasi."""

        prompt_usr = f"""Data Siswa Baru:
- Mata pelajaran yang dipilih: {', '.join(matpel_dipilih)}
- Emosi siswa saat ini: {emosi}

Hasil Pretest:
{context_str}

Hasilkan JSON dalam tag <REKOMENDASI>:
{{
  "pesan_empatik": "Kalimat penyambutan dan motivasi yang hangat untuk siswa baru.",
  "rekomendasi": [
    {{
      "urutan": 1,
      "matpel": "Nama Mata Pelajaran",
      "bab": "Nama Bab Spesifik",
      "alasan": "Alasan konkret berdasarkan data pretest.",
      "saran_aksi": "Langkah pertama yang bisa dilakukan siswa."
    }}
  ]
}}"""

    else:
        # ── Alur returning: berdasarkan riwayat progress ──
        riwayat_raw      = params.get("riwayat_progress", [])
        riwayat_filtered = _ambil_prioritas_belajar(riwayat_raw, top_n=5)
        matpel_dipilih   = params.get("matpel_dipilih", [])
        context_str      = json.dumps(riwayat_filtered, ensure_ascii=False, indent=2)

        prompt_sys = """Kamu adalah sistem rekomendasi belajar adaptif yang cerdas dan empatik.
TUGAS: Analisis riwayat progress siswa (sudah difilter ke 5 bab terlemah) dan rekomendasikan 3 bab yang paling perlu diulang atau diperkuat.
ATURAN WAJIB:
1. Output JSON murni. Bungkus dalam tag <REKOMENDASI> ... </REKOMENDASI>.
2. Pertimbangkan skor, tingkat pemahaman, emosi dominan, dan jumlah prompt (banyak prompt = sering bingung).
3. Rekomendasi diurutkan dari yang paling mendesak.
4. Sertakan alasan spesifik dan saran aksi yang adaptif."""

        prompt_usr = f"""Data Progress Siswa:
- Mata pelajaran yang dipelajari: {', '.join(matpel_dipilih)}
- Emosi siswa saat ini: {emosi}

5 Bab Paling Lemah (dari total riwayat yang lebih panjang, sudah difilter):
{context_str}

Hasilkan JSON dalam tag <REKOMENDASI>:
{{
  "pesan_empatik": "Kalimat motivasi yang personal berdasarkan progress siswa.",
  "rekomendasi": [
    {{
      "urutan": 1,
      "matpel": "Nama Mata Pelajaran",
      "bab": "Nama Bab Spesifik",
      "alasan": "Alasan konkret berdasarkan data progress.",
      "saran_aksi": "Langkah spesifik untuk meningkatkan pemahaman di bab ini."
    }}
  ]
}}"""

    result = _chat(system=prompt_sys, user=prompt_usr)
    match  = re.search(r"<REKOMENDASI>(.*?)</REKOMENDASI>", result, re.DOTALL | re.IGNORECASE)
    extracted = match.group(1).strip() if match else result
    return {"top_recommendations": extracted}


#----Flashcard----

def flashcard_node(state: AgentState) -> dict:
    """Spesialis 2: RAG + NotebookLM style Flashcards, adaptive difficulty berdasarkan nilai_siswa"""
    params      = state["request_params"]
    matpel      = params.get("matpel", "Umum")
    bab         = params.get("bab", "")
    nilai_siswa = params.get("nilai_siswa", None)
    query_rag   = f"{matpel} {bab}".strip()
    level_soal, instruksi_kesulitan = _hitung_level_soal(nilai_siswa)

    docs    = kb_sekolah.search(query_rag, k=8)
    context = "\n---\n".join([d.page_content.strip() for d in docs])

    prompt_sys = f"""Kamu adalah spesialis pembuat Flashcard berstandar tinggi bergaya NotebookLM.
TUGAS: Buat TEPAT 10-15 pasang pertanyaan (Front) dan jawaban (Back) tentang bab "{bab}" dari mata pelajaran {matpel},
berdasarkan referensi berikut.
TINGKAT KESULITAN: {level_soal}
{instruksi_kesulitan}

ATURAN NOTEBOOKLM (WAJIB):
1. Field "front": Berisi pertanyaan ringkas yang menguji konsep utama.
2. Field "back": WAJIB SANGAT SINGKAT DAN JELAS. Maksimal 1 hingga 2 kalimat pendek! Flashcard tidak boleh berupa paragraf panjang.

ATURAN OUTPUT:
Keluarkan output dalam format JSON Array murni. Bungkus dalam tag <FLASHCARD> ... </FLASHCARD>."""

    prompt_usr = f"""Mata Pelajaran: {matpel}
Bab: {bab}
Tingkat Kesulitan: {level_soal}

Referensi:
{context}

Format (DALAM TAG <FLASHCARD>):
[
  {{
    "front": "Pertanyaan {level_soal} tentang {bab}?",
    "back": "Jawaban sangat singkat dan jelas (maks 2 kalimat singkat)."
  }}
]"""

    result    = _chat(system=prompt_sys, user=prompt_usr)
    match     = re.search(r"<FLASHCARD>(.*?)</FLASHCARD>", result, re.DOTALL | re.IGNORECASE)
    extracted = match.group(1).strip() if match else result
    return {"flashcards_data": extracted}


#----Mindmap----

def mindmap_node(state: AgentState) -> dict:
    """Spesialis 3: RAG + Peta Konsep hierarki komprehensif, bahasa adaptif berdasarkan nilai_siswa"""
    params      = state["request_params"]
    matpel      = params.get("matpel", "Umum")
    bab         = params.get("bab", "")
    nilai_siswa = params.get("nilai_siswa", None)
    query_rag   = f"{matpel} {bab}".strip()

    # Mindmap selalu komprehensif — yang berubah hanya gaya bahasa
    if nilai_siswa is None or int(nilai_siswa) <= 70:
        gaya_bahasa = (
            "Tulis setiap 'penjelasan' dan 'deskripsi' dengan bahasa sederhana, "
            "gunakan analogi sehari-hari jika memungkinkan. Hindari jargon teknis."
        )
    else:
        gaya_bahasa = (
            "Tulis setiap 'penjelasan' dan 'deskripsi' dengan bahasa teknis yang akurat dan presisi. "
            "Gunakan istilah ilmiah yang tepat."
        )

    docs    = kb_sekolah.search(query_rag, k=10)
    context = "\n---\n".join([d.page_content.strip() for d in docs])

    prompt_sys = f"""Kamu adalah ahli pembuat Peta Konsep (Mindmap) yang komprehensif.
TUGAS: Buat struktur hierarki (parent-child) LENGKAP untuk semua konsep dalam bab "{bab}" ({matpel}).
STRUKTUR: Tampilkan SEMUA relasi penting antar konsep — jangan disederhanakan atau dipotong.
GAYA BAHASA: {gaya_bahasa}
ATURAN OUTPUT: JSON murni dalam tag <MINDMAP> ... </MINDMAP>."""

    prompt_usr = f"""Mata Pelajaran: {matpel}
Bab: {bab}

Referensi:
{context}

Format (DALAM TAG <MINDMAP>):
{{
  "konsep_utama": "{bab}",
  "deskripsi": "Deskripsi singkat 1 kalimat",
  "children": [
    {{
      "sub_konsep": "Nama Sub-konsep",
      "penjelasan": "Penjelasan relasi ke konsep utama",
      "children": [
        {{"sub_konsep": "Detail", "penjelasan": "...", "children": []}}
      ]
    }}
  ]
}}"""

    result    = _chat(system=prompt_sys, user=prompt_usr)
    match     = re.search(r"<MINDMAP>(.*?)</MINDMAP>", result, re.DOTALL | re.IGNORECASE)
    extracted = match.group(1).strip() if match else result
    return {"mindmap_data": extracted}


# ---- Helper: Adaptive Difficulty ----

def _hitung_level_soal(nilai_pretest) -> tuple:
    """
    Tentukan level kesulitan soal berdasarkan nilai siswa (pretest ATAU nilai quiz sebelumnya).
    Parameter `nilai_siswa` bisa diisi dari:
      - Nilai pretest awal (akses pertama)
      - Nilai quiz PG/uraian terakhir di topik/matpel yang sama (akses berikutnya)
    Returns: (label_level, instruksi_kesulitan_untuk_prompt)
    """
    if nilai_pretest is None:
        return (
            "Dasar",
            "Buat soal tingkat DASAR. Fokus pada pengenalan dan pemahaman konsep fundamental. "
            "Pertanyaan bersifat langsung, tidak membutuhkan analisis mendalam."
        )
    nilai = int(nilai_pretest)
    if nilai <= 40:
        return (
            "Dasar",
            "Buat soal tingkat DASAR. Fokus pada pengenalan dan hafalan konsep dasar. "
            "Gunakan pertanyaan faktual yang jawabannya langsung dari teks/referensi."
        )
    elif nilai <= 70:
        return (
            "Menengah",
            "Buat soal tingkat MENENGAH. Campurkan 60% soal pemahaman konsep dan 40% soal aplikasi/penerapan. "
            "Siswa perlu memahami 'mengapa' dan 'bagaimana', bukan hanya fakta."
        )
    else:
        return (
            "HOTS",
            "Buat soal tingkat HOTS (Higher Order Thinking Skills). Fokus pada analisis, evaluasi, dan sintesis. "
            "Siswa harus berpikir kritis, membandingkan konsep, dan menarik kesimpulan sendiri. "
            "Hindari soal faktual sederhana — semua soal harus membutuhkan penalaran."
        )


#----Quiz PG — Generate----

def quiz_node(state: AgentState) -> dict:
    """Spesialis 4a: NotebookLM-style Quiz PG — 5 soal fixed, adaptive difficulty"""
    params        = state["request_params"]
    matpel        = params.get("matpel", "Umum")
    bab           = params.get("bab", "")
    nilai_siswa   = params.get("nilai_siswa", None)  # nilai pretest ATAU nilai quiz sebelumnya
    jumlah        = 5  # FIXED
    query_rag     = f"{matpel} {bab}".strip()
    level_soal, instruksi_kesulitan = _hitung_level_soal(nilai_siswa)

    docs = kb_sekolah.search(query_rag, k=5)
    context = "\n---\n".join([d.page_content.strip() for d in docs])
    sumber_list = list({
        f"{d.metadata.get('subject', 'Umum').title()} — {d.metadata.get('topic', '?').replace('_', ' ').title()}"
        for d in docs
    })

    prompt_sys = f"""Kamu adalah pembuat soal ujian profesional bergaya NotebookLM.
TUGAS: Buat tepat {jumlah} soal pilihan ganda KHUSUS tentang bab "{bab}" dari mata pelajaran {matpel}.
TINGKAT KESULITAN: {level_soal}
{instruksi_kesulitan}
ATURAN WAJIB:
1. Soal HARUS berhubungan langsung dengan bab "{bab}". Abaikan referensi yang tidak relevan.
2. Setiap pertanyaan harus memiliki 4 pilihan jawaban (A, B, C, D).
3. Field "jawaban_benar" hanya berisi satu huruf kapital (A/B/C/D).
4. Field "pembahasan" berisi alasan mengapa jawaban tersebut benar.
5. DILARANG membuat soal di luar bab "{bab}".
6. Output: JSON Array murni dalam tag <QUIZ> ... </QUIZ>."""

    prompt_usr = f"""Mata Pelajaran: {matpel}
Bab: {bab}
Jumlah Soal: {jumlah}
Tingkat Kesulitan: {level_soal}

Referensi:
{context}

Format (DALAM TAG <QUIZ>):
[
  {{
    "nomor": 1,
    "pertanyaan": "Pertanyaan {level_soal} tentang {bab}?",
    "pilihan": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
    "jawaban_benar": "A",
    "pembahasan": "Penjelasan mengapa A benar."
  }}
]"""

    result = _chat(system=prompt_sys, user=prompt_usr)
    match  = re.search(r"<QUIZ>(.*?)</QUIZ>", result, re.DOTALL | re.IGNORECASE)
    extracted = match.group(1).strip() if match else result
    return {"quiz_data": extracted}


# ---- Evaluasi Quiz PG (deterministik, NO LLM) ----

def evaluasi_quiz_node(state: AgentState) -> dict:
    """Spesialis 4b: Evaluasi Quiz PG — deterministik, tanpa LLM, match by soal_id"""
    params        = state["request_params"]
    soal_pg       = params.get("soal_pg", [])
    jawaban_siswa = params.get("jawaban_siswa", [])
    skor_per_soal = params.get("skor_per_soal", 10)

    lookup = {s["soal_id"]: s for s in soal_pg if isinstance(s, dict) and "soal_id" in s}

    detail = []
    for jawaban in jawaban_siswa:
        if not isinstance(jawaban, dict): continue
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

    return {"evaluasi_quiz_result": json.dumps(detail, ensure_ascii=False)}


#----Quiz Uraian — Generate----

def quiz_uraian_node(state: AgentState) -> dict:
    """Spesialis 4c: Quiz Uraian — 5 soal fixed, adaptive difficulty, RAG + LLM"""
    params        = state["request_params"]
    matpel        = params.get("matpel", "Umum")
    bab           = params.get("bab", "")
    nilai_siswa   = params.get("nilai_siswa", None)  # nilai pretest ATAU nilai quiz sebelumnya
    jumlah        = 5  # FIXED
    query_rag     = f"{matpel} {bab}".strip()
    level_soal, instruksi_kesulitan = _hitung_level_soal(nilai_siswa)

    docs = kb_sekolah.search(query_rag, k=5)
    context = "\n---\n".join([d.page_content.strip() for d in docs])

    prompt_sys = f"""Kamu adalah pembuat soal ujian esai profesional bergaya NotebookLM.
TUGAS: Buat tepat {jumlah} soal uraian/esai KHUSUS tentang bab "{bab}" dari mata pelajaran {matpel}.
TINGKAT KESULITAN: {level_soal}
{instruksi_kesulitan}
ATURAN WAJIB:
1. Soal HARUS berhubungan langsung dengan bab "{bab}". Abaikan referensi yang tidak relevan.
2. Setiap soal bersifat terbuka (uraian), bukan pilihan ganda.
3. Field "kunci_jawaban" berisi jawaban ideal yang lengkap dan terstruktur.
4. Field "skor_maksimal" diisi angka 20 untuk semua soal.
5. DILARANG membuat soal di luar bab "{bab}".
6. Output: JSON Array murni dalam tag <QUIZ_URAIAN> ... </QUIZ_URAIAN>."""

    prompt_usr = f"""Mata Pelajaran: {matpel}
Bab: {bab}
Jumlah Soal: {jumlah}
Tingkat Kesulitan: {level_soal}

Referensi:
{context}

Format (DALAM TAG <QUIZ_URAIAN>):
[
  {{
    "nomor": 1,
    "pertanyaan": "Pertanyaan uraian tentang {bab}?",
    "kunci_jawaban": "Jawaban panjang ideal untuk evaluasi.",
    "skor_maksimal": 20
  }}
]"""

    result = _chat(system=prompt_sys, user=prompt_usr)
    match  = re.search(r"<QUIZ_URAIAN>(.*?)</QUIZ_URAIAN>", result, re.DOTALL | re.IGNORECASE)
    extracted = match.group(1).strip() if match else result
    return {"quiz_uraian_data": extracted}


# ---- Evaluasi Uraian (LLM per soal + Python overall) ----

def _hitung_tingkat_pemahaman(persentase: float) -> tuple:
    """
    Mapping persentase skor → tingkat_pemahaman + label_level.
    Konsisten dengan _hitung_level_soal (4 level).
    Returns: (tingkat_pemahaman, catatan_level)
    """
    if persentase >= 86:
        return "Paham Mendalam", "Siswa menunjukkan penguasaan konsep yang sangat baik."
    elif persentase >= 71:
        return "Paham", "Siswa memahami konsep dengan baik, ada ruang untuk pendalaman."
    elif persentase >= 41:
        return "Paham Dasar", "Siswa memahami sebagian konsep, perlu latihan lebih pada bagian yang lemah."
    else:
        return "Belum Paham", "Siswa memerlukan pengulangan materi secara menyeluruh."


def evaluasi_uraian_node(state: AgentState) -> dict:
    """Evaluasi Uraian — LLM per soal + Python-side overall assessment"""
    params        = state["request_params"]
    topik         = params.get("topik", params.get("matpel", "Tidak diketahui"))
    soal_uraian   = params.get("soal_uraian", [])
    jawaban_siswa = params.get("jawaban_siswa", [])

    lookup = {s["soal_id"]: s for s in soal_uraian if isinstance(s, dict) and "soal_id" in s}

    detail = []
    for jawaban in jawaban_siswa:
        if not isinstance(jawaban, dict): continue
        sid        = jawaban.get("soal_id", "")
        ans_siswa  = jawaban.get("jawaban", "").strip()
        soal       = lookup.get(sid, {})
        pertanyaan = soal.get("pertanyaan", "")
        kunci      = soal.get("kunci_jawaban", "")
        skor_maks  = soal.get("skor_maksimal", 20)

        if not pertanyaan or not kunci:
            detail.append({"soal_id": sid, "nomor": soal.get("nomor", "-"),
                           "skor": 0, "skor_maksimal": skor_maks, "feedback": "Data soal tidak lengkap."})
            continue

        llm_result = _chat(
            system=f"""Kamu adalah penilai jawaban esai siswa yang objektif dan konstruktif.
TUGAS: Nilai jawaban siswa berdasarkan kunci jawaban. Skala: 0-{skor_maks}.
OUTPUT: JSON murni dalam tag <NILAI> ... </NILAI>.""",
            user=f"""Pertanyaan: {pertanyaan}
Kunci Jawaban: {kunci}
Jawaban Siswa: {ans_siswa if ans_siswa else "(tidak menjawab)"}
Skor Maksimal: {skor_maks}

Hasilkan JSON dalam tag <NILAI>:
{{"skor": <0-{skor_maks}>, "feedback": "Umpan balik spesifik dan konstruktif."}}"""
        )
        m = re.search(r"<NILAI>(.*?)</NILAI>", llm_result, re.DOTALL | re.IGNORECASE)
        nilai_obj = clean_json_from_llm(m.group(1).strip() if m else llm_result)
        skor = max(0, min(int(nilai_obj.get("skor", 0)) if isinstance(nilai_obj, dict) else 0, skor_maks))
        detail.append({
            "soal_id":       sid,
            "nomor":         soal.get("nomor", "-"),
            "skor":          skor,
            "skor_maksimal": skor_maks,
            "feedback":      nilai_obj.get("feedback", "Tidak ada feedback.") if isinstance(nilai_obj, dict) else "Error parsing."
        })

    # ── Overall assessment — Python-side deterministik (no LLM, consistent + fast) ──
    skor_total       = sum(d.get("skor", 0) for d in detail)
    skor_maks_total  = sum(d.get("skor_maksimal", 20) for d in detail) or 1
    persentase       = round((skor_total / skor_maks_total) * 100, 1)
    tingkat, catatan = _hitung_tingkat_pemahaman(persentase)

    # Identifikasi soal terlemah dan terkuat
    soal_terlemah = min(detail, key=lambda d: d["skor"] / max(d["skor_maksimal"], 1), default=None)
    soal_terkuat  = max(detail, key=lambda d: d["skor"] / max(d["skor_maksimal"], 1), default=None)

    overall = {
        "skor_total":        skor_total,
        "skor_maksimal":     skor_maks_total,
        "persentase":        persentase,
        "tingkat_pemahaman": tingkat,
        "catatan":           catatan,
        "nomor_terlemah":    soal_terlemah["nomor"] if soal_terlemah else None,
        "nomor_terkuat":     soal_terkuat["nomor"]  if soal_terkuat  else None,
    }

    return {"evaluasi_uraian_result": json.dumps(
        {"detail": detail, "overall": overall, "topik": topik},
        ensure_ascii=False
    )}


# ---- Mekanik 5: Konten Belajar (RAG + LLM Long) ----

def konten_belajar_node(state: AgentState) -> dict:
    """Generate konten pembelajaran panjang terstruktur via RAG + LLM."""
    topik = state["request_params"].get("topik", "Sains")
    bab   = state["request_params"].get("bab", topik)
    level = state["request_params"].get("level", "SMA")

    docs = kb_sekolah.search(f"{topik} {bab}", k=4)
    context = "\n---\n".join([d.page_content.strip() for d in docs])
    sumber_list = list({
        f"{d.metadata.get('subject', 'Umum').title()} — {d.metadata.get('topic', '?').replace('_', ' ').title()}"
        for d in docs
    })

    prompt_sys = f"""Kamu adalah guru berpengalaman yang menulis materi pelajaran menarik dan mudah dipahami.
TUGAS: Buat materi pembelajaran lengkap tentang "{bab}" untuk siswa tingkat {level}.
ATURAN WAJIB:
1. Materi HARUS berdasarkan referensi yang diberikan. Jangan mengarang di luar referensi.
2. Gunakan bahasa yang ramah, mudah dipahami, dan mengalir natural.
3. Buat TEPAT 5 sub-bab yang berurutan dan saling terhubung.
4. Setiap sub-bab minimal 3-4 kalimat penjelasan yang substantif — JANGAN terpotong.
5. Sertakan contoh nyata dan analogi yang relatable untuk siswa {level}.
6. Output: JSON murni dalam tag <KONTEN> ... </KONTEN>."""

    prompt_usr = f"""Topik: {topik}
Bab: {bab}
Level Siswa: {level}

Referensi:
{context}

Format yang diminta (DALAM TAG <KONTEN>):
{{
  "judul_konten": "Judul menarik untuk bab ini",
  "konten": [
    {{"sub_bab": "1. Pengantar", "isi": "Pembuka yang menarik perhatian siswa..."}},
    {{"sub_bab": "2. Konsep Utama", "isi": "Penjelasan mendalam konsep inti..."}},
    {{"sub_bab": "3. Detail dan Pendalaman", "isi": "Penjelasan lanjutan dan nuansa..."}},
    {{"sub_bab": "4. Contoh di Kehidupan Nyata", "isi": "Contoh dan analogi yang relatable..."}},
    {{"sub_bab": "5. Poin Kunci & Rangkuman", "isi": "Ringkasan poin-poin penting..."}}
  ]
}}"""

    result  = _chat_long(system=prompt_sys, user=prompt_usr)
    match   = re.search(r"<KONTEN>(.*?)</KONTEN>", result, re.DOTALL | re.IGNORECASE)
    extracted = match.group(1).strip() if match else result
    return {"konten_belajar_data": extracted}


# ---- Mekanik 6: RAG Query (Pure RAG, NO LLM) ----

def rag_query_node(state: AgentState) -> dict:
    """Pure RAG retriever — tanpa LLM, untuk T&A real-time chatbot Tim 5."""
    params = state["request_params"]
    query  = params.get("query", "")
    topik  = params.get("topik", "")
    k      = int(params.get("k", 6))

    search_query = f"{topik} {query}".strip() if topik else query
    docs = kb_sekolah.search(search_query, k=6)

    import os
    chunks_list = []
    for d in docs:
        isi = d.page_content.strip()
        sumber = "Dokumen Sekolah"
        if d.metadata and "source" in d.metadata:
            import os
            sumber = os.path.basename(d.metadata["source"]).title()
            
        raw_asli = d.metadata.get("raw_sebelum_regex", "N/A") if d.metadata else "N/A"
        kena_regex = d.metadata.get("apakah_kena_regex_di_router", False) if d.metadata else False
        
        chunks_list.append({
            "isi": isi,
            "raw_sebelum_regex": raw_asli,
            "kena_regex_router": kena_regex,
            "sumber": sumber
        })
    
    result_dict = {
        "query": query,
        "chunks": chunks_list
    }
    return {"rag_query_result": json.dumps(result_dict, ensure_ascii=False)}


# ================================================================
# 4. STRUCTURER
# ================================================================

def structurer_node(state: AgentState) -> dict:
    """Format output RAW JSON dari node spesialis menjadi standar respon API yang seragam."""
    task = state.get("task", "")
    req_params = state.get("request_params", {})
    
    matpel = req_params.get("matpel", "Umum")
    bab    = req_params.get("bab", "")

    # ── Rekomendasi ──
    if task == "rekomendasi":
        raw_data    = state.get("top_recommendations", "")
        parsed_json = clean_json_from_llm(raw_data)
        student_id  = req_params.get("student_id", "siswa_tester")
        first_time  = req_params.get("first_time", True)
        emosi       = req_params.get("emotion", {}).get("emosi", "netral") if isinstance(req_params.get("emotion"), dict) else "netral"

        if isinstance(parsed_json, dict):
            pesan       = parsed_json.get("pesan_empatik", f"Siswa merasa {emosi}")
            rekomendasi = parsed_json.get("rekomendasi", [])
        else:
            pesan       = f"Siswa merasa {emosi}"
            rekomendasi = [{"error": "Data rusak dari spesialis"}]

        final_payload = util_format_recommender(student_id, first_time, pesan, rekomendasi)

    # ── Flashcard ──
    elif task == "flashcard":
        raw_data    = state.get("flashcards_data", "[]")
        parsed_json = clean_json_from_llm(raw_data)
        if not isinstance(parsed_json, list):
            if isinstance(parsed_json, dict):
                for k, v in parsed_json.items():
                    if isinstance(v, list): parsed_json = v; break
            if not isinstance(parsed_json, list): parsed_json = [{"error": "Data rusak"}]
            
        import os
        docs = kb_sekolah.search(f"{matpel} {bab}", k=1)
        sumber_text = "Materi Sekolah"
        if docs and docs[0].metadata and "source" in docs[0].metadata:
            sumber_text = os.path.basename(docs[0].metadata["source"]).replace(".md", "").replace("_", " ").title()

        # Inject kutipan_sumber by python logic
        for card in parsed_json:
            if isinstance(card, dict):
                card["kutipan_sumber"] = f"Sumber: {sumber_text}"

        final_payload = util_format_flashcard(matpel, bab, parsed_json)

    # ── Mindmap ──
    elif task == "mindmap":
        raw_data    = state.get("mindmap_data", "{}")
        parsed_json = clean_json_from_llm(raw_data)
        if not isinstance(parsed_json, dict) or "error" in parsed_json:
            parsed_json = {"error": "Format mindmap rusak"}
        nodes_arr = [parsed_json] if isinstance(parsed_json, dict) else parsed_json
        final_payload = util_format_mindmap(matpel, bab, nodes_arr)

    # ── Quiz PG generate ──
    elif task == "quiz":
        raw_data    = state.get("quiz_data", "[]")
        parsed_json = clean_json_from_llm(raw_data)
        if not isinstance(parsed_json, list):
            if isinstance(parsed_json, dict):
                for k, v in parsed_json.items():
                    if isinstance(v, list): parsed_json = v; break
            if not isinstance(parsed_json, list): parsed_json = [{"error": "Data rusak"}]
            
        import os
        docs = kb_sekolah.search(f"{matpel} {bab}", k=1)
        sumber_text = "Materi Sekolah"
        if docs and docs[0].metadata and "source" in docs[0].metadata:
            sumber_text = os.path.basename(docs[0].metadata["source"]).replace(".md", "").replace("_", " ").title()

        for s in parsed_json:
            if isinstance(s, dict):
                s["sumber"] = f"Sumber: {sumber_text}"
                
        final_payload = util_format_quiz(matpel, bab, parsed_json)

    # ── Quiz Uraian generate ──
    elif task == "quiz_uraian":
        raw_data    = state.get("quiz_uraian_data", "[]")
        parsed_json = clean_json_from_llm(raw_data)
        if not isinstance(parsed_json, list):
            if isinstance(parsed_json, dict):
                for k, v in parsed_json.items():
                    if isinstance(v, list): parsed_json = v; break
            if not isinstance(parsed_json, list): parsed_json = [{"error": "Data rusak"}]
            
        import os
        docs = kb_sekolah.search(f"{matpel} {bab}", k=1)
        sumber_text = "Materi Sekolah"
        if docs and docs[0].metadata and "source" in docs[0].metadata:
            sumber_text = os.path.basename(docs[0].metadata["source"]).replace(".md", "").replace("_", " ").title()

        for s in parsed_json:
            if isinstance(s, dict):
                s["sumber"] = f"Sumber: {sumber_text}"
                
        final_payload = util_format_quiz_uraian(matpel, bab, parsed_json)

    # ── Evaluasi Quiz PG ──
    elif task == "evaluasi_quiz":
        raw_data = state.get("evaluasi_quiz_result", "[]")
        detail   = json.loads(raw_data) if isinstance(raw_data, str) and raw_data else []
        if not isinstance(detail, list): detail = [{"error": "Data evaluasi rusak"}]
        final_payload = util_format_evaluasi_quiz(matpel, bab, detail)

    # ── Evaluasi Uraian ──
    elif task == "evaluasi_uraian":
        raw_data = state.get("evaluasi_uraian_result", "{}")
        parsed   = json.loads(raw_data) if isinstance(raw_data, str) and raw_data else {}
        if not isinstance(parsed, dict): parsed = {}
        final_payload = util_format_evaluasi_uraian(
            matpel,
            bab,
            parsed.get("detail", []),
            parsed.get("overall", {})
        )

    # ── Konten Belajar ──
    elif task == "konten_belajar":
        raw_data    = state.get("konten_belajar_data", "{}")
        parsed_json = clean_json_from_llm(raw_data)
        if not isinstance(parsed_json, dict) or "error" in parsed_json:
            parsed_json = {"judul_konten": bab, "sumber": [], "konten": [{"sub_bab": "Error", "isi": "Gagal generate konten."}]}
            
        import os
        docs = kb_sekolah.search(f"{matpel} {bab}", k=4)
        sumber_list = list(set([
            os.path.basename(d.metadata["source"]).replace(".md", "").replace("_", " ").title()
            for d in docs if getattr(d, 'metadata', None) and "source" in d.metadata
        ]))
        if not sumber_list:
            sumber_list = ["Materi Sekolah"]

        final_payload = util_format_konten_belajar(
            matpel, bab,
            parsed_json.get("judul_konten", bab),
            parsed_json.get("konten", []),
            sumber_list
        )

    # ── RAG Query ──
    elif task == "rag_query":
        raw_data = state.get("rag_query_result", "{}")
        parsed   = json.loads(raw_data) if isinstance(raw_data, str) and raw_data else {}
        if not isinstance(parsed, dict): parsed = {}
        final_payload = util_format_rag_query(
            parsed.get("query", req_params.get("query", "")),
            matpel,
            bab,
            parsed.get("chunks", [])
        )

    else:
        final_payload = {"error": f"Task '{task}' tidak dikenal"}

    return {"final_payload": final_payload}


# ================================================================
# 5. GRAPH COMPILATION
# ================================================================

workflow = StateGraph(AgentState)

workflow.add_node("recommender",    recommender_node)
workflow.add_node("konten_belajar", konten_belajar_node)
workflow.add_node("rag_query",      rag_query_node)
workflow.add_node("flashcard",      flashcard_node)
workflow.add_node("mindmap",        mindmap_node)
workflow.add_node("quiz",           quiz_node)
workflow.add_node("quiz_uraian",    quiz_uraian_node)
workflow.add_node("evaluasi_quiz",  evaluasi_quiz_node)
workflow.add_node("evaluasi_uraian",evaluasi_uraian_node)
workflow.add_node("structurer",     structurer_node)

workflow.add_conditional_edges(
    START,
    router_task,
    {
        "to_recommender":    "recommender",
        "to_konten_belajar": "konten_belajar",
        "to_rag_query":      "rag_query",
        "to_flashcard":      "flashcard",
        "to_mindmap":        "mindmap",
        "to_quiz":           "quiz",
        "to_quiz_uraian":    "quiz_uraian",
        "to_evaluasi_quiz":  "evaluasi_quiz",
        "to_evaluasi_uraian":"evaluasi_uraian",
        "to_structurer":     "structurer",
    }
)

for node in ["recommender", "konten_belajar", "rag_query", "flashcard", "mindmap",
             "quiz", "quiz_uraian", "evaluasi_quiz", "evaluasi_uraian"]:
    workflow.add_edge(node, "structurer")

workflow.add_edge("structurer", END)

router_agent_app = workflow.compile()

png = router_agent_app.get_graph().draw_mermaid_png()


with open("router_agent.png", "wb") as f:
    f.write(png)
