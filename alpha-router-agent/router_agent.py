import os, re, json
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from router_state import AgentState
from router_tools import (
    kb_sekolah, clean_json_from_llm, generate_soal_id,
    LEVEL_LABELS, LEVEL_INSTRUKSI,
    _ambil_prioritas_belajar, util_format_recommender,
    util_format_bacaan_multi, util_format_flashcard_multi,
    util_format_quiz_multi, util_format_quiz_uraian_multi,
    util_format_mindmap, util_format_evaluasi_quiz,
    util_format_evaluasi_uraian, util_format_rag_query,
    _get_sumber_from_docs,
)

# ================================================================
# 1. LLM SETUP
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
        self.client = InferenceClient(model=self.model_id, token=HF_TOKEN)

    @property
    def _llm_type(self) -> str:
        return "hf-chat"

    def _generate(self, messages, stop=None, **kwargs) -> ChatResult:
        hf_msgs = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                hf_msgs.append({"role": "system",    "content": msg.content})
            elif isinstance(msg, HumanMessage):
                hf_msgs.append({"role": "user",      "content": msg.content})
            elif isinstance(msg, AIMessage):
                hf_msgs.append({"role": "assistant", "content": msg.content})
            else:
                hf_msgs.append({"role": "user",      "content": str(msg.content)})
        response = self.client.chat_completion(
            messages=hf_msgs, max_tokens=self.max_tokens,
            temperature=self.temperature, stop=stop or [],
        )
        content = response.choices[0].message.content
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

llm      = HFChatModel()
llm_long = HFChatModel(max_tokens=3000)

def _chat(system: str, user: str) -> str:
    return llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content.strip()

def _chat_long(system: str, user: str) -> str:
    return llm_long.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content.strip()

# ================================================================
# 2. ROUTER — Rule-based
# ================================================================
def router_task(state: AgentState) -> str:
    task = state.get("task", "").lower()
    if "rekomendasi"     in task: return "to_recommender"
    if "bacaan"          in task: return "to_bacaan"
    if "rag_query"       in task: return "to_rag_query"
    if "flashcard"       in task: return "to_flashcard"
    if "mindmap"         in task: return "to_mindmap"
    if "evaluasi_uraian" in task: return "to_evaluasi_uraian"
    if "evaluasi_quiz"   in task: return "to_evaluasi_quiz"
    if "quiz_uraian"     in task: return "to_quiz_uraian"
    if "quiz"            in task: return "to_quiz"
    return "to_structurer"

# ================================================================
# 3. HELPER — Teacher Context
# ================================================================
def _get_teacher_context(params: dict) -> tuple:
    jenjang = params.get("jenjang", "10")
    kelas   = params.get("kelas", "")
    matpel  = params.get("mata_pelajaran", "Umum")
    elemen  = params.get("elemen", "")
    materi  = params.get("materi", "")
    atp_raw = params.get("atp", [])
    atp_str = "\n".join(f"{i+1}. {a}" for i, a in enumerate(atp_raw)) if atp_raw else "-"
    return jenjang, kelas, matpel, elemen, materi, atp_str

def _extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()

# ================================================================
# 4. NODE — Rekomendasi 
# ================================================================
def recommender_node(state: AgentState) -> dict:
    params     = state["request_params"]
    student_id = params.get("student_id", "unknown")
    first_time = params.get("first_time", True)

    if first_time:
        hasil_pretest  = params.get("hasil_pretest", [])
        matpel_dipilih = params.get("matpel_dipilih", [])
        ctx = json.dumps(hasil_pretest, ensure_ascii=False, indent=2)
        sys_p = ("Kamu adalah sistem rekomendasi belajar cerdas.\n"
                 "TUGAS: Analisis hasil pretest siswa, rekomendasikan 3 bab paling perlu dipelajari.\n"
                 "ATURAN: Output JSON murni dalam tag <REKOMENDASI>...</REKOMENDASI>.")
        usr_p = (f"Mata pelajaran dipilih: {', '.join(matpel_dipilih)}\n"
                 f"Hasil Pretest:\n{ctx}\n\n"
                 "Hasilkan JSON dalam tag <REKOMENDASI>:\n"
                 '{"pesan_empatik":"...","rekomendasi":[{"urutan":1,"matpel":"...","bab":"...","alasan":"...","saran_aksi":"..."}]}')
    else:
        riwayat        = _ambil_prioritas_belajar(params.get("riwayat_progress", []), top_n=5)
        matpel_dipilih = params.get("matpel_dipilih", [])
        ctx = json.dumps(riwayat, ensure_ascii=False, indent=2)
        sys_p = ("Kamu adalah sistem rekomendasi belajar adaptif.\n"
                 "TUGAS: Analisis 5 bab terlemah siswa, rekomendasikan 3 bab untuk diperkuat.\n"
                 "ATURAN: Output JSON murni dalam tag <REKOMENDASI>...</REKOMENDASI>.")
        usr_p = (f"Mata pelajaran: {', '.join(matpel_dipilih)}\n"
                 f"5 Bab Terlemah:\n{ctx}\n\n"
                 "Hasilkan JSON dalam tag <REKOMENDASI>:\n"
                 '{"pesan_empatik":"...","rekomendasi":[{"urutan":1,"matpel":"...","bab":"...","alasan":"...","saran_aksi":"..."}]}')

    result    = _chat(system=sys_p, user=usr_p)
    extracted = _extract_tag(result, "REKOMENDASI")
    return {"top_recommendations": extracted}

# ================================================================
# 5. NODE — Bacaan (3 level: LOTS / MOTS / HOTS)
# ================================================================
def bacaan_node(state: AgentState) -> dict:
    jenjang, kelas, matpel, elemen, materi, atp_str = _get_teacher_context(state["request_params"])
    query = f"{matpel} {materi}".strip()
    docs  = kb_sekolah.search(query, k=4)
    ctx   = "\n---\n".join(d.page_content.strip() for d in docs)
    sumber = [_get_sumber_from_docs(docs)] if docs else ["Materi Sekolah"]

    hasil = {}
    for level in ("LOTS", "MOTS", "HOTS"):
        tag = f"BACAAN_{level}"
        sys_p = (f"Kamu adalah guru berpengalaman yang menulis materi pelajaran.\n"
                 f"TUGAS: Buat materi bacaan tentang \"{materi}\" ({matpel}, Kelas {jenjang}).\n"
                 f"{LEVEL_INSTRUKSI[level]}\n"
                 f"ATURAN:\n"
                 f"1. Berdasarkan referensi yang diberikan. Jangan mengarang.\n"
                 f"2. Buat TEPAT 5 sub-bab berurutan dan saling terhubung.\n"
                 f"3. Setiap sub-bab minimal 3-4 kalimat substantif.\n"
                 f"4. Sesuaikan gaya bahasa dan kedalaman dengan level {level}.\n"
                 f"5. Output: JSON murni dalam tag <{tag}>...</{tag}>.")
        usr_p = (f"Mata Pelajaran: {matpel} | Elemen: {elemen} | Kelas: {jenjang} ({kelas})\n"
                 f"Materi: {materi}\n"
                 f"Alur Tujuan Pembelajaran:\n{atp_str}\n\n"
                 f"Referensi:\n{ctx}\n\n"
                 f"Format (dalam tag <{tag}>):\n"
                 '{"judul_konten":"Judul menarik","konten":['
                 '{"sub_bab":"1. Pengantar","isi":"..."},'
                 '{"sub_bab":"2. Konsep Utama","isi":"..."},'
                 '{"sub_bab":"3. Pendalaman","isi":"..."},'
                 '{"sub_bab":"4. Contoh Nyata","isi":"..."},'
                 '{"sub_bab":"5. Rangkuman","isi":"..."}]}')
        raw       = _chat_long(system=sys_p, user=usr_p)
        extracted = _extract_tag(raw, tag)
        parsed    = clean_json_from_llm(extracted)
        if isinstance(parsed, dict):
            parsed["sumber"] = sumber
        hasil[level] = parsed if isinstance(parsed, dict) else {"judul_konten": materi, "konten": [], "sumber": sumber}

    return {
        "bacaan_lots_data": json.dumps(hasil["LOTS"], ensure_ascii=False),
        "bacaan_mots_data": json.dumps(hasil["MOTS"], ensure_ascii=False),
        "bacaan_hots_data": json.dumps(hasil["HOTS"], ensure_ascii=False),
    }

# ================================================================
# 6. NODE — Flashcard (3 level: LOTS / MOTS / HOTS)
# ================================================================
def flashcard_node(state: AgentState) -> dict:
    jenjang, kelas, matpel, elemen, materi, atp_str = _get_teacher_context(state["request_params"])
    query = f"{matpel} {materi}".strip()
    docs  = kb_sekolah.search(query, k=8)
    ctx   = "\n---\n".join(d.page_content.strip() for d in docs)
    sumber_text = _get_sumber_from_docs(docs)

    hasil = {}
    for level in ("LOTS", "MOTS", "HOTS"):
        tag   = f"FLASHCARD_{level}"
        sys_p = (f"Kamu adalah spesialis pembuat Flashcard bergaya NotebookLM.\n"
                 f"TUGAS: Buat TEPAT 10 pasang pertanyaan (Front) dan jawaban (Back) tentang \"{materi}\" ({matpel}).\n"
                 f"{LEVEL_INSTRUKSI[level]}\n"
                 f"ATURAN:\n"
                 f"1. Field 'front': pertanyaan ringkas sesuai level {level}.\n"
                 f"2. Field 'back': SANGAT SINGKAT, maksimal 2 kalimat pendek.\n"
                 f"3. Output: JSON Array murni dalam tag <{tag}>...</{tag}>.")
        usr_p = (f"Mata Pelajaran: {matpel} | Elemen: {elemen} | Kelas: {jenjang} ({kelas})\n"
                 f"Materi: {materi}\n"
                 f"Alur Tujuan Pembelajaran:\n{atp_str}\n\n"
                 f"Referensi:\n{ctx}\n\n"
                 f"Format (dalam tag <{tag}>):\n"
                 f'[{{"front":"Pertanyaan {level}?","back":"Jawaban singkat (maks 2 kalimat).","kutipan_sumber":"Sumber: {sumber_text}"}}]')
        raw       = _chat(system=sys_p, user=usr_p)
        extracted = _extract_tag(raw, tag)
        parsed    = clean_json_from_llm(extracted)
        cards     = parsed if isinstance(parsed, list) else []
        for card in cards:
            if isinstance(card, dict):
                card["level"] = level
                if "kutipan_sumber" not in card:
                    card["kutipan_sumber"] = f"Sumber: {sumber_text}"
        hasil[level] = cards

    return {
        "flashcard_lots_data": json.dumps(hasil["LOTS"], ensure_ascii=False),
        "flashcard_mots_data": json.dumps(hasil["MOTS"], ensure_ascii=False),
        "flashcard_hots_data": json.dumps(hasil["HOTS"], ensure_ascii=False),
    }

# ================================================================
# 7. NODE — Mindmap (1 versi lengkap, tanpa level)
# ================================================================
def mindmap_node(state: AgentState) -> dict:
    jenjang, kelas, matpel, elemen, materi, atp_str = _get_teacher_context(state["request_params"])
    query = f"{matpel} {materi}".strip()
    docs  = kb_sekolah.search(query, k=10)
    ctx   = "\n---\n".join(d.page_content.strip() for d in docs)

    sys_p = (f"Kamu adalah ahli pembuat Peta Konsep (Mindmap) yang komprehensif.\n"
             f"TUGAS: Buat struktur hierarki LENGKAP dan DETAIL untuk semua konsep dalam \"{materi}\" ({matpel}).\n"
             f"STRUKTUR: Tampilkan SEMUA relasi penting antar konsep — jangan disederhanakan atau dipotong.\n"
             f"ATURAN: JSON murni dalam tag <MINDMAP>...</MINDMAP>.")
    usr_p = (f"Mata Pelajaran: {matpel} | Kelas: {jenjang} ({kelas})\n"
             f"Materi: {materi}\n"
             f"Alur Tujuan Pembelajaran:\n{atp_str}\n\n"
             f"Referensi:\n{ctx}\n\n"
             'Format (dalam tag <MINDMAP>):\n'
             '{"konsep_utama":"nama materi","deskripsi":"1 kalimat","children":['
             '{"sub_konsep":"...","penjelasan":"...","children":[{"sub_konsep":"...","penjelasan":"...","children":[]}]}]}')

    raw       = _chat(system=sys_p, user=usr_p)
    extracted = _extract_tag(raw, "MINDMAP")
    return {"mindmap_data": extracted}

# ================================================================
# 8. NODE — Quiz PG (3 level: LOTS / MOTS / HOTS)
# ================================================================
def quiz_node(state: AgentState) -> dict:
    jenjang, kelas, matpel, elemen, materi, atp_str = _get_teacher_context(state["request_params"])
    query = f"{matpel} {materi}".strip()
    docs  = kb_sekolah.search(query, k=5)
    ctx   = "\n---\n".join(d.page_content.strip() for d in docs)
    sumber_text = _get_sumber_from_docs(docs)

    hasil = {}
    for level in ("LOTS", "MOTS", "HOTS"):
        tag   = f"QUIZ_{level}"
        sys_p = (f"Kamu adalah pembuat soal ujian profesional.\n"
                 f"TUGAS: Buat TEPAT 5 soal pilihan ganda tentang \"{materi}\" ({matpel}, Kelas {jenjang}).\n"
                 f"{LEVEL_INSTRUKSI[level]}\n"
                 f"ATURAN WAJIB:\n"
                 f"1. Soal HARUS spesifik tentang \"{materi}\".\n"
                 f"2. Setiap soal: 4 pilihan (A/B/C/D), field 'jawaban_benar' satu huruf kapital.\n"
                 f"3. Field 'pembahasan': alasan jawaban benar.\n"
                 f"4. Output: JSON Array murni dalam tag <{tag}>...</{tag}>.")
        usr_p = (f"Mata Pelajaran: {matpel} | Elemen: {elemen} | Kelas: {jenjang} ({kelas})\n"
                 f"Materi: {materi}\n"
                 f"Alur Tujuan Pembelajaran:\n{atp_str}\n\n"
                 f"Referensi:\n{ctx}\n\n"
                 f"Format (dalam tag <{tag}>):\n"
                 '[{"nomor":1,"level":"' + level + '","pertanyaan":"...","pilihan":{"A":"...","B":"...","C":"...","D":"..."},'
                 '"jawaban_benar":"A","pembahasan":"...","sumber":"Sumber: ' + sumber_text + '"}]')
        raw       = _chat(system=sys_p, user=usr_p)
        extracted = _extract_tag(raw, tag)
        parsed    = clean_json_from_llm(extracted)
        soal_list = parsed if isinstance(parsed, list) else []
        for s in soal_list:
            if isinstance(s, dict):
                s["level"] = level
                if "soal_id" not in s:
                    s["soal_id"] = generate_soal_id("pg", level, materi)
        hasil[level] = soal_list

    return {
        "quiz_lots_data": json.dumps(hasil["LOTS"], ensure_ascii=False),
        "quiz_mots_data": json.dumps(hasil["MOTS"], ensure_ascii=False),
        "quiz_hots_data": json.dumps(hasil["HOTS"], ensure_ascii=False),
    }


# ================================================================
# 9. NODE — Quiz Uraian (3 level: LOTS / MOTS / HOTS)
# ================================================================
def quiz_uraian_node(state: AgentState) -> dict:
    jenjang, kelas, matpel, elemen, materi, atp_str = _get_teacher_context(state["request_params"])
    query = f"{matpel} {materi}".strip()
    docs  = kb_sekolah.search(query, k=5)
    ctx   = "\n---\n".join(d.page_content.strip() for d in docs)
    sumber_text = _get_sumber_from_docs(docs)

    hasil = {}
    for level in ("LOTS", "MOTS", "HOTS"):
        tag   = f"URAIAN_{level}"
        sys_p = (f"Kamu adalah pembuat soal esai profesional.\n"
                 f"TUGAS: Buat TEPAT 5 soal uraian/esai tentang \"{materi}\" ({matpel}, Kelas {jenjang}).\n"
                 f"{LEVEL_INSTRUKSI[level]}\n"
                 f"ATURAN WAJIB:\n"
                 f"1. Soal HARUS spesifik tentang \"{materi}\".\n"
                 f"2. Setiap soal terbuka (uraian), field 'kunci_jawaban' berisi jawaban ideal.\n"
                 f"3. Field 'skor_maksimal' = 20 untuk semua soal.\n"
                 f"4. Output: JSON Array murni dalam tag <{tag}>...</{tag}>.")
        usr_p = (f"Mata Pelajaran: {matpel} | Elemen: {elemen} | Kelas: {jenjang} ({kelas})\n"
                 f"Materi: {materi}\n"
                 f"Alur Tujuan Pembelajaran:\n{atp_str}\n\n"
                 f"Referensi:\n{ctx}\n\n"
                 f"Format (dalam tag <{tag}>):\n"
                 '[{"nomor":1,"level":"' + level + '","pertanyaan":"...","kunci_jawaban":"...","skor_maksimal":20,'
                 '"sumber":"Sumber: ' + sumber_text + '"}]')
        raw       = _chat(system=sys_p, user=usr_p)
        extracted = _extract_tag(raw, tag)
        parsed    = clean_json_from_llm(extracted)
        soal_list = parsed if isinstance(parsed, list) else []
        for s in soal_list:
            if isinstance(s, dict):
                s["level"] = level
                if "soal_id" not in s:
                    s["soal_id"] = generate_soal_id("uraian", level, materi)
        hasil[level] = soal_list

    return {
        "quiz_uraian_lots_data": json.dumps(hasil["LOTS"], ensure_ascii=False),
        "quiz_uraian_mots_data": json.dumps(hasil["MOTS"], ensure_ascii=False),
        "quiz_uraian_hots_data": json.dumps(hasil["HOTS"], ensure_ascii=False),
    }


# ================================================================
# 10. NODE — Evaluasi Quiz PG (deterministik, NO LLM)
# ================================================================
def evaluasi_quiz_node(state: AgentState) -> dict:
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
            "level":         soal.get("level", ""),
            "benar":         benar,
            "jawaban_siswa": ans,
            "jawaban_benar": benar_ans,
            "skor":          skor_per_soal if benar else 0,
            "skor_maksimal": skor_per_soal,
            "pembahasan":    soal.get("pembahasan", ""),
        })
    return {"evaluasi_quiz_result": json.dumps(detail, ensure_ascii=False)}


# ================================================================
# 11. NODE — Evaluasi Uraian (LLM per soal)
# ================================================================
def _hitung_tingkat_pemahaman(persentase: float) -> tuple:
    if persentase >= 86: return "Paham Mendalam", "Penguasaan konsep sangat baik."
    if persentase >= 71: return "Paham",          "Memahami konsep dengan baik."
    if persentase >= 41: return "Paham Dasar",    "Perlu latihan lebih pada bagian yang lemah."
    return "Belum Paham", "Perlu pengulangan materi secara menyeluruh."


def evaluasi_uraian_node(state: AgentState) -> dict:
    params        = state["request_params"]
    topik         = params.get("topik", params.get("mata_pelajaran", "Tidak diketahui"))
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
                           "level": soal.get("level", ""), "skor": 0,
                           "skor_maksimal": skor_maks, "feedback": "Data soal tidak lengkap."})
            continue

        llm_result = _chat(
            system=(f"Kamu adalah penilai jawaban esai siswa yang objektif.\n"
                    f"TUGAS: Nilai jawaban siswa berdasarkan kunci. Skala 0-{skor_maks}.\n"
                    f"OUTPUT: JSON murni dalam tag <NILAI>...</NILAI>."),
            user=(f"Pertanyaan: {pertanyaan}\nKunci: {kunci}\n"
                  f"Jawaban Siswa: {ans_siswa or '(tidak menjawab)'}\nSkor Maksimal: {skor_maks}\n\n"
                  f'Hasilkan JSON dalam tag <NILAI>:\n{{"skor":<0-{skor_maks}>,"feedback":"Umpan balik konstruktif."}}')
        )
        m = re.search(r"<NILAI>(.*?)</NILAI>", llm_result, re.DOTALL | re.IGNORECASE)
        nilai_obj = clean_json_from_llm(m.group(1).strip() if m else llm_result)
        skor = max(0, min(int(nilai_obj.get("skor", 0)) if isinstance(nilai_obj, dict) else 0, skor_maks))
        detail.append({
            "soal_id":       sid,
            "nomor":         soal.get("nomor", "-"),
            "level":         soal.get("level", ""),
            "skor":          skor,
            "skor_maksimal": skor_maks,
            "feedback":      nilai_obj.get("feedback", "Tidak ada feedback.") if isinstance(nilai_obj, dict) else "Error.",
        })

    skor_total      = sum(d.get("skor", 0) for d in detail)
    skor_maks_total = sum(d.get("skor_maksimal", 20) for d in detail) or 1
    persentase      = round((skor_total / skor_maks_total) * 100, 1)
    tingkat, catatan = _hitung_tingkat_pemahaman(persentase)
    soal_terlemah   = min(detail, key=lambda d: d["skor"] / max(d["skor_maksimal"], 1), default=None)
    soal_terkuat    = max(detail, key=lambda d: d["skor"] / max(d["skor_maksimal"], 1), default=None)

    overall = {
        "skor_total": skor_total, "skor_maksimal": skor_maks_total,
        "persentase": persentase, "tingkat_pemahaman": tingkat, "catatan": catatan,
        "nomor_terlemah": soal_terlemah["nomor"] if soal_terlemah else None,
        "nomor_terkuat":  soal_terkuat["nomor"]  if soal_terkuat  else None,
    }
    return {"evaluasi_uraian_result": json.dumps(
        {"detail": detail, "overall": overall, "topik": topik}, ensure_ascii=False
    )}


# ================================================================
# 12. NODE — RAG Query (Pure RAG, NO LLM)
# ================================================================
def rag_query_node(state: AgentState) -> dict:
    params = state["request_params"]
    query  = params.get("query", "")
    matpel = params.get("matpel", "")
    k      = int(params.get("k", 6))

    search_query = f"{matpel} {query}".strip() if matpel else query
    docs = kb_sekolah.search(search_query, k=k)

    chunks_list = []
    for d in docs:
        sumber = "Dokumen Sekolah"
        if d.metadata and "source" in d.metadata:
            sumber = os.path.basename(d.metadata["source"]).title()
        chunks_list.append({
            "isi":              d.page_content.strip(),
            "raw_sebelum_regex": d.metadata.get("raw_sebelum_regex", "N/A"),
            "kena_regex":       d.metadata.get("apakah_kena_regex", False),
            "sumber":           sumber,
        })
    return {"rag_query_result": json.dumps({"query": query, "chunks": chunks_list}, ensure_ascii=False)}


# ================================================================
# 13. STRUCTURER
# ================================================================
def structurer_node(state: AgentState) -> dict:
    task   = state.get("task", "")
    params = state.get("request_params", {})
    jenjang, kelas, matpel, elemen, materi, _ = _get_teacher_context(params)

    def _load(key: str):
        raw = state.get(key, "[]")
        return json.loads(raw) if isinstance(raw, str) and raw else []

    def _load_dict(key: str):
        raw = state.get(key, "{}")
        return json.loads(raw) if isinstance(raw, str) and raw else {}

    if task == "rekomendasi":
        raw_data   = state.get("top_recommendations", "")
        parsed     = clean_json_from_llm(raw_data)
        student_id = params.get("student_id", "unknown")
        first_time = params.get("first_time", True)
        pesan      = parsed.get("pesan_empatik", "Semangat belajar!") if isinstance(parsed, dict) else "Semangat belajar!"
        rekomendasi = parsed.get("rekomendasi", []) if isinstance(parsed, dict) else []
        final_payload = util_format_recommender(student_id, first_time, pesan, rekomendasi)

    elif task == "bacaan":
        lots = _load_dict("bacaan_lots_data")
        mots = _load_dict("bacaan_mots_data")
        hots = _load_dict("bacaan_hots_data")
        final_payload = util_format_bacaan_multi(jenjang, kelas, matpel, elemen, materi, lots, mots, hots)

    elif task == "flashcard":
        lots = _load("flashcard_lots_data")
        mots = _load("flashcard_mots_data")
        hots = _load("flashcard_hots_data")
        final_payload = util_format_flashcard_multi(jenjang, kelas, matpel, elemen, materi, lots, mots, hots)

    elif task == "quiz":
        lots = _load("quiz_lots_data")
        mots = _load("quiz_mots_data")
        hots = _load("quiz_hots_data")
        final_payload = util_format_quiz_multi(jenjang, kelas, matpel, elemen, materi, lots, mots, hots)

    elif task == "quiz_uraian":
        lots = _load("quiz_uraian_lots_data")
        mots = _load("quiz_uraian_mots_data")
        hots = _load("quiz_uraian_hots_data")
        final_payload = util_format_quiz_uraian_multi(jenjang, kelas, matpel, elemen, materi, lots, mots, hots)

    elif task == "mindmap":
        raw_data  = state.get("mindmap_data", "{}")
        parsed    = clean_json_from_llm(raw_data)
        nodes_arr = [parsed] if isinstance(parsed, dict) else parsed if isinstance(parsed, list) else []
        final_payload = util_format_mindmap(matpel, materi, nodes_arr)

    elif task == "evaluasi_quiz":
        detail = _load("evaluasi_quiz_result")
        if not isinstance(detail, list): detail = [{"error": "Data evaluasi rusak"}]
        final_payload = util_format_evaluasi_quiz(matpel, materi, detail)

    elif task == "evaluasi_uraian":
        parsed = _load_dict("evaluasi_uraian_result")
        final_payload = util_format_evaluasi_uraian(
            matpel, materi, parsed.get("detail", []), parsed.get("overall", {})
        )

    elif task == "rag_query":
        parsed = _load_dict("rag_query_result")
        final_payload = util_format_rag_query(
            parsed.get("query", params.get("query", "")),
            matpel, materi, parsed.get("chunks", [])
        )

    else:
        final_payload = {"error": f"Task '{task}' tidak dikenal"}

    return {"final_payload": final_payload}


# ================================================================
# 14. GRAPH COMPILATION
# ================================================================
workflow = StateGraph(AgentState)

workflow.add_node("recommender",     recommender_node)
workflow.add_node("bacaan",          bacaan_node)
workflow.add_node("flashcard",       flashcard_node)
workflow.add_node("mindmap",         mindmap_node)
workflow.add_node("quiz",            quiz_node)
workflow.add_node("quiz_uraian",     quiz_uraian_node)
workflow.add_node("evaluasi_quiz",   evaluasi_quiz_node)
workflow.add_node("evaluasi_uraian", evaluasi_uraian_node)
workflow.add_node("rag_query",       rag_query_node)
workflow.add_node("structurer",      structurer_node)

workflow.add_conditional_edges(
    START, router_task,
    {
        "to_recommender":     "recommender",
        "to_bacaan":          "bacaan",
        "to_flashcard":       "flashcard",
        "to_mindmap":         "mindmap",
        "to_quiz":            "quiz",
        "to_quiz_uraian":     "quiz_uraian",
        "to_evaluasi_quiz":   "evaluasi_quiz",
        "to_evaluasi_uraian": "evaluasi_uraian",
        "to_rag_query":       "rag_query",
        "to_structurer":      "structurer",
    }
)

for node in ["recommender", "bacaan", "flashcard", "mindmap", "quiz",
             "quiz_uraian", "evaluasi_quiz", "evaluasi_uraian", "rag_query"]:
    workflow.add_edge(node, "structurer")

workflow.add_edge("structurer", END)

router_agent_app = workflow.compile()

if __name__ == "__main__":
    # Generate diagram hanya jika dijalankan langsung
    try:
        png = router_agent_app.get_graph().draw_mermaid_png()
        with open("router_agent.png", "wb") as f:
            f.write(png)
        print("✅ Diagram router_agent.png berhasil dibuat.")
    except Exception as e:
        print(f"⚠️ Gagal generate diagram PNG: {e}")
        try:
            mmd = router_agent_app.get_graph().draw_mermaid()
            with open("arsitektur_route.mmd", "w", encoding="utf-8") as f:
                f.write(mmd)
            print("✅ Diagram Mermaid disimpan ke arsitektur_route.mmd")
        except Exception as e2:
            print(f"⚠️ Gagal generate diagram Mermaid: {e2}")
 