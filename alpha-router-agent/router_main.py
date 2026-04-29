from router_agent import router_agent_app
from router_state import AgentState
import json


# ================================================================
# Helper: build empty initial state
# ================================================================
def _initial_state(task: str, params: dict) -> AgentState:
    return {
        "task": task,
        "request_params": params,
        "bacaan_lots_data": "",
        "bacaan_mots_data": "",
        "bacaan_hots_data": "",
        "flashcard_lots_data": "",
        "flashcard_mots_data": "",
        "flashcard_hots_data": "",
        "quiz_lots_data": "",
        "quiz_mots_data": "",
        "quiz_hots_data": "",
        "quiz_uraian_lots_data": "",
        "quiz_uraian_mots_data": "",
        "quiz_uraian_hots_data": "",
        "mindmap_data": "",
        "top_recommendations": "",
        "evaluasi_quiz_result": "",
        "evaluasi_uraian_result": "",
        "rag_query_result": "",
        "final_payload": {},
    }


# ================================================================
# Runner
# ================================================================
def run_simulation(scenario_name: str, task: str, request_params: dict):
    print(f"\n{'='*66}")
    print(f"🚀 SCENARIO: {scenario_name}")
    print(f"   Task: {task}")
    print(f"{'='*66}")

    initial_state = _initial_state(task, request_params)
    final_output  = None

    print("Memulai stream eksekusi Graph...\n")
    for step in router_agent_app.stream(initial_state, stream_mode="updates"):
        for node_name, node_data in step.items():
            print(f"📍 [NODE]: {node_name.upper()}")
            if node_name == "structurer":
                final_output = node_data.get("final_payload", {})
            print("-" * 50)

    print("\n📦 === FINAL OUTPUT JSON ===")
    try:
        if isinstance(final_output, dict):
            print("✅ VALID Dictionary")
            print(json.dumps(final_output, indent=2, ensure_ascii=False)[:2000])  # Truncate preview
        else:
            raise ValueError("Bukan dictionary")
    except Exception as e:
        print(f"❌ ERROR: {e}\nRaw: {final_output}")

    return final_output


# ================================================================
# Contoh teacher input (reusable)
# ================================================================
GURU_PARAMS_BILANGAN = {
    "jenjang": "10",
    "kelas": "10A",
    "mata_pelajaran": "Matematika",
    "elemen": "Bilangan",
    "materi": "Bilangan Berpangkat",
    "atp": [
        "Peserta didik dapat memahami konsep dan sifat bilangan berpangkat",
        "Peserta didik dapat menyelesaikan operasi bilangan berpangkat positif, negatif, dan nol",
        "Peserta didik dapat menerapkan sifat bilangan berpangkat dalam pemecahan masalah",
    ]
}

GURU_PARAMS_NEWTON = {
    "jenjang": "10",
    "kelas": "10B",
    "mata_pelajaran": "Fisika",
    "elemen": "Mekanika",
    "materi": "Hukum Newton",
    "atp": [
        "Peserta didik dapat menjelaskan bunyi Hukum Newton I, II, dan III",
        "Peserta didik dapat menggunakan rumus F = m × a dalam perhitungan",
        "Peserta didik dapat mengidentifikasi pasangan gaya aksi-reaksi dalam kehidupan sehari-hari",
    ]
}

GURU_PARAMS_FOTOSINTESIS = {
    "jenjang": "11",
    "kelas": "11A",
    "mata_pelajaran": "Biologi",
    "elemen": "Sistem Kehidupan",
    "materi": "Fotosintesis",
    "atp": [
        "Peserta didik dapat menjelaskan pengertian dan persamaan reaksi fotosintesis",
        "Peserta didik dapat membedakan reaksi terang dan reaksi gelap (Siklus Calvin)",
        "Peserta didik dapat menganalisis faktor-faktor yang mempengaruhi laju fotosintesis",
    ]
}


def main():
    print("Mempersiapkan koneksi Qdrant dan model embedding...\n")

    # ------------------------------------------------------------------
    # Skenario 1 — Quiz PG (3 level sekaligus)
    # ------------------------------------------------------------------
    quiz_output = run_simulation(
        scenario_name="Generate Quiz PG — Bilangan Berpangkat (LOTS+MOTS+HOTS)",
        task="quiz",
        request_params=GURU_PARAMS_BILANGAN,
    )

    # ------------------------------------------------------------------
    # Skenario 2 — Evaluasi Quiz PG (ambil soal dari skenario 1)
    # ------------------------------------------------------------------
    soal_pg_list   = []
    jawaban_dummy  = []
    if isinstance(quiz_output, dict):
        per_level = quiz_output.get("soal_per_level", {})
        for level_data in per_level.values():
            for s in level_data.get("soal", []):
                soal_pg_list.append({
                    "soal_id":       s.get("soal_id", ""),
                    "nomor":         s.get("nomor", 1),
                    "level":         s.get("level", ""),
                    "jawaban_benar": s.get("jawaban_benar", "A"),
                    "pembahasan":    s.get("pembahasan", ""),
                })
                jawaban_dummy.append({"soal_id": s.get("soal_id", ""), "jawaban": "A"})

    run_simulation(
        scenario_name="Evaluasi Quiz PG — Bilangan Berpangkat (Jawaban Dummy)",
        task="evaluasi_quiz",
        request_params={
            "mata_pelajaran": "Matematika",
            "materi": "Bilangan Berpangkat",
            "skor_per_soal": 10,
            "soal_pg": soal_pg_list,
            "jawaban_siswa": jawaban_dummy,
        },
    )

    # ------------------------------------------------------------------
    # Skenario 3 — Quiz Uraian (3 level sekaligus)
    # ------------------------------------------------------------------
    uraian_output = run_simulation(
        scenario_name="Generate Quiz Uraian — Hukum Newton (LOTS+MOTS+HOTS)",
        task="quiz_uraian",
        request_params=GURU_PARAMS_NEWTON,
    )

    # ------------------------------------------------------------------
    # Skenario 4 — Evaluasi Uraian (ambil soal dari skenario 3)
    # ------------------------------------------------------------------
    soal_uraian_list  = []
    jawaban_uraian    = []
    if isinstance(uraian_output, dict):
        per_level = uraian_output.get("soal_per_level", {})
        for level_data in per_level.values():
            for s in level_data.get("soal", []):
                soal_uraian_list.append({
                    "soal_id":       s.get("soal_id", ""),
                    "nomor":         s.get("nomor", 1),
                    "level":         s.get("level", ""),
                    "pertanyaan":    s.get("pertanyaan", ""),
                    "kunci_jawaban": s.get("kunci_jawaban", ""),
                    "skor_maksimal": s.get("skor_maksimal", 20),
                })
                jawaban_uraian.append({
                    "soal_id": s.get("soal_id", ""),
                    "jawaban": "Hukum Newton I menyatakan benda diam tetap diam jika tidak ada gaya."
                })

    run_simulation(
        scenario_name="Evaluasi Quiz Uraian — Hukum Newton (Jawaban Dummy)",
        task="evaluasi_uraian",
        request_params={
            "mata_pelajaran": "Fisika",
            "materi": "Hukum Newton",
            "soal_uraian": soal_uraian_list,
            "jawaban_siswa": jawaban_uraian,
        },
    )

    # ------------------------------------------------------------------
    # Skenario 5 — Flashcard (3 level)
    # ------------------------------------------------------------------
    run_simulation(
        scenario_name="Generate Flashcard — Fotosintesis (LOTS+MOTS+HOTS)",
        task="flashcard",
        request_params=GURU_PARAMS_FOTOSINTESIS,
    )

    # ------------------------------------------------------------------
    # Skenario 6 — Bacaan (3 level)
    # ------------------------------------------------------------------
    run_simulation(
        scenario_name="Generate Bacaan — Hukum Newton (LOTS+MOTS+HOTS)",
        task="bacaan",
        request_params=GURU_PARAMS_NEWTON,
    )

    # ------------------------------------------------------------------
    # Skenario 7 — Mindmap (1 versi lengkap)
    # ------------------------------------------------------------------
    run_simulation(
        scenario_name="Generate Mindmap — Fotosintesis (1 versi lengkap)",
        task="mindmap",
        request_params=GURU_PARAMS_FOTOSINTESIS,
    )

    # ------------------------------------------------------------------
    # Skenario 8 — RAG Query (pure RAG, no LLM)
    # ------------------------------------------------------------------
    run_simulation(
        scenario_name="RAG Query — Pertanyaan Siswa tentang Inersia",
        task="rag_query",
        request_params={
            "query": "apa yang dimaksud dengan inersia dan bagaimana contohnya?",
            "matpel": "Fisika",
            "k": 3,
        },
    )

    # ------------------------------------------------------------------
    # Skenario 9 — Rekomendasi First Time
    # ------------------------------------------------------------------
    run_simulation(
        scenario_name="Rekomendasi — Siswa Baru (Berdasarkan Pretest)",
        task="rekomendasi",
        request_params={
            "student_id": "siswa-baru-001",
            "first_time": True,
            "matpel_dipilih": ["Matematika", "Fisika", "Biologi"],
            "hasil_pretest": [
                {"matpel": "Matematika", "skor": 45, "topik_lemah": ["Bilangan Berpangkat", "Persamaan Kuadrat"]},
                {"matpel": "Fisika",     "skor": 30, "topik_lemah": ["Hukum Newton", "Gerak Lurus"]},
                {"matpel": "Biologi",    "skor": 62, "topik_lemah": ["Fotosintesis"]},
            ],
        },
    )

    # ------------------------------------------------------------------
    # Skenario 10 — Rekomendasi Returning
    # ------------------------------------------------------------------
    run_simulation(
        scenario_name="Rekomendasi — Siswa Lama (8 riwayat bab)",
        task="rekomendasi",
        request_params={
            "student_id": "siswa-lama-042",
            "first_time": False,
            "matpel_dipilih": ["Matematika", "Fisika"],
            "riwayat_progress": [
                {"matpel": "Matematika", "bab": "Bilangan Berpangkat",  "skor_terakhir": 20, "tingkat_pemahaman": "Belum Paham",   "jumlah_prompt": 22},
                {"matpel": "Fisika",     "bab": "Hukum Newton",         "skor_terakhir": 35, "tingkat_pemahaman": "Belum Paham",   "jumlah_prompt": 18},
                {"matpel": "Matematika", "bab": "Persamaan Kuadrat",    "skor_terakhir": 50, "tingkat_pemahaman": "Paham Dasar",   "jumlah_prompt": 10},
                {"matpel": "Fisika",     "bab": "Gerak Lurus Beraturan","skor_terakhir": 55, "tingkat_pemahaman": "Paham Dasar",   "jumlah_prompt": 7},
                {"matpel": "Matematika", "bab": "Sistem Persamaan",     "skor_terakhir": 65, "tingkat_pemahaman": "Paham Dasar",   "jumlah_prompt": 8},
                {"matpel": "Fisika",     "bab": "Energi Mekanik",       "skor_terakhir": 70, "tingkat_pemahaman": "Paham Mendalam","jumlah_prompt": 5},
                {"matpel": "Matematika", "bab": "Statistika Dasar",     "skor_terakhir": 80, "tingkat_pemahaman": "Paham Mendalam","jumlah_prompt": 3},
                {"matpel": "Fisika",     "bab": "Gelombang",            "skor_terakhir": 85, "tingkat_pemahaman": "Paham Mendalam","jumlah_prompt": 4},
            ],
        },
    )


if __name__ == "__main__":
    main()
