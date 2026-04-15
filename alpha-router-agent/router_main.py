from router_agent import router_agent_app
from router_state import AgentState
import json

def run_simulation(scenario_name: str, task: str, request_params: dict, emotion: dict):
    print(f"\n==================================================================")
    print(f"🚀 SCENARIO: {scenario_name}")
    print(f"   Task: {task} | Emosi: {emotion.get('emosi')}")
    print(f"==================================================================")
    
    initial_state: AgentState = {
        "task": task,
        "request_params": request_params,
        "emotion_input": emotion,
        "top_recommendations": "",
        "flashcards_data": "",
        "mindmap_data": "",
        "quiz_data": "",
        "quiz_uraian_data": "",
        "grade_quiz_result": "",
        "grade_uraian_result": "",
        "final_payload": {}
    }
    
    final_output = None
    
    print("Memulai stream eksekusi Graph...\n")
    for step in router_agent_app.stream(initial_state, stream_mode="updates"):
        for node_name, node_data in step.items():
            print(f"📍 [NODE EXECUTED]: {node_name.upper()}")
            if node_name == "structurer":
                 final_output = node_data.get("final_payload", {})
            print("-" * 50)
            
    print("\n📦 === FINAL OUTPUT JSON (Siap dikonsumsi UI Tim 6) ===")
    try:
        if isinstance(final_output, dict):
            print("✅ VALID: Output berbentuk Dictionary Python/JSON Strict")
            print(json.dumps(final_output, indent=2, ensure_ascii=False))
        else:
            raise ValueError("Bukan data dictionary")
    except Exception as e:
        print("❌ ERROR: Output agen formatnya rusak.")
        print(f"Detail: {e}")
        print(f"Raw Output:\n{final_output}")
    
    return final_output


def main():
    print("Mempersiapkan RAG Sekolah DB sebelum simulasi...\n")
    
    # ------------------------------------------------------------------
    # Skenario 1 — Rekomendasi (Spesialis 1)
    # ------------------------------------------------------------------
    run_simulation(
        scenario_name="Rekomendasi Topik Evaluasi",
        task="rekomendasi",
        request_params={"nilai_pretest": 60, "nilai_ujian": 45},
        emotion={"emosi": "sedih", "confidence": 0.9}
    )
    
    # ------------------------------------------------------------------
    # Skenario 2 — Flashcard (Mekanik 2)
    # ------------------------------------------------------------------
    run_simulation(
        scenario_name="Generate Flashcards — Hukum Newton",
        task="flashcard",
        request_params={"topik": "Hukum Newton"},
        emotion={"emosi": "fokus", "confidence": 0.8}
    )

    # ------------------------------------------------------------------
    # Skenario 3 — Mindmap (Mekanik 3)
    # ------------------------------------------------------------------
    run_simulation(
        scenario_name="Generate Mindmap — Fotosintesis",
        task="mindmap",
        request_params={"topik": "Fotosintesis"},
        emotion={"emosi": "penasaran", "confidence": 0.85}
    )
    
    # ------------------------------------------------------------------
    # Skenario 4 — Quiz PG Generate (Mekanik 4a)
    # ------------------------------------------------------------------
    quiz_pg_output = run_simulation(
        scenario_name="Generate Quiz PG — Teorema Pythagoras",
        task="quiz",
        request_params={"topik": "Teorema Pythagoras", "jumlah_soal": 3},
        emotion={"emosi": "semangat", "confidence": 0.9}
    )

    # ------------------------------------------------------------------
    # Skenario 5 — Grade Quiz PG (Mekanik 4b) — simulasi jawaban dummy
    # Ambil soal dari output skenario 4 (jika berhasil), lalu buat jawaban dummy
    # ------------------------------------------------------------------
    soal_pg_list = []
    jawaban_dummy_pg = []
    if isinstance(quiz_pg_output, dict) and "soal" in quiz_pg_output:
        for s in quiz_pg_output["soal"]:
            soal_pg_list.append({
                "soal_id":      s.get("soal_id", ""),
                "nomor":        s.get("nomor", 1),
                "jawaban_benar": s.get("jawaban_benar", "A"),
                "pembahasan":   s.get("pembahasan", "")
            })
            # Dummy: siswa selalu jawab "A" (beberapa benar, beberapa salah)
            jawaban_dummy_pg.append({
                "soal_id": s.get("soal_id", ""),
                "jawaban": "A"
            })

    run_simulation(
        scenario_name="Evaluasi Quiz PG — Teorema Pythagoras (Dummy Jawaban)",
        task="evaluasi_quiz",
        request_params={
            "topik": "Teorema Pythagoras",
            "skor_per_soal": 10,
            "soal_pg": soal_pg_list,
            "jawaban_siswa": jawaban_dummy_pg
        },
        emotion={"emosi": "netral", "confidence": 0.8}
    )

    # ------------------------------------------------------------------
    # Skenario 6 — Quiz Uraian Generate (Mekanik 4c)
    # ------------------------------------------------------------------
    quiz_uraian_output = run_simulation(
        scenario_name="Generate Quiz Uraian — Proklamasi Kemerdekaan",
        task="quiz_uraian",
        request_params={"topik": "Proklamasi Kemerdekaan", "jumlah_soal": 3},
        emotion={"emosi": "penasaran", "confidence": 0.85}
    )

    # ------------------------------------------------------------------
    # Skenario 7 — Grade Uraian (Mekanik 4d) — simulasi jawaban dummy
    # ------------------------------------------------------------------
    soal_uraian_list = []
    jawaban_dummy_uraian = []
    if isinstance(quiz_uraian_output, dict) and "soal" in quiz_uraian_output:
        for s in quiz_uraian_output["soal"]:
            soal_uraian_list.append({
                "soal_id":       s.get("soal_id", ""),
                "nomor":         s.get("nomor", 1),
                "pertanyaan":    s.get("pertanyaan", ""),
                "kunci_jawaban": s.get("kunci_jawaban", ""),
                "skor_maksimal": s.get("skor_maksimal", 20)
            })
            # Dummy: jawaban singkat tidak lengkap
            jawaban_dummy_uraian.append({
                "soal_id": s.get("soal_id", ""),
                "jawaban": "Proklamasi Indonesia dibacakan oleh Soekarno pada 17 Agustus 1945."
            })

    run_simulation(
        scenario_name="Evaluasi Quiz Uraian — Proklamasi Kemerdekaan (Dummy Jawaban)",
        task="evaluasi_uraian",
        request_params={
            "topik": "Proklamasi Kemerdekaan",
            "soal_uraian": soal_uraian_list,
            "jawaban_siswa": jawaban_dummy_uraian
        },
        emotion={"emosi": "netral", "confidence": 0.8}
    )

    # ------------------------------------------------------------------
    # Skenario 8 — Rekomendasi FIRST TIME (siswa baru, berdasarkan pretest)
    # ------------------------------------------------------------------
    run_simulation(
        scenario_name="Rekomendasi v2 — First Time (Hasil Pretest)",
        task="rekomendasi",
        request_params={
            "student_id": "siswa-baru-001",
            "first_time": True,
            "matpel_dipilih": ["Matematika", "Fisika", "Biologi"],
            "hasil_pretest": [
                {"matpel": "Matematika", "skor": 45, "topik_lemah": ["Teorema Pythagoras", "Persamaan Kuadrat"]},
                {"matpel": "Fisika",     "skor": 30, "topik_lemah": ["Hukum Newton", "Gerak Lurus"]},
                {"matpel": "Biologi",    "skor": 62, "topik_lemah": ["Fotosintesis"]}
            ]
        },
        emotion={"emosi": "gugup", "confidence": 0.8}
    )

    # ------------------------------------------------------------------
    # Skenario 9 — Rekomendasi RETURNING (8 riwayat bab → filter ke 5 terburuk)
    # ------------------------------------------------------------------
    run_simulation(
        scenario_name="Rekomendasi v2 — Returning (8 bab riwayat)",
        task="rekomendasi",
        request_params={
            "student_id": "siswa-lama-042",
            "first_time": False,
            "matpel_dipilih": ["Matematika", "Fisika"],
            "riwayat_progress": [
                {"matpel": "Matematika", "bab": "Teorema Pythagoras",    "skor_terakhir": 20, "jumlah_evaluasi": 3, "tingkat_pemahaman": "Belum Paham",   "emosi_dominan": "bingung", "jumlah_prompt": 22},
                {"matpel": "Fisika",     "bab": "Hukum Newton",         "skor_terakhir": 35, "jumlah_evaluasi": 2, "tingkat_pemahaman": "Belum Paham",   "emosi_dominan": "frustrasi","jumlah_prompt": 18},
                {"matpel": "Matematika", "bab": "Persamaan Kuadrat",    "skor_terakhir": 50, "jumlah_evaluasi": 2, "tingkat_pemahaman": "Paham Dasar",   "emosi_dominan": "netral",  "jumlah_prompt": 10},
                {"matpel": "Fisika",     "bab": "Gerak Lurus Beraturan","skor_terakhir": 55, "jumlah_evaluasi": 1, "tingkat_pemahaman": "Paham Dasar",   "emosi_dominan": "fokus",   "jumlah_prompt": 7},
                {"matpel": "Matematika", "bab": "Sistem Persamaan",     "skor_terakhir": 65, "jumlah_evaluasi": 2, "tingkat_pemahaman": "Paham Dasar",   "emosi_dominan": "netral",  "jumlah_prompt": 8},
                {"matpel": "Fisika",     "bab": "Energi Mekanik",       "skor_terakhir": 70, "jumlah_evaluasi": 1, "tingkat_pemahaman": "Paham Mendalam","emosi_dominan": "semangat","jumlah_prompt": 5},
                {"matpel": "Matematika", "bab": "Statistika Dasar",     "skor_terakhir": 80, "jumlah_evaluasi": 1, "tingkat_pemahaman": "Paham Mendalam","emosi_dominan": "semangat","jumlah_prompt": 3},
                {"matpel": "Fisika",     "bab": "Gelombang",            "skor_terakhir": 85, "jumlah_evaluasi": 1, "tingkat_pemahaman": "Paham Mendalam","emosi_dominan": "fokus",   "jumlah_prompt": 4},
            ]
        },
        emotion={"emosi": "semangat", "confidence": 0.9}
    )

    # ------------------------------------------------------------------
    # Skenario 10 — Konten Belajar (materi panjang terstruktur)
    # ------------------------------------------------------------------
    run_simulation(
        scenario_name="Konten Belajar — Hukum Newton (SMA)",
        task="konten_belajar",
        request_params={
            "topik": "Hukum Newton",
            "bab": "Hukum Newton I dan Inersia",
            "level": "SMA"
        },
        emotion={"emosi": "semangat", "confidence": 0.85}
    )

    # ------------------------------------------------------------------
    # Skenario 11 — RAG Query (pure RAG, no LLM — T&A real-time)
    # ------------------------------------------------------------------
    run_simulation(
        scenario_name="RAG Query — Pertanyaan Siswa tentang Inersia (no LLM)",
        task="rag_query",
        request_params={
            "query": "apa yang dimaksud dengan inersia dan bagaimana contohnya?",
            "topik": "Hukum Newton",
            "k": 3
        },
        emotion={"emosi": "penasaran", "confidence": 0.8}
    )


if __name__ == "__main__":
    main()
