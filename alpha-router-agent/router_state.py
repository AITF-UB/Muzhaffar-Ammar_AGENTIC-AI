from typing import TypedDict, Dict, Any, List

class AgentState(TypedDict):
    # ================================================================
    # 1. Task Identifier & Params
    # ================================================================
    task: str
    # Generate (teacher-centric):
    #   "bacaan" | "flashcard" | "mindmap" | "quiz" | "quiz_uraian"
    # Evaluasi (student-side):
    #   "evaluasi_uraian"
    # Lainnya:
    #   "rag_query" | "rekomendasi"

    request_params: Dict[str, Any]
    # Teacher input:
    #   jenjang        : str  → "10" | "11" | "12"
    #   kelas          : str  → "10A" | "10B" | "11C" dst.
    #   mata_pelajaran : str  → "Matematika" | "Fisika" dst.
    #   elemen         : str  → Elemen Capaian Pembelajaran
    #   materi         : str  → Sub-materi / topik spesifik
    #   atp            : List[str] → Alur Tujuan Pembelajaran
    #
    # Evaluasi input:
    #   matpel, bab, soal_pg/uraian, jawaban_siswa, dll.

    # ================================================================
    # 2. Output: Bacaan (3 level — LOTS / MOTS / HOTS)
    # ================================================================
    bacaan_lots_data: str
    bacaan_mots_data: str
    bacaan_hots_data: str

    # ================================================================
    # 3. Output: Flashcard (3 level — LOTS / MOTS / HOTS)
    # ================================================================
    flashcard_lots_data: str
    flashcard_mots_data: str
    flashcard_hots_data: str

    # ================================================================
    # 4. Output: Quiz PG (3 level — LOTS / MOTS / HOTS)
    # ================================================================
    quiz_lots_data: str
    quiz_mots_data: str
    quiz_hots_data: str

    # ================================================================
    # 5. Output: Quiz Uraian (3 level — LOTS / MOTS / HOTS)
    # ================================================================
    quiz_uraian_lots_data: str
    quiz_uraian_mots_data: str
    quiz_uraian_hots_data: str

    # ================================================================
    # 6. Output: Mindmap (1 versi lengkap & detail)
    # ================================================================
    mindmap_data: str

    # ================================================================
    # 7. Output: Rekomendasi, Evaluasi, RAG
    # ================================================================
    top_recommendations: str

    evaluasi_uraian_result: str
    rag_query_result: str

    # ================================================================
    # 8. Final Output JSON
    # ================================================================
    final_payload: Dict[str, Any]
