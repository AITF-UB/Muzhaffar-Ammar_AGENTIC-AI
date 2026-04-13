from typing import TypedDict, Dict, Any

class AgentState(TypedDict):
    # 1. Input Utama Request
    task: str # "rekomendasi" | "flashcard" | "mindmap" | "quiz" | "quiz_uraian" | "grade_quiz" | "grade_uraian"
    request_params: Dict[str, Any]
    emotion_input: Dict[str, Any]
    
    # 2. State Internal Spesifik per Jalur (Isolasi Memori)
    top_recommendations: str
    flashcards_data: str
    mindmap_data: str
    quiz_data: str           # PG generate output
    quiz_uraian_data: str    # Uraian generate output
    grade_quiz_result: str   # PG grading output (deterministik)
    grade_uraian_result: str # Uraian grading output (LLM)
    
    # 3. Output Final JSON Strict untuk UI Tim 6
    final_payload: Dict[str, Any]
