from typing import TypedDict, Dict, Any

class AgentState(TypedDict):
    # 1. Input Utama Request
    task: str # HARUS: "rekomendasi", "flashcard", "mindmap", atau "quiz"
    request_params: Dict[str, Any]
    emotion_input: Dict[str, Any]
    
    # 2. State Internal Spesifik per Jalur (Isolasi Memori)
    top_recommendations: str
    flashcards_data: str
    mindmap_data: str
    quiz_data: str
    
    # 3. Output Final JSON Strict untuk UI Tim 6
    final_payload: Dict[str, Any]
