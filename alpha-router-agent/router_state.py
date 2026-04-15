from typing import TypedDict, Dict, Any

class AgentState(TypedDict):
    # 1. Input Utama Request
    task: str # "rekomendasi" | "konten_belajar" | "rag_query" | "flashcard" | "mindmap"
              # "quiz" | "quiz_uraian" | "evaluasi_quiz" | "evaluasi_uraian"
    request_params: Dict[str, Any]
    emotion_input: Dict[str, Any]
    
    # 2. State Internal Spesifik per Jalur 
    top_recommendations: str
    flashcards_data: str
    mindmap_data: str
    quiz_data: str              # PG generate output
    quiz_uraian_data: str       # Uraian generate output
    evaluasi_quiz_result: str   # PG evaluasi output (deterministik)
    evaluasi_uraian_result: str # Uraian evaluasi output (LLM)
    konten_belajar_data: str    # Konten panjang generate output
    rag_query_result: str       # Pure RAG
    
    # 3. Output Final JSON Strict untuk UI Tim 6
    final_payload: Dict[str, Any]
