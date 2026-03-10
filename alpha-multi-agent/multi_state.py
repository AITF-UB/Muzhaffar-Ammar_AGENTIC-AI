from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # 1. Riwayat percakapan (untuk log pipeline)
    messages: Annotated[list, add_messages]
    
    # 2. Input Parameter dari MVP (Tim 6) - cth: topik, tingkat SMA, riwayat nilai
    request_params: Dict[str, Any]
    
    # 3. Input Emosi Real-time (dari Tim 1) - cth: frustrasi, senang
    emotion_input: Dict[str, Any]
    
    # 4. Pipeline Internal State
    retrieved_documents: List[str]
    documents_relevant: bool
    draft_content: str
    adapted_content: str
    
    # 5. Pipeline Evaluasi Internal
    quality_score: int
    quality_feedback: str
    revision_count: int
    max_revisions: int
    
    # 6. Output Final JSON Strict yang dituntut oleh UI Tim 6
    final_payload: Dict[str, Any]