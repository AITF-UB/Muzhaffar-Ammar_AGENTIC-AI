from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # 1. Riwayat pemikiran dan tindakan agen (untuk ReAct)
    messages: Annotated[list, add_messages]
    
    # 2. Input Parameter dari MVP (Tim 6) - cth: topik, tingkat SMA
    request_params: Dict[str, Any]
    
    # 3. Input Emosi Real-time (dari Tim 1) - cth: frustrasi, senang
    emotion_input: Dict[str, Any]
    
    # 4. Draft materi mentah dari RAG sebelum diformat
    raw_content_context: str
    
    # 5. Output Final JSON Strict yang dituntut oleh UI Tim 6
    final_payload: Dict[str, Any]