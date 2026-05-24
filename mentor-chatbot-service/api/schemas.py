from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class Konteks(BaseModel):
    emosi: Optional[str] = None
    publish_id: Optional[str] = None
    bacaan: Optional[str] = None

class MessageHistory(BaseModel):
    role: str
    teks: str

class ChatRequest(BaseModel):
    siswa_id: str
    sesi_id: str
    mapel_id: str
    elemen_id: str
    elemen_label: str
    materi: str
    materi_id: str
    atp: str
    level: str
    pesan: str
    konteks: Optional[Konteks] = None
    history: List[MessageHistory] = Field(default_factory=list)

class EvaluasiRequest(BaseModel):
    siswa_id: str
    sesi_id: str
    hasil_quiz_id: str
    mapel_id: str
    elemen_id: str
    elemen_label: str
    materi: str
    materi_id: str
    level: str
    atp: str
    quiz_data: Optional[Dict[str, Any]] = None
    history: List[MessageHistory] = Field(default_factory=list)

class ChatResponseData(BaseModel):
    balasan: str
    sesi_id: str

class ChatResponse(BaseModel):
    data: ChatResponseData
    meta: Optional[Any] = None
    error: Optional[Any] = None
