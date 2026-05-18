from pydantic import BaseModel, Field, ConfigDict
from typing import Any, Dict, List, Optional

# --- Envelope Models (V3.6) ---
class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

class StandardResponse(BaseModel):
    data: Optional[Any] = None
    meta: Optional[Dict[str, Any]] = None
    error: Optional[ErrorDetail] = None

# --- Generate / Regenerate Models ---
class GenerateRequest(BaseModel):
    mapel_id: str
    elemen_id: str
    elemen_label: str
    materi: Optional[str] = ""
    materi_id: Optional[str] = ""
    kelas_id: Optional[str] = ""
    jenjang: str
    atp: Optional[str] = ""
    tipe: str = Field(description="bacaan, quiz_pg, quiz_essay, flashcard, mindmap")
    level: Optional[str] = Field(default=None, description="Low, Mid, or High (Null for mindmap)")

class RegenerateRequest(GenerateRequest):
    konten_id: str
    instruksi_revisi: str

# --- Quiz Submission Models ---
class MCSubmitRequest(BaseModel):
    publish_id: str
    mapel_id: str
    elemen_id: str
    elemen_label: str
    materi: str
    materi_id: str
    level: str
    jawaban: Dict[str, str]

class EssaySubmitRequest(BaseModel):
    publish_id: str
    mapel_id: str
    elemen_id: str
    elemen_label: str
    materi: str
    materi_id: str
    level: str
    soal: Dict[str, str]
    rubrik: Dict[str, str]
    jawaban: Dict[str, str]

# --- RAG Specific Models ---
class RekomendasiRequest(BaseModel):
    siswa_id: str
    levels: Dict[str, str] = Field(default_factory=dict)
    sudah_selesai_ids: List[str] = Field(default_factory=list)
    sedang_dipelajari_ids: List[str] = Field(default_factory=list)

class InsightRequest(BaseModel):
    siswa_id: str
    nama: str
    streak: int
    total_topik: int
    total_poin_kuiz: int
    total_durasi: int
