import os
import json
import re
import uuid
import torch
import numpy as np
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

# ================================================================
# Models Configuration
# ================================================================
QDRANT_HOST        = os.getenv("QDRANT_HOST", "https://vbfbs-175-45-190-1.run.pinggy-free.link")
# QDRANT_PORT        = int(os.getenv("QDRANT_PORT", 6333))
TEXT_COLLECTION    = os.getenv("QDRANT_TEXT_COLLECTION", "semantic_chunks")
IMAGE_COLLECTION   = os.getenv("QDRANT_IMAGE_COLLECTION", "Image_Clip_retriever")

TEXT_MODEL_NAME    = "microsoft/harrier-oss-v1-0.6b"
CLIP_MODEL_NAME    = "openai/clip-vit-large-patch14"
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

EXTRACTION_BASE_DIR = Path(__file__).resolve().parent.parent / "extraction"

# Lazy load globals
_sentence_model = None
_clip_model     = None
_clip_processor = None

# ================================================================
# 1. Models Loader
# ================================================================
def get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer(TEXT_MODEL_NAME)
    return _sentence_model

def get_clip_model():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        _clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _clip_model.eval()
    return _clip_model, _clip_processor

def embed_text_for_text_vdb(query: str) -> list:
    model = get_sentence_model()
    prefixed = f"query: {query.strip()}"
    vector = model.encode([prefixed], normalize_embeddings=True, convert_to_numpy=True)[0]
    return vector.tolist()

def embed_text_for_image_vdb(query: str) -> list:
    clip_model, clip_processor = get_clip_model()
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        features = clip_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0].tolist()


# ================================================================
# 2. Qdrant Search Engine
# ================================================================
def _search_qdrant(collection: str, vector: list, top_k: int, filter_payload: Optional[dict] = None) -> list:
    url = f"{QDRANT_HOST}/collections/{collection}/points/search"
    payload = {"vector": vector, "limit": top_k, "with_payload": True}
    if filter_payload:
        payload["filter"] = filter_payload
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("result", [])
    except Exception as e:
        print(f"❌ Error query Qdrant: {e}")
        return []

def _build_qdrant_filter(asset_type: Optional[str] = None, source: Optional[str] = None) -> Optional[dict]:
    conditions = []
    if asset_type: conditions.append({"key": "asset_type", "match": {"value": asset_type}})
    if source: conditions.append({"key": "source", "match": {"value": source}})
    return {"must": conditions} if conditions else None

def retrieve_text(query: str, top_k: int = 5) -> list:
    if not query.strip(): return []
    vector = embed_text_for_text_vdb(query)
    hits = _search_qdrant(TEXT_COLLECTION, vector, top_k)
    results = []
    for hit in hits:
        payload = hit.get("payload", {})
        results.append({
            "score": hit.get("score", 0.0),
            "text": payload.get("text", payload.get("page_content", "N/A")),
            "metadata": payload.get("metadata", {}),
            "source_file": payload.get("source_file", payload.get("metadata", {}).get("source_file", "N/A"))
        })
    return results

def retrieve_images(query: str, top_k: int = 6) -> list:
    if not query.strip(): return []
    vector = embed_text_for_image_vdb(query)
    hits = _search_qdrant(IMAGE_COLLECTION, vector, top_k)
    results = []
    for hit in hits:
        payload = hit.get("payload", {})
        raw_fp = payload.get("filepath", "")
        fname = payload.get("filename", "unknown")
        results.append({
            "score": hit.get("score", 0.0),
            "filepath": raw_fp,
            "filename": fname,
            "page_num": payload.get("page_num", "?"),
            "asset_type": payload.get("asset_type", "?")
        })
    return results

def extract_source(chunks: List[dict]) -> List[str]:
    """Extract raw source file strings from chunks."""
    sources = set()
    for c in chunks:
        src = c.get("source_file")
        if src and src != "N/A":
            sources.add(src)
    return list(sources)

# ================================================================
# 3. Dynamic RAG Engine
# ================================================================
class RAGEngine:
    @staticmethod
    def get_k_for_type(tipe: str) -> int:
        if tipe == "bacaan": return 10
        if tipe == "flashcard": return 3
        if tipe == "mindmap": return 7
        return 5

    @staticmethod
    def unified_search(query: str, tipe: str) -> Dict[str, Any]:
        """Perform search with dynamic chunk sizing and multimodal capabilities."""
        k_text = RAGEngine.get_k_for_type(tipe)
        texts = retrieve_text(query, top_k=k_text)
        
        # Ekstrak gambar langsung dari metadata 'has_visual_content' di chunk teks
        images = []
        if tipe in ["quiz_pg", "quiz_essay"]:
            for t in texts:
                vis = t.get("metadata", {}).get("has_visual_content", [])
                if isinstance(vis, list):
                    for img in vis:
                        if img not in images:
                            images.append(img)
                elif isinstance(vis, str):
                    if vis not in images:
                        images.append(vis)
            
        return {
            "text": texts,
            "images": images
        }

# ================================================================
# 4. Utilities
# ================================================================
def clean_json_from_llm(raw_text: str) -> dict | list:
    """Robust JSON parser to extract dirty JSON string from LLM."""
    clean_text = re.sub(r'```(?:json)?', '', raw_text).strip()
    idx_brace = clean_text.find('{')
    idx_bracket = clean_text.find('[')
    is_array = (idx_bracket != -1 and (idx_brace == -1 or idx_bracket < idx_brace))

    if is_array:
        start_idx = idx_bracket
        end_idx = clean_text.rfind(']')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            try: return json.loads(clean_text[start_idx:end_idx+1])
            except json.JSONDecodeError: pass
            
        text_to_parse = clean_text[start_idx:].strip()
        if text_to_parse.endswith(','): text_to_parse = text_to_parse[:-1]
        for fix in [']', '}]']:
            try: return json.loads(text_to_parse + fix)
            except json.JSONDecodeError: pass

    start_idx = clean_text.find('{')
    end_idx = clean_text.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try: return json.loads(clean_text[start_idx:end_idx+1])
        except json.JSONDecodeError: pass
            
    if start_idx != -1:
        try: return json.loads(clean_text[start_idx:] + '}')
        except json.JSONDecodeError: pass

    return {"error": "Gagal parsing JSON dari LLM", "raw": raw_text[:200]}

# extract_source dipindahkan ke atas

def generate_konten_id(tipe: str, level: str, materi_id: str) -> str:
    """Generate konten_id specified in API Contract."""
    import time
    lvl_str = (level or "all").lower()
    mat_clean = materi_id.split("__")[-1] if "__" in materi_id else materi_id
    return f"konten_{mat_clean}_{tipe}_{lvl_str}_{int(time.time())}"
