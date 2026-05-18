"""
execution/unified_retriever.py
──────────────────────────────
Unified Text + Image Retriever
Menggabungkan dua Vector Database Qdrant:
  1. Text VDB  : semantic_chunks        (SentenceTransformer dim=1024)
  2. Image VDB : Image_Clip_retriever   (CLIP ViT-L/14 dim=768)

Dengan satu query teks, sistem akan meng-retrieve:
  - Chunk teks relevan dari buku pelajaran
  - Gambar/tabel relevan dari buku pelajaran

Referensi: directives/unified_retriever.md
"""

import torch
import numpy as np
import requests
import json
from pathlib import Path
from typing import Optional

# ─── Lazy imports (model berat, di-load saat dibutuhkan) ─────────────────────
_sentence_model = None
_clip_model     = None
_clip_processor = None

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
QDRANT_HOST        = "http://76.13.195.1"
QDRANT_PORT        = 6333
TEXT_COLLECTION    = "semantic_chunks"
IMAGE_COLLECTION   = "Image_Clip_retriever"

TEXT_MODEL_NAME    = "microsoft/harrier-oss-v1-0.6b"
CLIP_MODEL_NAME    = "openai/clip-vit-large-patch14"
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

# Folder lokal tempat gambar disimpan setelah ekstraksi PDF
# Struktur: extraction/output_multimodal_{source}/extracted_assets/{filename}
EXTRACTION_BASE_DIR = Path(__file__).resolve().parent.parent / "extraction"


# ─── Path Resolution ──────────────────────────────────────────────────────────

def resolve_filepath(filepath: str, filename: str, source: str) -> Path:
    raw = Path(filepath)
    if raw.exists():
        return raw

    constructed = EXTRACTION_BASE_DIR / f"output_multimodal_{source}" / "extracted_assets" / filename
    if constructed.exists():
        return constructed
    matches = list(EXTRACTION_BASE_DIR.rglob(filename))
    if matches:
        return matches[0] 

    return constructed


# ─── Helper: Load Models ──────────────────────────────────────────────────────

def get_sentence_model():
    """Lazy-load SentenceTransformer untuk text embedding."""
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"⏳ Loading text model: {TEXT_MODEL_NAME}")
        _sentence_model = SentenceTransformer(TEXT_MODEL_NAME)
        print("✅ Text model loaded")
    return _sentence_model


def get_clip_model():
    """Lazy-load CLIP model dan processor untuk image embedding."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        print(f"⏳ Loading CLIP model: {CLIP_MODEL_NAME}")
        _clip_model     = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _clip_model.eval()
        print(f"✅ CLIP model loaded  |  Device: {DEVICE}")
    return _clip_model, _clip_processor


# ─── Embedding Functions ───────────────────────────────────────────────────────

def embed_text_for_text_vdb(query: str) -> list:
    """
    Embed query menggunakan SentenceTransformer untuk mencari di Text VDB.
    Prefix 'query: ' diperlukan oleh model harrier.
    """
    model = get_sentence_model()
    prefixed = f"query: {query.strip()}"
    vector = model.encode(
        [prefixed],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    return vector.tolist()


def embed_text_for_image_vdb(query: str) -> list:
    """
    Embed query menggunakan CLIP text encoder untuk mencari di Image VDB.
    Menghasilkan vektor dim=768 yang compatible dengan Image_Clip_retriever.
    """
    clip_model, clip_processor = get_clip_model()
    inputs = clip_processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        features = clip_model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
    # L2 normalize agar kompatibel dengan cosine similarity di Qdrant
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0].tolist()


# ─── Qdrant REST Search ────────────────────────────────────────────────────────

def _search_qdrant(collection: str, vector: list, top_k: int,
                   filter_payload: Optional[dict] = None) -> list:
    """
    Generik search ke Qdrant via REST API.
    Returns list of hit dicts: {id, score, payload}
    """
    url = f"{QDRANT_HOST}:{QDRANT_PORT}/collections/{collection}/points/search"
    payload = {
        "vector":       vector,
        "limit":        top_k,
        "with_payload": True,
    }
    if filter_payload:
        payload["filter"] = filter_payload

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("result", [])
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"❌ Tidak bisa terhubung ke Qdrant: {url}")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"❌ HTTP Error dari Qdrant: {e}")
    except Exception as e:
        raise RuntimeError(f"❌ Error saat query Qdrant: {e}")


# ─── Build Qdrant Filter ──────────────────────────────────────────────────────

def _build_qdrant_filter(asset_type: Optional[str] = None,
                          source: Optional[str] = None) -> Optional[dict]:
    """Buat filter Qdrant berdasarkan asset_type dan/atau source."""
    conditions = []
    if asset_type:
        conditions.append({"key": "asset_type", "match": {"value": asset_type}})
    if source:
        conditions.append({"key": "source", "match": {"value": source}})
    if conditions:
        return {"must": conditions}
    return None


# ─── Retrieval Functions ───────────────────────────────────────────────────────

def retrieve_text(query: str, top_k: int = 5) -> list:
    """
    Retrieve chunk teks relevan dari VDB 'semantic_chunks'.

    Returns:
        list of dict: [{score, text, metadata}]
    """
    if not query.strip():
        raise ValueError("Query tidak boleh kosong.")

    vector  = embed_text_for_text_vdb(query)
    hits    = _search_qdrant(TEXT_COLLECTION, vector, top_k)

    results = []
    for hit in hits:
        payload = hit.get("payload", {})
        results.append({
            "score":    hit.get("score", 0.0),
            "text":     payload.get("text", payload.get("page_content", "N/A")),
            "metadata": payload.get("metadata", {}),
            "source_file": payload.get("source_file", "N/A"),
        })
    return results


def retrieve_images(query: str, top_k: int = 6,
                    asset_type: Optional[str] = None,
                    source: Optional[str] = None) -> list:
    """
    Retrieve gambar/tabel relevan dari VDB 'Image_Clip_retriever'.

    Args:
        query:      Query teks pencarian
        top_k:      Jumlah hasil yang dikembalikan
        asset_type: Filter 'picture' atau 'table' (opsional)
        source:     Filter berdasarkan sumber buku (opsional)

    Returns:
        list of dict: [{score, filepath, filename, page_num, asset_type, source}]
    """
    if not query.strip():
        raise ValueError("Query tidak boleh kosong.")

    vector  = embed_text_for_image_vdb(query)
    filt    = _build_qdrant_filter(asset_type, source)
    hits    = _search_qdrant(IMAGE_COLLECTION, vector, top_k, filt)

    results = []
    for hit in hits:
        payload    = hit.get("payload", {})
        raw_fp     = payload.get("filepath", "")
        fname      = payload.get("filename", "unknown")
        src        = payload.get("source", "?")

        # Resolve path ke file lokal yang sebenarnya
        local_path = resolve_filepath(raw_fp, fname, src)

        results.append({
            "score":      hit.get("score", 0.0),
            "filepath":   raw_fp,                  # path asli dari Qdrant (referensi)
            "local_path": str(local_path),          # path lokal yang telah di-resolve
            "file_exists": local_path.exists(),     # apakah file ditemukan di disk
            "filename":   fname,
            "page_num":   payload.get("page_num", "?"),
            "asset_type": payload.get("asset_type", "?"),
            "source":     src,
        })
    return results


# ─── Unified Retrieve ─────────────────────────────────────────────────────────

def unified_retrieve(
    query: str,
    top_k_text: int = 5,
    top_k_image: int = 6,
    asset_type: Optional[str] = None,
    source: Optional[str] = None,
) -> dict:
    """
    Unified retriever: satu query → teks + gambar.

    Args:
        query:       Query pencarian teks
        top_k_text:  Jumlah chunk teks yang dikembalikan
        top_k_image: Jumlah gambar yang dikembalikan
        asset_type:  Filter jenis gambar (opsional)
        source:      Filter sumber buku (opsional)

    Returns:
        {
          "query":  str,
          "text":   list of text results,
          "images": list of image results,
        }
    """
    print(f"\n🔍 Query: '{query}'")
    print("─" * 60)

    print(f"📄 Mencari chunk teks (top_{top_k_text})...")
    text_results = retrieve_text(query, top_k_text)

    print(f"🖼️  Mencari gambar relevan (top_{top_k_image})...")
    image_results = retrieve_images(query, top_k_image, asset_type, source)

    print(f"\n✅ Ditemukan {len(text_results)} chunk teks, {len(image_results)} gambar")
    return {
        "query":  query,
        "text":   text_results,
        "images": image_results,
    }


# ─── Display Functions ─────────────────────────────────────────────────────────

def display_text_results(text_results: list):
    """Print chunk teks hasil retrieval ke stdout."""
    print("\n" + "═" * 60)
    print("  📄  HASIL TEKS")
    print("═" * 60)
    if not text_results:
        print("  (Tidak ada hasil teks)")
        return
    for i, r in enumerate(text_results, 1):
        meta = r.get("metadata", {})
        section = meta.get("section", "-")
        src     = r.get("source_file", meta.get("source_file", "-"))
        print(f"\n  [{i}]  Score: {r['score']:.4f}  |  {src}  |  {section}")
        print(f"       {r['text']}{'...' if len(r['text']) > 200 else ''}")
        print("  " + "─" * 56)


def display_image_results(image_results: list, show_images: bool = True):
    """
    Display gambar hasil retrieval.
    Requires matplotlib dan PIL jika show_images=True.
    """
    print("\n" + "═" * 60)
    print("  🖼️   HASIL GAMBAR")
    print("═" * 60)
    if not image_results:
        print("  (Tidak ada hasil gambar)")
        return

    found_count = sum(1 for r in image_results if r.get("file_exists", False))
    print(f"  📊 {found_count}/{len(image_results)} gambar ditemukan di disk lokal\n")

    for i, r in enumerate(image_results, 1):
        found_icon = "✅" if r.get("file_exists", False) else "❌"
        print(f"  [{i}] {found_icon} Score: {r['score']:.4f}  |  {r['source']}  "
              f"|  Hal.{r['page_num']}  |  {r['asset_type']}")
        print(f"       File     : {r['filename']}")
        if not r.get("file_exists", False):
            print(f"       ⚠️  Path  : {r.get('local_path', r['filepath'])}")

    if show_images:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            n    = len(image_results)
            cols = 3
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            axes = np.array(axes).flatten() if n > 1 else [axes]

            for i, r in enumerate(image_results):
                ax = axes[i]
                # Gunakan local_path (hasil resolve) bukan filepath mentah dari Qdrant
                display_path = r.get("local_path", r["filepath"])
                try:
                    img = mpimg.imread(display_path)
                    ax.imshow(img)
                except Exception as e:
                    ax.text(
                        0.5, 0.5,
                        f"[Gambar tidak ditemukan]\n{Path(display_path).name}",
                        ha="center", va="center",
                        transform=ax.transAxes, color="gray",
                        fontsize=8, wrap=True
                    )
                ax.set_title(
                    f"#{i+1}  Score: {r['score']:.4f}\n"
                    f"Hal.{r['page_num']} | {r['asset_type']}\n"
                    f"{r['source']}\n{r['filename'][:28]}...",
                    fontsize=7
                )
                ax.axis("off")

            for j in range(n, len(axes)):
                axes[j].set_visible(False)

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("  ⚠️  matplotlib/PIL tidak tersedia. Install untuk menampilkan gambar.")


# ─── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "hukum termodinamika"
    results = unified_retrieve(query, top_k_text=5, top_k_image=6)
    display_text_results(results["text"])
    display_image_results(results["images"], show_images=False)
