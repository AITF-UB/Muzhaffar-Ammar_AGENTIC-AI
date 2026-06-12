# -*- coding: utf-8 -*-
"""
benchmark_retriever.py
======================
Analisis Kecepatan Retriever Qdrant — Dense & Hybrid Pipeline

Menjalankan benchmark terhadap retriever yang sudah dibangun di tools.py,
mengukur latency setiap stage (embedding, qdrant search, BM25, SPLADE,
RRF fusion, chunk expansion, reranking) dan menghasilkan report HTML
interaktif dengan grafik perbandingan.

Cara menjalankan:
    python benchmark_retriever.py                         # default 5 query × 3 iterasi
    python benchmark_retriever.py --iterations 10         # 5 query × 10 iterasi
    python benchmark_retriever.py --queries "query1" "query2"  # custom queries
    python benchmark_retriever.py --mode dense             # hanya benchmark dense
    python benchmark_retriever.py --mode hybrid            # hanya benchmark hybrid
    python benchmark_retriever.py --mode both              # benchmark keduanya (default)
    python benchmark_retriever.py --top_k 10               # ubah top_k
"""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import re
import sys
import time
import statistics
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict

import requests
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# ── Import dari project yang sudah ada ──────────────────────────────────────
from model_registry import get_dense_model, get_sparse_model, get_reranker
from rank_bm25 import BM25Okapi

# ── Konfigurasi Qdrant (sama seperti tools.py) ─────────────────────────────
QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost")
if QDRANT_HOST.startswith("http://"):
    QDRANT_HOST = QDRANT_HOST[7:]
elif QDRANT_HOST.startswith("https://"):
    QDRANT_HOST = QDRANT_HOST[8:]
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
TEXT_COLLECTION = os.getenv("QDRANT_TEXT_COLLECTION", "hybrid_new")

BM25_CACHE_PATH = Path(__file__).resolve().parent / f"bm25_{TEXT_COLLECTION}.pkl"

# ── Default Queries untuk benchmark ─────────────────────────────────────────
DEFAULT_QUERIES = [
    "Apa itu fotosintesis dan bagaimana prosesnya?",
    "Jelaskan struktur sel hewan dan fungsinya",
    "Apa perbedaan antara mitosis dan meiosis?",
    "Bagaimana sistem peredaran darah manusia bekerja?",
    "Jelaskan tentang ekosistem dan rantai makanan",
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StageTimings:
    """Waktu per stage dalam satu kali retrieval."""
    embedding_ms: float = 0.0
    qdrant_dense_ms: float = 0.0
    qdrant_splade_ms: float = 0.0
    bm25_ms: float = 0.0
    rrf_fusion_ms: float = 0.0
    dedup_ms: float = 0.0
    chunk_expansion_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0
    num_results: int = 0


@dataclass
class QueryBenchmark:
    """Hasil benchmark satu query."""
    query: str
    mode: str
    iterations: int
    timings: List[StageTimings] = field(default_factory=list)


@dataclass
class BenchmarkReport:
    """Report keseluruhan."""
    timestamp: str
    qdrant_host: str
    qdrant_port: int
    collection: str
    search_modes: List[str]
    top_k: int
    num_queries: int
    iterations_per_query: int
    collection_info: Dict[str, Any] = field(default_factory=dict)
    benchmarks: List[QueryBenchmark] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — mereplikasi logic dari tools.py tapi dengan timing instrumentation
# ══════════════════════════════════════════════════════════════════════════════

def tokenize(text: str):
    text = text.lower()
    return re.findall(r"\w+", text)


def _timed(fn, *args, **kwargs) -> Tuple[Any, float]:
    """Jalankan fungsi dan return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = (time.perf_counter() - t0) * 1000
    return result, elapsed


async def _timed_async(coro) -> Tuple[Any, float]:
    """Jalankan coroutine dan return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = await coro
    elapsed = (time.perf_counter() - t0) * 1000
    return result, elapsed


# ── Embedding ───────────────────────────────────────────────────────────────

def embed_dense(query: str) -> Tuple[list, float]:
    model = get_dense_model()
    prefixed = f"query: {query.strip()}"
    t0 = time.perf_counter()
    vector = model.encode([prefixed], normalize_embeddings=True, convert_to_numpy=True)[0]
    elapsed = (time.perf_counter() - t0) * 1000
    return vector.tolist(), elapsed


# ── Qdrant Dense Search ────────────────────────────────────────────────────

def search_qdrant_dense(vector: list, top_k: int) -> Tuple[list, float]:
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{TEXT_COLLECTION}/points/search"
    payload = {
        "vector": {"name": "dense", "vector": vector},
        "limit": top_k,
        "with_payload": {"exclude": ["has_visual_content"]},
    }
    t0 = time.perf_counter()
    try:
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code == 400 and "Not existing vector name error" in response.text:
            payload["vector"] = vector
            response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        results = response.json().get("result", [])
    except Exception as e:
        print(f"❌ Qdrant Dense Error: {e}")
        results = []
    elapsed = (time.perf_counter() - t0) * 1000
    return results, elapsed


# ── Qdrant SPLADE Search ──────────────────────────────────────────────────

def search_qdrant_splade(query: str, top_k: int) -> Tuple[list, float]:
    model = get_sparse_model()
    t0 = time.perf_counter()
    try:
        sparse_vector = model.encode_query(query)
        url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{TEXT_COLLECTION}/points/search"
        payload = {
            "vector": {"name": "sparse", "vector": sparse_vector},
            "limit": top_k,
            "with_payload": {"exclude": ["has_visual_content"]},
        }
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        hits = response.json().get("result", [])
        results = []
        for hit in hits:
            p = hit.get("payload", {})
            results.append({
                "score": hit.get("score", 0.0),
                "text": p.get("text", p.get("page_content", "N/A")),
                "metadata": p,
                "source_file": p.get("source_file", "N/A"),
                "retrieval_type": "splade"
            })
    except Exception as e:
        print(f"❌ SPLADE Error: {e}")
        results = []
    elapsed = (time.perf_counter() - t0) * 1000
    return results, elapsed


# ── BM25 Search ────────────────────────────────────────────────────────────

_bm25 = None
_bm25_docs = []

def _load_bm25():
    global _bm25, _bm25_docs
    if _bm25 is not None:
        return
    if BM25_CACHE_PATH.exists():
        with open(BM25_CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        _bm25 = cache["bm25"]
        _bm25_docs = cache["docs"]
        print(f"📦 BM25 loaded from cache: {len(_bm25_docs)} docs")
    else:
        print("⚠️ BM25 cache not found, BM25 benchmark akan di-skip")


def search_bm25(query: str, top_k: int) -> Tuple[list, float]:
    _load_bm25()
    if _bm25 is None:
        return [], 0.0

    t0 = time.perf_counter()
    tokenized_query = tokenize(query)
    scores = _bm25.get_scores(tokenized_query)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    results = []
    for idx in ranked_idx[:top_k]:
        doc = dict(_bm25_docs[idx])
        doc["bm25_score"] = float(scores[idx])
        doc["retrieval_type"] = "sparse"
        results.append(doc)
    elapsed = (time.perf_counter() - t0) * 1000
    return results, elapsed


# ── RRF Fusion ─────────────────────────────────────────────────────────────

def rrf_fusion(dense_results, splade_results, bm25_results, k=60) -> Tuple[list, float]:
    t0 = time.perf_counter()
    fused_scores = {}

    def add(results, type_key):
        for rank, doc in enumerate(results):
            text = doc.get("text", "")
            rrf_score = 1 / (k + rank + 1)
            if text not in fused_scores:
                fused_scores[text] = {"doc": dict(doc), "score": 0}
            fused_scores[text]["score"] += rrf_score

    add(dense_results, "dense")
    add(splade_results, "splade")
    add(bm25_results, "bm25")

    reranked = sorted(fused_scores.values(), key=lambda x: x["score"], reverse=True)
    final = [item["doc"] for item in reranked]
    elapsed = (time.perf_counter() - t0) * 1000
    return final, elapsed


# ── Deduplication ──────────────────────────────────────────────────────────

def dedup(docs: list) -> Tuple[list, float]:
    t0 = time.perf_counter()
    unique = []
    seen = set()
    for doc in docs:
        metadata = doc.get("metadata", {})
        chunk_index = metadata.get("chunk_index")
        source_file = doc.get("source_file", "")
        uid = (source_file, chunk_index) if chunk_index is not None else doc.get("text", "")
        if uid in seen:
            continue
        seen.add(uid)
        unique.append(doc)
    elapsed = (time.perf_counter() - t0) * 1000
    return unique, elapsed


# ── Rerank ─────────────────────────────────────────────────────────────────

def rerank(query: str, docs: list, top_k: int) -> Tuple[list, float]:
    if not docs:
        return [], 0.0
    reranker = get_reranker()
    pairs = [(query, doc.get("expanded_text", doc.get("text", ""))) for doc in docs]
    t0 = time.perf_counter()
    scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    result = []
    for score, doc in ranked[:top_k]:
        doc["rerank_score"] = float(score)
        result.append(doc)
    elapsed = (time.perf_counter() - t0) * 1000
    return result, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def benchmark_dense(query: str, top_k: int) -> StageTimings:
    """Benchmark dense-only pipeline."""
    retrieve_k = max(top_k * 3, 20)

    # Stage 1: Embedding
    vector, embed_ms = embed_dense(query)

    # Stage 2: Qdrant dense search
    hits, qdrant_ms = search_qdrant_dense(vector, retrieve_k)

    # Format results
    results = []
    for hit in hits:
        p = hit.get("payload", {})
        results.append({
            "score": hit.get("score", 0.0),
            "text": p.get("text", p.get("page_content", "N/A")),
            "metadata": p,
            "source_file": p.get("source_file", "N/A"),
            "retrieval_type": "dense",
        })

    # Stage 3: Dedup
    unique_results, dedup_ms = dedup(results)

    total = embed_ms + qdrant_ms + dedup_ms

    return StageTimings(
        embedding_ms=round(embed_ms, 2),
        qdrant_dense_ms=round(qdrant_ms, 2),
        dedup_ms=round(dedup_ms, 2),
        total_ms=round(total, 2),
        num_results=min(len(unique_results), top_k),
    )


def benchmark_hybrid(query: str, top_k: int) -> StageTimings:
    """Benchmark hybrid pipeline (dense + splade + bm25 + rrf + rerank)."""
    retrieve_k = max(top_k * 5, 10)

    # Stage 1: Dense embedding
    vector, embed_ms = embed_dense(query)

    # Stage 2: Qdrant dense search
    dense_hits, qdrant_dense_ms = search_qdrant_dense(vector, retrieve_k)
    dense_results = []
    for hit in dense_hits:
        p = hit.get("payload", {})
        dense_results.append({
            "score": hit.get("score", 0.0),
            "text": p.get("text", p.get("page_content", "N/A")),
            "metadata": p,
            "source_file": p.get("source_file", "N/A"),
            "retrieval_type": "dense"
        })

    # Stage 3: SPLADE search (termasuk encoding sparse)
    splade_results, splade_ms = search_qdrant_splade(query, retrieve_k)

    # Stage 4: BM25 search
    bm25_results, bm25_ms = search_bm25(query, retrieve_k)

    # Stage 5: RRF fusion
    fused_results, rrf_ms = rrf_fusion(dense_results, splade_results, bm25_results)

    # Stage 6: Dedup
    unique_results, dedup_ms = dedup(fused_results)

    docs_to_rerank = unique_results[:15] 


    # Stage 7: Rerank
    reranked, rerank_ms = rerank(query, docs_to_rerank, top_k)

    total = embed_ms + qdrant_dense_ms + splade_ms + bm25_ms + rrf_ms + dedup_ms + rerank_ms

    return StageTimings(
        embedding_ms=round(embed_ms, 2),
        qdrant_dense_ms=round(qdrant_dense_ms, 2),
        qdrant_splade_ms=round(splade_ms, 2),
        bm25_ms=round(bm25_ms, 2),
        rrf_fusion_ms=round(rrf_ms, 2),
        dedup_ms=round(dedup_ms, 2),
        rerank_ms=round(rerank_ms, 2),
        total_ms=round(total, 2),
        num_results=len(reranked),
    )


def get_collection_info() -> Dict[str, Any]:
    """Ambil info collection dari Qdrant."""
    url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{TEXT_COLLECTION}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("result", {})
        return {
            "name": TEXT_COLLECTION,
            "vectors_count": data.get("vectors_count", "N/A"),
            "points_count": data.get("points_count", "N/A"),
            "segments_count": data.get("segments_count", "N/A"),
            "status": data.get("status", "N/A"),
            "optimizer_status": data.get("optimizer_status", "N/A"),
        }
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# HTML REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_html_report(report: BenchmarkReport) -> str:
    """Generate sebuah HTML report interaktif dengan Chart.js."""

    # ── Aggregate stats per mode ──
    mode_stats = {}
    for bm in report.benchmarks:
        if bm.mode not in mode_stats:
            mode_stats[bm.mode] = {
                "queries": [],
                "all_timings": [],
            }
        mode_stats[bm.mode]["queries"].append(bm.query)
        mode_stats[bm.mode]["all_timings"].extend(bm.timings)

    # ── Hitung summary stats ──
    summary_data = {}
    for mode, data in mode_stats.items():
        timings = data["all_timings"]
        totals = [t.total_ms for t in timings]
        stages = {}
        stage_fields = [
            ("embedding_ms", "Embedding"),
            ("qdrant_dense_ms", "Qdrant Dense"),
            ("qdrant_splade_ms", "Qdrant SPLADE"),
            ("bm25_ms", "BM25"),
            ("rrf_fusion_ms", "RRF Fusion"),
            ("dedup_ms", "Dedup"),
            ("chunk_expansion_ms", "Chunk Expansion"),
            ("rerank_ms", "Rerank"),
        ]
        for field_name, label in stage_fields:
            vals = [getattr(t, field_name) for t in timings if getattr(t, field_name) > 0]
            if vals:
                stages[label] = {
                    "mean": round(statistics.mean(vals), 2),
                    "median": round(statistics.median(vals), 2),
                    "min": round(min(vals), 2),
                    "max": round(max(vals), 2),
                    "stdev": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0,
                }

        summary_data[mode] = {
            "total_runs": len(totals),
            "mean_total_ms": round(statistics.mean(totals), 2),
            "median_total_ms": round(statistics.median(totals), 2),
            "min_total_ms": round(min(totals), 2),
            "max_total_ms": round(max(totals), 2),
            "p95_total_ms": round(sorted(totals)[int(len(totals) * 0.95)] if len(totals) >= 20 else max(totals), 2),
            "stdev_total_ms": round(statistics.stdev(totals), 2) if len(totals) > 1 else 0,
            "stages": stages,
        }

    # ── Per-query averages ──
    per_query_data = {}
    for bm in report.benchmarks:
        key = f"{bm.mode}|{bm.query}"
        totals = [t.total_ms for t in bm.timings]
        per_query_data[key] = {
            "mode": bm.mode,
            "query": bm.query[:60] + "..." if len(bm.query) > 60 else bm.query,
            "mean_ms": round(statistics.mean(totals), 2),
            "median_ms": round(statistics.median(totals), 2),
            "min_ms": round(min(totals), 2),
            "max_ms": round(max(totals), 2),
        }

    # ── Serialize to JSON for JS ──
    summary_json = json.dumps(summary_data, ensure_ascii=False)
    per_query_json = json.dumps(list(per_query_data.values()), ensure_ascii=False)
    collection_json = json.dumps(report.collection_info, ensure_ascii=False, indent=2)

    # ── Raw timings for iteration chart ──
    iteration_data = []
    for bm in report.benchmarks:
        for i, t in enumerate(bm.timings):
            iteration_data.append({
                "mode": bm.mode,
                "query": bm.query[:40],
                "iteration": i + 1,
                "total_ms": t.total_ms,
            })
    iteration_json = json.dumps(iteration_data, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Retriever Speed Analysis — Qdrant Benchmark</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #0a0e1a;
            --bg-secondary: #111827;
            --bg-card: #1a2235;
            --bg-card-hover: #1f2a42;
            --border: #2d3a52;
            --text-primary: #e8ecf4;
            --text-secondary: #8b97b0;
            --text-muted: #5a667a;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-cyan: #06b6d4;
            --accent-green: #10b981;
            --accent-orange: #f59e0b;
            --accent-red: #ef4444;
            --accent-pink: #ec4899;
            --gradient-primary: linear-gradient(135deg, #3b82f6, #8b5cf6);
            --gradient-secondary: linear-gradient(135deg, #06b6d4, #10b981);
            --gradient-warm: linear-gradient(135deg, #f59e0b, #ef4444);
            --shadow-glow: 0 0 30px rgba(59, 130, 246, 0.15);
            --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.3);
            --radius: 16px;
            --radius-sm: 10px;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }}

        .bg-mesh {{
            position: fixed;
            inset: 0;
            z-index: 0;
            background:
                radial-gradient(circle at 20% 20%, rgba(59,130,246,.08) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(139,92,246,.08) 0%, transparent 50%),
                radial-gradient(circle at 50% 50%, rgba(6,182,212,.05) 0%, transparent 60%);
            pointer-events: none;
        }}

        .container {{
            max-width: 1360px;
            margin: 0 auto;
            padding: 40px 24px;
            position: relative;
            z-index: 1;
        }}

        /* ── Header ── */
        .header {{
            text-align: center;
            margin-bottom: 48px;
        }}

        .header h1 {{
            font-size: 2.8rem;
            font-weight: 900;
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
            letter-spacing: -1px;
        }}

        .header .subtitle {{
            font-size: 1.1rem;
            color: var(--text-secondary);
            font-weight: 400;
        }}

        .header .timestamp {{
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 6px;
        }}

        /* ── Stat Cards Row ── */
        .stats-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 36px;
        }}

        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 24px;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}

        .stat-card::before {{
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: var(--gradient-primary);
            opacity: 0;
            transition: opacity 0.3s;
        }}

        .stat-card:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-glow);
            border-color: var(--accent-blue);
        }}

        .stat-card:hover::before {{ opacity: 1; }}

        .stat-card .label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            color: var(--text-muted);
            margin-bottom: 8px;
            font-weight: 600;
        }}

        .stat-card .value {{
            font-size: 2rem;
            font-weight: 800;
            color: var(--text-primary);
        }}

        .stat-card .unit {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            font-weight: 400;
        }}

        .stat-card.highlight .value {{
            background: var(--gradient-primary);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        /* ── Section ── */
        .section {{
            margin-bottom: 36px;
        }}

        .section-title {{
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .section-title .icon {{
            font-size: 1.5rem;
        }}

        /* ── Cards ── */
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 28px;
            box-shadow: var(--shadow-card);
            transition: all 0.3s ease;
        }}

        .card:hover {{
            border-color: rgba(59, 130, 246, 0.3);
        }}

        /* ── Charts Grid ── */
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 36px;
        }}

        .chart-container {{
            position: relative;
            height: 360px;
        }}

        /* ── Table ── */
        .table-wrapper {{
            overflow-x: auto;
            border-radius: var(--radius-sm);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th {{
            background: rgba(59, 130, 246, 0.1);
            color: var(--accent-blue);
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            padding: 14px 16px;
            text-align: left;
            border-bottom: 2px solid var(--border);
        }}

        td {{
            padding: 12px 16px;
            border-bottom: 1px solid rgba(45, 58, 82, 0.5);
            font-size: 0.92rem;
            color: var(--text-secondary);
        }}

        tr:hover td {{
            background: rgba(59, 130, 246, 0.04);
            color: var(--text-primary);
        }}

        .badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .badge-dense {{
            background: rgba(59, 130, 246, 0.15);
            color: var(--accent-blue);
        }}

        .badge-hybrid {{
            background: rgba(139, 92, 246, 0.15);
            color: var(--accent-purple);
        }}

        .value-good {{ color: var(--accent-green); font-weight: 600; }}
        .value-warn {{ color: var(--accent-orange); font-weight: 600; }}
        .value-bad  {{ color: var(--accent-red); font-weight: 600; }}

        /* ── Collection Info ── */
        .collection-info {{
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            padding: 16px 20px;
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.85rem;
            color: var(--text-secondary);
            overflow-x: auto;
            white-space: pre-wrap;
        }}

        /* ── Footer ── */
        .footer {{
            text-align: center;
            padding: 32px 0 16px;
            color: var(--text-muted);
            font-size: 0.8rem;
        }}

        /* ── Responsive ── */
        @media (max-width: 768px) {{
            .charts-grid {{ grid-template-columns: 1fr; }}
            .header h1 {{ font-size: 1.8rem; }}
            .stats-row {{ grid-template-columns: repeat(2, 1fr); }}
        }}

        /* ── Animations ── */
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .animate {{ animation: fadeInUp 0.5s ease-out forwards; }}
        .delay-1 {{ animation-delay: 0.1s; opacity: 0; }}
        .delay-2 {{ animation-delay: 0.2s; opacity: 0; }}
        .delay-3 {{ animation-delay: 0.3s; opacity: 0; }}
        .delay-4 {{ animation-delay: 0.4s; opacity: 0; }}
    </style>
</head>
<body>
    <div class="bg-mesh"></div>
    <div class="container">

        <!-- Header -->
        <div class="header animate">
            <h1>🚀 Retriever Speed Analysis</h1>
            <div class="subtitle">Qdrant Vector Database — Benchmark Report</div>
            <div class="timestamp">Generated: {report.timestamp} &nbsp;|&nbsp; Collection: <strong>{report.collection}</strong></div>
        </div>

        <!-- Summary Stats -->
        <div class="stats-row animate delay-1" id="stats-row"></div>

        <!-- Charts -->
        <div class="charts-grid animate delay-2">
            <div class="card">
                <h3 style="margin-bottom: 16px; font-size: 1rem; color: var(--text-secondary);">⚡ Stage Breakdown (Rata-rata)</h3>
                <div class="chart-container"><canvas id="stageChart"></canvas></div>
            </div>
            <div class="card">
                <h3 style="margin-bottom: 16px; font-size: 1rem; color: var(--text-secondary);">📈 Latency per Iterasi</h3>
                <div class="chart-container"><canvas id="iterationChart"></canvas></div>
            </div>
        </div>

        <div class="charts-grid animate delay-3">
            <div class="card">
                <h3 style="margin-bottom: 16px; font-size: 1rem; color: var(--text-secondary);">📊 Perbandingan per Query</h3>
                <div class="chart-container"><canvas id="queryChart"></canvas></div>
            </div>
            <div class="card">
                <h3 style="margin-bottom: 16px; font-size: 1rem; color: var(--text-secondary);">🎯 Distribusi Waktu Total</h3>
                <div class="chart-container"><canvas id="distributionChart"></canvas></div>
            </div>
        </div>

        <!-- Per-query Table -->
        <div class="section animate delay-4">
            <div class="section-title"><span class="icon">📋</span> Detail per Query</div>
            <div class="card">
                <div class="table-wrapper">
                    <table id="queryTable">
                        <thead>
                            <tr>
                                <th>Mode</th>
                                <th>Query</th>
                                <th>Mean (ms)</th>
                                <th>Median (ms)</th>
                                <th>Min (ms)</th>
                                <th>Max (ms)</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Stage Detail Tables per Mode -->
        <div class="section animate delay-4">
            <div class="section-title"><span class="icon">🔬</span> Detail Stage Timing</div>
            <div id="stage-tables"></div>
        </div>

        <!-- Collection Info -->
        <div class="section animate delay-4">
            <div class="section-title"><span class="icon">💾</span> Qdrant Collection Info</div>
            <div class="card">
                <div class="collection-info">{collection_json}</div>
            </div>
        </div>

        <div class="footer">
            Benchmark Report — Qdrant Retriever Analysis &nbsp;•&nbsp; {report.timestamp}
        </div>

    </div>

    <script>
        const summaryData = {summary_json};
        const perQueryData = {per_query_json};
        const iterationData = {iteration_json};

        // ── Color palette ──
        const COLORS = {{
            dense: {{ bg: 'rgba(59,130,246,0.7)', border: '#3b82f6' }},
            hybrid: {{ bg: 'rgba(139,92,246,0.7)', border: '#8b5cf6' }},
        }};

        const STAGE_COLORS = [
            'rgba(59,130,246,0.8)',    // Blue
            'rgba(6,182,212,0.8)',     // Cyan
            'rgba(139,92,246,0.8)',    // Purple
            'rgba(236,72,153,0.8)',    // Pink
            'rgba(245,158,11,0.8)',    // Amber
            'rgba(16,185,129,0.8)',    // Green
            'rgba(239,68,68,0.8)',     // Red
            'rgba(99,102,241,0.8)',    // Indigo
        ];

        // ── Defaults ──
        Chart.defaults.color = '#8b97b0';
        Chart.defaults.borderColor = 'rgba(45,58,82,0.4)';
        Chart.defaults.font.family = "'Inter', sans-serif";

        // ── Build Stats Row ──
        const statsRow = document.getElementById('stats-row');
        const modes = Object.keys(summaryData);

        function addStatCard(label, value, unit, highlight) {{
            const div = document.createElement('div');
            div.className = 'stat-card' + (highlight ? ' highlight' : '');
            div.innerHTML = `<div class="label">${{label}}</div><div class="value">${{value}}<span class="unit"> ${{unit}}</span></div>`;
            statsRow.appendChild(div);
        }}

        addStatCard('Collection', '{TEXT_COLLECTION}', '', false);
        addStatCard('Queries', '{report.num_queries}', '', false);
        addStatCard('Iterasi', '{report.iterations_per_query}', '×', false);

        modes.forEach(mode => {{
            const s = summaryData[mode];
            addStatCard(`${{mode.toUpperCase()}} Mean`, s.mean_total_ms, 'ms', true);
            addStatCard(`${{mode.toUpperCase()}} P95`, s.p95_total_ms, 'ms', false);
        }});

        // ── Stage Breakdown Chart (Stacked Bar) ──
        const stageLabels = new Set();
        modes.forEach(m => Object.keys(summaryData[m].stages).forEach(s => stageLabels.add(s)));
        const stageList = Array.from(stageLabels);

        const stageDatasets = stageList.map((stage, i) => ({{
            label: stage,
            data: modes.map(m => (summaryData[m].stages[stage] || {{}}).mean || 0),
            backgroundColor: STAGE_COLORS[i % STAGE_COLORS.length],
            borderRadius: 4,
        }}));

        new Chart(document.getElementById('stageChart'), {{
            type: 'bar',
            data: {{
                labels: modes.map(m => m.toUpperCase()),
                datasets: stageDatasets,
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'bottom', labels: {{ padding: 12, usePointStyle: true }} }},
                    tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{ctx.raw.toFixed(1)}} ms` }} }},
                }},
                scales: {{
                    x: {{ stacked: true, grid: {{ display: false }} }},
                    y: {{ stacked: true, title: {{ display: true, text: 'Waktu (ms)' }} }},
                }},
            }},
        }});

        // ── Iteration Chart (Line) ──
        const iterGroups = {{}};
        iterationData.forEach(d => {{
            const key = `${{d.mode}} — ${{d.query}}`;
            if (!iterGroups[key]) iterGroups[key] = {{ mode: d.mode, points: [] }};
            iterGroups[key].points.push({{ x: d.iteration, y: d.total_ms }});
        }});

        const iterDatasets = Object.entries(iterGroups).map(([label, g], i) => ({{
            label: label,
            data: g.points,
            borderColor: g.mode === 'dense' ? COLORS.dense.border : COLORS.hybrid.border,
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 3,
            pointBackgroundColor: g.mode === 'dense' ? COLORS.dense.border : COLORS.hybrid.border,
            tension: 0.3,
        }}));

        new Chart(document.getElementById('iterationChart'), {{
            type: 'line',
            data: {{ datasets: iterDatasets }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'bottom', labels: {{ padding: 12, usePointStyle: true, font: {{ size: 10 }} }} }},
                    tooltip: {{ callbacks: {{ label: ctx => `${{ctx.raw.y.toFixed(1)}} ms` }} }},
                }},
                scales: {{
                    x: {{ type: 'linear', title: {{ display: true, text: 'Iterasi' }}, ticks: {{ stepSize: 1 }} }},
                    y: {{ title: {{ display: true, text: 'Latency (ms)' }} }},
                }},
            }},
        }});

        // ── Per-Query Comparison (Horizontal Bar) ──
        const queryLabels = perQueryData.map(d => d.query);
        const queryModes = [...new Set(perQueryData.map(d => d.mode))];

        const qDatasets = queryModes.map(mode => ({{
            label: mode.toUpperCase(),
            data: perQueryData.filter(d => d.mode === mode).map(d => d.mean_ms),
            backgroundColor: mode === 'dense' ? COLORS.dense.bg : COLORS.hybrid.bg,
            borderRadius: 4,
        }}));

        // Build unique query labels (use first mode's queries)
        const uniqueQueries = perQueryData.filter(d => d.mode === queryModes[0]).map(d => d.query);

        new Chart(document.getElementById('queryChart'), {{
            type: 'bar',
            data: {{
                labels: uniqueQueries,
                datasets: qDatasets,
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {{
                    legend: {{ position: 'bottom', labels: {{ padding: 12, usePointStyle: true }} }},
                    tooltip: {{ callbacks: {{ label: ctx => `${{ctx.dataset.label}}: ${{ctx.raw.toFixed(1)}} ms` }} }},
                }},
                scales: {{
                    x: {{ title: {{ display: true, text: 'Mean Latency (ms)' }} }},
                    y: {{ ticks: {{ font: {{ size: 10 }} }} }},
                }},
            }},
        }});

        // ── Distribution Doughnut ──
        if (modes.length > 0) {{
            const firstMode = modes[0];
            const stages = summaryData[firstMode].stages;
            const stageNames = Object.keys(stages);
            const stageVals = stageNames.map(s => stages[s].mean);

            new Chart(document.getElementById('distributionChart'), {{
                type: 'doughnut',
                data: {{
                    labels: stageNames,
                    datasets: [{{
                        data: stageVals,
                        backgroundColor: STAGE_COLORS.slice(0, stageNames.length),
                        borderWidth: 0,
                    }}],
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    cutout: '55%',
                    plugins: {{
                        legend: {{ position: 'bottom', labels: {{ padding: 12, usePointStyle: true }} }},
                        title: {{ display: true, text: `${{firstMode.toUpperCase()}} — Proporsi Waktu per Stage`, color: '#8b97b0', font: {{ size: 13 }} }},
                        tooltip: {{ callbacks: {{ label: ctx => `${{ctx.label}}: ${{ctx.raw.toFixed(1)}} ms` }} }},
                    }},
                }},
            }});
        }}

        // ── Per-Query Table ──
        const tbody = document.querySelector('#queryTable tbody');
        perQueryData.forEach(d => {{
            const cls = d.mean_ms < 500 ? 'value-good' : d.mean_ms < 2000 ? 'value-warn' : 'value-bad';
            const badgeCls = d.mode === 'dense' ? 'badge-dense' : 'badge-hybrid';
            tbody.innerHTML += `<tr>
                <td><span class="badge ${{badgeCls}}">${{d.mode}}</span></td>
                <td>${{d.query}}</td>
                <td class="${{cls}}">${{d.mean_ms.toFixed(1)}}</td>
                <td>${{d.median_ms.toFixed(1)}}</td>
                <td>${{d.min_ms.toFixed(1)}}</td>
                <td>${{d.max_ms.toFixed(1)}}</td>
            </tr>`;
        }});

        // ── Stage Detail Tables ──
        const stageTables = document.getElementById('stage-tables');
        modes.forEach(mode => {{
            const stages = summaryData[mode].stages;
            let rows = '';
            Object.entries(stages).forEach(([name, s]) => {{
                const cls = s.mean < 100 ? 'value-good' : s.mean < 500 ? 'value-warn' : 'value-bad';
                rows += `<tr>
                    <td>${{name}}</td>
                    <td class="${{cls}}">${{s.mean.toFixed(1)}}</td>
                    <td>${{s.median.toFixed(1)}}</td>
                    <td>${{s.min.toFixed(1)}}</td>
                    <td>${{s.max.toFixed(1)}}</td>
                    <td>${{s.stdev.toFixed(1)}}</td>
                </tr>`;
            }});

            stageTables.innerHTML += `
                <div class="card" style="margin-bottom: 16px;">
                    <h4 style="margin-bottom: 12px; color: ${{mode === 'dense' ? '#3b82f6' : '#8b5cf6'}}">
                        ${{mode.toUpperCase()}} — Stage Timings
                    </h4>
                    <div class="table-wrapper">
                        <table>
                            <thead><tr>
                                <th>Stage</th><th>Mean (ms)</th><th>Median (ms)</th>
                                <th>Min (ms)</th><th>Max (ms)</th><th>Std Dev</th>
                            </tr></thead>
                            <tbody>${{rows}}</tbody>
                        </table>
                    </div>
                </div>`;
        }});
    </script>
</body>
</html>"""
    return html


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Benchmark Retriever Qdrant")
    parser.add_argument("--queries", nargs="+", default=None, help="Custom queries to benchmark")
    parser.add_argument("--iterations", type=int, default=3, help="Iterations per query (default: 3)")
    parser.add_argument("--top_k", type=int, default=5, help="Top K results (default: 5)")
    parser.add_argument("--mode", choices=["dense", "hybrid", "both"], default="both", help="Search mode to benchmark")
    parser.add_argument("--output", type=str, default=None, help="Output HTML file path")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations (not counted)")
    args = parser.parse_args()

    queries = args.queries or DEFAULT_QUERIES
    modes_to_test = ["dense", "hybrid"] if args.mode == "both" else [args.mode]
    output_path = args.output or f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

    print("=" * 70)
    print("🚀 RETRIEVER SPEED ANALYSIS — Qdrant Benchmark")
    print("=" * 70)
    print(f"📍 Qdrant:      {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"📦 Collection:  {TEXT_COLLECTION}")
    print(f"🔍 Modes:       {', '.join(m.upper() for m in modes_to_test)}")
    print(f"❓ Queries:     {len(queries)}")
    print(f"🔄 Iterations:  {args.iterations}")
    print(f"🔥 Warmup:      {args.warmup}")
    print(f"🎯 Top K:       {args.top_k}")
    print(f"📄 Output:      {output_path}")
    print("=" * 70)

    # Ambil collection info
    print("\n📊 Fetching collection info...")
    collection_info = get_collection_info()
    print(f"   Points: {collection_info.get('points_count', 'N/A')}")

    report = BenchmarkReport(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        collection=TEXT_COLLECTION,
        search_modes=modes_to_test,
        top_k=args.top_k,
        num_queries=len(queries),
        iterations_per_query=args.iterations,
        collection_info=collection_info,
    )

    for mode in modes_to_test:
        print(f"\n{'─' * 50}")
        print(f"🔬 Benchmarking [{mode.upper()}] mode")
        print(f"{'─' * 50}")

        benchmark_fn = benchmark_dense if mode == "dense" else benchmark_hybrid

        for qi, query in enumerate(queries, 1):
            print(f"\n  [{qi}/{len(queries)}] {query[:60]}...")

            qb = QueryBenchmark(query=query, mode=mode, iterations=args.iterations)

            # Warmup
            if args.warmup > 0:
                print(f"    🔥 Warmup ({args.warmup}x)...", end=" ", flush=True)
                for _ in range(args.warmup):
                    benchmark_fn(query, args.top_k)
                print("done")

            # Actual benchmark
            for it in range(args.iterations):
                timings = benchmark_fn(query, args.top_k)
                qb.timings.append(timings)
                print(f"    ⏱️  Iter {it+1}: {timings.total_ms:.1f} ms  "
                      f"(embed={timings.embedding_ms:.1f}, qdrant={timings.qdrant_dense_ms:.1f}"
                      f"{f', splade={timings.qdrant_splade_ms:.1f}' if timings.qdrant_splade_ms > 0 else ''}"
                      f"{f', bm25={timings.bm25_ms:.1f}' if timings.bm25_ms > 0 else ''}"
                      f"{f', rerank={timings.rerank_ms:.1f}' if timings.rerank_ms > 0 else ''}"
                      f", results={timings.num_results})")

            avg = statistics.mean([t.total_ms for t in qb.timings])
            print(f"    📊 Mean: {avg:.1f} ms")

            report.benchmarks.append(qb)

    # ── Generate HTML Report ──
    print(f"\n{'═' * 70}")
    print("📝 Generating HTML report...")
    html = generate_html_report(report)

    output_file = Path(output_path)
    output_file.write_text(html, encoding="utf-8")
    print(f"✅ Report saved: {output_file.resolve()}")

    # ── Also save raw JSON ──
    json_path = output_file.with_suffix(".json")
    raw_data = {
        "timestamp": report.timestamp,
        "config": {
            "qdrant_host": report.qdrant_host,
            "qdrant_port": report.qdrant_port,
            "collection": report.collection,
            "modes": report.search_modes,
            "top_k": report.top_k,
            "queries": report.num_queries,
            "iterations": report.iterations_per_query,
        },
        "collection_info": report.collection_info,
        "results": {},
    }
    for bm in report.benchmarks:
        key = f"{bm.mode}|{bm.query}"
        raw_data["results"][key] = {
            "mode": bm.mode,
            "query": bm.query,
            "timings": [asdict(t) for t in bm.timings],
        }
    json_path.write_text(json.dumps(raw_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Raw data saved: {json_path.resolve()}")

    # ── Print summary ──
    print(f"\n{'═' * 70}")
    print("📊 SUMMARY")
    print(f"{'═' * 70}")

    for mode in modes_to_test:
        mode_timings = [t for bm in report.benchmarks if bm.mode == mode for t in bm.timings]
        totals = [t.total_ms for t in mode_timings]
        print(f"\n  [{mode.upper()}]")
        print(f"    Mean:   {statistics.mean(totals):.1f} ms")
        print(f"    Median: {statistics.median(totals):.1f} ms")
        print(f"    Min:    {min(totals):.1f} ms")
        print(f"    Max:    {max(totals):.1f} ms")
        if len(totals) > 1:
            print(f"    StdDev: {statistics.stdev(totals):.1f} ms")

    print(f"\n🎉 Benchmark selesai! Buka {output_file.name} di browser untuk report interaktif.")


if __name__ == "__main__":
    main()
