import json
import os

OUT_PATH = "C:/Users/Ammar/Projek/agentic-ai/alpha-router-agent/alpha_router_all_in_one_local.ipynb"

# Read actual python files
with open("C:/Users/Ammar/Projek/agentic-ai/alpha-router-agent/router_state.py", "r", encoding="utf-8") as f:
    state_code = f.read()

with open("C:/Users/Ammar/Projek/agentic-ai/alpha-router-agent/router_tools.py", "r", encoding="utf-8") as f:
    tools_code = f.read()

with open("C:/Users/Ammar/Projek/agentic-ai/alpha-router-agent/router_agent.py", "r", encoding="utf-8") as f:
    agent_code = f.read()
TRANSFORMERS_LLM_SETUP = """
# ================================================================
# 1. SETUP HUGGINGFACE LLM (LOCAL TRANSFORMERS - A100 COLAB)
# ================================================================
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False
)

def _generate_response(system: str, user: str, max_tokens: int = 2000) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(
        prompt, 
        max_new_tokens=max_tokens, 
        temperature=0.3, 
        do_sample=True,
    )
    return outputs[0]["generated_text"].strip()

def _chat(system: str, user: str) -> str:
    return _generate_response(system, user, max_tokens=2000)

def _chat_long(system: str, user: str) -> str:
    return _generate_response(system, user, max_tokens=3000)

"""

import re
# Replace API client setup with local transformers setup
agent_code = re.sub(
    r"# ================================================================\n# 1\. SETUP HUGGINGFACE LLM.*?(?=# ================================================================\n# 2\. ROUTER)",
    TRANSFORMERS_LLM_SETUP,
    agent_code,
    flags=re.DOTALL
)

# Remove imports
agent_code_clean = []
for line in agent_code.split('\n'):
    if line.startswith("from router_state import") or line.startswith("from router_tools import"):
        continue
    if line.strip() in [
        "kb_sekolah,", "clean_json_from_llm,", "generate_soal_id,", "_ambil_prioritas_belajar,",
        "util_format_recommender,", "util_format_flashcard,", "util_format_mindmap,",
        "util_format_quiz,", "util_format_quiz_uraian,", "util_format_evaluasi_quiz,",
        "util_format_evaluasi_uraian,", "util_format_konten_belajar,", "util_format_rag_query,", ")"
    ]:
        continue
    agent_code_clean.append(line)
        
agent_code = "\n".join(agent_code_clean)

MAIN_CELL = """# ================================================================
# 4. SIMULASI TERMINAL (Bisa di-Run langsung di Colab)
# ================================================================

import json

def run_simulation(scenario_name: str, task: str, request_params: dict, emotion: dict):
    print(f"\\n==================================================================")
    print(f"🚀 SCENARIO: {scenario_name}")
    print(f"   Task: {task} | Emosi: {emotion.get('emosi')}")
    print(f"==================================================================")
    
    initial_state = {
        "task": task,
        "request_params": request_params,
        "emotion_input": emotion,
    }
    
    final_output = None
    
    print("Memulai stream eksekusi Graph...\\n")
    for step in workflow.compile().stream(initial_state, stream_mode="updates"):
        for node_name, node_data in step.items():
            print(f"📍 [NODE EXECUTED]: {node_name.upper()}")
            if node_name == "structurer":
                 final_output = node_data.get("final_payload", {})
            print("-" * 50)
            
    print("\\n📦 === FINAL OUTPUT JSON ===")
    try:
        if isinstance(final_output, dict):
            print("✅ VALID: Output berbentuk Dictionary Python/JSON Strict")
            print(json.dumps(final_output, indent=2, ensure_ascii=False))
        else:
            raise ValueError("Bukan data dictionary")
    except Exception as e:
        print("❌ ERROR: Output agen formatnya rusak.")
        print(f"Raw Output:\\n{final_output}")
    
    return final_output

def main():
    print("Mempersiapkan RAG Sekolah DB sebelum simulasi...\\n")
    
    run_simulation(
        scenario_name="Rekomendasi Topik Evaluasi",
        task="rekomendasi",
        request_params={
            "student_id": "siswa123", 
            "first_time": True, 
            "matpel_dipilih":["Fisika"], 
            "hasil_pretest":[{"matpel":"Fisika", "skor":60, "topik_lemah":["Hukum Newton"]}]
        },
        emotion={"emosi": "sedih", "confidence": 0.9}
    )
    
    run_simulation(
        scenario_name="Generate Flashcards — Hukum Newton",
        task="flashcard",
        request_params={"matpel": "Fisika", "bab": "Hukum Newton"},
        emotion={"emosi": "fokus", "confidence": 0.8}
    )

    run_simulation(
        scenario_name="Generate Mindmap — Fotosintesis",
        task="mindmap",
        request_params={"matpel": "Biologi", "bab": "Fotosintesis"},
        emotion={"emosi": "penasaran", "confidence": 0.85}
    )
    
    quiz_pg_output = run_simulation(
        scenario_name="Generate Quiz PG — Pythagoras",
        task="quiz",
        request_params={"matpel": "Matematika", "bab": "pythagoras", "jumlah_soal": 3},
        emotion={"emosi": "semangat", "confidence": 0.9}
    )

    soal_pg_list = []
    jawaban_dummy_pg = []
    if isinstance(quiz_pg_output, dict) and "soal" in quiz_pg_output:
        for s in quiz_pg_output["soal"]:
            soal_pg_list.append(s)
            jawaban_dummy_pg.append({
                "soal_id": s.get("soal_id", ""),
                "jawaban": "A"
            })

    run_simulation(
        scenario_name="Evaluasi Quiz PG — Pythagoras",
        task="evaluasi_quiz",
        request_params={
            "matpel": "Matematika",
            "bab": "pythagoras",
            "skor_per_soal": 10,
            "soal_pg": soal_pg_list,
            "jawaban_siswa": jawaban_dummy_pg
        },
        emotion={"emosi": "netral", "confidence": 0.8}
    )

    quiz_uraian_output = run_simulation(
        scenario_name="Generate Quiz Uraian — Proklamasi",
        task="quiz_uraian",
        request_params={"matpel": "Sejarah", "bab": "proklamasi", "jumlah_soal": 3},
        emotion={"emosi": "penasaran", "confidence": 0.85}
    )

    soal_uraian_list = []
    jawaban_dummy_uraian = []
    if isinstance(quiz_uraian_output, dict) and "soal" in quiz_uraian_output:
        for s in quiz_uraian_output["soal"]:
            soal_uraian_list.append(s)
            jawaban_dummy_uraian.append({
                "soal_id": s.get("soal_id", ""),
                "jawaban": "Proklamasi dibacakan di pegangsaan timur."
            })

    run_simulation(
        scenario_name="Evaluasi Quiz Uraian — Proklamasi",
        task="evaluasi_uraian",
        request_params={
            "matpel": "Sejarah",
            "bab": "proklamasi",
            "soal_uraian": soal_uraian_list,
            "jawaban_siswa": jawaban_dummy_uraian
        },
        emotion={"emosi": "netral", "confidence": 0.8}
    )

    run_simulation(
        scenario_name="Konten Belajar — Hukum Newton",
        task="konten_belajar",
        request_params={
            "matpel": "Fisika",
            "bab": "hukum_newton",
            "level": "SMA"
        },
        emotion={"emosi": "semangat", "confidence": 0.85}
    )

    run_simulation(
        scenario_name="RAG Query — Pertanyaan Siswa",
        task="rag_query",
        request_params={
            "query": "inersia?",
            "matpel": "Fisika",
            "bab": "hukum_newton",
            "k": 3
        },
        emotion={"emosi": "penasaran", "confidence": 0.8}
    )

if __name__ == '__main__':
    main()
"""

notebook = {
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": "# Cell 1: Install Dependencies (Mendukung Colab A100 Lokal)\n!pip install -q transformers accelerate torch langchain-community langchain-core langgraph pydantic chromadb sentence-transformers sentencepiece"
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": "# Cell 2: router_state.py (State DAG)\n" + state_code
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": "# Cell 3: router_tools.py (RAG & Tools)\n" + tools_code
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": "# Cell 4: router_agent.py (Agen & Workflow)\n" + agent_code
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": "# Cell 5: Main Tryout (Tanpa FastAPI, Langsung Test Logic)\n" + MAIN_CELL
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Berhasil update notebook")
