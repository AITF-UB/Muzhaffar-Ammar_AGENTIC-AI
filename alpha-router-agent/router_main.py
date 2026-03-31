from router_agent import router_agent_app
from router_state import AgentState
import json

def run_simulation(scenario_name: str, task: str, request_params: dict, emotion: dict):
    print(f"\n==================================================================")
    print(f"🚀 SCENARIO: {scenario_name}")
    print(f"   Task: {task} | Emosi: {emotion.get('emosi')}")
    print(f"==================================================================")
    
    initial_state: AgentState = {
        "task": task,
        "request_params": request_params,
        "emotion_input": emotion,
        "top_recommendations": "",
        "flashcards_data": "",
        "mindmap_data": "",
        "final_payload": {}
    }
    
    final_output = None
    
    print("Memulai stream eksekusi Graph...\n")
    for step in router_agent_app.stream(initial_state, stream_mode="updates"):
        for node_name, node_data in step.items():
            print(f"📍 [NODE EXECUTED]: {node_name.upper()}")
            
            if node_name == "structurer":
                 final_output = node_data.get("final_payload", {})
                 
            print("-" * 50)
            
    print("\n📦 === FINAL OUTPUT JSON (Siap dikonsumsi UI Tim 6) ===")
    try:
        if isinstance(final_output, dict):
            print("✅ VALID: Output berbentuk Dictionary Python/JSON Strict")
            print(json.dumps(final_output, indent=2, ensure_ascii=False))
        else:
            raise ValueError("Bukan data dictionary")
            
    except Exception as e:
        print("❌ ERROR: Output agen formatnya rusak.")
        print(f"Detail: {e}")
        print(f"Raw Output:\n{final_output}")


def main():
    print("Mempersiapkan RAG Sekolah DB sebelum simulasi...\n")
    
    # Skenario 1 (Jalur Recommender Agent)
    # Siswa sedang sedih setelah ujian yang jelek
    run_simulation(
        scenario_name="Rekomendasi Topik Evaluasi",
        task="rekomendasi",
        request_params={"nilai_pretest": 60, "nilai_ujian": 45},
        emotion={"emosi": "sedih", "confidence": 0.9}
    )
    
    # Skenario 2 (Jalur Flashcard Agent)
    # Siswa minta dibuatkan flashcard untuk Hukum Newton dengan NotebookLM Sources
    run_simulation(
        scenario_name="Generate Flashcards",
        task="flashcard",
        request_params={"topik": "Hukum Newton"},
        emotion={"emosi": "fokus", "confidence": 0.8}
    )
    
    # Skenario 3 (Jalur Mindmap Agent)
    # Siswa butuh peta hierarki konsep Fotosintesis
    run_simulation(
        scenario_name="Generate Mindmap",
        task="mindmap",
        request_params={"topik": "Fotosintesis"},
        emotion={"emosi": "penasaran", "confidence": 0.85}
    )

if __name__ == "__main__":
    main()
