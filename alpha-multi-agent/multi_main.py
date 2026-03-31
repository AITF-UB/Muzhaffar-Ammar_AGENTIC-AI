from multi_agent import alpha_agent_app
from multi_state import AgentState
import json

def main():
    print("=== MEMPERSIAPKAN INPUT SIMULASI (TIM 1 & TIM 6) ===")
    
    # 1. Simulasi Payload Request dari Tim 6 (MVP)
    mock_request_params = {
        "user_id": "SR-1029",
        "topik": "Hukum Newton",
        "tingkat": "SMA",
        "riwayat_nilai_rata_rata": 75  # Nilai menengah
    }
    
    # 2. Simulasi Payload Data Emosi dari Tim 1 (Computer Vision)
    mock_emotion = {
        "emosi": "bingung",  # Emosi negatif agar Emotion Adapter Tool merespons suportif
        "confidence": 0.88
    }
    
    # 3. Merakit Initial State DAG
    initial_state: AgentState = {
        "messages": [],
        "request_params": mock_request_params,
        "emotion_input": mock_emotion,
        "retrieved_documents": [],
        "documents_relevant": False,
        "draft_content": "",
        "adapted_content": "",
        "quality_score": 0,
        "quality_feedback": "",
        "revision_count": 0,
        "max_revisions": 2,
        "final_payload": {}
    }
    
    print("\n=== MENJALANKAN PIPELINE AI (DAG) ===")
    print("Memulai log aliran node berurutan...\n")
    
    # 4. Stream iterasi Graph per-node layaknya debug loop
    final_output = None
    
    for step in alpha_agent_app.stream(initial_state, stream_mode="updates"):
        for node_name, node_data in step.items():
            print(f"📍 [LOG] Node aktif: {node_name.upper()}")
            if "messages" in node_data and node_data["messages"]:
                snippet = str(node_data["messages"][-1].content).replace('\n', ' ')
                if len(snippet) > 100:
                    snippet = snippet[:100] + " ... [dipotong]"
                print(f"   💬 ACTIVITY: {snippet}")
                
            # Cek field spesifik untuk tracking log pipeline
            if node_name == "retriever":
                print(f"   🔍 FOUND DOCS: {len(node_data.get('retrieved_documents', []))}")
            elif node_name == "grader":
                print(f"   ⚖️ RELEVANCE: {node_data.get('documents_relevant', False)}")
            elif node_name == "emotion_adapter":
                print(f"   🎭 EMOTION APPLIED: {mock_emotion['emosi']}")
            elif node_name == "quality_checker":
                print(f"   ✅ QC SCORE: {node_data.get('quality_score')} (Rev: {node_data.get('revision_count')})")
            elif node_name == "structurer":
                final_output = node_data.get("final_payload")
                
            print("-" * 60)
            
    # 5. Output Validasi Format JSON
    print("\n=== HASIL AKHIR (OUTPUT UNTUK UI TIM 6) ===")
    print("Agen berhenti. Memeriksa format Final Answer...\n")
    
    try:
        if isinstance(final_output, dict):
            print("✅ VALIDASI SUKSES: Output berbentuk JSON (Dictionary Python/JSON) strict yang aman untuk dirender Tim 6 Frontend.")
            print(json.dumps(final_output, indent=2))
        else:
            raise ValueError("Bukan data dictionary")
            
    except Exception as e:
        print("❌ VALIDASI GAGAL: Output agen formatnya rusak.")
        print(f"Error: {e}")
        print(f"Output mentah agen:\n{final_output}")

if __name__ == "__main__":
    main()