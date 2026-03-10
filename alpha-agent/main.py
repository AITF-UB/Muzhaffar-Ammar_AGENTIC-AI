from agent import alpha_agent_app
import json

def main():
    print("=== MEMPERSIAPKAN INPUT SIMULASI (TIM 1 & TIM 6) ===")
    
    # 1. Simulasi Payload Request dari Tim 6 (MVP)
    mock_request_params = {
        "user_id": "SR-1029",
        "topik": "Pernyataan Kondisional",
        "tingkat": "SMA",
        "riwayat_nilai_rata_rata": 75  # Nilai menengah, kita lihat apakah tool tingkat kesulitan merespons
    }
    
    # 2. Simulasi Payload Data Emosi dari Tim 1 (Computer Vision)
    mock_emotion = {
        "emosi": "bingung",  # Emosi negatif agar Emotion Adapter Tool dipanggil
        "confidence": 0.88
    }
    
    # 3. Merakit Initial State
    # Kita memberikan instruksi awal (prompt) sebagai pemicu berjalannya ReAct Loop
    initial_state = {
        "messages": [("user", "Tolong buatkan materi pembelajaran terstruktur sesuai dengan parameter request dan kondisi emosi siswa saat ini. Ikuti hukum mutlakmu.")],
        "request_params": mock_request_params,
        "emotion_input": mock_emotion,
        "raw_content_context": "",
        "final_payload": {}
    }
    
    print("\n=== MENJALANKAN AGENTIC AI (TIM 3) ===")
    print("Memulai proses ReAct Loop...\n")
    
    # Menjalankan graph secara streaming agar kita bisa mengaudit setiap keputusan AI
    final_output = None
    
    for step in alpha_agent_app.stream(initial_state, stream_mode="updates"):
        for node_name, node_data in step.items():
            print(f"📍 [LOG] Node aktif: {node_name.upper()}")
            if "messages" in node_data:
                last_msg = node_data["messages"][-1]
                
                # Cek jika agen memutuskan untuk menggunakan Tool
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    tools_dipanggil = [t['name'] for t in last_msg.tool_calls]
                    print(f"   🔧 ACTION: Memanggil Tool -> {tools_dipanggil}")
                
                # Jika pesan datang dari ToolNode (tipe ToolMessage)
                elif last_msg.__class__.__name__ == 'ToolMessage':
                    snippet = str(last_msg.content).replace('\n', ' ')
                    if len(snippet) > 100:
                        snippet = snippet[:100] + " ... [dipotong]"
                    print(f"   ⚙️ TOOL OUTPUT: {snippet}")
                    
                    # Jika tool yang baru saja selesai adalah content_structurer_tool, ini adalah kebenaran mutlaknya
                    if last_msg.name == "content_structurer_tool":
                        final_output = last_msg.content  # Kunci output murni di sini
                
                # Jika pesan dari agen (AIMessage)
                elif hasattr(last_msg, 'content') and last_msg.content:
                    snippet = str(last_msg.content).replace('\n', ' ')
                    if len(snippet) > 100:
                        snippet = snippet[:100] + " ... [dipotong]"
                    print(f"   💬 OBSERVATION/REASONING: {snippet}")
                    # Kita TIDAK LAGI menimpa final_output dengan ocehan akhir agen
    
            print("-" * 60)
            
    print("\n=== HASIL AKHIR (OUTPUT UNTUK UI TIM 6) ===")
    print("Agen berhenti. Memeriksa format Final Answer...\n")
    
    # Karena kita memaksa agen untuk merespons dalam format JSON di akhir, kita coba parse
    try:
        # Mencari blok JSON jika agen membungkusnya dalam markdown ```json ... ```
        clean_output = final_output
        if "```json" in clean_output:
            clean_output = clean_output.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_output:
            clean_output = clean_output.split("```")[1].strip()
            
        parsed_json = json.loads(clean_output)
        print("✅ VALIDASI SUKSES: Output berbentuk JSON strict.")
        print(json.dumps(parsed_json, indent=2))
        
    except Exception as e:
        print("❌ VALIDASI GAGAL: Output agen BUKAN JSON strict atau formatnya rusak.")
        print(f"Error: {e}")
        print(f"Output mentah agen:\n{final_output}")

if __name__ == "__main__":
    main()