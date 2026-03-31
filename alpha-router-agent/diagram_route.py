from langchain_core.runnables.graph import MermaidDrawMethod
from router_agent import router_agent_app

def main():
    print("Menggambar arsitektur DAG Tim 3...")
    
    try:
        # Karena API Mermaid di-block dan Pyppeteer butuh install library, 
        # kita simpan Graph ke dalam bentuk teks Mermaid mentah (.mmd)
        diagram_text = router_agent_app.get_graph().draw_mermaid()

        # Menyimpan data tersebut ke dalam file fisik
        with open("arsitektur_route_tim3.mmd", "w", encoding="utf-8") as f:
            f.write(diagram_text)

        print("✅ Diagram Mermaid mentah berhasil disimpan sebagai 'arsitektur_route_tim3.mmd'!")
        print("💡 Silahkan copy isi file tersebut dan paste ke https://mermaid.live untuk melihat grafiknya.")
        
    except Exception as e:
        print(f"❌ Gagal membuat diagram. Pastikan koneksi internet aktif karena draw_mermaid_png butuh akses API eksternal.")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()