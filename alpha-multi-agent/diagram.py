from multi_agent import alpha_agent_app

def main():
    print("Menggambar arsitektur DAG Tim 3...")
    
    try:
        # Mengambil graf dan merendernya menjadi data gambar PNG via API Mermaid
        gambar_png = alpha_agent_app.get_graph().draw_mermaid_png()

        # Menyimpan data tersebut ke dalam file fisik
        with open("arsitektur_dag_tim3.png", "wb") as f:
            f.write(gambar_png)

        print("✅ Diagram berhasil disimpan sebagai 'arsitektur_dag_tim3.png' di folder ini!")
        
    except Exception as e:
        print(f"❌ Gagal membuat diagram. Pastikan koneksi internet aktif karena draw_mermaid_png butuh akses API eksternal.")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()