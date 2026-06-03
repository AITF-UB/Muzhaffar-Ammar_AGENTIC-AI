import json
import os
import re
from collections import OrderedDict

# Path folder utama berdasarkan struktur proyek kamu
target_root = r"C:\Local D\Galeri Belajar\Project\SR_02\data_prep_for_sft_v3\data\chunk"

def normalize_key(text):
    """Membersihkan string agar pencocokan key lebih akurat."""
    if not text: return ""
    text = text.strip().lower()
    text = re.sub(r'\.+$', '', text) # Hapus titik di akhir
    text = re.sub(r'\s+', ' ', text) # Normalisasi spasi ganda
    return text

# Mapping ATP - Awalan 'Mampu' telah dihapus agar langsung ke kata kerja
atp_map = {
    # --- MATEMATIKA UMUM ---
    "definisi eksponen": "Menyatakan perkalian bilangan bulat berulang sebagai bilangan berpangkat (eksponen).",
    "sifat-sifat eksponen": "Menggeneralisasi sifat-sifat bilangan berpangkat.",
    "fungsi eksponensial": "Menyelesaikan persamaan eksponensial dengan bilangan pokok yang sama serta mengidentifikasi dan menjelaskan makna grafik fungsi eksponensial (naik atau turun).",
    "sistem persamaan linear": "Menentukan perbedaan antara persamaan dan pertidaksamaan linear dua variabel, menyajikan grafik, serta menentukan daerah penyelesaiannya.",
    "sistem pertidaksamaan linear": "Menentukan perbedaan antara persamaan dan pertidaksamaan linear dua variabel, menyajikan grafik, serta menentukan daerah penyelesaian pertidaksamaan linear.",
    "persamaan kuadrat": "Menentukan akar-akar, menganalisis hubungan diskriminan, serta menyusun persamaan kuadrat.",
    "fungsi kuadrat": "Membedakan fungsi dan bukan fungsi persamaan kuadrat serta menganalisis karakteristik grafiknya.",
    "perbandingan trigonometri": "Mengidentifikasi hubungan sudut dan sisi segitiga siku-siku, menjelaskan definisi perbandingan trigonometri sudut lancip, serta menggunakan hubungan sinus dan cosinus.",
    "pemanfaatan perbandingan trigonometri": "Menggunakan perbandingan trigonometri untuk menyelesaikan masalah kontekstual yang melibatkan segitiga siku-siku dan sudut lancip.",
    "statistik deskriptif": "Menentukan jangkauan kuartil dan interkuartil dari data.",
    "representasi data lanjutan": "Merepresentasikan data dalam bentuk boxplot, membandingkan data, membaca berbagai jenis diagram (histogram, dot plot, pencar), serta mengevaluasi grafik di media.",

    # --- BAHASA INDONESIA ---
    "membandingkan informasi yang akurat dalam laporan hasil observasi": "Membandingkan temuan informasi dari dua media laporan hasil observasi visual yang berbeda tentang satu objek yang sama.",
    "mengidentifikasi makna kata dan informasi faktual dalam laporan hasil observasi dan sumber lainnya yang mendukung": "Menemukan kosakata sulit/istilah khusus, menentukan arti, serta mengidentifikasi informasi faktual dan membedakan fakta dengan opini.",
    "menggunakan kaidah kebahasaan dalam laporan hasil observasi": "Menulis teks laporan hasil observasi secara logis, kritis, dan kreatif dengan memperhatikan unsur kebahasaan teks LHO.",
    "menulis laporan hasil observasi yang objektif": "Menulis teks laporan hasil observasi secara logis, kritis, dan kreatif dengan memperhatikan struktur teks LHO.",
    "menyajikan laporan hasil observasi dalam bentuk buku tempel": "Mengevaluasi gagasan utama dan pesan dari teks laporan hasil observasi visual yang ditempel di dinding kelas.",
    "mempresentasikan laporan hasil observasi": "Menyajikan hasil observasi dalam bentuk teks laporan hasil observasi secara lisan dengan memperhatikan struktur teks, intonasi, pelafalan, dan gestur secara tepat.",
    "mengevaluasi pesan dari menyimak teks monolog lawakan tunggal": "Menentukan dan mengevaluasi masalah, pendapat (tesis), serta bukti pendukung tersirat dalam teks aural.",
    "menentukan struktur teks dari menyimak monolog lawakan tunggal dan/atau anekdot": "Mengevaluasi struktur teks (tesis, rangkaian argumen, dan penegasan ulang) dan ciri kebahasaan secara akurat.",
    "menulis teks eksposisi hasil penelitian sederhana sebagai bahan untuk menyampaikan kritik sosial": "Menulis gagasan dan pandangan dalam teks eksposisi secara logis dengan mengembangkan kerangka menjadi teks utuh minimal empat paragraf.",

    # --- IPS ---
    "kajian ilmu sejarah": "Menjelaskan sejarah, tokoh, dan konsep dasar Ilmu Sejarah serta menerapkan cara berpikir kronologis, diakronis, dan sinkronis.",
    "kajian geografi": "Menjelaskan sejarah, tokoh, dan konsep dasar Geografi serta menerapkan pendekatan Geografi untuk menganalisis fenomena geosfer.",
    "penelitian sosial suatu pengantar": "Menjelaskan definisi, fungsi, tahapan, dan etika dalam penelitian sosial.",
    "kekhasan penelitian sejarah": "Memahami kekhasan penelitian Sejarah yang meliputi heuristik, verifikasi, interpretasi, dan historiografi.",
    "kekhasan penelitian geografi": "Memahami kekhasan penelitian Geografi (peta, penginderaan jauh, SIG) dan merancang penelitian sederhana berbasis spasial.",
    "kehidupan masyarakat pada masa kerajaan hindu–buddha": "Menganalisis proses masuknya pengaruh Hindu-Buddha serta mengidentifikasi corak kehidupan dan peninggalan budayanya di Nusantara.",
    "kehidupan masyarakat pada masa kerajaan islam": "Menganalisis proses masuknya pengaruh Islam serta mengidentifikasi corak kehidupan dan peninggalan budaya hasil akulturasi.",
    "lingkungan geosfer fisikal indonesia: litosfer": "Menjelaskan karakteristik litosfer serta dampak tektonisme dan vulkanisme bagi kehidupan.",
    "lingkungan geosfer fisikal indonesia: atmosfer": "Menjelaskan karakteristik atmosfer serta pengaruh iklim dan cuaca bagi kehidupan.",
    "lingkungan geosfer fisikal indonesia: hidrosfer": "Menjelaskan karakteristik hidrosfer serta potensi dan pemanfaatan perairan darat dan laut.",
    "struktur sosial dalam masyarakat": "Menjelaskan konsep stratifikasi sosial, status, peran, dan mobilitas sosial.",
    "ragam gejala sosial dalam masyarakat": "Mengidentifikasi dan menganalisis berbagai gejala sosial yang ada di dalam masyarakat.",
    "diferensiasi sosial budaya": "Menjelaskan konsep diferensiasi sosial serta menganalisis keragaman suku bangsa, religi, gender, dan profesi di Indonesia.",
    "masyarakat, pasar, dan terbentuknya harga pasar": "Menjelaskan mekanisme permintaan, penawaran, dan terbentuknya harga keseimbangan di pasar.",
    "masyarakat dan peran lembaga keuangan": "Membedakan peran lembaga keuangan bank/OJK serta mengidentifikasi produk Industri Keuangan Non-Bank (IKNB) dan ekonomi digital."
}

def process_files():
    if not os.path.exists(target_root):
        print(f"Folder tidak ditemukan: {target_root}")
        return

    updated_count = 0
    for root, _, files in os.walk(target_root):
        for filename in files:
            if filename.endswith(".json"):
                file_path = os.path.join(root, filename)
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except: continue

                modified = False
                for item in data:
                    # 1. Update ATP (dan pastikan urutan: sub_bab -> atp)
                    if "query" in item and "sub_bab" in item["query"]:
                        norm_name = normalize_key(item["query"]["sub_bab"])
                        if norm_name in atp_map:
                            # Reconstruct dictionary to control key order
                            new_query = OrderedDict()
                            for k, v in item["query"].items():
                                new_query[k] = v
                                if k == "sub_bab":
                                    new_query["atp"] = atp_map[norm_name]
                            item["query"] = dict(new_query)
                            modified = True
                    
                    # 2. Hapus id di dalam chunks
                    if "chunks" in item:
                        for chunk in item["chunks"]:
                            if "id" in chunk:
                                del chunk["id"]
                                modified = True
                
                if modified:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print(f"Berhasil memperbarui: {os.path.relpath(file_path, target_root)}")
                    updated_count += 1

    print(f"\nSelesai! Total {updated_count} file JSON telah diproses.")

if __name__ == "__main__":
    process_files()