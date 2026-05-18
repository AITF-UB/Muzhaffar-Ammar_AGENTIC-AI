let lastGeneratedData = null; // Menyimpan data terakhir yang digenerate untuk keperluan evaluasi essay

document.getElementById("inp-tipe").addEventListener("change", (e) => {
    const levelGroup = document.getElementById("group-level");
    // Sembunyikan pilihan level jika tipe = mindmap
    levelGroup.style.display = e.target.value === "mindmap" ? "none" : "block";
});

async function handleGenerate() {
    const btn = document.getElementById("btn-gen");
    const loading = document.getElementById("loading");
    
    // 1. Ambil data dari form
    const tipe = document.getElementById("inp-tipe").value;
    const payload = {
        mapel_id: document.getElementById("inp-mapel").value,
        elemen_id: document.getElementById("inp-elemen-id").value,
        elemen_label: document.getElementById("inp-elemen-label").value,
        materi: document.getElementById("inp-materi").value,
        atp: document.getElementById("inp-atp").value,
        jenjang: "X",
        tipe: tipe
    };
    if (tipe !== "mindmap") {
        payload.level = document.getElementById("inp-level").value;
    }

    // 2. Reset UI
    document.querySelectorAll(".view-container").forEach(el => el.classList.remove("active"));
    document.getElementById("debug-json").style.display = "none";
    btn.disabled = true;
    loading.style.display = "flex";

    try {
        // 3. Panggil API Utama Beta Agentic
        const response = await fetch("http://localhost:8000/konten/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        
        const json = await response.json();
        console.log("Raw Response:", json);
        
        if (json.error) {
            alert("Error: " + json.error.message);
            return;
        }

        const data = json.data;
        const content = data.content;
        lastGeneratedData = data;

        // 4. Tampilkan RAW JSON untuk keperluan debug
        document.getElementById("debug-json").textContent = JSON.stringify(data, null, 2);
        document.getElementById("debug-json").style.display = "block";

        // 5. Render konten menggunakan script modular spesifik
        if (tipe === "bacaan") renderBacaan(content, data.source);
        else if (tipe === "quiz_pg") renderQuizPG(content);
        else if (tipe === "quiz_essay") renderQuizEssay(content);
        else if (tipe === "flashcard") renderFlashcard(content);
        else if (tipe === "mindmap") renderMindmap(content);

    } catch (err) {
        alert("Network error: " + err.message);
    } finally {
        btn.disabled = false;
        loading.style.display = "none";
    }
}
