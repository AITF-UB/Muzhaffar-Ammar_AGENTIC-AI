function renderQuizPG(content) {
    const container = document.getElementById("view-quiz-pg");
    container.innerHTML = "";
    
    content.soal.forEach((q, idx) => {
        let imgHtml = q.image_path ? `<img src="${q.image_path}" alt="Konteks Soal">` : "";
        
        let optsHtml = q.pilihan.map((opt, i) => `
            <div class="quiz-opt" onclick="this.parentElement.querySelectorAll('.quiz-opt').forEach(el=>el.classList.remove('selected')); this.classList.add('selected'); document.getElementById('expl-${q.id}').style.display='block';">
                <strong>${String.fromCharCode(65+i)}.</strong> ${opt}
            </div>
        `).join('');

        container.innerHTML += `
            <div class="quiz-card">
                <div class="quiz-q">${idx+1}. ${q.soal}</div>
                ${imgHtml}
                <div class="opts-container">${optsHtml}</div>
                <div class="quiz-explanation" id="expl-${q.id}">
                    <strong>Kunci: ${String.fromCharCode(65+q.jawaban)}</strong><br>
                    ${q.penjelasan}
                </div>
            </div>
        `;
    });
    
    container.classList.add("active");
}
