function renderBacaan(content, source) {
    // 1. Parsing Markdown menjadi HTML dan memperbaiki URL gambar
    let mdText = content.text;
    mdText = mdText.replace(/!\[(.*?)\]\((?!http)(.*?)\)/g, "![$1](http://localhost:8000/extraction/$2)");
    const html = marked.parse(mdText);
    document.getElementById("bacaan-html").innerHTML = html;
    
    // 2. Format Sumber Referensi (Array of Strings)
    let srcHtml = "<strong>Sumber Referensi:</strong><br>";
    if (source && source.length > 0) {
        source.forEach(s => { srcHtml += `- ${s}<br>`; });
    } else {
        srcHtml += "- (Generated without DB Source)";
    }
    
    document.getElementById("bacaan-source").innerHTML = srcHtml;
    
    // 3. Tampilkan kontainer
    document.getElementById("view-bacaan").classList.add("active");
}
