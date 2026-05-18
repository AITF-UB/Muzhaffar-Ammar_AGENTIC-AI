function renderBacaan(content, source) {
    // 1. Parsing Markdown menjadi HTML
    const html = marked.parse(content.text);
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
