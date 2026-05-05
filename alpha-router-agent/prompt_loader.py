"""
prompt_loader.py — Jinja2 Template Loader untuk Router Agent
    from prompt_loader import render_prompt

    sys_p = render_prompt("mindmap.j2", role="system",
                          matpel=matpel, materi=materi, jenjang=jenjang)
    usr_p = render_prompt("mindmap.j2", role="user",
                          matpel=matpel, materi=materi, jenjang=jenjang,
                          elemen=elemen, atp_str=atp_str, konteks=ctx,
                          kelas=kelas)

TEMPLATE TERSEDIA:
    bacaan.j2          → node bacaan (butuh: level, level_instruksi, konteks, ...)
    flashcard.j2       → node flashcard (butuh: level, level_instruksi, sumber_text, ...)
    mindmap.j2         → node mindmap (IMPROVED: struktur radial)
    quiz.j2            → node quiz PG (butuh: level, level_instruksi, sumber_text, ...)
    quiz_uraian.j2     → node quiz uraian (butuh: level, level_instruksi, sumber_text, ...)
    evaluasi_uraian.j2 → node evaluasi jawaban (butuh: pertanyaan, kunci, jawaban_siswa, skor_maks)
    recommender.j2     → node rekomendasi (butuh: first_time, matpel_dipilih, konteks)
"""

import os
from jinja2 import Environment, FileSystemLoader, StrictUndefined

# Resolve path template relatif terhadap file ini
_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")

_env = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    undefined=StrictUndefined,   # error jelas kalau variable hilang
    trim_blocks=True,            # hapus newline setelah {% block %}
    lstrip_blocks=True,          # hapus indent sebelum {% block %}
)


def render_prompt(template_name: str, **kwargs) -> str:
    """
    Render Jinja2 template dan return string prompt yang bersih.

    Args:
        template_name: nama file template (misal "mindmap.j2")
        **kwargs: variabel yang diinjeksi ke template (wajib include 'role')

    Returns:
        String prompt yang sudah di-render, whitespace dibersihkan.

    Raises:
        UndefinedError: jika variable yang dibutuhkan template tidak diberikan
        TemplateNotFound: jika nama template salah
    """
    template = _env.get_template(template_name)
    rendered = template.render(**kwargs)
    return rendered.strip()


def render_system(template_name: str, **kwargs) -> str:
    """Shortcut: render dengan role='system'."""
    return render_prompt(template_name, role="system", **kwargs)


def render_user(template_name: str, **kwargs) -> str:
    """Shortcut: render dengan role='user'."""
    return render_prompt(template_name, role="user", **kwargs)


# ================================================================
# Quick test — python prompt_loader.py
# ================================================================
if __name__ == "__main__":
    ctx_dummy = {
        "matpel":         "Fisika",
        "materi":         "Hukum Newton",
        "jenjang":        "10",
        "kelas":          "10A",
        "elemen":         "Mekanika",
        "atp_str":        "1. Menjelaskan Hukum Newton I, II, III\n2. Menerapkan F=ma",
        "konteks":        "[dummy konteks dari RAG]",
        "level":          "LOTS",
        "level_instruksi":"Fokus pada mengingat dan memahami.",
        "sumber_text":    "Buku Fisika Kelas 10",
    }

    print("=" * 60)
    print("MINDMAP SYSTEM PROMPT:")
    print("=" * 60)
    print(render_system("mindmap.j2",
                        matpel=ctx_dummy["matpel"],
                        materi=ctx_dummy["materi"],
                        jenjang=ctx_dummy["jenjang"]))

    print("\n" + "=" * 60)
    print("BACAAN SYSTEM PROMPT (LOTS):")
    print("=" * 60)
    print(render_system("bacaan.j2", **ctx_dummy))

    print("\n✅ Template loader berfungsi dengan benar.")
