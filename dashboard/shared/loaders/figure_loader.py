import hashlib
from pathlib import Path
import streamlit as st
from shared.utils.paths import APP_ROOT


@st.cache_resource(show_spinner=False)
def render_pdf(run_key, content):
    import pymupdf
    with pymupdf.open(stream=content, filetype="pdf") as document:
        if not len(document):
            raise ValueError("Empty PDF")
        png = document[0].get_pixmap(dpi=200).tobytes("png")
    directory = APP_ROOT / ".cache/figures" / run_key
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (hashlib.sha256(content).hexdigest() + ".png")).write_bytes(png)
    return png
