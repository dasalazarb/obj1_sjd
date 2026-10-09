from pathlib import Path
import base64
from html import escape
import streamlit as st
from shared.loaders.figure_loader import render_pdf
from shared.components.evidence import missing_output, evidence_footer


@st.cache_data(show_spinner=False)
def original_image_uri(run_key,figure_id,data,suffix):
    mime='image/svg+xml' if suffix=='.svg' else 'image/png'
    return 'data:'+mime+';base64,'+base64.b64encode(data).decode()


def image_original(display,suffix,title,run_key,figure_id):
    # Native st.image resizes large assets internally. A raw img URI preserves bytes.
    uri=original_image_uri(run_key,figure_id,display,suffix)
    st.html('<img class="sjd-figure" alt="'+escape(title,quote=True)+'" src="'+uri+'">')


def figure_viewer(figure_id,run,title='',metadata=None):
    entry=run.registry.get(figure_id)
    try:
        if metadata and metadata.get('generated_run') and metadata['generated_run'] not in {run.key,run.run_id}:
            raise ValueError('Figure and table run identities differ')
        data=run.figure_bytes(figure_id)
        suffix=Path(entry.path).suffix.lower()
        st.download_button('Download original figure',data,file_name=Path(entry.path).name,key='download_'+figure_id)
        try:
            display=render_pdf(run.key,data) if suffix=='.pdf' else data
            if suffix=='.png':
                from io import BytesIO
                from PIL import Image
                with Image.open(BytesIO(display)) as original:
                    original.verify() # Integrity check only; no editing or recompression.
            image_original(display,'.png' if suffix=='.pdf' else suffix,title,run.key,figure_id)
            if title: st.caption(title)
            @st.dialog('Figure zoom',width='large')
            def zoom():
                image_original(display,'.png' if suffix=='.pdf' else suffix,title,run.key,figure_id)
            if st.button('Zoom figure',key='zoom_'+figure_id): zoom()
        except Exception as exc:
            missing_output('Figure cannot be rendered; original download is available',str(exc))
    except (ValueError,OSError) as exc:
        missing_output(title or 'Figure not found for this run',entry.path+' · '+str(exc))
    evidence_footer([figure_id+':figure'],run)
