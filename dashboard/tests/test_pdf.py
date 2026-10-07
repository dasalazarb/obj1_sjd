from io import BytesIO
from PIL import Image
import pymupdf
from shared.loaders import figure_loader


def test_pdf_first_page_render_cache_is_local(tmp_path,monkeypatch):
    monkeypatch.setattr(figure_loader,'APP_ROOT',tmp_path)
    figure_loader.render_pdf.clear()
    document=pymupdf.open();page=document.new_page(width=144,height=72)
    page.insert_text((10,30),'MOCK — NOT RESULTS')
    document.new_page(width=72,height=72)
    original=document.tobytes();document.close()
    rendered=figure_loader.render_pdf('MOCK_run',original)
    with Image.open(BytesIO(rendered)) as image:
        assert image.size==(400,200) # First page, 200 dpi; no scientific data involved.
    cache=list((tmp_path/'.cache/figures/MOCK_run').glob('*.png'))
    assert len(cache)==1 and cache[0].read_bytes()==rendered
