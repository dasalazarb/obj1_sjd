import re
from shared.utils.paths import APP_ROOT


def test_editorial_text_has_no_result_literals():
    paths=list((APP_ROOT/'objectives').rglob('narrative.md'))+list((APP_ROOT/'objectives').rglob('claims.yml'))
    for path in paths:
        text=path.read_text()
        text=re.sub(r'\{\{ref:.*?\}\}|t\d+_[^\n]+|SF-36|Pop[1-3]|Pop1–3','',text)
        assert not re.search(r'\b\d+\.\d+\b|\b[1-9]\d+\b',text),path
