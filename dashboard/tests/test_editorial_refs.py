"""Software-only MOCK checks for scientific-cell traceability in editorial text."""
from shared.components.evidence import template_html
from shared.loaders.refs import resolve


def test_editorial_numbers_keep_cell_refs_and_source_tooltips(mock_run):
    ref='t11_overall:Cohort / demographics|N patients@Summary#raw'
    html=template_html('Patients: {{ref:'+ref+'}}. <unsafe>',mock_run)
    assert resolve(ref,mock_run).display in html
    assert 'data-ref="'+ref+'"' in html
    assert '11_table1_overall.csv' in html
    assert mock_run.key in html
    assert '&lt;unsafe&gt;' in html


def test_malformed_editorial_cell_shows_raw_warning(mock_run):
    frame=mock_run.tables['t11_overall']
    frame.loc[frame.Variable.eq('N patients'),'Summary']='MOCK unrecognized'
    ref='t11_overall:Cohort / demographics|N patients@Summary#raw'
    html=template_html('{{ref:'+ref+'}}',mock_run)
    assert 'MOCK unrecognized' in html
    assert 'Unrecognized source-cell format' in html
