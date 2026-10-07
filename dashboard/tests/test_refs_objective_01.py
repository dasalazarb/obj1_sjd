"""Reference plumbing tests on MOCK cells; historical export validation is separate."""
import pytest
import yaml
from shared.loaders.refs import resolve,metadata_counts
from shared.loaders.registry import MissingOutput,SchemaError
from shared.utils.paths import APP_ROOT
from scripts.validate_export import configured_refs


def test_every_configured_ref_on_mock_cells(mock_run):
    cfg=yaml.safe_load((APP_ROOT/'objectives/objective_01/config.yml').read_text())
    refs=list(configured_refs(cfg))
    assert len(refs)>100
    for ref in refs:
        value=resolve(ref,mock_run)
        assert value.source.run_key==mock_run.key
        assert value.kind!='failure',ref


def test_no_positional_or_fuzzy_selection(mock_run):
    with pytest.raises(MissingOutput): resolve('t11_overall:Disease activity| ESSDAI@Summary#raw',mock_run)


def test_missing_source_is_not_zero(mock_run):
    mock_run.tables.pop('t11_overall')
    with pytest.raises(MissingOutput): resolve('t11_overall:Cohort / demographics|N patients@Summary#raw',mock_run)


def test_availability_consistency(mock_run):
    frame=mock_run.tables['t11_overall']; frame.loc[frame.Variable.eq('essdai_total'),'N available']='99'
    with pytest.raises(SchemaError): resolve('t11_overall:Disease activity|essdai_total@Summary#n',mock_run)


def test_raw_failure_and_na_preserved(mock_run):
    frame=mock_run.tables['t11_overall']; frame.loc[frame.Variable.eq('essdai_total'),'Summary']='malformed MOCK'
    value=resolve('t11_overall:Disease activity|essdai_total@Summary#median',mock_run)
    assert value.display=='malformed MOCK' and value.warning
    frame.loc[frame.Variable.eq('essdai_total'),'Summary']='NA'
    assert resolve('t11_overall:Disease activity|essdai_total@Summary#median',mock_run).display=='NA'


def test_metadata_only_count(mock_run): assert metadata_counts(mock_run)==(1,1)
