import importlib.util
from pathlib import Path
import pandas as pd
import pytest

p=Path(__file__).resolve().parents[1]/"src/block_A/08_merge_medication_documentation.py"
spec=importlib.util.spec_from_file_location("merge_med",p)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def sample():
    a=pd.DataFrame([dict(audit_record_id="11D:0",source="11D",
      linked_patient_pseudonymous="patientA",drug_raw="Plaquenil",ingredient_raw="")])
    b=pd.DataFrame([dict(audit_record_id="11D:0",source="11D",
      linked_patient_pseudonymous="patientA",date_of_documentation="2021-02-03",
      documentation_window_lower_bound="2021-02-03",
      documentation_window_upper_bound="2021-02-03",
      documentation_window_precision="day",
      source_start_lower_bound="2021-02-03",
      source_start_upper_bound="2021-02-03",
      source_start_precision="day",
      normalized_dated_usable=True,documentation_dated_usable=True,
      drug_short_label="Hydroxychloroquine",ingredient_canonical="hydroxychloroquine",
      clinical_category_primary="systemic_sjd",clinical_category_detail="sjd_systemic_candidate",
      classification_confidence="high_exact",study_category_rule_status="candidate_not_adjudicated",
      route_canonical="oral",classification_review_reasons="",qc_flags="")])
    c=pd.DataFrame([dict(patient_pseudonym="patientA",clinical_episode_id="ep1",
      row_pseudonym="row1",slot=1,medication_name_raw="Restasis",
      visit_documentation_date="2021-02-01",medication_date_within_episode=True,
      classification_confidence="high_exact",clinical_category_primary="symptomatic_sjd",
      ingredient_canonical="cyclosporine",clinical_category_detail="sjd_ocular_symptomatic_candidate",
      study_category_rule_status="candidate_not_adjudicated",route_canonical="ophthalmic",
      clinical_visit=True,episode_start_date="2021-01-01",
      episode_end_date="2021-03-31",clinical_anchor_date="2021-02-01",
      source_reported_start_lower="",source_reported_start_upper="",
      source_reported_start_precision="")])
    return b,a,c,dict(version="08-medication-documentation-release-1.1.0",
      source_classification_version="08-classification-2.1.10",n_documented_named=2),dict(
      version="08-ctdb-visit-1.1",n_original_visit_medication_mentions=1)

def test_source_grains_and_no_exposure_inference():
    b,a,c,rs,cs=sample()
    rs["n_documented_named"]=1
    allrows,dated,normalized,qc,s=mod.build(b,a,c,rs,cs)
    assert s["n_total_documentation_records"]==2
    assert s["n_btris_records_with_episode_assignment"]==0
    assert s["n_ctdb_authoritatively_linked_episodes"]==1
    assert len(dated)==len(normalized)==2
    assert not allrows.actual_exposure_confirmed.any()
    assert set(allrows.source_kind)=={"BTRIS_order","CTDB_reported_current"}

def test_reject_source_mismatch():
    b,a,c,rs,cs=sample()
    a.loc[0,"linked_patient_pseudonymous"]="wrong"
    rs["n_documented_named"]=1
    with pytest.raises(ValueError,match="patient_identity_disagreement"):
        mod.build(b,a,c,rs,cs)

def test_no_duplicate_ctdb_original_slots():
    b,a,c,rs,cs=sample()
    rs["n_documented_named"]=1
    with pytest.raises(ValueError,match="duplicate_original_visit_slot"):
        mod.build(b,a,pd.concat([c,c]),rs,{**cs,"n_original_visit_medication_mentions":2})
