"""Synthetic-only contracts for Step 08 v2; no clinical inputs are needed."""
import importlib.util
import json
import stat
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "src/block_A/08_treatment_audit_v2.py"
SPEC = importlib.util.spec_from_file_location("treatment_audit_v2", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
SECRET = b"synthetic-test-pseudonym-key-only!"


def btris(**overrides):
    values = {c: None for c in audit.BTRIS_MEDICATION_COLUMNS}
    values.update({"Protocol Number": "11-D-0172", "MRN": "00123", "Order ID": "order-1",
                   "Medication Name": "Plaquenil", "Drug Ingredient": "hydroxychloroquine",
                   "Order Route": "PO", "Order Date/Start Date": "06/08/2015", "Order Status": "ACTIVE",
                   "Order Type": "outpatient", "Data Source": "synthetic",
                   "Order Dose/Strength": "200", "Order UOM": "mg", "Order Frequency": "daily"})
    values.update(overrides)
    return values


def ctdb(patient="p1", **values):
    return {audit.CTDB_PATIENT_ID: patient, audit.CTDB_VISIT_DATE: "2015-06-08", **values}


def complete_ctdb(rows):
    return pd.DataFrame(rows).reindex(columns=[audit.CTDB_PATIENT_ID, audit.CTDB_VISIT_DATE, *audit.CTDB_COLUMNS])


def link(eligible=None, pairs=None):
    return audit.validate_linkage(pd.DataFrame(pairs or {"MRN": ["00123"], "patient_id": ["p1"]}),
                                  set(eligible or ["p1"]), authorized=True)


def records(rows, source="11D", linkage=None, eligible=None):
    return audit.records_frame(audit.btris_records(pd.DataFrame(rows), source, linkage or link(),
                                                  set(eligible or ["p1"]), SECRET))


def args_for(tmp_path, *, crosswalk=True):
    root = tmp_path / "inputs"
    root.mkdir()
    complete_ctdb([ctdb(**{"medications__rx_1_name": "HCQ", "medications__med1_strt_date": "06/2015",
                          "past_medical_history__rx_1_name": "Latex",
                          "past_medical_history__rx_allergies": "Latex"})]).to_parquet(root / "ctdb.parquet")
    pd.DataFrame([btris()]).to_parquet(root / "11d.parquet")
    pd.DataFrame([btris(**{"Protocol Number": "15-D-0051", "MRN": "00456", "Order ID": "order-2"})]).to_parquet(root / "15d.parquet")
    pd.DataFrame({"patient_id": ["p1", "p2", "p3", "p4"]}).to_csv(root / "eligible.csv", index=False)
    pd.DataFrame({"MRN": ["00123", "00456", "00789", "00999"], "patient_id": ["p1", "p2", "p3", "p4"]}).to_csv(root / "crosswalk.csv", index=False)
    manifest = root / "crosswalk.json"
    manifest.write_text(json.dumps(dict(path="crosswalk.csv", mrn_column="MRN", patient_column="patient_id",
                                        authorization_evidence="Synthetic test authorization, not real patient linkage")))
    return Namespace(input=root / "ctdb.parquet", eligibility=root / "eligible.csv", btris_11d=root / "11d.parquet",
                     btris_15d=root / "15d.parquet", crosswalk_manifest=manifest if crosswalk else None,
                     ctdb_protocol_column=None, ctdb_index_column=None, previous_summary=None,
                     pseudonym_key_file=None, output_dir=tmp_path / "audit-v2")


def table(args, name):
    return pd.read_csv(args.output_dir / "restricted" / name)


@pytest.mark.parametrize("value,parsed,precision,lower,upper,rule", [
    ("06/08/2015", "2015-06-08", "day", "2015-06-08", "2015-06-08", "MM/DD/YYYY"),
    ("03/27/2015", "2015-03-27", "day", "2015-03-27", "2015-03-27", "MM/DD/YYYY"),
    ("06/2015", "", "month", "2015-06-01", "2015-06-30", "MM/YYYY"),
    ("02/2020", "", "month", "2020-02-01", "2020-02-29", "MM/YYYY"),
    ("2015", "", "year", "2015-01-01", "2015-12-31", "YYYY"),
    ("2015-06-08", "2015-06-08", "day", "2015-06-08", "2015-06-08", "YYYY-MM-DD"),
])
def test_strict_dates_preserve_precision(value, parsed, precision, lower, upper, rule):
    assert audit.parse_date(value) == dict(raw=value, parsed=parsed, precision=precision,
                                          lower_bound=lower, upper_bound=upper, parse_rule=rule, status="valid")


@pytest.mark.parametrize("value", ["02/30/2015", "2015-02-29", "13/2015", "0000", "99/99/2015"])
def test_invalid_calendar_dates_are_not_silently_coerced(value):
    assert audit.parse_date(value)["status"] == "invalid_date"


def test_timezone_is_preserved_and_unapproved_slash_format_blocks():
    for value in ["2015-06-08T09:30:00-04:00", datetime(2015, 6, 8, 9, 30, tzinfo=timezone.utc)]:
        result = audit.parse_date(value)
        assert result["precision"] == "datetime"
        assert result["parsed"].endswith(("-04:00", "+00:00"))
    assert audit.parse_date("06/08/2015", slash_order="unresolved")["status"] == "ambiguous_format"


def test_month_day_comparison_uses_bounds_and_never_fabricates_observed_day():
    month = audit.parse_date("06/2015")
    assert not month["parsed"]
    assert audit.compare_dates(month, audit.parse_date("06/08/2015")) == "compatible"
    assert audit.compare_dates(month, audit.parse_date("07/08/2015")) == "incompatible"
    assert audit.compare_dates(month, audit.parse_date(None)) == "indeterminate"


def test_all_28_literal_headers_and_no_silent_alias(tmp_path):
    assert len(audit.BTRIS_MEDICATION_COLUMNS) == 28
    path = tmp_path / "source.parquet"
    data = pd.DataFrame([btris()]).rename(columns={"Order Dose/Strength": "Order Dose Strength"})
    data.to_parquet(path)
    frame, info = audit.load_source(path, audit.BTRIS_MEDICATION_COLUMNS, parquet_only=True)
    assert info["missing_columns"] == ["Order Dose/Strength"]
    assert info["unexpected_columns"] == ["Order Dose Strength"]
    assert info["n_row_groups"] == 1 and info["metadata_rows"] == 1
    assert info["physical_schema"]["MRN"] == "BYTE_ARRAY"
    schema = pd.DataFrame(audit.schema_audit(frame, info, audit.BTRIS_MEDICATION_COLUMNS, "11D")).set_index("column")
    assert not schema.loc["Order Dose/Strength", "column_present"]
    assert schema.loc["Order Dose Strength", "schema_status"] == "unexpected_column"
    record = records([data.iloc[0].to_dict()]).iloc[0]
    assert record.dose_raw == ""


def test_missing_and_corrupt_sources_are_not_zero_patients(tmp_path):
    for path in [tmp_path / "missing.parquet", tmp_path / "corrupt.parquet"]:
        if path.name == "corrupt.parquet":
            path.write_bytes(b"not a parquet file")
        frame, info = audit.load_source(path, audit.BTRIS_MEDICATION_COLUMNS, parquet_only=True)
        assert frame is None and info["n_rows"] is None
        assert info["status"] == "BLOCKED_SOURCE_UNAVAILABLE"


def test_null_blank_token_and_absent_header_are_distinguished(tmp_path):
    path = tmp_path / "schema.parquet"
    data = pd.DataFrame([btris(**{"Order Status": v}) for v in [None, "", "unknown", "cancelled"]])
    data.to_parquet(path)
    frame, info = audit.load_source(path, audit.BTRIS_MEDICATION_COLUMNS)
    schema = pd.DataFrame(audit.schema_audit(frame, info, audit.BTRIS_MEDICATION_COLUMNS, "11D")).set_index("column")
    row = schema.loc["Order Status"]
    assert row.n_null == 1 and row.n_empty == 1 and row.n_missing_tokens == 3 and row.pct_null == 25


def test_identity_preserves_zeroes_and_never_links_using_names():
    data = records([btris(), btris(**{"MRN": None, "Subject Name": "Same Person"}), btris(**{"MRN": 123})])
    assert data.eligible_linked.tolist() == [True, False, False]
    assert data.linkage_status.tolist() == ["linked_one_to_one", "invalid_key", "nontext_unverified"]
    assert data.iloc[0].source_patient_pseudonymous == audit.pseudonym(SECRET, "mrn", "00123")
    assert audit.key("00123", mrn=True) != audit.key("123", mrn=True)
    assert not {"MRN", "Subject Name", "Date of Birth"} & set(data.columns)


def test_non_one_to_one_crosswalk_excludes_every_conflicted_edge():
    result = audit.validate_linkage(pd.DataFrame({"MRN": ["001", "001", "002", "003", "004"],
                                                 "patient_id": ["a", "b", "b", "c", "c"]}),
                                    {"a", "b", "c"}, authorized=True)
    assert not result["mapping"] and result["status"] == "blocked_linkage"
    assert result["ambiguous_mrns"] == {"001", "002", "003", "004"}
    assert audit.validate_linkage(pd.DataFrame({"MRN": ["00123"], "patient_id": ["p1"]}), {"p1"})["status"] == "blocked_linkage"


def test_duplicate_pairs_allowed_but_missing_eligible_crosswalk_blocks_neither():
    result = link(pairs={"MRN": ["00123", "00123"], "patient_id": ["p1", "p1"]})
    assert result["status"] == "validated" and result["n_crosswalk_pairs"] == 1
    assert link(eligible=["p1", "p2"])["status"] == "blocked_linkage"


def test_orders_statuses_reexports_revisions_and_missing_ids_remain_distinct():
    active = btris()
    rows = [active, active, btris(**{"Order Status": "CANCELLED"}), btris(**{"Order ID": "order-2", "Order Status": "COMPLETED"}),
            btris(**{"Order ID": None, "Order Status": "NEW_UNKNOWN_VALUE"})]
    result = audit.duplicate_qc(records(rows))
    assert len(result) == 5
    assert result.iloc[0].qc_flags.find("exact_reexport_duplicate_retained") >= 0
    assert "same_order_id_revision_or_discordance_retained" in result.iloc[2].qc_flags
    assert "same_order_id_revision_or_discordance_retained" not in result.iloc[3].qc_flags
    assert result.order_semantic_class.eq("unvalidated_observed_value").all()
    dictionary = audit.status_dictionary({"11D": pd.DataFrame(rows)})
    assert set(dictionary[dictionary.field.eq("Order Status")].observed_value) == {"ACTIVE", "CANCELLED", "COMPLETED", "NEW_UNKNOWN_VALUE"}
    assert dictionary.needs_validation.all()


def test_order_id_is_not_globally_unique_and_cross_protocol_is_not_two_patients():
    left = records([btris(), btris(**{"MRN": "00999", "Order Status": "CANCELLED"})])
    right = records([btris(**{"Protocol Number": "15-D-0051"})], source="15D")
    result = audit.duplicate_qc(pd.concat([left, right], ignore_index=True))
    assert "cross_protocol_equal_except_protocol" in result.iloc[0].qc_flags
    assert "same_order_id_revision_or_discordance_retained" not in result.iloc[0].qc_flags
    assert "cross_protocol_possible_duplicate_retained" not in result.iloc[1].qc_flags
    assert result[result.eligible_linked].linked_patient_pseudonymous.nunique() == 1


@pytest.mark.parametrize("drug,ingredient,route,normalized,drug_class", [
    ("Plaquenil", "", "PO", "hydroxychloroquine", "nonbiologic_systemic"),
    ("Evoxac", "", "PO", "cevimeline", "symptomatic_sjd_oral"),
    ("Restasis", "cyclosporine", "ophthalmic", "cyclosporine ophthalmic", "symptomatic_sjd_ocular"),
    ("Neoral", "cyclosporine", "oral", "cyclosporine", "nonbiologic_systemic"),
    ("Myfortic", "", "PO", "mycophenolic acid sodium", "nonbiologic_systemic"),
    ("CellCept", "", "PO", "mycophenolate mofetil", "nonbiologic_systemic"),
    ("Ruxience", "rituximab-pvvr", "IV", "rituximab", "biologic"),
    ("prednisolone drops", "prednisolone", "ophthalmic", "prednisolone", "other_symptomatic"),
    ("prednisone", "", "PO", "prednisone", "systemic_glucocorticoid"),
    ("prednisone", "", "", "prednisone", "needs_review"),
    ("unknown product", "", "", "", "unmapped"),
    ("", "methotrexate", "PO", "methotrexate", "nonbiologic_systemic"),
])
def test_conservative_drug_dictionary(drug, ingredient, route, normalized, drug_class):
    result = audit.normalize_drug(drug, ingredient, route)
    assert result["drug_normalized"] == normalized and result["drug_class"] == drug_class
    assert result["drug_raw"] == drug and result["ingredient_raw"] == ingredient
    assert result["indication"] == "indeterminate"


def test_combinations_and_ingredient_product_conflicts_are_not_silently_collapsed():
    result = audit.normalize_drug("generic combination", "trimethoprim/sulfamethoxazole", "PO")
    assert result["ingredient_normalized"] == "sulfamethoxazole + trimethoprim"
    conflict = audit.normalize_drug("Plaquenil", "methotrexate", "PO")
    assert "ingredient_product_conflict" in conflict["normalization_flags"]
    assert conflict["drug_class"] == "needs_review"


def test_ctdb_allergy_quarantine_irregular_fields_orphan_dates_and_unassigned_doses():
    data = pd.DataFrame([ctdb(**{
        "medications__rx_3_name": "HCQ", "medications__rx_3_frequency": "1", "medications__rx_3_frequency_": "12",
        "medications__rx_5_name": "MTX", "medications__rx_5_frequency_": "15",
        "medications__rx_6_name": "prednisone", "medications__rx_6_frequency_": "1",
        "medications__med2_strt_date": "06/2015", "medications__anticogaulant_1_dose": "5 mg",
        "past_medical_history__rx_1_name": "amoxicillin", "past_medical_history__rx_allergies": "amoxicillin",
    })])
    current, corrections, doses = audit.ctdb_records(data, {"p1"}, SECRET)
    assert len(current) == 4 and sum(r["name_present"] for r in current) == 3
    assert next(r for r in current if r["slot"] == 3)["frequency_normalized"] == "Weekly"
    assert corrections[0]["source_semantics"] == "unverified_possible_allergy" and corrections[0]["excluded_from_medication_counts"]
    assert corrections[0]["allergy_text_exact_match"]
    assert doses[0]["slot_mapping"] == "unresolved_not_assigned"
    assert all(not r["dose_raw"] for r in current)


def test_ctdb_concatenated_visits_no_arbitrary_date_and_preindex_retained():
    data = pd.DataFrame([ctdb(**{audit.CTDB_VISIT_DATE: "2015-06-08 | 2015-07-08", "index": "2016-01-01",
                                "medications__rx_1_name": "HCQ", "medications__med1_strt_date": "03/27/2015"})])
    rows, _, _ = audit.ctdb_records(data, {"p1"}, SECRET, index_column="index")
    assert rows[0]["visit_semantics"] == "concatenated_visits_no_arbitrary_date"
    assert "pre_index_historical_record_retained" in rows[0]["qc_flags"]
    assert rows[0]["protocol"] == "protocol_provenance_unavailable"


def test_btris_date_plausibility_does_not_filter_preconsent_and_end_is_not_stop():
    result = records([btris(**{"Order Date/Start Date": "03/27/2015", "Order End Date": "03/26/2015",
                              "Initial Consent Date": "2016-01-01", "Date of Death": "2015-03-01"})]).iloc[0]
    assert "end_definitely_before_start" in result.qc_flags
    assert "preconsent_historical_order_retained" in result.qc_flags
    assert "order_after_documented_death_verify" in result.qc_flags
    assert result.time_semantics == "order_or_start_unresolved"
    assert result.end_semantics == "order_validity_end_not_confirmed_discontinuation"


def test_dose_presence_does_not_convert_strength_to_daily_or_received_dose():
    result = records([btris(**{"Order Frequency": "PRN titration", "Order No. of Refills": "-1"}),
                      btris(**{"Order UOM": None}), btris(**{"Order Route": None})])
    assert result.prescription_description_complete.tolist() == [True, False, False]
    assert all(token in result.iloc[0].qc_flags for token in ["prn", "titration_or_variable", "refills_nonplausible_review"])
    assert not any(c in result for c in ["dose_daily", "received_dose", "adherence", "treatment_duration"])
    qc = audit.dose_qc(result)
    assert qc[qc.metric.eq("prescription_description_complete")].n_records.item() == 1


def test_overlap_reconciles_all_four_categories_and_deduplicates_protocols(tmp_path):
    args = args_for(tmp_path)
    # p1 both, p2 BTRIS only, p3 CTDB only, p4 neither.
    complete_ctdb([ctdb("p1", **{"medications__rx_1_name": "HCQ"}),
                   ctdb("p3", **{"medications__rx_1_name": "MTX"})]).to_parquet(args.input)
    pd.DataFrame([btris(**{"Protocol Number": "15-D-0051"}),
                  btris(**{"Protocol Number": "15-D-0051", "MRN": "00456"})]).to_parquet(args.btris_15d)
    summary = audit.run(args)
    union = next(r for r in summary["overlap"] if r["scope"] == "union")
    assert [union[f"n_{c}"] for c in ["ctdb_only", "btris_only", "both", "neither"]] == [1, 1, 1, 1]
    assert union["denominator"] == 4
    assert sum(union[f"pct_{c}"] for c in ["ctdb_only", "btris_only", "both", "neither"]) == 100
    orders = pd.read_parquet(args.output_dir / "restricted/08_btris_orders_audit_long.parquet")
    assert len(orders) == 3 and orders.linked_patient_pseudonymous.nunique() == 2


def test_without_crosswalk_audit_continues_but_joint_counts_are_null(tmp_path):
    args = args_for(tmp_path, crosswalk=False)
    summary = audit.run(args)
    assert summary["linkage"]["status"] == "blocked_linkage"
    assert all(r["n_neither"] is None and r["status"] == "blocked_linkage" for r in summary["overlap"])
    assert len(table(args, "08_btris_source_profile.csv")) == 2
    coverage = table(args, "08_treatment_combined_coverage.csv")
    assert coverage.cross_source_status.eq("blocked_linkage").all()


def test_end_to_end_preservation_private_permissions_and_no_shared_phi(tmp_path):
    args = args_for(tmp_path)
    data = pd.read_parquet(args.btris_11d)
    data["Subject Name"] = "PRIVATE PERSON NAME"
    data["Date of Birth"] = "01/02/1900"
    data["Order Instruction"] = "PRIVATE IDENTIFIABLE INSTRUCTION"
    data["Non Formulary Info"] = "PRIVATE FREE TEXT"
    data.to_parquet(args.btris_11d)
    old = tmp_path / "audit-v1.csv"
    old.write_text("old audit product\n")
    paths = [*args.input.parent.iterdir(), old, ROOT / "src/block_A/08_treatment_audit.py", ROOT / "src/block_A/10_build_integrated_longitudinal_dataset.py", ROOT / "src/block_A/14_risk_factors_progression.py"]
    before = {p: audit.file_fingerprint(p) for p in paths}
    summary = audit.run(args)
    for path, info in before.items():
        assert audit.file_fingerprint(path) == info
    manifest = json.loads((args.output_dir / "restricted/08_treatment_input_manifest.json").read_text())
    assert all(info.get("preserved", True) for info in manifest["inputs"].values())
    assert stat.S_IMODE(args.output_dir.stat().st_mode) == 0o700
    for path in (args.output_dir / "restricted").iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for path in args.output_dir.iterdir():
        if path.is_file():
            content = path.read_text()
            assert all(value not in content for value in ["PRIVATE PERSON NAME", "01/02/1900", "00123", "PRIVATE IDENTIFIABLE INSTRUCTION", "PRIVATE FREE TEXT"])
    assert summary["gates"]["clinical_prepost_ESSDAI"] == "not_evaluated_in_this_audit"
    assert summary["restrictions"]["no_master_data_changes"]
    audit.run(args)  # Preliminary v2 reruns overwrite known artifacts only.


def test_blocked_source_run_still_delivers_all_required_artifacts_and_exit_two(tmp_path):
    args = args_for(tmp_path)
    args.btris_15d.unlink()
    summary = audit.run(args)
    assert summary["status"] == "blocked"
    assert summary["source_counts"]["15D"]["n_rows"] is None
    union = next(r for r in summary["overlap"] if r["scope"] == "union")
    assert union["status"] == "blocked_source" and union["n_neither"] is None
    assert len(list((args.output_dir / "restricted").glob("08_*"))) >= 16
    result = subprocess.run([sys.executable, str(SCRIPT), "--input", str(tmp_path / "absent.parquet"),
                             "--eligibility", str(tmp_path / "absent.csv"),
                             "--btris-11d", str(tmp_path / "absent11.parquet"), "--btris-15d", str(tmp_path / "absent15.parquet"),
                             "--output-dir", str(tmp_path / "blocked")], capture_output=True, text=True, check=False)
    assert result.returncode == 2, result.stderr
    assert (tmp_path / "blocked/08_treatment_audit_v2_report.md").is_file()


def test_optional_v1_summary_compares_numbers_without_reusing_fixed_denominator(tmp_path):
    args = args_for(tmp_path)
    args.previous_summary = tmp_path / "v1-summary.json"
    args.previous_summary.write_text(json.dumps({"n_eligible_patients": 159, "n_historical_slots": 639}))
    summary = audit.run(args)
    comparison = {r["metric"]: r for r in summary["v1_comparison"]}
    assert comparison["n_eligible_patients"]["current"] == 4
    assert comparison["n_eligible_patients"]["difference"] == -155
    assert comparison["n_historical_slots"]["current"] == 1


def test_scope_no_downstream_reads_and_literal_schema_reuse(monkeypatch, tmp_path):
    args = args_for(tmp_path)
    original = audit.load_source
    reads = []
    def checked(path, *a, **kw):
        reads.append(Path(path))
        assert Path(path).parent == args.input.parent
        return original(path, *a, **kw)
    monkeypatch.setattr(audit, "load_source", checked)
    monkeypatch.setattr(pd, "read_excel", lambda *a, **kw: pytest.fail("Excel codebook must not be read"))
    audit.run(args)
    assert len(reads) == 5
    assert len(audit.CTDB_COLUMNS) == 66
    assert audit.CTDB_RX_FREQ_BY_SLOT[3] == "medications__rx_3_frequency_"
    assert audit.CTDB_RX_FREQ_BY_SLOT[5] == "medications__rx_5_frequency_"
    assert audit.CTDB_RX_FREQ_BY_SLOT[6] == "medications__rx_6_frequency_"
    assert "medications__anticogaulant_1_dose" in audit.CTDB_UNASSIGNED_DOSES


def test_full_source_with_empty_tables_is_available_not_missing(tmp_path):
    args = args_for(tmp_path)
    pd.DataFrame(columns=audit.BTRIS_MEDICATION_COLUMNS).to_parquet(args.btris_11d)
    pd.DataFrame(columns=audit.BTRIS_MEDICATION_COLUMNS).to_parquet(args.btris_15d)
    complete_ctdb([]).to_parquet(args.input)
    summary = audit.run(args)
    assert summary["source_counts"]["11D"]["n_rows"] == 0
    union = next(r for r in summary["overlap"] if r["scope"] == "union")
    assert union["n_neither"] == 4


def test_missing_crosswalk_manifest_continues_source_audit(tmp_path):
    args = args_for(tmp_path)
    args.crosswalk_manifest.unlink()
    summary = audit.run(args)
    assert summary["linkage"]["status"] == "blocked_linkage"
    assert summary["source_counts"]["11D"]["n_rows"] == 1
    assert summary["source_counts"]["11D"]["n_eligible_linked_patients"] is None
    profile = table(args, "08_btris_source_profile.csv")
    assert profile.n_eligible_linked_patients.isna().all()
    manifest = json.loads((args.output_dir / "restricted/08_treatment_input_manifest.json").read_text())
    assert manifest["inputs"]["crosswalk_manifest"]["status"] == "blocked_linkage"


def test_partial_crosswalk_does_not_fill_neither_or_eligible_counts_with_zero(tmp_path):
    args = args_for(tmp_path)
    pd.DataFrame({"MRN": ["00123"], "patient_id": ["p1"]}).to_csv(args.input.parent / "crosswalk.csv", index=False)
    summary = audit.run(args)
    assert summary["linkage"]["status"] == "blocked_linkage"
    assert all(row["n_neither"] is None for row in summary["overlap"])
    assert table(args, "08_btris_drug_inventory.csv").n_eligible_linked_patients.isna().all()
    assert "CTDB+BTRIS" not in set(table(args, "08_treatment_combined_coverage.csv").source)


def test_exact_signatures_distinguish_null_blank_and_unexpected_column_changes():
    result = audit.duplicate_qc(records([btris(**{"Order Instruction": None, "extra": "a"}),
                                        btris(**{"Order Instruction": "", "extra": "a"}),
                                        btris(**{"Order Instruction": None, "extra": "b"})]))
    assert result.content_signature.nunique() == 3
    assert not result.qc_flags.str.contains("exact_reexport_duplicate_retained").any()


def test_cross_protocol_record_matching_blocked_without_validated_identity():
    no_links = audit.validate_linkage(None, {"p1"})
    left = records([btris()], linkage=no_links)
    right = records([btris(**{"Protocol Number": "15-D-0051"})], source="15D", linkage=no_links)
    result = audit.duplicate_qc(pd.concat([left, right], ignore_index=True))
    assert not result.qc_flags.str.contains("cross_protocol_possible_duplicate_retained").any()


def test_uncertain_ingredient_or_conflicting_route_requires_review():
    result = audit.normalize_drug("Plaquenil", "unknown component + another agent", "PO")
    assert result["drug_class"] == "needs_review"
    assert "ingredient_unverified_product_fallback" in result["normalization_flags"]
    ocular = audit.normalize_drug("Restasis", "cyclosporine", "PO")
    assert ocular["drug_class"] == "needs_review" and "formulation_route_conflict" in ocular["normalization_flags"]
    assert ocular["ingredient_normalized"] == "cyclosporine"
    assert audit.normalize_drug("", "hydroxychloroquine sulfate", "PO")["ingredient_normalized"] == "hydroxychloroquine"


def test_documentation_intervals_flag_overlap_without_deriving_durations():
    rows = [btris(**{"Order Date/Start Date": "01/01/2015", "Order End Date": "06/01/2015"}),
            btris(**{"Order ID": "order-2", "Order Date/Start Date": "05/01/2015", "Order End Date": "07/01/2015"})]
    result = audit.interval_qc(records(rows))
    assert result.qc_flags.str.contains("documented_intervals_overlap_requires_review").all()
    assert "duration" not in result


def test_concordance_temporal_bounds_and_unresolved_formulations():
    rows = [ctdb(**{"medications__rx_1_name": "HCQ", "medications__rx_1_route": "PO",
                    "medications__med1_strt_date": "06/2015"})]
    current = audit.records_frame(audit.ctdb_records(pd.DataFrame(rows), {"p1"}, SECRET)[0])
    result = audit.concordance(current, records([btris()]), link(), {s: True for s in ["11D", "15D", "CTDB"]})
    assert result.level.item() == "temporally_compatible" and result.date_compatibility.item() == "compatible"
    assert current.start_parsed.item() == ""
    current["formulation"] = "unresolved"
    result = audit.concordance(current, records([btris()]), link(), {s: True for s in ["11D", "15D", "CTDB"]})
    assert result.level.item() == "potential_match_unresolved"


def test_union_coverage_counts_same_validated_patient_once_across_sources(tmp_path):
    args = args_for(tmp_path)
    audit.run(args)
    coverage = table(args, "08_treatment_combined_coverage.csv")
    combined = coverage[coverage.source.eq("CTDB+BTRIS")]
    assert combined.n_patients.item() == 2
    assert combined.n_eligible_linked_patients.item() == 2
    assert combined.n_records.item() == 3  # two orders and one visit slot remain separate.
    assert combined.percent_eligible_cohort.item() == 50
    assert combined.evidence_type.item() == "mixed_documentation_records_not_exposure"


def test_all_28_schema_rows_reported_for_each_file_and_unknown_orders_keep_gates(tmp_path):
    args = args_for(tmp_path)
    summary = audit.run(args)
    schema = table(args, "08_btris_source_schema_audit.csv")
    assert schema.groupby("source").size().to_dict() == {"11D": 28, "15D": 28}
    assert summary["input_fingerprints"]["11D"]["sha256"] == audit.v1.file_hash(args.btris_11d)
    gates = table(args, "08_treatment_combined_feasibility.csv")
    assert gates[gates.gate.eq("clinical_prepost_ESSDAI")].status.eq("not_evaluated_in_this_audit").all()
    assert gates[gates.gate.eq("actual_exposure")].n_records.isna().all()


def test_reusable_key_makes_pseudonyms_reproducible_and_is_not_copied_or_printed(tmp_path, capsys):
    args = args_for(tmp_path)
    args.pseudonym_key_file = tmp_path / "approved-synthetic-key"
    args.pseudonym_key_file.write_bytes(SECRET)
    audit.run(args)
    first = pd.read_parquet(args.output_dir / "restricted/08_btris_orders_audit_long.parquet")
    args.output_dir = tmp_path / "audit-v2-second"
    audit.run(args)
    second = pd.read_parquet(args.output_dir / "restricted/08_btris_orders_audit_long.parquet")
    assert first.patient_id_pseudonymous.tolist() == second.patient_id_pseudonymous.tolist()
    assert not (args.output_dir / "restricted/.pseudonym_key").exists()
    assert SECRET.decode() not in capsys.readouterr().out


@pytest.mark.parametrize("path", ["10_integrated.parquet", "14_risk.parquet", "15_future.csv", "studies/pharma/source.parquet"])
def test_downstream_sources_are_rejected_before_reading(path, tmp_path):
    args = args_for(tmp_path)
    args.input = tmp_path / path
    with pytest.raises(ValueError, match="does not read downstream"):
        audit.run(args)
    assert not args.output_dir.exists()
