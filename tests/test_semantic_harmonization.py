"""Synthetic regression tests for the semantic candidate and local report."""
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from openpyxl import Workbook

from src.semantic_harmonization.inference import LocalModel, profile, supports, validate_extraction
from src.semantic_harmonization.inventory import CONTEXT_FIELDS, route
from src.semantic_harmonization.parsers import new_finding, parse
from src.semantic_harmonization.rag import CODEBOOK_FIELDS
from src.semantic_harmonization.run_pipeline import run
from src.semantic_harmonization.settings import MODULE, ROOT, digest, file_hash, pipeline_config, read_yaml, serialize
from src.semantic_harmonization.study_input import SOURCE_MANIFEST, verify_source

VOCAB = {"ucum": {"%", "mg", "dL"}, "ipd": {"g_group": {"DRB1*04:01:01G"}}, "rxnorm": {}}
KEYS = ["patient_id", "clinical_episode_id"]


@pytest.fixture
def cfg():
    configs = MODULE / "configs"
    return read_yaml(configs / "pipeline.yaml") | {
        name: read_yaml(configs / f"{name}.yaml")
        for name in ("concepts", "normalization", "prompts", "resources")
    }


def parsed(cfg, variable, raw, **context):
    concept, family = route(variable, cfg)
    return parse(dict(variable=variable, concept=concept, family=family, raw_json=serialize(raw),
                      context_json=serialize(dict.fromkeys(CONTEXT_FIELDS) | context)), cfg, VOCAB)


@pytest.mark.parametrize("text, comparator, expected, bound", [
    ("<10", "LT", None, "10"), (">1300", "GT", None, "1300"), ("< 0.2", "LT", None, "0.2"),
    ("25.4", "EQ", "25.4", None), ("≤ 2.5e-2", "LE", None, "0.025"),
    ("<3%", "LT", None, "3"), ("0", "EQ", "0", None), ("2025-01-03", None, None, None),
    ("10 reviewed", None, None, None),
])
def test_numeric_grammar(cfg, text, comparator, expected, bound):
    result = parsed(cfg, "lab__anti_ro60__text", text)
    assert (result.comparator, result.exact_value, result.limit_value, result.qualitative_result) == (
        comparator, expected, bound, None
    )


def test_ana_urine_status_and_metadata(cfg):
    ana = parsed(cfg, "lab__ana_hep2_status__text", "<1:80 (Negative)")
    assert (ana.titer_numerator, ana.titer_denominator, ana.titer_comparator, ana.qualitative_result) == (1, 80, "LT", "negative")
    assert parsed(cfg, "lab__ana_status__text", "Positive 1:640").titer_denominator == 640
    assert parsed(cfg, "lab__ana_status__text", "not negative").qualitative_result is None
    assert parsed(cfg, "lab__rheumatoid_factor__text", "not performed").result_availability == "not_performed"
    assert parsed(cfg, "lab__rheumatoid_factor__text", "HEMOLYZED").specimen_state == "hemolyzed"
    assert parsed(cfg, "lab__urine_protein__text", "NORMAL").qualitative_result == "normal"
    assert parsed(cfg, "lab__urine_protein__text", "NEGATIVE").qualitative_result == "negative"
    assert parsed(cfg, "lab__urine_color__text", "Light-Yellow").category == "light yellow"
    assert parsed(cfg, "lab__urine_bacteria__text", "Occassional").category == "occasional"
    assert parsed(cfg, "lab__unknown__text", "Occassional").category is None
    result = parsed(cfg, "lab__urine_protein__text", "1+(30 mg/dL)")
    assert (result.ordinal_grade, result.exact_value, result.raw_unit, result.ucum_unit) == ("1+", "30", "mg/dL", "mg/dL")
    result = parsed(cfg, "lab__urine_protein__text", "+1(30-70)")
    assert (result.ordinal_grade, result.range_low, result.range_high, result.raw_unit) == ("1+", "30", "70", None)
    result = parsed(cfg, "lab__urine_protein__text", "Trace(10-20")
    assert (result.ordinal_grade, result.comparator) == ("trace", None)
    result = parsed(cfg, "lab__urine_blood__text", "Non Hemolyzed Trace")
    assert (result.ordinal_grade, result.descriptor, result.specimen_state) == ("trace", "non hemolyzed", None)


def test_narratives_hla_and_medications(cfg):
    summary = parsed(cfg, "lab__cryoglobulins__text", "negative at 24h, positive at 7d")
    assert [(f.incubation_hours, f.assertion) for f in summary.findings] == [(24, "absent"), (168, "present")]
    summary = parsed(cfg, "lab__cryoglobulins__text", "Negative at 24 hours, reviewed again at 7 days")
    assert [(f.incubation_hours, f.temporality) for f in summary.findings] == [(24, "current"), (168, "planned")]
    summary = parsed(cfg, "lab__cryoglobulins__text", "3% of Cryoprecipitate; Previously identified as IgM Kappa and Polyclonal IgG")
    assert [(f.entity, f.temporality) for f in summary.findings] == [
        ("cryoprecipitate", "unspecified"), ("immunoglobulin", "historical"), ("immunoglobulin", "historical")]
    assert parsed(cfg, "lab__cryoglobulins__text", "<3%").qualitative_result is None
    assert parsed(cfg, "lab__immunofixation__text", "No anomalous immunoglobulins").findings[0].assertion == "absent"
    assert parsed(cfg, "lab__immunofixation__text", "Faint Free Lambda").findings[0].pattern is None
    assert not parsed(cfg, "lab__immunofixation__text", "No IgG and IgM").findings
    carried = parsed(cfg, "lab__immunofixation__text", "SEE BELOW", observation_comment="Faint Free Lambda")
    assert carried.findings[0].light_chain == "lambda"
    assert any(e.source == "context.observation_comment" for e in carried.evidence)
    result = parsed(cfg, "lab__hla_drb1__text", "04:01:01G, *")
    assert [(a.group, a.allele, a.suffix, a.ipd_match) for a in result.alleles] == [
        ("DRB1*04:01:01G", None, "G", "g_group"), (None, None, None, None)]
    assert len(parsed(cfg, "lab__hla_a__text", "*,*").alleles) == 2
    ambiguous, = parsed(cfg, "lab__hla_drb__text", "DRB1*04:01:01G").alleles
    assert ambiguous.reported == "DRB1*04:01:01G" and ambiguous.allele is None
    assert ambiguous.ipd_match == "ambiguous_locus"
    assert parsed(cfg, "lab__hla_a__text", "07,17").alleles[0].allele is None
    result = parsed(cfg, "medication__btris_order_nearby_1y_names", "drug a / drug b; drug c 10 mg PO daily")
    assert [m.name for m in result.medications] == ["drug a / drug b", "drug c"]
    assert all(m.temporal_availability == "retrospective" and m.documentary_evidence == "order" for m in result.medications)
    assert (result.medications[1].dose, result.medications[1].route) == ("10", "PO")


def test_model_evidence_respects_opposite_polarity_and_temporality():
    text = "previously positive, now negative"
    assert supports(new_finding("cryoglobulin", text, assertion="present", temporality="historical"), text, "cryoglobulin")
    assert supports(new_finding("cryoglobulin", text, assertion="absent", temporality="current"), text, "cryoglobulin")
    assert not supports(new_finding("cryoglobulin", text, assertion="present", temporality="current"), text, "cryoglobulin")
    assert not supports(new_finding("immunoglobulin", "IgG kappa", immunoglobulin_class="IgG", light_chain="kappa"),
                        "No evidence of IgG kappa.", "immunofixation")
    row = dict(family="immunofixation", raw_json='"No anomalous immunoglobulins."', context_json="{}")
    unsupported = new_finding("anomalous_immunoglobulins", "No anomalous immunoglobulins.").model_dump()
    with pytest.raises(ValueError, match="source span"):
        validate_extraction(serialize({"findings": [unsupported]}), row, [])


class Encoder:
    """Deterministic test embedding with the production interface."""
    batches = 0

    def encode(self, terms):
        self.batches += 1
        return np.array([[1.0, 0.0] for _ in terms], dtype="float32")

    def count(self, text):
        return len(text.split())


class Model:
    """Inference double returning only a supported explicit observation."""
    calls = 0

    def __init__(self, cfg):
        self.cfg = cfg

    def smoke(self, fixtures=None):
        self.__class__.calls += 1
        return SmokeModel(self.cfg).smoke(fixtures)

    def messages(self, row, retrieved, retry):
        return row, retrieved

    def chat(self, messages):
        results = []
        for row, retrieved in messages:
            if row["family"] == "immunofixation":
                assert any(item["source"] == "local_rules" for item in retrieved)
                finding = new_finding("light_chain", "faint lambda", light_chain="lambda", intensity_descriptor="faint")
                results.append(serialize({"findings": [finding.model_dump()]}))
            else:
                results.append('{"findings":[]}')
        return results


def synthetic(tmp_path, cfg):
    paths = cfg["paths"]
    names = {"integrated": "step10.parquet", "context": "context.parquet", "source_events": "events.parquet",
             "codebook": "codebook.xlsx", "knowledge_dir": "knowledge", "pharma_registry": "pharma.yaml",
             "output": "report", "analytic": "analytic", "obj1_root": "obj1"}
    paths.update({name: str(tmp_path / file) for name, file in names.items()})
    cfg["validation"]["profile_pharma"] = False
    cfg["report"]["page_size"] = 1
    master = pd.DataFrame(dict(patient_id=["synthetic-A", "synthetic-A", "synthetic-B"],
                               clinical_episode_id=["e0", "e1", "e2"],
                               clinical_anchor_date=pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01"]),
                               is_clinical_baseline=[True, False, True],
                               clinical_visit=[True, True, True], clinical_visit_number=[1, 2, 1]))
    additions = {
        "pop__status": ["Pop1", "Pop2", "Pop3"], "essdai__total": [0, 5, 2], "esspri__total": [1, 2, 3],
        "lab__anti_ro60__text": [">1300", ">1300", " "],
        "lab__anti_ro60__value": pd.Series([1300.0, 1300.0, None], dtype="Float64"),
        "lab__immunofixation__text": ["A faint lambda band was detected.", "SEE NOTE | IgG(lambda) band detected.", None],
        "lab__urinalysis_includes_microscopic__hyalin_casts_urinalysis__text": ["<1", None, None],
        "lab__urinalysis_includes_microscopic__renal_epithelial_cells_urinalysis__text": ["<1", None, None],
        "lab__urinalysis_includes_microscopic__unclassified_casts_urinalysis__text": ["0-2", "3-5", "6-10"],
        "lab__urine_bacteria__text": [">10000", None, None],
        "lab__ana_status__text": ["Positive 1:640", "not performed", "negative"],
        "sero__ana_status__ever_positive_through_episode": pd.Series([True, True, None], dtype="boolean"),
        "lab__anti_sm__text": ["NEG", "POSITIVE", "  Unknown  "],
        "lab__other__text": ['</script><img src=x onerror=alert(2)> &', "", None],
        "spine__sjogrens_class_patient_values": ["4; 2", "1; 4", "2; 1"],
        "demo__name": ["private-synthetic-a", "private-synthetic-a", "private-synthetic-b"],
        "ext__source_identifier": ["hidden1", "hidden2", "hidden3"],
        "ext__score": [0, 10, 13],
    }
    master = master.assign(**additions)
    master.to_parquet(paths["integrated"], index=False)
    master[KEYS].assign(**{"lab__anti_ro60__unit": ["mg/dL", "IU/L", None]}).iloc[::-1].to_parquet(paths["context"], index=False)
    pd.DataFrame(dict(
        patient_id=["synthetic-A", "synthetic-B", "synthetic-B"], matched_clinical_episode_id=["e0", "e2", None],
        canonical_analyte=["ana_status"] * 3, lab_date=["2023-12-31"] * 3,
        result_available_date=["2024-01-15", "2023-12-31", None],
        result_raw=["positive", "negative", "positive"], result_valid_for_analysis=[True] * 3,
    )).to_parquet(paths["source_events"], index=False)
    wb = Workbook()
    ws = wb.active
    ws.title = "Consolidated_Codebook"
    ws.append(CODEBOOK_FIELDS)
    ws.append(["Sheet", "P", "Form", "F_Q", "Q", "1", "Q", "ana_status", "Generic metadata", "Yes", "01"])
    wb.save(paths["codebook"])
    Path(paths["pharma_registry"]).write_text("registry_version: test\nconcepts: {}\n")
    resources = {"codebook": {"status": "available", "sha256": "book"}, "gpt_oss": {"status": "available", "sha256": "gpt"},
                 "sapbert": {"status": "available", "sha256": "encoder"}}
    resources.update({name: {"status": "unavailable"} for name in ("ipd", "ucum", "rxnorm", "protocols")})
    return master, {"resources": resources}


def test_pipeline_integration_cache_and_offline_report(tmp_path, cfg):
    original, manifest = synthetic(tmp_path, cfg)
    Model.calls = 0
    options = dict(encoder=Encoder(), model_factory=Model, resource_verify=lambda cfg: manifest)
    analytic, output = run(cfg, **options)
    candidate = pd.read_parquet(analytic / "10_integrated_longitudinal_clinical_episode_harmonized.parquet")
    pd.testing.assert_frame_equal(original, candidate[original.columns])
    assert candidate["harm__lab__anti_ro60__text__limit_value"].iloc[:2].tolist() == [1300.0, 1300.0]
    assert "harm__lab__anti_ro60__text__exact_value" not in candidate
    microscopy = "harm__lab__urinalysis_includes_microscopic__"
    for name in ("hyalin_casts", "renal_epithelial_cells"):
        base = microscopy + name + "_urinalysis__text__"
        assert candidate[base + "limit_value"].iloc[0] == 1
        assert candidate[base + "comparator"].iloc[0] == "LT"
        assert base + "exact_value" not in candidate
    base = microscopy + "unclassified_casts_urinalysis__text__"
    assert candidate[base + "range_low"].tolist() == [0, 3, 6]
    assert candidate[base + "range_high"].tolist() == [2, 5, 10]
    assert candidate["harm__lab__urine_bacteria__text__limit_value"].iloc[0] == 10000
    assert candidate["harm__lab__immunofixation__text__igg_lambda_present"].iloc[1] == True
    assert pd.isna(candidate["harm__lab__immunofixation__text__igg_lambda_present"].iloc[0])
    assert not {"harm__lab__immunofixation__text__igg_kappa_present",
                "harm__lab__immunofixation__text__abnormal_band_present"} & set(candidate)
    for directory in (analytic, output):
        for path in directory.glob("*.parquet"):
            assert path.with_suffix(".csv").is_file()
            assert path.with_suffix(".csv.schema.json").is_file()
    comment = pq.read_table(output / "comment_prefix_audit.parquet").to_pylist()
    assert comment[0]["disposition"] == "fully_structured"
    assert len({r["mapping_id"] for r in comment}) == 1
    audit = pq.read_table(output / "semantic_materialization_audit.parquet").to_pylist()
    assert any(r["status"] == "linked" and r["wide_fields"] == ["harm__lab__immunofixation__text__igg_lambda_present"] for r in audit)
    assert candidate["harm__sero__ana_status__ever_positive_through_episode"].iloc[1] == True
    assert pd.isna(candidate["harm__sero__ana_status__ever_positive_through_episode"].iloc[0])
    assert candidate["harm__sero__ana_status__ever_positive_through_episode"].iloc[2] == False
    source = pq.read_table(analytic / "harmonized_source_events.parquet")
    assert source.num_rows == 3
    mappings = pq.read_table(output / "semantic_dictionary.parquet").to_pylist()
    censor = [row for row in mappings if row["variable"] == "lab__anti_ro60__text" and row["comparator"] == "GT"]
    assert len(censor) == 2 and len({row["mapping_id"] for row in censor}) == 2
    assert not {row["variable"] for row in mappings} & {"spine__sjogrens_class_patient_values", "demo__name", "ext__source_identifier"}
    html = output / "semantic_harmonization_final.html"
    content = html.read_text()
    assert "</script><img" not in content and "synthetic-A" not in content and "private-synthetic-a" not in content
    assert "\\u003c/script\\u003e" in content and "\\u0026" in content and "connect-src 'none'" in content
    browser_js = Path(__file__).with_name("semantic_report_browser.cjs")
    node = shutil.which("node")
    browser = shutil.which("chromium") or shutil.which("google-chrome")
    if node and browser and subprocess.run([node, "-p", "require.resolve('playwright')"],
                                           capture_output=True, cwd=browser_js.parent).returncode == 0:
        env = os.environ | {"XDG_CONFIG_HOME": str(tmp_path / "browser-config"),
                            "XDG_CACHE_HOME": str(tmp_path / "browser-cache")}
        checked = subprocess.run([node, str(browser_js), str(html), browser], env=env,
                                 capture_output=True, timeout=30, text=True)
        assert checked.returncode == 0, checked.stderr
    assert stat_mode(html) == 0o600
    assert run(cfg, **options) == (analytic, output)
    assert Model.calls == 1
    cfg["prompts"]["version"] = "revised-fixture-v2"
    next_analytic, _ = run(cfg, **options)
    assert next_analytic != analytic and Model.calls == 2


def stat_mode(path: Path):
    return os.stat(path).st_mode & 0o777


def test_context_cardinality_failure_publishes_no_candidate(tmp_path, cfg):
    _, manifest = synthetic(tmp_path, cfg)
    path = Path(cfg["paths"]["context"])
    frame = pd.read_parquet(path)
    pd.concat([frame, frame.iloc[[0]]]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="duplicate episode keys"):
        run(cfg, encoder=Encoder(), model_factory=Model, resource_verify=lambda cfg: manifest)
    assert not list(Path(cfg["paths"]["analytic"]).glob("*"))


def test_protocol_threshold_uses_the_whole_interval(cfg):
    def measure(raw, unit="mg/dL"):
        row = dict(variable="lab__anti_ro60__text", concept="anti_ro60", family="qualitative",
                   raw_json=serialize(raw), context_json=serialize(dict.fromkeys(CONTEXT_FIELDS) |
                       dict(assay_id="synthetic-assay", method="synthetic-method", unit=unit, relevant_protocol_version="s1")))
        vocab = VOCAB | {"protocol_rules": [
            dict(rule_id="synthetic-rule", concept_id="anti_ro60", assay_key="synthetic-assay",
                 method="synthetic-method", protocol_version="s1", units=["mg/dL"],
                 positive_threshold=dict(operator="GE", value="10", otherwise="negative"))
        ]}
        return parse(row, cfg, vocab)

    assert measure("<10").qualitative_result == "negative"
    assert measure("<=10").qualitative_result is None
    assert measure(">=10").qualitative_result == "positive"
    assert measure(">8").qualitative_result is None
    assert measure(">10", unit="IU/L").qualitative_result is None


def test_candidate_bundle_pairs_real_pharma_and_graph_inputs(tmp_path, cfg, monkeypatch):
    from src.studies import economic
    from src.studies.graph import _pipeline
    from src.studies.graph._lab_contract import load_current_profile
    from src.studies.graph.stage02.s02_contracts import resolve_sources
    from src.studies.graph.stage03 import s03_release_reader

    _, resources = synthetic(tmp_path, cfg)
    cfg["paths"]["obj1_root"] = str(ROOT)
    cfg["paths"]["pharma_registry"] = str(ROOT / "src/studies/pharma/expected_lab_concepts.yaml")
    cfg["validation"]["profile_pharma"] = True
    options = dict(encoder=Encoder(), model_factory=Model, resource_verify=lambda cfg: resources)
    analytic, report = run(cfg, **options)
    selected = verify_source(analytic.name, analytic_root=analytic.parent, report_root=report.parent)
    assert json.loads((report / SOURCE_MANIFEST).read_text())["run_id"] == analytic.name
    assert selected.require("context").parent == analytic
    assert selected.require("pharma_candidates").is_relative_to(report)
    assert selected.require("feature_temporal_coverage").parent == analytic
    semantic = pq.read_table(selected.require("pharma_semantic_profile")).to_pylist()
    assert {row["study"] for row in semantic} == {"Pharma", "Graph"}
    assert any(row["encoding"] == "censored" for row in semantic if row["study"] == "Pharma")
    assert all(row["encoding"] != "exact_numeric" for row in semantic if "__limit_value" in row["feature_name"])
    master = pd.read_parquet(selected.require("integrated"))
    _, graph_report = load_current_profile(master, selected.require("integrated"), selected.require("pharma_candidates"), "exploratory")
    assert graph_report["input_hash"] == selected.sha256["integrated"]
    impact = pd.read_csv(report / "downstream_impact.csv")
    assert {"Pharma eligibility profile", "Graph exploratory eligibility"} <= set(impact.study)
    pharma_summary = json.loads(selected.require("pharma_provenance").read_text())
    assert pharma_summary["input_hashes"][0]["path"] == str(selected.require("integrated"))

    monkeypatch.setattr(economic, "active_study_input", lambda: selected)
    assert economic.resolve_inputs() == (selected.require("integrated"), selected.require("context"), selected.provenance())
    monkeypatch.setattr(_pipeline, "active_study_input", lambda: selected)
    graph_cfg = _pipeline.load_pipeline()
    assert resolve_sources(graph_cfg) == {role: selected.require(role) for role in ("integrated", "pharma_candidates")}
    assert _pipeline.output_root(graph_cfg, "02").name == selected.run_id
    other = tmp_path / "other_graph.csv"
    graph_cfg["paths"]["pharma_candidates"] = str(other)
    with pytest.raises(ValueError, match="conflicts"):
        resolve_sources(graph_cfg)
    monkeypatch.setattr(s03_release_reader, "validate_release",
                        lambda path: ({"schema_version": "graph02_current_v1"}, Path(path)))
    with pytest.raises(ValueError, match="different semantic run"):
        s03_release_reader.read_release(graph_cfg, tmp_path / "foreign_graph_run.json")
    selected.require("pharma_selection").write_text("changed")
    with pytest.raises(ValueError, match="hash or path mismatch: pharma_selection"):
        verify_source(analytic.name, analytic_root=analytic.parent, report_root=report.parent)


def test_selection_is_opt_in_and_rejects_unpublished_runs(tmp_path):
    from src.semantic_harmonization.study_input import selection_id

    (tmp_path / "input_selection.yaml").write_text("mode: original\nrun_id: null\n")
    assert selection_id(tmp_path) is None
    local = tmp_path / "input_selection.local.yaml"
    local.write_text("mode: candidate\nrun_id: ../cross-run\n")
    with pytest.raises(ValueError, match="run_id"):
        selection_id(tmp_path)
    run_id = "a" * 64
    local.write_text(f"mode: candidate\nrun_id: {run_id}\n")
    assert selection_id(tmp_path) == run_id
    with pytest.raises(ValueError, match="run directory"):
        verify_source(run_id, analytic_root=tmp_path, report_root=tmp_path)
    local.write_text("mode: original\nrun_id: null\n")
    assert selection_id(tmp_path) is None


def test_episode_only_candidate_does_not_require_upstream(tmp_path, cfg):
    _, resources = synthetic(tmp_path, cfg)
    cfg["paths"]["source_events"] = None
    cfg["paths"]["context"] = None
    original_hash = file_hash(Path(cfg["paths"]["integrated"]))
    analytic, report = run(cfg, encoder=Encoder(), model_factory=Model, resource_verify=lambda cfg: resources)
    selected = verify_source(analytic.name, analytic_root=analytic.parent, report_root=report.parent)
    assert selected.select("context") is None
    assert not (analytic / "harmonized_source_events.parquet").exists()
    assert file_hash(Path(cfg["paths"]["integrated"])) == original_hash
    assert pipeline_config()["paths"]["obj1_root"] == str(ROOT)


def test_pharma_followups_require_the_same_run(tmp_path, cfg, monkeypatch):
    from argparse import Namespace
    from src.studies.pharma import _semantic_input as pharma

    _, resources = synthetic(tmp_path, cfg)
    analytic, report = run(cfg, encoder=Encoder(), model_factory=Model, resource_verify=lambda cfg: resources)
    source = verify_source(analytic.name, analytic_root=analytic.parent, report_root=report.parent)
    monkeypatch.setattr(pharma, "active_study_input", lambda: source)
    default = tmp_path / "intervals.parquet"
    target = pharma.versioned(default, source)
    target.parent.mkdir(parents=True)
    target.write_text("synthetic staged product")
    args = Namespace(analytic=default)
    with pytest.raises(FileNotFoundError, match="preceding Pharma"):
        pharma.follow_up(args, {"analytic": default})
    pharma.publish_provenance(source, target.parent, (target.name,))
    replacement, provenance = pharma.follow_up(args, {"analytic": default})
    assert (replacement.analytic, provenance) == (target, source)
    with pytest.raises(ValueError, match="conflicts"):
        pharma.follow_up(Namespace(analytic=tmp_path / "different"), {"analytic": default})
    target.write_text("different contents")
    with pytest.raises(ValueError, match="provenance or hash"):
        pharma.follow_up(args, {"analytic": default})


def test_launchers_resolve_repo_from_foreign_and_slurm_working_directories(tmp_path):
    repo = tmp_path / "obj1 runtime"
    module = repo / "src" / "semantic_harmonization"
    (repo / "src" / "block_A").mkdir(parents=True)
    (repo / "src" / "block_A" / "10_build_integrated_longitudinal_dataset.py").touch()
    module.mkdir()
    for name in ("run_prepare.sh", "run_harmonization.sbatch", "repository_root.sh",
                 "00_prepare_resources.py", "run_pipeline.py"):
        shutil.copyfile(MODULE / name, module / name)
    venv = tmp_path / "fake-venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text("export PATH=\"$SJD_SEMANTIC_VENV/bin:$PATH\"\n")
    python = venv / "bin" / "python"
    python.write_text('#!/usr/bin/env bash\n[[ $(pwd -P) == "$EXPECTED_REPO" ]] || exit 1\n'
                      'if [[ $1 == -c ]]; then exit 0; fi\n'
                      'printf "%s|%s|%s\\n" "$1" "$#" "${HF_HUB_OFFLINE:-unset}"\n')
    python.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in {"HF_HUB_OFFLINE", "SLURM_SUBMIT_DIR", "OBJ1_ROOT"}}
    env.update(SJD_SEMANTIC_VENV=str(venv), EXPECTED_REPO=str(repo))
    prepared = subprocess.run(["bash", str(module / "run_prepare.sh")], env=env, cwd=tmp_path,
                              capture_output=True, text=True, check=True)
    assert prepared.stdout.strip() == f"{module / '00_prepare_resources.py'}|1|unset"
    spool = tmp_path / "slurm-spool.sh"
    shutil.copyfile(module / "run_harmonization.sbatch", spool)
    invalid = subprocess.run(["bash", str(spool)], env=env | {"SLURM_SUBMIT_DIR": str(tmp_path)},
                             cwd=tmp_path, capture_output=True, text=True)
    assert invalid.returncode != 0 and "obj1_sjd" in invalid.stderr
    base = tmp_path / "canonical" / "salazarda"
    (base / "containers").mkdir(parents=True)
    (base / "containers" / "vllm-0.11.2.sif").touch()
    supplemental = base / "sjd_container_env" / "bin"
    supplemental.mkdir(parents=True)
    shutil.copyfile(python, supplemental / "python")
    (supplemental / "python").chmod(0o755)
    fake_module = venv / "bin" / "module"
    fake_module.write_text('#!/usr/bin/env bash\n[[ $* == "load apptainer" ]]\n')
    fake_module.chmod(0o755)
    apptainer = venv / "bin" / "apptainer"
    apptainer.write_text(
        '#!/usr/bin/env bash\n'
        '[[ $1 == exec && $2 == --nv && $3 == --cleanenv ]] || exit 2\n'
        'while [[ $1 != --pwd ]]; do shift; done\n'
        'printf "%s|%s|%s\\n" "$2" "$5" "$APPTAINERENV_HF_HUB_OFFLINE"\n'
        'cat >/dev/null\n')
    apptainer.chmod(0o755)
    alias = tmp_path / "alias"
    alias.symlink_to(base, target_is_directory=True)
    inside = base / "obj1 runtime"
    repo.rename(inside)
    spool.write_text(spool.read_text().replace("BASE=/data/salazarda", f'BASE="{alias}"'))
    subprocess.run(["bash", str(spool)], env=env | {"OBJ1_ROOT": str(inside), "SLURM_JOB_ID": "123",
                   "PATH": f'{venv / "bin"}:{env["PATH"]}'},
                   cwd=tmp_path, capture_output=True, text=True, check=True)
    log = inside / "work" / "semantic_harmonization" / "job-123.log"
    assert log.read_text().splitlines()[-1] == f"{alias / inside.name}|-u|1"


@pytest.mark.parametrize("concept, raw, family, expected", [
    ("anti_sm", "POSITIVE", "qualitative", "positive"),
    ("anti_rnp", " NEGATIVE ", "qualitative", "negative"),
    ("anti_scl70", "NEGATIVE", "qualitative", "negative"),
    ("anti_jo1", "NEGATIVE", "qualitative", "negative"),
    ("hepatitis_b_surface_antigen", "REPEAT REACTIVE", "qualitative", "reactive"),
    ("sars_cov2_anti_n", "indeterminate", "qualitative", "indeterminate"),
    ("urine_pregnancy_test", "Negative", "qualitative", "negative"),
])
def test_v2_exact_profiles_resolve_without_model(cfg, concept, raw, family, expected):
    from src.semantic_harmonization.inference import routing
    from src.semantic_harmonization.semantic_facts import parse_mapping, verify_contract

    variable = f"lab__{concept}__text"
    assert route(variable, cfg) == (concept, family)
    assert route(f"lab__{concept}__value", cfg) == (concept, "lab_numeric")
    row = dict(mapping_id="fixture", variable=variable, concept=concept, family=family,
               raw_json=serialize(raw), context_json="{}")
    semantic, segments, facts = parse_mapping(row, cfg, VOCAB)
    verify_contract([row], segments, facts)
    assert facts[0].payload.category == expected
    if expected == "reactive":
        assert facts[0].payload.descriptor == "REPEAT REACTIVE"
    assert not routing(row, semantic, segments, facts)["send"]


def v2_row(cfg, key, raw, **context):
    variable = f"lab__{key}__text"
    concept, family = route(variable, cfg)
    return dict(mapping_id=f"fixture-{key}", variable=variable, concept=concept, family=family,
                raw_json=serialize(raw), context_json=serialize(context))


def test_v2_segments_isolate_reference_history_and_transport(cfg):
    from src.semantic_harmonization.semantic_facts import parse_mapping, verify_contract

    row = v2_row(cfg, "anti_ccp", "  12 mg/dL | Reference range: Negative <5; Positive >=5 ",
                 observation_comment="Method: synthetic control review", reference_status="Positive")
    semantic, segments, facts = parse_mapping(row, cfg, VOCAB)
    verify_contract([row], segments, facts)
    assert semantic.exact_value == "12"
    assert semantic.qualitative_result is None
    assert {part.role for part in segments} == {"patient_result", "reference_rule", "assay_method", "unresolved"}
    assert [fact.fact_type for fact in facts].count("numeric_result") == 1
    assert "qualitative_result" not in {fact.fact_type for fact in facts}
    bad = segments[0].model_copy(update={"start": 1})
    with pytest.raises(ValueError, match="reconstruction"):
        verify_contract([row], [bad, *segments[1:]], facts)


@pytest.mark.parametrize("key, raw, expected", [
    ("quantiferon_final_interpretation", "Sample was not processed", "not_performed"),
    ("anti_sm", "see comment", None),
    ("anti_sm", "sent to external laboratory", None),
    ("cbc_diff__bands_blood", "Grouped with neutrophils", None),
    ("urinalysis_includes_microscopic__digital_user_review_urine", "Done", None),
])
def test_v2_process_markers_emit_metadata_only(cfg, key, raw, expected):
    from src.semantic_harmonization.semantic_facts import parse_mapping

    semantic, _, facts = parse_mapping(v2_row(cfg, key, raw), cfg, VOCAB)
    assert semantic.result_availability == expected
    assert not semantic.qualitative_result
    assert {fact.fact_type for fact in facts} == {"observation_metadata"}


def test_v2_urine_ana_spike_and_morphology(cfg):
    from src.semantic_harmonization.semantic_facts import parse_mapping

    floor = parsed(cfg, "lab__ana_status__text", "Negative (<1:40)")
    assert (floor.titer_denominator, floor.titer_comparator, floor.qualitative_result) == (40, "LT", "negative")
    assert parsed(cfg, "lab__ana_status__text", "Positive >= 1:640").titer_comparator == "GE"
    row = v2_row(cfg, "ana_hep2_cytoplasmic_pattern", "Cytoplasmic reticular may be associated with synthetic antibody")
    _, _, facts = parse_mapping(row, cfg, VOCAB)
    assert [(fact.payload.compartment, fact.payload.pattern) for fact in facts if fact.fact_type == "ana_pattern"] == [
        ("cytoplasmic", "cytoplasmic reticular")]
    _, _, facts = parse_mapping(v2_row(cfg, row["concept"], "Positive"), cfg, VOCAB)
    assert any(fact.fact_type == "ana_pattern" and fact.payload.pattern is None for fact in facts)
    for raw in ("not detected", "No monoclonal band detected"):
        _, _, facts = parse_mapping(v2_row(cfg, "monoclonal_band_m_spike", raw), cfg, VOCAB)
        assert facts[0].payload.detection == "absent" and facts[0].payload.concentration is None
    urine = parsed(cfg, "lab__urine_glucose_qualitative__text", "1+ (25 mg/dL)")
    assert (urine.ordinal_grade, urine.exact_value, urine.ordinal_score) == ("1+", "25", None)
    assert parsed(cfg, "lab__urinalysis_includes_microscopic__clarity_urinalysis__text", "Cloudy").category == "cloudy"
    urine = parsed(cfg, "lab__urine_bacteria__text", "Few (5-10)")
    assert urine.category == "few" and urine.comparator is None
    _, _, facts = parse_mapping(v2_row(cfg, "cbc_diff__anisocytosis_blood", "anisocytosis 10%"), cfg, VOCAB)
    assert facts[0].payload.percentage.exact_value == "10"


def test_v2_narrative_hla_and_conflicting_scalars(cfg):
    from src.semantic_harmonization.semantic_facts import parse_mapping

    immuno = parsed(cfg, "lab__immunofixation_serum_interpretation__text",
                    "SEE BELOW | Trace IgA (lambda) band detected; Previously identified as IgM kappa")
    assert [(f.immunoglobulin_class, f.light_chain, f.pattern, f.temporality) for f in immuno.findings] == [
        ("IgA", "lambda", None, "unspecified"), ("IgM", "kappa", None, "historical")]
    immune = parsed(cfg, "lab__immunofixation_serum_interpretation__text", "See note: No anomalous immunoglobulins")
    assert [(f.entity, f.assertion) for f in immune.findings] == [("anomalous_immunoglobulins", "absent")]
    hla = v2_row(cfg, "hla_drb1", "04:01:01G, * | Method: DRB1*04:01:02G")
    _, _, facts = parse_mapping(hla, cfg, VOCAB)
    assert [(f.payload.token, f.payload.nomenclature) for f in facts if f.fact_type == "hla_result"] == [
        ("04:01:01G", "group"), ("*", "unresolved")]
    _, _, facts = parse_mapping(v2_row(cfg, "hla_drb", "04:01/04:02"), cfg, VOCAB)
    assert facts[0].payload.alternatives == ["04:01", "04:02"] and facts[0].payload.allele is None
    semantic, _, facts = parse_mapping(v2_row(cfg, "anti_sm", "Positive | Negative"), cfg, VOCAB)
    assert semantic.qualitative_result is None and semantic.interpretation_note.startswith("Conflicting")
    assert {f.payload.category for f in facts} == {"positive", "negative"}


def test_v2_model_requires_verbatim_profile_spans_and_licensed_roles(cfg):
    from src.semantic_harmonization.inference import apply_extraction, validate_profile
    from src.semantic_harmonization.schemas import Extraction, Semantic
    from src.semantic_harmonization.segmentation import segment

    row = v2_row(cfg, "immunofixation_serum_interpretation", "Reference range: No anomalous immunoglobulins")
    finding = new_finding("anomalous_immunoglobulins", "No anomalous immunoglobulins", assertion="absent")
    with pytest.raises(ValueError, match="source span"):
        validate_extraction(serialize({"findings": [finding.model_dump()]}), row, [])
    row = v2_row(cfg, "immunofixation_serum_interpretation", "A faint lambda band was detected.")
    finding = new_finding("light_chain", "faint lambda", light_chain="lambda", intensity_descriptor="faint")
    answer = Extraction(findings=[finding])
    semantic = Semantic()
    assert len(apply_extraction(semantic, answer, row, "synthetic")) == 1
    assert not apply_extraction(semantic, answer, row, "synthetic")
    contradicted = Semantic(findings=[finding.model_copy(update={"assertion": "absent"})])
    with pytest.raises(ValueError, match="conflicts"):
        apply_extraction(contradicted, answer, row, "synthetic")
    row = v2_row(cfg, "ana_hep2_cytoplasmic_pattern", "Positive")
    part, = segment(row)
    proposal = dict(segment_id=part.segment_id, start=0, end=8, quote="Positive", document_ids=[],
                    payload=dict(compartment="cytoplasmic", pattern=None, state="positive"))
    assert len(validate_profile(serialize({"candidates": [proposal]}), row, [], cfg, VOCAB)) == 1
    for invalid in (dict(quote="positive"), dict(document_ids=["unknown"]), dict(extra="unexpected")):
        with pytest.raises(ValueError):
            validate_profile(serialize({"candidates": [proposal | invalid]}), row, [], cfg, VOCAB)


def test_v2_publication_reuses_checkpoints_and_compares_published_baseline(tmp_path, monkeypatch, cfg):
    from src.semantic_harmonization import run_pipeline as pipeline

    _, manifest = synthetic(tmp_path, cfg)
    options = dict(encoder=Encoder(), model_factory=Model, resource_verify=lambda cfg: manifest)
    original_report = pipeline.report

    def stop(*args, **kwargs):
        raise RuntimeError("synthetic stop before publication")

    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "report", stop)
        with pytest.raises(RuntimeError, match="synthetic stop"):
            pipeline.run(cfg, **options)
    assert not list(Path(cfg["paths"]["analytic"]).iterdir())
    assert not list(Path(cfg["paths"]["output"]).glob("*/run_manifest.json"))
    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "parse_mapping", stop)
        patch.setattr(pipeline, "report", original_report)
        analytic, baseline = pipeline.run(cfg, **options)
    cfg["paths"]["baseline"] = str(baseline)
    _, report_dir = pipeline.run(cfg, **options)
    summary = json.loads((report_dir / "validation_summary.json").read_text())
    assert summary["v2"]["baseline"]["status"] == "reconciled"
    assert summary["v2"]["baseline"]["added_facts"] == 0
    assert summary["v2"]["llm_outcomes"] == {"added": 1}
    assert summary["clinical_evaluation"]["status"] == "not performed"
    facts = pq.read_table(report_dir / "semantic_facts.parquet").to_pylist()
    view = pq.read_table(analytic / "patient_visit_semantic_observations.parquet").to_pylist()
    assert len({row["observation_id"] for row in view}) == len(view)
    assert {key for row in view for key in row["fact_ids"]} <= {fact["fact_id"] for fact in facts}
    future = [row for row in view if row["result_available_date"] == "2024-01-15"]
    assert len(future) == 1 and future[0]["fact_ids"] and future[0]["as_of_fact_ids"] == []


def test_v2_measurement_counts_are_nullable_and_integral():
    from src.semantic_harmonization.inventory import measurement_count

    assert measurement_count(float("nan")) is None
    assert measurement_count(3.0) == 3
    with pytest.raises(ValueError):
        measurement_count(1.5)


def test_report_reconciles_final_evidence_and_does_not_repeat_a_citation(cfg):
    from src.semantic_harmonization.parsers import narrative
    from src.semantic_harmonization.report_progress import coverage, work_for
    from src.semantic_harmonization.segmentation import segment
    from src.semantic_harmonization.semantic_facts import make_fact

    row = dict(mapping_id="mapping", variable="lab__immunofixation__text", family="immunofixation",
               concept="immunofixation", context_json="{}", raw_type="string",
               raw_json=json.dumps("IgG kappa detected; IgM lambda detected"))
    parts = segment(row)
    first, last = (part for part in parts if part.role == "patient_result")
    facts = []
    for part in (first, last):
        finding, = narrative(part.text, row["family"])
        facts.append(make_fact(row, part, "immunofixation_finding", finding, "llm_rag", "test", finding.quote))
    route = dict(reason="unparsed_result", residual_segments=2)
    full, _ = work_for(row, parts, [], facts, None, False, cfg, route, "added", False)
    assert len(full) == 2 and all(item["state_v2"] == "resolved" for item in full)
    assert all(item["state_m1"] == "pending" for item in full)
    partial, _ = work_for(row, parts, [], facts[:1], None, False, cfg, route, "added", False)
    assert [item["state_v2"] for item in partial] == ["resolved", "pending"]
    assert "IgM lambda" in partial[1]["remaining_fragment"]
    assert coverage(parts, [facts[0], facts[0]]) == coverage(parts, facts[:1])
    historical = facts[0].model_copy(update={"temporality": "historical"})
    assert not coverage([first], [historical])[first.segment_id][0]
    for raw, family, note in (("not performed", "immunofixation", "legitimate missing / administrative only"),
                              ("*", "hla", "hla_resolution_unavailable")):
        example = dict(row, raw_json=json.dumps(raw), family=family)
        work, notes = work_for(example, segment(example), [], [], None, False, cfg, {}, None, False)
        assert not work and note in notes


def test_historical_report_keeps_publications_and_disables_incompatible_deltas(tmp_path, cfg):
    from src.semantic_harmonization.report_existing import generate
    from src.semantic_harmonization.report_sources import file_manifest
    from src.semantic_harmonization.settings import digest, write_json

    _, resources = synthetic(tmp_path, cfg)
    options = dict(encoder=Encoder(), model_factory=Model, resource_verify=lambda _: resources)
    analytic, target = run(cfg, **options)
    original_qc = file_manifest(target)
    original_analytic = file_manifest(analytic)
    output = tmp_path / "historical"
    first, summary = generate(cfg, {}, output)
    assert summary["scope"] == "original_v2" and summary["cumulative"] is None
    assert stat_mode(first) == 0o700 and all(stat_mode(f) == 0o600 for f in first.iterdir())
    assert generate(cfg, {}, output)[0] == first
    mapping = pq.read_table(first / "mapping_comparison.parquet").to_pylist()
    assert any(r["raw_json"] == "0" for r in mapping)
    assert any(r["raw_json"] == "null" for r in mapping)
    assert any(r["raw_json"] == '""' for r in mapping)
    assert any(r["raw_json"] == '" "' for r in mapping)
    assert all(not r["v1_available"] and r["facts_added"] is None for r in mapping)
    raw = json.loads((target / "run_manifest.json").read_text())
    legacy_identity = dict(raw["identity"], source="synthetic-v1")
    old_id = digest(legacy_identity)
    old_qc, old_analytic = target.parent / old_id, analytic.parent / old_id
    shutil.copytree(target, old_qc)
    shutil.copytree(analytic, old_analytic)
    for name in ("semantic_facts.parquet", "deterministic_facts.parquet", "semantic_segments.parquet"):
        (old_qc / name).unlink()
    legacy_manifest = dict(raw, identity=legacy_identity, run_id=old_id, schema_version="1.0",
                           outputs=dict(report=file_manifest(old_qc, skip={"run_manifest.json"}), analytic=file_manifest(old_analytic)))
    write_json(old_qc / "run_manifest.json", legacy_manifest)
    selection = dict(target_run_id=target.name, baseline_run_id=old_id)
    second, paired = generate(cfg, selection, output)
    assert paired["scope"] == "paired" and paired["baseline_projection"] == "legacy"
    paired_rows = pq.read_table(second / "mapping_comparison.parquet").to_pylist()
    assert len(paired_rows) == len(mapping) and all(r["comparable"] for r in paired_rows)
    assert paired["affected"]["patients"] <= 2
    changed_identity = dict(legacy_identity, inputs=dict(legacy_identity["inputs"], source_events="changed"))
    changed_id = digest(changed_identity)
    changed_qc = old_qc.rename(old_qc.with_name(changed_id))
    old_analytic.rename(old_analytic.with_name(changed_id))
    write_json(changed_qc / "run_manifest.json", dict(legacy_manifest, run_id=changed_id, identity=changed_identity,
                                                    inputs=changed_identity["inputs"]))
    third, unpaired = generate(cfg, dict(selection, baseline_run_id=changed_id), output)
    assert unpaired["scope"] == "separate_scenarios" and unpaired["historical"] is None
    assert unpaired["cumulative"] is None and unpaired["incremental"] is None
    assert pq.read_table(third / "mapping_comparison.parquet").num_rows == 2 * len(mapping)
    assert file_manifest(target) == original_qc and file_manifest(analytic) == original_analytic
    with (changed_qc / "variable_inventory.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksums differ"):
        generate(cfg, dict(selection, baseline_run_id=changed_id), output)


def test_closure_recovers_prefixed_clauses_with_literal_evidence(cfg):
    from src.semantic_harmonization.closure_audit import comment_rows
    from src.semantic_harmonization.semantic_facts import parse_mapping, verify_contract
    from src.semantic_harmonization.inference import routing

    row = v2_row(cfg, "ana_hep2_status",
                 "sEe\u00a0BeLoW: | <1:80 (Negative); Method: Synthetic HEp-2")
    semantic, parts, facts = parse_mapping(row, cfg, VOCAB)
    verify_contract([row], parts, facts)
    assert (semantic.titer_numerator, semantic.titer_denominator, semantic.titer_comparator) == (1, 80, "LT")
    assert semantic.qualitative_result == "negative" and semantic.exact_value is None
    assert not routing(row, semantic, parts, facts)["send"]

    repeated = "SEE BELOW | <1:80 (Negative)"
    copied = v2_row(cfg, "ana_hep2_status", repeated, observation_comment=repeated)
    _, copied_parts, copied_facts = parse_mapping(copied, cfg, VOCAB)
    verify_contract([copied], copied_parts, copied_facts)
    titer, = [fact for fact in copied_facts if fact.fact_type == "titer_result"]
    assert titer.source_field == "raw_text"
    assert [span.source_field for span in titer.duplicate_source_spans] == ["context.observation_comment"]

    pattern = "ana_hep_2_substrate_igg__antinuclear_antibody_ana_hep_2_substrate_comment_blood"
    row = v2_row(cfg, pattern, "SEE BELOW | Cytoplasmic speckled staining pattern observed. May be associated with synthetic antibody.")
    semantic, parts, facts = parse_mapping(row, cfg, VOCAB)
    assert semantic.pattern_terms == ["cytoplasmic speckled"]
    assert semantic.qualitative_result is None and not any(f.fact_type == "qualitative_result" for f in facts)
    assert any(part.role == "general_interpretation" and "synthetic antibody" in part.text for part in parts)
    for case in ("SEE NOTE | Cytoplasmic reticular/AMA", "SEE NOTE:", "This contains see notebook"):
        row = v2_row(cfg, pattern, case, observation_comment="Cytoplasmic reticular/AMA" if case.endswith(":") else None)
        semantic, parts, facts = parse_mapping(row, cfg, VOCAB)
        verify_contract([row], parts, facts)
        if case.startswith("SEE"):
            assert semantic.pattern_terms == ["cytoplasmic reticular/ama"]
            assert {p.source_field for p in parts if p.role == "patient_result"} == (
                {"context.observation_comment"} if case.endswith(":") else {"raw_text"})
        else:
            assert not any(p.role_rule == "comment_marker" for p in parts)

    for key, body in (("anti_ccp", "<20 Units Negative, 20-39 Weak Positive, >=40 Positive"),
                      ("anti_dsdna", "<=30 Negative Measures only high avidity anti-ds DNA")):
        row = v2_row(cfg, key, f"SEE COMMENT | {body}")
        semantic, parts, facts = parse_mapping(row, cfg, VOCAB)
        counts = dict(occurrences=1, episodes=1, patients=1, source_events=0)
        records = comment_rows(row, parts, facts, facts, counts)
        assert records and {r["disposition"] for r in records} == {"metadata_only"}
        assert semantic.exact_value is None and semantic.qualitative_result is None
        assert all(f.fact_type == "observation_metadata" for f in facts)

    assert parsed(cfg, "lab__anti_sm__text", "Method: synthetic. Patient result: Negative").qualitative_result == "negative"
    result = parsed(cfg, "lab__cryoglobulin__text", "SEE BELOW | Negative at 24 hours. Will be reviewed again at 7 days.")
    assert [(f.assertion, f.temporality, f.incubation_hours, f.incubation_stage) for f in result.findings] == [
        ("absent", "current", 24, None), ("uncertain", "planned", 168, None)]
    final = parsed(cfg, "lab__cryoglobulin__text", "Final negative at 7 days").findings
    assert len(final) == 1 and final[0].incubation_stage == "final"


def test_closure_morphology_hla_and_mixed_immunofixation(cfg):
    from src.semantic_harmonization.semantic_facts import parse_mapping
    from src.semantic_harmonization.report_progress import required_components

    for name, raw, bounds, grade in (
        ("anisocytosis", "6-14% (Mild)", ("6", "14"), "mild"),
        ("hypochromasia", "6-14% (Few)", ("6", "14"), "few"),
        ("macrocytosis", "2-5% (Mild)", ("2", "5"), "mild"),
        ("poikilocytosis", "1.5-2.4% (Mild)", ("1.5", "2.4"), "mild"),
        ("polychromasia", "2-5% (Mild)", ("2", "5"), "mild"),
    ):
        _, _, facts = parse_mapping(v2_row(cfg, f"cbc_diff__{name}_blood", raw), cfg, VOCAB)
        morph, = [f.payload for f in facts if f.fact_type == "morphology_result"]
        assert (morph.percentage.range_low, morph.percentage.range_high) == bounds
        assert morph.percentage.exact_value is None and morph.percentage.raw_unit == "%"
        assert morph.descriptor == grade
    for key in ("lab__hla_dpb__text", "sero__hla_dpb__asof_value"):
        value = "03:01"
        concept, family = route(key, cfg)
        row = dict(v2_row(cfg, "hla_dpb", value), variable=key, concept=concept, family=family)
        semantic, parts, facts = parse_mapping(row, cfg, VOCAB | {"ipd": {"allele": {"DPB*03:01"}}})
        allele, = [f.payload for f in facts if f.fact_type == "hla_result"]
        assert (allele.token, allele.locus, allele.resolution, allele.allele) == ("03:01", "DPB", 2, "DPB*03:01")
        assert semantic.titer_denominator is None
        assert not required_components(parts[0], facts, family)
    result = parsed(cfg, "lab__immunofixation_serum_interpretation__text",
                    "SEE NOTE | IgG(kappa) band detected + polyclonal increase + oligoclonal lambda light chains.")
    assert [(f.immunoglobulin_class, f.light_chain, f.pattern) for f in result.findings] == [
        ("IgG", "kappa", None), (None, None, "polyclonal"), (None, "lambda", "oligoclonal")]


def test_closure_csv_roundtrip_distinguishes_blank_null_false_and_zero(tmp_path):
    import csv
    from src.semantic_harmonization.table_exports import export_csv, unescape

    table = pa.table(dict(raw=["", None, "=formula", "'literal", r"\N"],
                          flag=[False, True, None, None, False],
                          value=[0, None, -1, 2, 4],
                          nested=[[{"measurement": 0}], None, [], [None], []]))
    path = tmp_path / "fixture.parquet"
    pq.write_table(table, path)
    export_csv(path)
    with path.with_suffix(".csv").open(newline="") as stream:
        reader = csv.reader(stream)
        assert next(reader) == table.column_names
        recovered = [[unescape(value, field.type) for field, value in zip(table.schema, row)] for row in reader]
    assert recovered == [list(row.values()) for row in table.to_pylist()]
    assert path.with_suffix(".csv.schema.json").is_file()
    assert stat_mode(path.with_suffix(".csv")) == 0o600


def test_closure_compares_explicit_previous_v2_and_frozen_work(tmp_path, cfg):
    from src.semantic_harmonization.report_existing import generate
    from src.semantic_harmonization.report_sources import file_manifest

    _, resources = synthetic(tmp_path, cfg)
    options = dict(encoder=Encoder(), model_factory=Model, resource_verify=lambda _: resources)
    root = tmp_path / "comparisons"
    _, previous = run(cfg, **options)
    historical, _ = generate(cfg, {"target_run_id": previous.name}, root)
    immutable = file_manifest(historical)
    cfg["comparison"].update(previous_v2_run_id=previous.name, historical_report_id=historical.name)
    cfg["prompts"]["version"] += "-final-fixture"
    _, final = run(cfg, **options)
    path, summary = generate(cfg, {"target_run_id": final.name, "previous_v2_run_id": previous.name}, root)
    assert summary["previous_v2_scope"] == "paired" and summary["previous_v2_comparable_mappings"] > 0
    frozen = pq.read_table(path / "historical_work_crosswalk.parquet").to_pylist()
    assert frozen and summary["frozen_work"]["units"] == len(frozen)
    assert {r["reconciliation"] for r in frozen} == {"matched"}
    assert file_manifest(historical) == immutable
    assert generate(cfg, {"target_run_id": final.name, "previous_v2_run_id": previous.name}, root)[0] == path


@pytest.mark.parametrize("value,dtype", [
    (None, pa.string()), ("", pa.string()), ("  Unknown  ", pa.string()),
    ("\t Ñội\n原文 '\\N", pa.string()), ("*,*", pa.large_string()),
    (False, pa.bool_()), (0, pa.int64()), (2**63 - 1, pa.int64()),
    (float("nan"), pa.float64()), (-0.0, pa.float32()), (float("inf"), pa.float64()),
    (Decimal("103.545"), pa.decimal128(10, 3)), (1730419200000000123, pa.timestamp("ns", "UTC")),
    (17, pa.date32()), (24735, pa.time64("ns")),
    (["x", "", None], pa.list_(pa.string())),
    ([300, None, 400], pa.list_(pa.timestamp("ns"))),
    ({"x": Decimal("2.30"), "b": b"\xff"}, pa.struct([("x", pa.decimal128(10, 2)), ("b", pa.binary())])),
    ([("x", 1), ("x", 2)], pa.map_(pa.string(), pa.int64())),
    ("observed", pa.dictionary(pa.int32(), pa.string())),
])
def test_effective_arrow_scalar_codec(value, dtype):
    from src.semantic_harmonization.effective_values import decode_value, encode_value

    raw = pa.scalar(value, type=dtype)
    encoded = encode_value(raw, dtype)
    restored = decode_value(encoded, dtype)
    assert restored.type == raw.type and restored.is_valid == raw.is_valid
    assert encode_value(restored, dtype) == encoded
    if not (isinstance(value, float) and math.isnan(value)):
        assert raw.equals(restored)


@pytest.mark.parametrize("concept,value,after,origin", [
    ("anti_sm", "NEG", "negative", "current_validated"),
    ("anti_sm", "POSITIVE", "positive", "current_validated"),
    ("anti_sm", "negative", "negative", "original_unchanged"),
    ("anti_sm", "  Unknown  ", "  Unknown  ", "original_fallback"),
    ("anti_sm", "", "", "original_fallback"),
    ("anti_sm", None, None, "original_null"),
    ("anti_sm", "NEG. Unknown clause", "NEG. Unknown clause", "original_fallback"),
    ("anti_sm", "NEG | Positive", "NEG | Positive", "original_fallback"),
    ("anti_sm", "SEE BELOW | Complex finding", "SEE BELOW | Complex finding", "original_fallback"),
    ("anti_sm", "SEE NOTE | Negative; unresolved statement", "SEE NOTE | Negative; unresolved statement", "original_partial"),
    ("anti_sm", "SEE BELOW | Reference range: Negative", "SEE BELOW | Reference range: Negative", "original_fallback"),
    ("hla_dpb", "04:01:01", "04:01:01", "original_fallback"),
    ("hla_a", "*,*", "*,*", "original_fallback"),
    ("anti_ro60", "<0.2", "<0.2", "original_fallback"),
    ("ana_hep2_status", "<1:80 (Negative) ADDITIONAL INFORMATION: Synthetic associations",
     "<1:80 (Negative) ADDITIONAL INFORMATION: Synthetic associations", "original_fallback"),
    ("cryoglobulins", "SEE BELOW | Negative at 24 hours. Reviewed again at 7 days.",
     "SEE BELOW | Negative at 24 hours. Reviewed again at 7 days.", "original_fallback"),
])
def test_effective_decision_keeps_every_unrendered_component(cfg, concept, value, after, origin):
    from src.semantic_harmonization.effective_values import choose_effective
    from src.semantic_harmonization.semantic_facts import parse_mapping

    row = v2_row(cfg, concept, value)
    parsed, parts, facts = parse_mapping(row, cfg, VOCAB)
    dtype = pa.string()
    choice = choose_effective(pa.scalar(value, type=dtype), dtype, row | parsed.model_dump(),
                              parts, facts, {}, cfg["normalization"])
    assert choice.origin == origin
    assert (choice.value if choice.fact_id else value) == after
    if choice.fact_id:
        assert choice.fact_id in {fact.fact_id for fact in facts}


def test_effective_wide_and_long_are_verified_together_without_inference(tmp_path, cfg):
    from src.semantic_harmonization.effective_values import EFFECTIVE
    from src.semantic_harmonization.study_input import verify_source

    _, resources = synthetic(tmp_path, cfg)
    path = Path(cfg["paths"]["integrated"])
    original = pq.read_table(path)
    original = original.set_column(original.schema.get_field_index("lab__immunofixation__text"),
                                   "lab__immunofixation__text",
                                   pa.array(["IgG kappa band detected", None, None]))
    original = original.append_column("ext__physical_float", pa.array([float("nan"), None, -0.0]))
    original = original.append_column("ext__big", pa.array([2**63 - 1, None, 0], type=pa.int64()))
    original = original.append_column("ext__list", pa.array([["a", "", None], None, []], type=pa.list_(pa.string())))
    original = original.append_column("ext__boolean", pa.array([False, None, True]))
    pq.write_table(original, path)

    def no_inference(_):
        pytest.fail("deterministic publication initialized GPT-OSS")

    analytic, qc = run(cfg, encoder=object(), model_factory=no_inference, resource_verify=lambda _: resources)
    effective = pq.read_table(analytic / EFFECTIVE)
    assert effective.schema.equals(pq.read_schema(path), check_metadata=True)
    assert effective["lab__anti_sm__text"].to_pylist() == ["negative", "positive", "  Unknown  "]
    assert effective["ext__big"].to_pylist() == [2**63 - 1, None, 0]
    assert effective["ext__boolean"].to_pylist() == [False, None, True]
    assert effective["ext__list"].to_pylist() == [["a", "", None], None, []]
    assert effective["ext__physical_float"].is_null().to_pylist() == [False, True, False]
    rows = pq.read_table(analytic / "effective_observations.parquet").to_pylist()
    decisions = {row["variable"]: [item for item in rows if item["variable"] == row["variable"]
                                  and item["source_grain"] == "episode_variable"] for row in rows}
    assert len(rows) == 3 * len(decisions) + 3
    assert [r["raw_json"] for r in decisions["ext__physical_float"]] == ['"nan"', "null", "-0.0"]
    assert len({r["mapping_id"] for r in decisions["ext__physical_float"]}) == 3
    assert all(r["temporal_availability"] == "unverified" for r in decisions["lab__anti_sm__text"])
    assert all(r["effective_json"] == r["raw_json"] for r in rows if r["effective_origin"].startswith("original_"))
    assert all(r["decision_fact_id"] in r["accepted_fact_ids"] for r in rows if r["decision_fact_id"])
    summary = json.loads((qc / "validation_summary.json").read_text())
    assert summary["inference"]["model_initialized"] is False
    assert summary["effective_values"]["current_validated"] == 2
    assert summary["effective_values"]["new_nulls"] == 0
    selected = verify_source(analytic.name, analytic_root=analytic.parent, report_root=qc.parent)
    assert selected.require("effective_integrated") == analytic / EFFECTIVE
    assert selected.require("effective_integrated_csv_schema").is_file()
    payload = json.loads((qc / "semantic_harmonization_final.html").read_text().split(
        '<script id="payload" type="application/json">', 1)[1].split("</script>", 1)[0])
    assert any(item["effective_json"] == '"negative"' and item["effective_origin"] == "current_validated"
               for row in payload["mappings"] for item in row.get("effective_values") or [])


def test_hla_resolution_verifies_without_extending_the_reported_allele(cfg):
    from src.semantic_harmonization.semantic_facts import parse_mapping

    vocab = VOCAB | {"ipd": {"allele": {"DPB1*04:01:01", "DPB1*04:01:02", "DPB1*04:10"},
                             "g_group": {"DPB1*04:01:01G"}, "p_group": {"DPB1*04:01P"}}}
    value = "SEE BELOW | HLA typing: 04:01, 04:01:01, 04:01:01G, 04:01P, *"
    semantic, parts, facts = parse_mapping(v2_row(cfg, "hla_dpb", value), cfg, vocab)
    assert [(a.reported, a.allele, a.ipd_match, a.group) for a in semantic.alleles] == [
        ("04:01", None, "compatible_resolution", None),
        ("04:01:01", "DPB1*04:01:01", "allele", None),
        ("04:01:01G", None, "g_group", "DPB1*04:01:01G"),
        ("04:01P", None, "p_group", "DPB1*04:01P"),
        ("*", None, None, None),
    ]
    assert all(value[f.start:f.end] == f.quote for f in facts)
    incompatible, _, _ = parse_mapping(v2_row(cfg, "hla_dpb", "04:1:01"), cfg, vocab)
    assert not incompatible.alleles
    partial, _, _ = parse_mapping(v2_row(cfg, "hla_dpb", "04:11"), cfg, vocab)
    assert partial.alleles[0].ipd_match == "no_match"


@pytest.mark.parametrize("operator,unit,status", [("LT", "g/dL", "equivalent"),
                                                  ("EQ", "g/dL", "unexplained_removal"),
                                                  ("LT", "mg/dL", "unexplained_removal")])
def test_regression_checks_semantics_and_evidence_for_m_spike(cfg, operator, unit, status):
    from src.semantic_harmonization.closure_audit import regression_rows
    from src.semantic_harmonization.report_progress import canonical_storage
    from src.semantic_harmonization.schemas import NumericResult
    from src.semantic_harmonization.semantic_facts import make_fact, parse_mapping

    row = v2_row(cfg, "m_spike", "<0.2 g/dL")
    _, parts, current = parse_mapping(row, cfg, VOCAB)
    old = make_fact(row, parts[0], "numeric_result",
                    NumericResult(comparator=operator, raw_unit=unit,
                                  **({"exact_value": "0.20"} if operator == "EQ" else {"limit_value": "0.20"})),
                    "numeric_parser", "legacy")
    audit, = regression_rows(row | {"occurrences": 1}, canonical_storage(current), canonical_storage([old]),
                            None, parts, current_facts=current, reference_facts={"V1": [old]})
    assert audit["change"] == status
    assert audit["reference_evidence"] == "<0.2 g/dL"


def test_historical_value_uses_a_revalidated_fact_and_refuses_conflict(cfg):
    from src.semantic_harmonization.effective_values import choose_effective
    from src.semantic_harmonization.schemas import QualitativeResult
    from src.semantic_harmonization.semantic_facts import make_fact, parse_mapping

    row = v2_row(cfg, "anti_sm", "NEG")
    parsed, parts, facts = parse_mapping(row, cfg, VOCAB)
    prior, = facts
    source = pa.scalar("NEG")
    historical = {prior.fact_id: "sealed-reference"}
    selected = choose_effective(source, source.type, row | parsed.model_dump(), parts, [prior],
                                historical, cfg["normalization"])
    assert (selected.origin, selected.value) == ("previous_validated", "negative")
    conflict = make_fact(row, parts[0], "qualitative_result", QualitativeResult(category="positive"),
                         "fixture", "fixture")
    selected = choose_effective(source, source.type, row | parsed.model_dump(), parts, [prior, conflict],
                                historical, cfg["normalization"])
    assert selected.origin not in {"current_validated", "previous_validated"} and selected.value is None


def test_selector_requires_comparable_history_and_complete_effective_assets():
    from src.semantic_harmonization.run_pipeline import selector_blockers

    valid = dict(regression_counts={}, assessment={"V1": "paired", "Previous V2": "paired", "Recent V2": "paired"},
                 frozen_work={"units": 10}, effective_values={"integrity_failures": 0})
    assert selector_blockers(valid) == []
    assert selector_blockers(valid | {"regression_counts": {"unexplained_removal": 2}})
    assert selector_blockers(valid | {"assessment": {"Recent V2": "unavailable"}})
    assert selector_blockers(valid | {"missing_materialization_links": 1})
    assert selector_blockers(valid | {"effective_values": None})


def test_compatible_sealed_residual_can_be_revalidated_on_cpu(tmp_path, cfg):
    _, resources = synthetic(tmp_path, cfg)
    analytic, qc = run(cfg, encoder=Encoder(), model_factory=Model, resource_verify=lambda _: resources)
    cfg["comparison"]["recent_v2_run_id"] = qc.name
    cfg["comparison"]["recent_report_id"] = None

    def no_inference(_):
        pytest.fail("compatible sealed residual initialized GPT-OSS")

    new_analytic, new_qc = run(cfg, encoder=Encoder(), model_factory=no_inference, resource_verify=lambda _: resources)
    assert new_analytic != analytic
    summary = json.loads((new_qc / "validation_summary.json").read_text())
    inference = summary["inference"]
    assert not inference["model_initialized"]
    assert inference["status"] == "not_run"
    assert inference["revalidated_historical_mappings"] == inference["accepted_historical_facts"] == 1
    assert inference["fresh_residual_mappings"] == 0
    audit = pq.read_table(new_qc / "effective_value_audit.parquet").to_pylist()
    assert any(ev["reference_run_id"] == qc.name for row in audit for ev in row["evidence"])


@pytest.mark.parametrize("answer,status", [
    (None, "context_limit"), ("", "empty_response"), ("{}", "invalid_schema"),
    ('{"findings":[]}', "validated"), (TimeoutError(), "timeout"),
])
def test_model_failures_leave_literal_effective_value(cfg, answer, status):
    from src.semantic_harmonization.effective_values import choose_effective
    from src.semantic_harmonization.inference import extract_batch
    from src.semantic_harmonization.semantic_facts import parse_mapping

    class Residual:
        def messages(self, *_):
            return None

        def chat(self, *_):
            if isinstance(answer, BaseException):
                raise answer
            return [answer]

    raw = "SEE BELOW | A synthetic, unresolved kappa result."
    row = v2_row(cfg, "immunofixation", raw)
    semantic, parts, facts = parse_mapping(row, cfg, VOCAB)
    outcome, = extract_batch(Residual(), [row], {row["mapping_id"]: []}, 0, cfg=cfg, vocab=VOCAB)
    assert outcome["status"] == status and not outcome["findings"]
    result = choose_effective(pa.scalar(raw), pa.string(), row | semantic.model_dump(), parts, facts, {},
                              cfg["normalization"])
    assert result.origin in {"original_fallback", "original_partial"} and result.value is None


class SmokeModel(LocalModel):
    """Exercise the production smoke validators with a synthetic engine."""

    def __init__(self, cfg, faults=None, *, fatal_on_batch=False):
        self.cfg = cfg
        self.faults = faults or {}
        self.fatal_on_batch = fatal_on_batch
        self.submitted = []
        self.attempted = []

    def messages(self, row, retrieved, retry=False):
        return row, retrieved, retry

    def chat(self, messages, active="finding"):
        from src.semantic_harmonization.inference import InferenceUnavailable

        self.submitted.append(active)
        if self.fatal_on_batch:
            raise InferenceUnavailable("synthetic engine failure")
        return super().chat(messages, active)

    def chat_completions(self, messages, active):
        from src.semantic_harmonization.inference import (
            Completion, EvidenceError, PROFILES, eligible_segments, literal_profile_value, validate_profile,
        )
        from src.semantic_harmonization.parsers import narrative, parse_segment
        from src.semantic_harmonization.semantic_facts import part_facts

        completions = []
        for row, retrieved, retry in messages:
            raw = json.loads(row["raw_json"])
            parts = eligible_segments(row)
            if active == "finding":
                values = [item.model_dump() for part in parts for item in narrative(part.text, row["family"])]
                if raw == "A faint lambda band was detected.":
                    values = [new_finding("light_chain", "faint lambda", light_chain="lambda",
                                          intensity_descriptor="faint").model_dump()]
                field = "findings"
            else:
                values = []
                field = "candidates"
                for part in parts:
                    parsed = parse_segment(row, self.cfg, VOCAB, literal_profile_value(part.text.strip()),
                                           part.source_field, part.role)
                    for fact in part_facts(row, part, parsed, self.cfg["normalization"]["version"]):
                        if fact.fact_type != PROFILES[active][1]:
                            continue
                        payload = fact.payload.model_dump()
                        if active == "hla":
                            for name in ("lookup_locus", "normalized_report", "group", "ipd_match", "allele"):
                                payload[name] = None
                        candidate = dict(segment_id=part.segment_id, start=part.start, end=part.end, quote=part.text,
                                         document_ids=[], payload=payload)
                        try:
                            validate_profile(serialize({field: [candidate]}), row, retrieved, self.cfg, VOCAB)
                        except EvidenceError:
                            continue
                        values.append(candidate)
            if "mapping_id" not in row:
                self.attempted.append((active, raw, retry))
                fault = self.faults.get((active, raw), self.faults.get(active))
                if fault:
                    changed = fault(field, values, retry)
                    if changed is not None:
                        completions.append(changed)
                        continue
            completions.append(Completion(serialize({field: values}), "stop", True))
        return completions


def empty_smoke(field, values, retry):
    from src.semantic_harmonization.inference import Completion

    return Completion("", "stop", False)


def test_all_smoke_cases_pass_with_literal_hla_and_abstention(cfg):
    model = SmokeModel(cfg)
    result = model.smoke()
    assert result.model_available
    assert {row.status for row in result.profiles.values()} == {"passed"}
    assert len(result.audit) == 11
    assert all(row["fixture_valid"] and row["attempt"] == 1 for row in result.audit)
    assert sum(row["abstention_passed"] is True for row in result.audit) == 5
    assert any(row["case_id"] == "hla_literal" and row["accepted_count"] == 1 for row in result.audit)


@pytest.mark.parametrize("failure,case,attempts", [
    ("invalid_schema", "positive", 2), ("unsupported_evidence", "positive", 2),
    ("missing_expected_fact", "positive", 2), ("unexpected_fact", "positive", 2),
    ("failed_abstention", "abstention", 2), ("truncated_response", "positive", 2),
    ("inference_runtime_error", "positive", 2), ("empty_response", "positive", 1),
])
def test_smoke_attempt_categories_and_bounded_retry(cfg, failure, case, attempts):
    from src.semantic_harmonization.inference import Completion

    def fault(field, values, retry):
        if failure == "inference_runtime_error":
            raise TimeoutError
        if failure == "empty_response":
            return None if retry else empty_smoke(field, values, retry)
        if failure == "invalid_schema":
            return Completion("{", "stop", True)
        if failure == "truncated_response":
            return Completion(serialize({field: values}), "length", True)
        if failure == "unsupported_evidence":
            values[0]["quote"] = "Unsupported synthetic citation"
        elif failure == "missing_expected_fact":
            values.pop()
        elif failure == "unexpected_fact":
            values.append(values[0])
        else:
            values = [new_finding("cryoglobulin", "not performed").model_dump()]
        return Completion(serialize({field: values}), "stop", True)

    raw, case_id = (("not performed", "cryoglobulin_not_performed") if case == "abstention" else
                    ("negative at 24h, positive at 7d", "cryoglobulin_timed"))
    result = SmokeModel(cfg, {("finding", raw): fault}).smoke()
    bad = [row for row in result.audit if row["reason_code"]]
    assert len(bad) == attempts
    assert [(row["case_id"], row["attempt"], row["reason_code"]) for row in bad] == [
        (case_id, i, failure) for i in range(1, attempts + 1)]
    assert {row.status for name, row in result.profiles.items() if name != "finding"} == {"passed"}
    assert result.profiles["finding"].status == ("disabled" if attempts == 2 else "passed")
    if failure == "truncated_response":
        assert all(row["finish_reason"] == "length" and row["has_final_channel"] and row["accepted_count"] == 0
                   for row in bad)
    if case == "abstention":
        assert all(row["abstention_passed"] is False for row in bad)


ANA_NARRATIVE = "lab__ana_hep_2_substrate_igg__antinuclear_antibody_ana_hep_2_substrate_comment_blood__text"
CBC_MORPHOLOGY = "lab__cbc_diff__anisocytosis_blood__text"


def with_profile_residuals(tmp_path, cfg):
    master, resources = synthetic(tmp_path, cfg)
    master[ANA_NARRATIVE] = ["  SEE BELOW | IgG abnormal  ", "", None]
    master[CBC_MORPHOLOGY] = ["Morphology: few", None, "   "]
    master["lab__hla_a__text"] = ["99", "A*99", None]
    master["lab__cryoglobulins__text"] = ["negative at 24h, positive at 7d | currently indeterminate", None, None]
    master.to_parquet(cfg["paths"]["integrated"], index=False)
    return resources


@pytest.mark.parametrize("failed", [(), ("ana_narrative",), ("hla",), ("finding",),
                                     ("finding", "ana", "ana_narrative", "hla", "cbc_morphology")])
def test_profile_gating_seals_lossless_candidate_and_audit(tmp_path, cfg, failed):
    from zipfile import ZipFile

    from src.semantic_harmonization.effective_values import EFFECTIVE
    from src.semantic_harmonization.report_existing import generate
    from src.semantic_harmonization.run_pipeline import comparative_root

    resources = with_profile_residuals(tmp_path, cfg)
    def timeout(*args):
        raise TimeoutError

    faults = {name: timeout if name == "ana_narrative" else empty_smoke for name in failed if name != "finding"}
    if "finding" in failed:
        faults["finding", "negative at 24h, positive at 7d"] = empty_smoke
    model = SmokeModel(cfg, faults)
    analytic, qc = run(cfg, encoder=Encoder(), model_factory=lambda _: model, resource_verify=lambda _: resources)
    original, effective = map(pq.read_table, (cfg["paths"]["integrated"], analytic / EFFECTIVE))
    assert original.schema.equals(effective.schema, check_metadata=True)
    for name in [*KEYS, ANA_NARRATIVE, CBC_MORPHOLOGY, "lab__hla_a__text", "lab__cryoglobulins__text",
                 "lab__immunofixation__text", "lab__anti_ro60__text", "spine__sjogrens_class_patient_values"]:
        assert effective[name].equals(original[name])
    assert not set(model.submitted) & set(failed)
    assert set(model.submitted) == {"finding", "ana_narrative", "cbc_morphology"} - set(failed)
    extractions = pq.read_table(qc / "llm_extractions.parquet").to_pylist()
    mappings = {row["mapping_id"]: row for row in pq.read_table(qc / "unique_values.parquet").to_pylist()}
    assert len(extractions) == len({row["mapping_id"] for row in extractions})
    for item in extractions:
        active = profile(mappings[item["mapping_id"]])
        if active in failed:
            assert item["status"] == "profile_disabled"
            assert item["reason_code"] == ("inference_runtime_error" if active == "ana_narrative" else "empty_response")
            assert not (item["findings"] or item["facts"])
    facts = pq.read_table(qc / "semantic_facts.parquet").to_pylist()
    hla = [f["hla_result"] for f in facts if f["fact_type"] == "hla_result"]
    assert {item["token"] for item in hla} == {"99", "A*99"}
    assert all(item["allele"] is None and item["ipd_match"] == "resource_unavailable" for item in hla)
    assert {f["cryoglobulin_finding"]["incubation_hours"] for f in facts if f["fact_type"] == "cryoglobulin_finding"} == {24, 168}
    assert any(f["fact_type"] == "morphology_result" and f["strategy"] == "llm_rag"
               for f in facts) == ("cbc_morphology" not in failed)
    assert any(f["fact_type"] == "immunofixation_finding" and f["strategy"] == "llm_rag"
               for f in facts) == ("finding" not in failed)
    validation = json.loads((qc / "validation_summary.json").read_text())
    assert validation["technical_integrity"] == "passed"
    assert validation["semantic_completeness"]["status"] == "incomplete"
    assert validation["downstream_activation"]["status"] == "retained"
    assert validation["inference"]["status"] == ("degraded" if failed else "available")
    assert validation["effective_values"]["new_nulls"] == 0
    assert set(validation["inference"]["profile_summary"]) == {"finding", "ana", "ana_narrative", "hla", "cbc_morphology"}
    for name in failed:
        assert validation["inference"]["profile_summary"][name]["status"] == "disabled"
        assert validation["inference"]["profile_summary"][name]["sent_to_model_mappings"] == 0
    report_dir, _ = generate(cfg, {"target_run_id": qc.name}, comparative_root(cfg))
    names = ["inference_smoke_audit", "inference_profile_summary"]
    for directory in (qc, report_dir):
        with ZipFile(directory / "inference_smoke_qc.zip") as archive:
            assert set(archive.namelist()) == {name + suffix for name in names
                                              for suffix in (".parquet", ".csv", ".csv.schema.json")}
            for name in names:
                schema = json.loads(archive.read(name + ".csv.schema.json"))
                assert schema["sha256"] == {file: file_hash(directory / file) for file in schema["sha256"]}
    assert not (comparative_root(cfg) / "current_v2.json").exists()
    if failed and len(failed) == 5:
        healthy = SmokeModel(cfg)
        next_analytic, next_qc = run(cfg, encoder=Encoder(), model_factory=lambda _: healthy,
                                    resource_verify=lambda _: resources)
        assert healthy.submitted and next_analytic != analytic and next_qc != qc
        assert verify_source(qc.name, analytic_root=analytic.parent, report_root=qc.parent).require("effective_integrated").is_file()


@pytest.mark.parametrize("where", ["initialization", "smoke", "extraction"])
def test_fatal_optional_inference_leaves_effective_and_is_never_reused(tmp_path, cfg, where):
    from src.semantic_harmonization.inference import InferenceUnavailable

    resources = with_profile_residuals(tmp_path, cfg)

    def fatal(*args):
        raise InferenceUnavailable("synthetic engine failure")

    model = SmokeModel(cfg, {"finding": fatal} if where == "smoke" else {}, fatal_on_batch=where == "extraction")
    factory = fatal if where == "initialization" else lambda _: model
    _, qc = run(cfg, encoder=Encoder(), model_factory=factory, resource_verify=lambda _: resources)
    assert len(model.submitted) == (1 if where == "extraction" else 0)
    assert len(model.attempted) == {"initialization": 0, "smoke": 1, "extraction": 11}[where]
    rows = pq.read_table(qc / "llm_extractions.parquet").to_pylist()
    assert rows and all(row["status"] == "model_unavailable" and not (row["findings"] or row["facts"]) for row in rows)


def test_fixture_or_unclassified_runtime_error_aborts_publication(tmp_path, cfg, monkeypatch, capsys):
    from src.semantic_harmonization import inference

    resources = with_profile_residuals(tmp_path, cfg)
    model = SmokeModel(cfg)
    monkeypatch.setattr(inference, "narrative", lambda *args: [])
    with pytest.raises(ValueError, match="fixture_invalid"):
        run(cfg, encoder=Encoder(), model_factory=lambda _: model, resource_verify=lambda _: resources)
    assert '"case_id":"immunofixation_negation"' in capsys.readouterr().out
    assert not model.attempted
    monkeypatch.undo()

    def fail(_):
        raise RuntimeError("unclassified fault")

    with pytest.raises(RuntimeError, match="unclassified fault"):
        run(cfg, encoder=Encoder(), model_factory=fail, resource_verify=lambda _: resources)
    assert not list(Path(cfg["paths"]["analytic"]).iterdir())


def test_disabled_inference_withholds_model_dependent_holdout_and_rejects_tampered_history(tmp_path, cfg):
    from src.semantic_harmonization.run_pipeline import build_identity, inputs_for

    _, resources = synthetic(tmp_path, cfg)
    auto_identity = build_identity(cfg, resources, inputs_for(cfg))
    options = dict(encoder=Encoder(), resource_verify=lambda _: resources)
    _, old = run(cfg, model_factory=Model, **options)
    dictionary = pq.read_table(old / "unique_values.parquet").to_pylist()
    annotated = [row for row in dictionary if row["family"] == "immunofixation" and row["raw_json"] != "null"]
    gold = tmp_path / "gold.jsonl"
    gold.write_text("".join(serialize(dict(mapping_id=row["mapping_id"], group=str(i), family="immunofixation",
                                           expected={})) + "\n" for i, row in enumerate(annotated)))
    cfg["paths"]["gold"] = str(gold)
    cfg["model"]["inference_mode"] = "disabled"
    disabled_identity = build_identity(cfg, resources, inputs_for(cfg))
    assert auto_identity["inference_profile"] == disabled_identity["inference_profile"]
    assert auto_identity["inference_mode"] != disabled_identity["inference_mode"]

    def no_model(_):
        pytest.fail("disabled inference instantiated a model")

    cfg["comparison"]["recent_v2_run_id"] = old.name
    cfg["comparison"]["recent_report_id"] = None
    _, qc = run(cfg, model_factory=no_model, **options)
    summary = json.loads((qc / "validation_summary.json").read_text())
    assert summary["clinical_evaluation"]["status"] == "not performed"
    assert summary["clinical_evaluation"]["unavailable_profiles"] == ["finding"]
    assert {p["status"] for p in summary["inference"]["profile_summary"].values()} == {"not_run"}
    assert summary["inference"]["revalidated_historical_mappings"] == 1
    with (old / "llm_extractions.parquet").open("ab") as stream:
        stream.write(b"synthetic corruption")
    with pytest.raises(ValueError, match="checksums differ"):
        run(cfg, model_factory=no_model, **options)


def test_preflight_checks_resource_hashes_without_gpu(tmp_path, cfg, monkeypatch, capsys):
    from src.semantic_harmonization import preflight, resources as local_resources

    synthetic(tmp_path, cfg)
    paths = cfg["paths"]
    for name in ("gpt_oss", "embedding_model"):
        path = tmp_path / name
        path.write_text("synthetic resource fingerprint fixture")
        paths[name] = str(path)
    harmony = Path(paths["knowledge_dir"]) / "harmony" / "o200k_base.tiktoken"
    harmony.parent.mkdir(parents=True)
    harmony.write_text("synthetic tokenizer fingerprint fixture")
    cfg["resources"]["harmony"]["sha256"] = file_hash(harmony)
    manifest = {}
    for resource, name in (("gpt_oss", "gpt_oss"), ("sapbert", "embedding_model"),
                           ("codebook", "codebook"), ("pharma_registry", "pharma_registry")):
        manifest[resource] = dict(status="available", path=paths[name],
                                  sha256=local_resources.fingerprint(Path(paths[name]))["sha256"])
    for name, spec in cfg["resources"]["optional"].items():
        manifest[name] = dict(status="unavailable", path=spec["path"], version=spec["version"])
    (harmony.parent.parent / "resources_manifest.json").write_text(serialize({"resources": manifest}))
    monkeypatch.setattr(preflight, "load_config", lambda: cfg)
    preflight.main()
    assert "Semantic preflight passed." in capsys.readouterr().out
    Path(paths["gpt_oss"]).write_text("changed fixture")
    with pytest.raises(SystemExit, match="preflight failed"):
        preflight.main()


def test_incompatible_historical_proposal_stays_rejected_in_disabled_mode(tmp_path, cfg, monkeypatch):
    from src.semantic_harmonization import run_pipeline as pipeline

    _, resources = synthetic(tmp_path, cfg)

    def bad_candidates(references, inputs, pending, *args):
        bad = new_finding("anomalous_immunoglobulins", "Unrelated synthetic evidence").model_dump()
        return {row["mapping_id"]: [("historic", dict(mapping_id=row["mapping_id"], status="validated",
                                                    findings=[bad], facts=[]))] for row in pending}

    def no_model(_):
        pytest.fail("disabled inference instantiated a model")

    cfg["model"]["inference_mode"] = "disabled"
    monkeypatch.setattr(pipeline, "incoming_reuse", bad_candidates)
    _, qc = run(cfg, encoder=Encoder(), model_factory=no_model, resource_verify=lambda _: resources)
    extraction, = pq.read_table(qc / "llm_extractions.parquet").to_pylist()
    assert extraction["status"] == "not_run" and not extraction["findings"]
    assert extraction["reason_code"] == "inference_disabled"
    summary = json.loads((qc / "validation_summary.json").read_text())
    assert summary["inference"]["revalidated_historical_mappings"] == 0


@pytest.mark.parametrize("failure", ["writing", "new_null"])
def test_effective_write_failure_or_changed_original_blocks_publication(tmp_path, cfg, monkeypatch, failure):
    from src.semantic_harmonization.effective_values import EFFECTIVE

    resources = with_profile_residuals(tmp_path, cfg)
    write_table = pq.write_table

    def write(table, path, *args, **kwargs):
        if Path(path).name == EFFECTIVE:
            if failure == "writing":
                raise OSError("synthetic write failure")
            field = table.schema.get_field_index("lab__hla_a__text")
            table = table.set_column(field, table.schema.field(field), pa.nulls(len(table), pa.string()))
        write_table(table, path, *args, **kwargs)

    monkeypatch.setattr(pq, "write_table", write)
    model = SmokeModel(cfg, {"finding": empty_smoke})
    error, match = (OSError, "synthetic write failure") if failure == "writing" else (
        ValueError, "Effective wide changed after Parquet round-trip")
    with pytest.raises(error, match=match):
        run(cfg, encoder=Encoder(), model_factory=lambda _: model, resource_verify=lambda _: resources)
    assert not list(Path(cfg["paths"]["analytic"]).iterdir())
    assert not list(Path(cfg["paths"]["output"]).glob("*/run_manifest.json"))


def test_cli_retains_selector_and_reports_new_run_and_report_ids(tmp_path, cfg, monkeypatch, capsys):
    from src.semantic_harmonization import run_pipeline as pipeline

    resources = with_profile_residuals(tmp_path, cfg)
    reference = pipeline.comparative_root(cfg)
    reference.mkdir(parents=True)
    pointer = reference / "current_v2.json"
    prior = serialize({"target_run_id": "a" * 64, "report_id": "b" * 64})
    pointer.write_text(prior)
    model = SmokeModel(cfg, {"finding": empty_smoke})
    monkeypatch.setattr(pipeline, "load_config", lambda: cfg)
    monkeypatch.setattr(pipeline.sys, "argv", ["run_pipeline"])
    monkeypatch.setattr(pipeline, "run", lambda cfg: run(
        cfg, encoder=Encoder(), model_factory=lambda _: model, resource_verify=lambda _: resources))
    pipeline.main()
    output = capsys.readouterr().out
    assert pointer.read_text() == prior
    assert "V2 selector retained (comparison unavailable" in output
    analytic, = Path(cfg["paths"]["analytic"]).iterdir()
    report_dir, = (p for p in reference.iterdir() if p.is_dir())
    assert f"V2 run ID: {analytic.name}" in output
    assert f"Comparative report ID: {report_dir.name}" in output
    assert "Effective Parquet:" in output and "Effective CSV:" in output
