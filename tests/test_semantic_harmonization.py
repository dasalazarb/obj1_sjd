"""Synthetic regression tests for the semantic candidate and local report."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from openpyxl import Workbook

from src.semantic_harmonization.inference import supports, validate_extraction
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
    assert [(a.allele, a.suffix, a.ipd_match) for a in result.alleles] == [("DRB1*04:01:01G", "G", "g_group"), (None, None, None)]
    assert len(parsed(cfg, "lab__hla_a__text", "*,*").alleles) == 2
    assert parsed(cfg, "lab__hla_drb__text", "DRB1*04:01:01G").alleles == []
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

    def smoke(self):
        self.__class__.calls += 1

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
        "lab__immunofixation__text": ["A faint lambda band was detected.", None, None],
        "lab__ana_status__text": ["Positive 1:640", "not performed", "negative"],
        "sero__ana_status__ever_positive_through_episode": pd.Series([True, True, None], dtype="boolean"),
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

    def stop(*args):
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
