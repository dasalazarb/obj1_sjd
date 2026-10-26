"""Reproducible synthetic Step 15 benchmark (no canonical data).

Run: python tests/benchmark_secondary_trajectories.py --root work/step15-benchmark
Each measured old/new scenario executes in a fresh interpreter. Results include
CLI wall time, process peak RSS, and scientific CSV equivalence (PNG hashes
excluded). Defaults retain 250 DPI for both implementations.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_secondary_trajectories_by_essdai_pattern import upstream, args_for, cli, step13, v2
from step15_reference import reference


def fixture(root):
    class Factory:
        def mktemp(self, name):
            path = root/name
            path.mkdir(parents=True)
            return path
    base = upstream.__wrapped__(Factory())
    frames = []
    for i in range(250):
        source = base["x"].loc[base["x"].patient_id.eq(f"synthetic_{i%5:03d}")]
        frame = source.iloc[np.arange(12) % len(source)].copy().reset_index(drop=True)
        dates = pd.date_range("2020-01-01", periods=12, freq="30D")
        frame["patient_id"] = f"benchmark_{i:04d}"
        frame["clinical_episode_id"] = [f"e{j:03d}" for j in range(12)]
        frame["clinical_anchor_date"] = dates
        frame["clinical_baseline_date"] = dates[0]
        frame["is_clinical_baseline"] = np.arange(12) == 0
        frame["clinical_visit_number"] = np.arange(1, 13)
        frame["time_since_clinical_baseline_days"] = (dates-dates[0]).days
        frame["time_since_clinical_baseline_years"] = (dates-dates[0]).days/365.25
        frames.append(frame)
    x = pd.concat(frames, ignore_index=True)
    # Wide upstream data, with irrelevant columns retained in the full catalog.
    x = pd.concat([x, pd.DataFrame({f"unused_{j:03d}": np.arange(len(x), dtype=float) for j in range(200)})], axis=1)
    b = x.loc[x.is_clinical_baseline].copy()
    x.to_parquet(base["integrated"], index=False)
    b.to_parquet(base["baseline"], index=False)
    context = x[["patient_id", "clinical_episode_id", "clinical_anchor_date"]].copy()
    context["lab__complement_c3__measurement_date"] = x.clinical_anchor_date.where(x["lab__complement_c3__value"].notna())
    context["lab__complement_c3__unit"] = "mg/dL"
    context["source_protocol"] = "synthetic_protocol"
    context.to_parquet(base["integrated"].with_name("10_integrated_longitudinal_context.parquet"), index=False)
    v2.write_observed_pattern_outputs_v2(x, b, step13.build_patient_absolute_change_endpoints(x), base["root"],
        input_origin="synthetic_validation", input_paths={"step10": base["integrated"], "step11": base["baseline"]})
    (root/"shape.json").write_text(json.dumps(dict(n_patients=x.patient_id.nunique(), n_rows=len(x), n_columns=len(x.columns))))
    return {k: str(base[k]) for k in ("root", "integrated", "baseline", "patterns", "manifest")}


SCENARIOS = {
    "delta3_three_dry": ["--threshold", "3", "--max-variables", "3", "--dry-run"],
    "delta3_full_dry": ["--threshold", "3", "--dry-run"],
    "both_full_dry": ["--threshold", "both", "--dry-run"],
    "delta3_numeric_png": ["--threshold", "3", "--variables", "esspri_total"],
    "both_full_png": ["--threshold", "both"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--worker", choices=("old", "new"))
    parser.add_argument("--scenario", choices=SCENARIOS)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.worker:
        inputs = {k: Path(v) for k, v in json.loads((root/"fixture.json").read_text()).items()}
        module = reference()[2] if args.worker == "old" else cli
        output = root/args.scenario/args.worker
        options = args_for(inputs, output, *SCENARIOS[args.scenario])
        options.dpi = 250
        options.progress = False
        started = time.perf_counter()
        report = module.run(options)
        wall = time.perf_counter()-started
        import resource
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        result = dict(wall_seconds=wall, peak_rss_mb=rss_mb, groups=report["groups"], variables=report["variables"], n_png=report["n_png"])
        (root/f"{args.scenario}-{args.worker}.json").write_text(json.dumps(result, indent=2)+"\n")
        return
    root.mkdir(parents=True)
    (root/"fixture.json").write_text(json.dumps(fixture(root)))
    import pyarrow.parquet as pq
    shape = pq.ParquetFile(json.loads((root/"fixture.json").read_text())["integrated"])
    caveats = f"synthetic; 250 patients, {shape.metadata.num_rows} rows, {len(shape.schema_arrow.names)} columns; 250 DPI"
    rows = []
    for scenario in SCENARIOS:
        for version in ("old", "new"):
            subprocess.run([sys.executable, __file__, "--root", str(root), "--worker", version, "--scenario", scenario], check=True)
        a, b = [json.loads((root/f"{scenario}-{v}.json").read_text()) for v in ("old", "new")]
        assert all(a[key] == b[key] for key in ("groups", "variables", "n_png"))
        scientific_equal = "report metadata; full tables checked by regression tests"
        if "png" in scenario:
            paths = list((root/scenario/"old").rglob("*.csv"))
            for path in paths:
                other = root/scenario/"new"/path.relative_to(root/scenario/"old")
                aa, bb = pd.read_csv(path), pd.read_csv(other)
                if path.name == "plot_manifest.csv":
                    aa, bb = aa.drop(columns="sha256"), bb.drop(columns="sha256")
                pd.testing.assert_frame_equal(aa, bb, check_dtype=False, rtol=1e-10, atol=1e-12)
            scientific_equal = f"yes ({len(paths)} CSVs)"
        rows.append(dict(scenario=scenario, old_wall_s=a["wall_seconds"], new_wall_s=b["wall_seconds"],
            speedup=a["wall_seconds"]/b["wall_seconds"], old_peak_rss_mb=a["peak_rss_mb"], new_peak_rss_mb=b["peak_rss_mb"],
            same_scientific_outputs=scientific_equal, caveats=caveats))
        pd.DataFrame(rows).to_csv(root/"results.csv", index=False)
        print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
