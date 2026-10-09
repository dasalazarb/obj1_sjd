# SjD Research Explorer

Streamlit presentation of published SjD results, kept in `obj1_sjd/dashboard/`.
The app reads registered aggregate tables and curated figures; it does not execute
analysis scripts or read patient-level datasets. The UI and scientific labels are
in English.

## Run on your PC

Requires Python 3.10 or newer. First clone this repository, or pull the merged
dashboard change into your existing clone:

```bash
git clone https://github.com/dasalazarb/obj1_sjd.git
cd obj1_sjd/dashboard
```

For an existing clone, run `git pull` on the branch containing the dashboard and
then `cd dashboard`. Before the PR is merged, check out its branch instead:
`git fetch origin` followed by `git switch --track origin/codex/add-local-research-dashboard`.

### Windows (PowerShell or Command Prompt)

From the `dashboard` directory, install once:

```powershell
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

If Python was installed without the `py` launcher, use `python -m venv .venv`.
Start the app with:

```powershell
.venv\Scripts\python.exe run_dashboard.py
```

These commands use the virtual environment directly, so PowerShell activation
and execution-policy changes are unnecessary.

### macOS / Linux

From the `dashboard` directory, install once:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Start the app with:

```bash
.venv/bin/python run_dashboard.py
```

Open **http://localhost:8501** in your browser. Stop the app with **Ctrl+C**.
The launcher uses the selected Python environment and sets the working directory
to `dashboard/`, so the app's `.streamlit/config.toml` is loaded. With the environment
activated, you can also run `python dashboard/run_dashboard.py` from the repository
root, or `python -m streamlit run app.py` from `dashboard/`.
To select a different port, append `--server.port=8502` to the launch command.

## Where the data come from

Defaults are relative to this checkout, with no cloud-specific paths:

- `SJD_REPO_ROOT`: the containing `obj1_sjd/` directory.
- `SJD_OUTPUTS_ROOT`: `SJD_REPO_ROOT/outputs/`.

The only registered file under `data/` is the Step 11 provenance JSON. Adjacent
patient-level CSV and Parquet files are rejected. Original results remain read
only. The dashboard creates its own audit reports and PDF cache inside `dashboard/`.

GitHub contains the code, not the generated scientific outputs. A fresh clone
therefore shows **Not available for this run** until the registered aggregate
outputs exist on your PC. Use the actual result folders from the same upstream run;
the app has no runtime sample-data or test-fixture fallback. Exact paths and
producers are listed in `config/output_registry.yml` and `DASHBOARD_SPEC.md` §5.2.

If results reside elsewhere, set `SJD_OUTPUTS_ROOT` before starting the app. For
example, PowerShell uses `$env:SJD_OUTPUTS_ROOT = "C:\path\to\outputs"`; macOS/Linux
uses `export SJD_OUTPUTS_ROOT=/path/to/outputs`. Set `SJD_REPO_ROOT` separately if
the provenance JSON belongs to another checkout. Setting only `SJD_REPO_ROOT`
also changes the default outputs root to that checkout's `outputs/`.

## Audit and verification

From `dashboard/`, with its environment activated:

```bash
python scripts/audit_outputs.py
python scripts/check_no_science.py
python -m pytest -q
```

You may use the platform-specific `.venv` Python path from the install instructions
instead of activating the environment. These are dashboard software checks;
run them independently of the analysis repository's scientific test suite.

The audit lists actual assets without altering upstream results or overwriting
`objectives/objective_01/metadata/figures.csv`. Candidate mappings require human
verification. Laboratory figures are served only after their mappings are verified.
The initial catalogue contains the supplied editorial seeds; no laboratory filename
or scientific definition is guessed. See `OPEN_QUESTIONS.md` for missing assets
and investigator/upstream decisions.

Once the researcher provides the aggregate historical exports in specification
§10.4, put them in `tests/fixtures/objective_01/` and run:

```bash
python scripts/validate_export.py
```

That command reports BLOCKED while real fixtures are absent. MOCK inputs are used
only for isolated software tests and do not establish historical result validity.
Slide equivalence and logo extraction also require `pres_po1.pptx`.

## Navigation and provenance

- `?inspect=1`: exact source cells and provenance.
- `?mode=present`: one section at a time; Previous/Next and arrow keys navigate.
- `?obj=objective_01&var=complement_c4&view=longitudinal`: explorer deep link.

Search, categories, views and coverage are manifest-driven. Unavailable linked
views are reported without substitution. Original images are served without
recoloring, cropping or reinterpretation. Refresh freezes a new snapshot; a
present invalid run manifest blocks capture rather than falling back silently.
Synthesized fingerprints identify file versions but cannot establish that upstream
scripts ran together. Reviewed interpretations require a human reviewer and a
matching evidence hash in `claims.yml`.

Add studies with `scripts/new_study.py`, verified registry paths and YAML navigation
entries. Planned destinations show Coming soon. Compare mode and thumbnails remain
optional future enhancements. Keep individual-point figures local/institutional
until the investigator resolves Q6; no external dashboard deployment is included.
