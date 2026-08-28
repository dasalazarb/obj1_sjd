"""Centralized repository paths for Sjögren's disease analyses.

All analysis scripts should import paths from this module instead of hardcoding
repository-relative locations. Paths are intentionally lightweight and do not
create files on import, except via helper functions called by scripts.
"""

from pathlib import Path

# Repository root is the directory containing this file.
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERMEDIATE_DATA_DIR = DATA_DIR / "intermediate"
BLOCKA_INTERMEDIATE_DATA_DIR = INTERMEDIATE_DATA_DIR / "block_A"
ANALYTIC_DATA_DIR = DATA_DIR / "analytic"
METADATA_DIR = PROJECT_ROOT / "metadata"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Canonical input defaults. CLI arguments in scripts may override these.
DEFAULT_ANALYTIC_DATASET = Path("/data/salazarda/data/obj1_sjd/data/raw") / "visits_long_collapsed_by_interval_codebook_corrected.parquet"
DEFAULT_POP_DISTRIBUTION_INPUT = DEFAULT_ANALYTIC_DATASET
SOURCE_EPISODE_SPINE = (
    Path("/data/salazarda/data/obj1_sjd/data/raw")
    / "clinical_episode_spine_sjd.parquet"
)
DEFAULT_CODEBOOK = METADATA_DIR / "Consolidated_Codebook_all_columns.xlsx"

# Block A outputs.
BLOCKA_TABLES_DIR = OUTPUTS_DIR / "tables" / "blockA"
BLOCKA_QC_DIR = OUTPUTS_DIR / "qc" / "blockA"
# Block B observation-process outputs.  Defining paths is deliberately free of
# filesystem side effects; ``ensure_output_dirs`` performs creation explicitly.
BLOCKB_TABLES_DIR = OUTPUTS_DIR / "tables" / "blockB"
BLOCKB_FIGURES_DIR = OUTPUTS_DIR / "figures" / "blockB"
BLOCKB_QC_DIR = OUTPUTS_DIR / "qc" / "blockB"
BLOCKB_LOGS_DIR = OUTPUTS_DIR / "logs"
VISIT_SPINE_PARQUET = INTERMEDIATE_DATA_DIR / "00_patient_visit_spine.parquet"
VISIT_SPINE_CSV = INTERMEDIATE_DATA_DIR / "00_patient_visit_spine.csv"
# Episode-based Step 00 outputs.  The legacy VISIT_SPINE_* paths above remain
# unchanged temporarily so that unmigrated downstream scripts do not start
# consuming the episode spine implicitly.
EPISODE_SPINE_PARQUET = INTERMEDIATE_DATA_DIR / "00_episode_spine_all.parquet"
EPISODE_SPINE_CSV = INTERMEDIATE_DATA_DIR / "00_episode_spine_all.csv"
CLINICAL_VISIT_SPINE_PARQUET = INTERMEDIATE_DATA_DIR / "00_clinical_visit_spine.parquet"
CLINICAL_VISIT_SPINE_CSV = INTERMEDIATE_DATA_DIR / "00_clinical_visit_spine.csv"
POP_LONGITUDINAL_PARQUET = INTERMEDIATE_DATA_DIR / "01_visit_level_classification.parquet"
POP_TRANSITION_INTERVALS_PARQUET = (
    INTERMEDIATE_DATA_DIR
    / "10_pop_transition_clinical_episode_intervals.parquet"
)
OVERLAP_LONGITUDINAL_PARQUET = (
    BLOCKA_INTERMEDIATE_DATA_DIR / "06_overlap_episode_level.parquet"
)
PROS_LONGITUDINAL_PARQUET = INTERMEDIATE_DATA_DIR / "09_pros_episode_level.parquet"
LABS_EPISODE_WIDE_PARQUET = BLOCKA_INTERMEDIATE_DATA_DIR / "01_labs_episode_wide.parquet"
SEROLOGY_EPISODE_PARQUET = BLOCKA_INTERMEDIATE_DATA_DIR / "01_serology_episode_level.parquet"
INTEGRATED_LONGITUDINAL_PARQUET = ANALYTIC_DATA_DIR / "10_integrated_longitudinal_clinical_episode.parquet"
TRANSITION_EPISODE_PARQUET = INTERMEDIATE_DATA_DIR / "23_transition_episode_dataset.parquet"
TRANSITION_EPISODE_CSV = INTERMEDIATE_DATA_DIR / "23_transition_episode_dataset.csv"


def ensure_output_dirs() -> None:
    """Create standard output directories used by analysis scripts."""
    for path in (
        INTERMEDIATE_DATA_DIR, BLOCKA_INTERMEDIATE_DATA_DIR, ANALYTIC_DATA_DIR,
        BLOCKA_TABLES_DIR, BLOCKA_QC_DIR,
        BLOCKB_TABLES_DIR, BLOCKB_FIGURES_DIR, BLOCKB_QC_DIR, BLOCKB_LOGS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
