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
ANALYTIC_DATA_DIR = DATA_DIR / "analytic"
METADATA_DIR = PROJECT_ROOT / "metadata"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

# Canonical input defaults. CLI arguments in scripts may override these.
DEFAULT_ANALYTIC_DATASET = Path("/data/salazarda/data/obj1_sjd/data/raw") / "visits_long_collapsed_by_interval_codebook_corrected.parquet"
DEFAULT_POP_DISTRIBUTION_INPUT = DEFAULT_ANALYTIC_DATASET
DEFAULT_CODEBOOK = METADATA_DIR / "Consolidated_Codebook_all_columns.xlsx"

# Block A outputs.
BLOCKA_TABLES_DIR = OUTPUTS_DIR / "tables" / "blockA"
BLOCKA_QC_DIR = OUTPUTS_DIR / "qc" / "blockA"
VISIT_SPINE_PARQUET = INTERMEDIATE_DATA_DIR / "00_patient_visit_spine.parquet"
VISIT_SPINE_CSV = INTERMEDIATE_DATA_DIR / "00_patient_visit_spine.csv"
POP_LONGITUDINAL_PARQUET = INTERMEDIATE_DATA_DIR / "01_visit_level_classification.parquet"
OVERLAP_LONGITUDINAL_PARQUET = (
    INTERMEDIATE_DATA_DIR / "06_overlap_longitudinal_dx_temporal_anchor_patient_visit.parquet"
)
PROS_LONGITUDINAL_PARQUET = INTERMEDIATE_DATA_DIR / "09_pros_longitudinal_patient_visit.parquet"
INTEGRATED_LONGITUDINAL_PARQUET = ANALYTIC_DATA_DIR / "10_integrated_longitudinal_patient_visit.parquet"


def ensure_output_dirs() -> None:
    """Create standard output directories used by analysis scripts."""
    for path in (INTERMEDIATE_DATA_DIR, ANALYTIC_DATA_DIR, BLOCKA_TABLES_DIR, BLOCKA_QC_DIR):
        path.mkdir(parents=True, exist_ok=True)
