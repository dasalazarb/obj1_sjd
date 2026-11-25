"""Input contract for the planned Economic analysis."""
from pathlib import Path

import common
from src.semantic_harmonization.study_input import active_study_input


def resolve_inputs(integrated: Path = common.INTEGRATED_LONGITUDINAL_PARQUET,
                   context: Path | None = None) -> tuple[Path, Path | None, dict | None]:
    """Resolve the common episode source for a future Economic producer."""
    selected = active_study_input()
    if selected is None:
        return integrated, context, None
    selected.select("integrated", integrated, default=common.INTEGRATED_LONGITUDINAL_PARQUET)
    return selected.require("integrated"), selected.select("context", context), selected.provenance()
