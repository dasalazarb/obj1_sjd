"""Audit name visibility from protected Step 08 catalog to Step 10 episode names.

Aggregate output ONLY; never exports patient-level records, pseudonyms, drug raw
text, medication dates, record IDs, or individual episode drug lists.
It does not alter Step 08 or Step 10 datasets or relax their QC gates.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

CATALOG = "medication_documentation_consolidated/08_medications_consolidated_documentation.parquet"
PRIMARY = "medication_episode_documentation_links/08_medication_episode_primary_links.parquet"
STATUS = "medication_episode_documentation_links/08_medication_documentation_record_status.parquet"
OUTPUT_DIR = "medication_visibility_audit"
OUTPUT_FILE = "10_medication_visibility_audit.json"
BAD_NAME = {"", "nan", "none", "null", "unknown", "unmapped", "indeterminate"}


def is_named(series: pd.Series) -> pd.Series:
    value = series.astype("string").fillna("").str.strip()
    return value.ne("") & ~value.str.lower().isin(BAD_NAME)


def audit(catalog: pd.DataFrame, primary: pd.DataFrame,
          status: pd.DataFrame) -> dict:
    required_catalog = {"documentation_record_id", "source",
                        "linked_patient_pseudonymous", "medication_name_raw",
                        "ingredient_canonical", "classification_confidence",
                        "clinical_category_primary"}
    required_primary = {"documentation_record_id", "source",
                        "linked_patient_pseudonymous", "clinical_episode_id",
                        "primary_documentation_relation",
                        "tolerance_sensitivity_only"}
    required_status = {"documentation_record_id", "n_primary_episode_relations",
                       "n_tolerance_sensitivity_relations", "original_match_status"}
    for label, frame, req in [("catalog", catalog, required_catalog),
                               ("primary", primary, required_primary),
                               ("status", status, required_status)]:
        missing = req.difference(frame.columns)
        if missing:
            raise ValueError(label + "_missing_columns:" + ",".join(sorted(missing)))
    if (catalog.documentation_record_id.isna().any()
            or catalog.documentation_record_id.duplicated().any()
            or status.documentation_record_id.isna().any()
            or status.documentation_record_id.duplicated().any()
            or primary.duplicated(["documentation_record_id", "clinical_episode_id"]).any()):
        raise ValueError("duplicate_or_missing_documentation_identity")
    if not primary.primary_documentation_relation.eq(True).all():
        raise ValueError("nonprimary_episode_relation_in_primary_input")
    if primary.tolerance_sensitivity_only.ne(False).any():
        raise ValueError("month_tolerance_in_primary_input")
    records = primary.merge(
        catalog[["documentation_record_id", "source",
                 "linked_patient_pseudonymous", "medication_name_raw",
                 "ingredient_canonical", "classification_confidence",
                 "clinical_category_primary"]],
        on="documentation_record_id", how="left", validate="many_to_one",
        suffixes=("", "_catalog"), indicator=True)
    if not records._merge.eq("both").all():
        raise ValueError("primary_document_missing_from_catalog")
    for col in ["source", "linked_patient_pseudonymous"]:
        if not records[col].astype("string").eq(
                records[col + "_catalog"].astype("string")).all():
            raise ValueError("primary_catalog_" + col + "_mismatch")
    linkage = status.merge(
        primary.groupby("documentation_record_id").size().rename("observed_n"),
        left_on="documentation_record_id", right_index=True, how="left",
        validate="one_to_one")
    if not linkage.n_primary_episode_relations.eq(
            linkage.observed_n.fillna(0)).all():
        raise ValueError("record_status_primary_counts_mismatch")
    if not set(catalog.documentation_record_id).eq(set(status.documentation_record_id)):
        raise ValueError("catalog_status_record_ids_disagree")

    catalog = catalog.copy()
    catalog["raw_present"] = is_named(catalog.medication_name_raw)
    catalog["ingredient_mapped"] = is_named(catalog.ingredient_canonical)
    records["raw_present"] = is_named(records.medication_name_raw)
    records["ingredient_mapped"] = is_named(records.ingredient_canonical)
    linked_ids = set(primary.documentation_record_id)
    catalog["primary_link"] = catalog.documentation_record_id.isin(linked_ids)
    catalog["only_raw"] = catalog.raw_present & ~catalog.ingredient_mapped
    records["only_raw"] = records.raw_present & ~records.ingredient_mapped

    def metrics(frame: pd.DataFrame) -> dict:
        return {
            "n_documentation_rows": int(len(frame)),
            "n_with_raw_medication_name": int(frame.raw_present.sum()),
            "n_with_mapped_ingredient": int(frame.ingredient_mapped.sum()),
            "n_raw_named_but_ingredient_unmapped": int(frame.only_raw.sum()),
            "n_without_raw_or_mapped_ingredient": int(
                (~frame.raw_present & ~frame.ingredient_mapped).sum()),
            "n_distinct_mapped_ingredient_strings": int(
                frame.loc[frame.ingredient_mapped, "ingredient_canonical"]
                     .str.strip().str.lower().nunique()),
            "n_distinct_raw_medication_strings": int(
                frame.loc[frame.raw_present, "medication_name_raw"]
                     .str.strip().str.lower().nunique()),
        }

    # Avoid pandas-version-specific GroupBy.apply(include_groups=...) API.
    group_keys = ["linked_patient_pseudonymous", "clinical_episode_id"]
    summary_docs = records.groupby(group_keys, dropna=False).agg(
        n_docs=("documentation_record_id", "size"),
        has_unmapped_raw=("only_raw", "any"),
    )
    mapped = records.loc[records.ingredient_mapped].copy()
    mapped["_ingredient_key"] = (
        mapped.ingredient_canonical.astype("string").str.strip().str.lower())
    unique_names = mapped.groupby(group_keys, dropna=False)["_ingredient_key"].nunique()
    episode_summary = summary_docs.join(
        unique_names.rename("n_unique_ingredients"), how="left")
    episode_summary["n_unique_ingredients"] = (
        episode_summary.n_unique_ingredients.fillna(0).astype(int))
    ep_with=episode_summary.n_unique_ingredients.gt(0)
    cohort_summary = {
        "n_episode_keys_with_documentation": int(len(episode_summary)),
        "n_episode_keys_with_mapped_medication_names": int(ep_with.sum()),
        "n_episode_keys_with_documents_but_no_mapped_names": int((~ep_with).sum()),
        "n_episode_keys_with_at_least_one_unmapped_raw_name": int(
            episode_summary.has_unmapped_raw.sum()),
        "median_documents_per_documented_episode": float(episode_summary.n_docs.median()),
        "median_distinct_mapped_ingredients_per_documented_episode": float(
            episode_summary.n_unique_ingredients.median()),
    }
    sources = {}
    for src, grp in catalog.groupby("source", dropna=False):
        source = str(src)
        active = grp.loc[grp.primary_link]
        links = records.loc[records.source.eq(src)]
        sources[source] = {
            "original_documentation": metrics(grp),
            "distinct_source_documents_with_primary_link": metrics(active),
            "primary_episode_relations": metrics(links),
            "n_original_documents_without_primary_link": int((~grp.primary_link).sum()),
            "classification_confidence_original": {
                str(k): int(v) for k, v in
                grp.classification_confidence.fillna("missing").value_counts().items()
            },
            "candidate_categories_original": {
                str(k): int(v) for k, v in
                grp.clinical_category_primary.fillna("missing").value_counts().items()
            },
        }

    return {
        "audit_version": "10-medication-visibility-1.0.0",
        "interpretation": "These are source documents and episode links, not dispensed/administered treatment.",
        "overall_original_documentation": metrics(catalog),
        "overall_original_documents_with_primary_link": metrics(
            catalog.loc[catalog.primary_link]),
        "overall_primary_episode_relations": metrics(records),
        "n_original_documents_without_primary_episode_link": int(
            (~catalog.primary_link).sum()),
        "n_tolerance_sensitivity_relations_excluded": int(
            status.n_tolerance_sensitivity_relations.sum()),
        "episode_name_visibility": cohort_summary,
        "sources": sources,
        "original_match_status_counts": {
            str(k): int(v) for k, v in
            status.original_match_status.fillna("missing").value_counts().items()
        },
        "next_review": (
            "Compare n_raw_named_but_ingredient_unmapped with mapped counts; "
            "review original restricted vocabulary if elevated. A short list of "
            "unique ingredients is not by itself evidence of lost source records."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.audit_root.resolve() / "restricted"
    tables = [root / CATALOG, root / PRIMARY, root / STATUS]
    if not all(p.is_file() for p in tables):
        raise FileNotFoundError("Step08 protected catalog/primary/status missing")
    result = audit(*(pd.read_parquet(p) for p in tables))
    folder = root / OUTPUT_DIR
    folder.mkdir(parents=True, mode=0o700, exist_ok=True)
    os.chmod(folder, 0o700)
    path = folder / OUTPUT_FILE
    import tempfile
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", prefix=".audit_", suffix=".json",
                dir=folder, delete=False) as out:
            tmp = Path(out.name)
            json.dump(result, out, indent=2, sort_keys=True)
            out.write("\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()
    print(json.dumps(result, indent=2, sort_keys=True))
    print("Protected summary:", path)


if __name__ == "__main__":
    main()
