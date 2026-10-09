# AGENTS.md — SjD Research Explorer

You build a presentation layer. You are NOT the owner of any scientific result.

## Hard rules
1. Only consume outputs produced by obj1_sjd (and other registered study repos). Never compute, derive, estimate, impute, normalize, or re-interpret scientific results.
2. Every number on screen resolves through a registered ref (config/output_registry.yml + shared/loaders/refs.py). No hard-coded result numbers in .py, .md or .yml.
3. Allowed operations are only: SELECT, PARSE_CELL, FORMAT (display rounding, ROUND_HALF_UP), ORDER, LAYOUT, COUNT_METADATA_ROWS, COMPARE_PUBLISHED_EQUALITY, HASH_EVIDENCE (see DASHBOARD_SPEC.md §8.2).
4. Do not redefine cohorts, baseline, Pop thresholds, overlap rules or denominators. Quote the source; do not reimplement it.
5. Never read patient-level files (DASHBOARD_SPEC.md §5.7), modify analytical scripts or upstream data/output files, or re-run the analysis scripts. The user authorized hosting this presentation layer inside obj1_sjd/dashboard; code, configuration, caches and audit files may be written only within dashboard/ (or temporary test directories).
6. If something is missing or ambiguous: render MissingOutput, append to OPEN_QUESTIONS.md, and STOP on that point. Do not guess, mock, or fill.
7. Interpretive text is editorial (claims.yml) and human-reviewed. Never generate conclusions from numbers.
8. UI language is English. Never show internal variable ids (e.g. sf36_mcs); use labels from config/labels.yml.
9. Scientific colors (Pop groups, figure panels) are fixed; do not recolor figures.
10. Tests in tests/ must pass before any change is considered done.
