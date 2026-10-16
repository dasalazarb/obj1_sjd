# Synthetic execution example — not the clinical cohort

These CSV/JSON files come from the healthy end-to-end regression fixture:
24 fictitious patients, three LAB concepts, four PRO scores and one overlap
feature; one requested bootstrap attempt per scenario. The formal production
configuration remains B=60. `metadata.json` records all fixture overrides.

Every CSV is labeled `SYNTHETIC_QC_ONLY_NOT_CLINICAL`. A technical PASS at B=1
does not validate topological stability or a clinical conclusion. The three
denominator examples use mocked computation failures and are independently
labeled `SYNTHETIC_DENOMINATOR_REGRESSION`.

The runtime outputs retain their normal filenames. This folder is a small,
inspectable example of their schemas and denominator behavior, not a replacement
for restricted clinical outputs. No actual patient records are included.
