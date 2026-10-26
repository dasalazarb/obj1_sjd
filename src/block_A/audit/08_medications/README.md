# Historical medication tools

Production is `src/block_A/08_build_medications.py`; see `docs/08_medications_analytic_contract.md`. No archive module is required by that producer or Step 10.

`legacy/08_treatment_audit.py`, `08_treatment_audit_v2.py`, `08_build_medication_dataset.py`, `08_ctdb_visit_recovery.py`, `08_merge_medication_documentation.py`, `08_match_btris_episode_windows.py`, `08_build_medication_episode_links.py` and `08_prepare_full_btris_medication_inputs.py` are **requires old inputs** tools. Their protected outputs and interpretation are historical, not the new contract. They can be invoked by their full paths with their original options. The legacy classification helper imports the maintained productive implementation, including the private helper functions required by original tests. Legacy v2 imports the central maintained date parser.

`legacy/08_rebuild_full_source_pipeline.py` is **disabled historical orchestration**. It exits before executing any step because its old Step08→Step10→Step08 flow is no longer supported. Use the two separate staged commands in the contract documentation.

`diagnostics/` tools are **requires old inputs** manual audit tools, with original protected products only. In particular `08_anydate_medication_associations.py` still describes the historical provisional-Step10 procedure and frozen count assertions; it is never called or imported in production. `08_btris_exception_sensitivity.py` resolves the moved legacy matcher explicitly. The existing exception-provenance, review, sensitivity and linkage tests remain under `tests/audit/08_medications/`.

The last known artifacts are the historical `outputs/audit/08_treatments_v2/` and `08_treatments_full_source/` restricted products, described by the supplied migration baseline. Their presence and hashes were not verified in this cloud environment. All tool output must remain inside the authorized NIH environment. Never export source rows, pseudonyms, medication detail, keys or generated parquets to GitHub.
