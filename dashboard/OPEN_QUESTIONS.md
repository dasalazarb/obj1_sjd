# Open questions

Unresolved scientific definitions and source assets are never filled by the dashboard. The following investigator/upstream questions are copied from specification §11.

## Investigator and upstream questions — 2026-10-07

| ID | Pregunta | Quién | Impacto |
|---|---|---|---|
| Q1 | La imagen del swimmer del PPTX (slide 4) tiene `n=162`, "Phase 1: Initial Full Evaluation" y eje en días. ¿Qué script/ejecución la produjo? El código actual dibuja un swimmer en años con N basal 159. ¿Se reemplaza por el output actual? | investigador | slide 4 |
| Q2 | Semántica de los paneles de laboratorio (`Class 1 history (no class 2)`, `Ever class 2`, `Class 4 only`) y de `Patient later transitioned to class 4`: ¿definición y origen (`SJOGRENS_CLASS`)? ¿Se publicarán N por panel en un sidecar JSON? | investigador / upstream | explorer, slides 15–19 |
| Q3 | ¿Escribir celdas vacías como `0/N (0.0%)` en `11_table1_by_pop.csv`? | upstream | H6, slide 12 |
| Q4 | ¿Publicar `summary.json` con `n_patients`, `n_clinical_episodes`, `n_variables_total`, `n_variables_table1`? | upstream | slide 5 |
| Q5 | ¿Publicar IgG y otras variables con más decimales o redondeo definido? | upstream | slide 11 |
| Q6 | ¿Las figuras con puntos individuales (swimmer, laboratorios) pueden mostrarse fuera de un entorno local/institucional? | investigador | despliegue |
| Q7 | La slide 2 se titula "diagnosis-anchored view of overlap" pero define baseline/seguimiento. ¿Mantener? | investigador | título |
| Q8 | ¿El script generador de figuras de laboratorio vive en otra rama o en el repo `eda_sjd`? ¿Nombres de archivo/convención? | investigador | `figures.csv` (H2) |
| Q9 | ¿Se reincorporan las slides retiradas (definiciones de Pop, limitaciones) como `MethodsDetails`? | investigador | §1.17 |
| Q10 | Step 12 sin `provenance.json` ni `run_manifest.json`: ¿se añadirá? | upstream | `RunBadge` |
| Q11 | Seguimiento por protocolo (11D/15D): el deck anterior lo mostraba y las exportaciones nuevas son solo `Overall`. ¿Se re-exportará con `Cohort` = `11D`/`15D`? | upstream | slides 13–14 |

## Audit findings — 2026-10-07

| ID | Section | Missing / searched | Upstream action |
|---|---|---|---|
| A1 | All data views | Only .gitkeep files exist in ../outputs. No aggregate export fixtures were attached. | Provide the aggregate CSV exports listed in specification §10.4; do not provide patient data. |
| A2 | Provenance | Registered Step 11 provenance JSON is absent. | Provide the JSON from the same run as the tables. |
| A3 | Figures | No swimmer PDF or laboratory figure files exist in outputs/figures. | Supply outputs with reviewed figures.csv mappings; candidates require human verification. |
| A4 | Visual verification | pres_po1.pptx and its logo are absent. | Supply the deck for slide comparison; the app follows the written design tokens. |
| A5 | Fixture verification | The historical export referenced by the specification is not available in this conversation. | Run the real-export validation command after supplying those aggregate files; no historical CSVs were reconstructed. |
| A6 | Source checkout | The original software audit used source checkout de835fb, while the specification cites 1dd1662. Integration is based on repository main; the actual result-run commit remains unknown. Producer names and registered output paths were checked in the available source code; no result files exist. | Provide provenance from the actual export run; the checkout commit is never substituted for a result commit. |

## Remaining validation gates

Real-export reference validation, slide equivalence and projector review remain blocked by A1–A4. The laboratory catalogue cannot be expanded beyond the provided editorial seeds without actual filenames and human curation. The interim run fingerprint freezes files but does not certify joint upstream execution; provide complete manifests to establish that contract. Optional Compare and thumbnail enhancements remain outside the current missing-source MVP.
