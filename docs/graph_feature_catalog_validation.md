# Validación del refactor Pharma 00 / Graph 01–02

Validación realizada el 2026-10-16 sobre `main` base `0c9c106`.
Todos los datos usados son sintéticos. No está disponible el parquet clínico real;
no se han inferido resultados clínicos ni alterado Step 10 o aprobaciones.

## Pruebas

```bash
python -m pytest -q \
  tests/test_graph_feature_catalog.py \
  tests/test_graph_lab_contract_v2.py \
  tests/test_graph_lab_ingestion.py \
  tests/test_pharma_lab_profile.py \
  tests/test_pharma_lab_profile_integration.py \
  tests/test_topological_phenotyping.py \
  tests/test_graph_representation_sensitivity.py \
  tests/test_graph_bootstrap_qc.py
```

Resultado final: **170 passed**, dos advertencias de deprecación de timedelta
preexistentes en un fixture. Los archivos Python modificados pasan compilación y
parseo con la gramática Python 3.10. `git diff --check` pasa. El intérprete local
es Python 3.12; no se afirma haber ejecutado el Python real de Biowulf.

Cobertura de regresiones: inventario sintético de 485 columnas / 177 stems;
12 PRO y 96 EXT; nombres compuestos; Schirmer derecho/izquierdo/mínimo;
representaciones valor/texto con coberturas independientes; NaN como texto y
float; censura y códigos inseguros; nominales y missing; ordinales explícitos;
restricciones temporales SERO; independencia entre elegibilidad técnica y
aprobación; ausencia de edad diagnóstica; cero LAB legibles; procedencia y
catálogos obsoletos; protección de runs existentes; catálogo compartido,
fingerprints, cinco familias, S4 incompleto sin impedir S0–S3 y bootstrap previo.
Las pruebas del antiguo contrato clínico se conservan con llamadas explícitas a
su helper histórico; la API normal y ambos CLI descubren el catálogo ampliado.

La colección de toda la suite falla por dos imports ya inexistentes en `main`:
`src/20_btris_visit_date_match_report.py` y
`src/studies/pharma/01_run_pharma.py`. Al excluir esos dos módulos, una ejecución
ampliada produjo **611 passed, 11 failed, 1 skipped**. Los 11 fallos se reprodujeron
en un worktree limpio del mismo `main`, sin este refactor:

* `test_comorbidities`: separación de cargas familiares.
* `test_cross_domain_correlations`: N pareado y fórmula PCS (2).
* `test_missingness_observation_process`: completitud, gap máximo y anotaciones
  Python 3.9 (3).
* `test_pharma_main_plots`: fixture sin `interval_days`.
* `test_pros_longitudinal_labels`: número de valores devueltos.
* `test_risk_factors_progression`: ponderación y resta de dominios (2).
* `test_visit_dates`: etiqueta `missing_date` frente a `no_valid_date`.

Estos fallos permanecen fuera del alcance. No se modifican sus fórmulas ni sus
expectativas para declarar una suite general verde.

## Ejecución integral sintética

Se ejecutaron Pharma 00, Graph 01 y Graph 02 completos con datos sintéticos,
`--quick-exploratory --dry-run`, para `phenotype_independent` y
`clinical_inclusive`. `dry-run` omite figuras; sí se ejecutan PCA, Mapper,
comunidades, bootstrap y comparaciones. Se conservaron las configuraciones
originales de los algoritmos; quick reduce sólo réplicas y desactiva homología.
S0 coincide con Graph 01: ambos fingerprints son iguales y la lista de
diferencias de matriz es vacía en las dos políticas. S3 y S4 conservaron las
mismas columnas y missingness. Los pesos se registraron por familia y subgrupo.

Ejemplo real de consola de esa prueba, **exclusivamente sintético**:

```text
analysis_mode=exploratory; clinical_validation=not_established
run_kind=quick_exploratory; feature_policy=phenotype_independent
episodes=72; patients=72; baselines=72; source_columns=43
namespaces: structural=7, pop=1, essdai=2, esspri=2, ovl=1,
            demo=2, lab=5, sero=4, pro=12, ext=7
PCA/Mapper: LAB=3, SERO=0, PRO=12, EXT=6, CLINICAL=3; total=24
ocular=4; Schirmer right/left/min: included/included/included
missing before imputation=12/1728 (0.6944%); after=0
PCA dimensions=10; cumulative variance=0.719291
Mapper nodes=17; connected components=5; communities=7; supported=3
SERO=EMPTY_AFTER_FILTERS: 1 future consensus + 3 temporal-not-verified
Other exclusions: outcomes, POP-definition domains, reserved ESSPRI component,
                  administrative/context fields, unverified lab text, empty EXT
```

| Política | Escenario | Features | PCA dimensiones | Comunidades |
| --- | --- | ---: | ---: | ---: |
| phenotype_independent | S0 | 24 | 10 | 7 |
| phenotype_independent | S1 | 24 | 13 | 7 |
| phenotype_independent | S2 | 22 | 13 | 6 |
| phenotype_independent | S3 | 22 | 12 | 7 |
| phenotype_independent | S4 | 22 | 12 | 7 |
| clinical_inclusive | S0 | 28 | 10 | 6 |
| clinical_inclusive | S1 | 28 | 13 | 5 |
| clinical_inclusive | S2 | 23 | 13 | 4 |
| clinical_inclusive | S3 | 23 | 13 | 5 |
| clinical_inclusive | S4 | 23 | 13 | 5 |

En ambos runs entraron seis EXT y las doce PRO sintéticas, mientras SERO quedó
vacío por las restricciones temporales. La variante clínica incorporó siete
features CLINICAL frente a tres de la independiente. Las dos réplicas quick
terminaron en los cinco escenarios, pero esa comprobación no establece
estabilidad científica ni independencia causal. El límite canónico de diez PCs
se conserva aunque no alcance 80% de varianza; los datos de consola lo muestran.

## Límites para el run real

Se debe reperfilar el parquet real en Biowulf y revisar denominadores, fuentes,
unidades, covariación y exclusiones antes de presentar resultados. Las cifras
497/159/485/177 del documento son referencias informativas, nunca condiciones
hardcodeadas. El inventario de 485 columnas de la prueba valida la forma del
contrato; no sustituye un encabezado o datos clínicos reales.

El mapa incluido tenía cinco filas de sólo cuatro campos: `MAPPED` ocupaba la
columna `s4_group_order`. Se trasladó a `mapping_status`, conservando exactamente
los grupos/subgrupos suministrados; no se inventó un orden ni disponibilidad de
columnas. No se dispone de los nombres exactos de los otros 23 stems mencionados
en el documento. Se deja su revisión clínica pendiente: el runtime los audita si
se seleccionan y únicamente S4 queda bloqueado. Los dos tests de etiquetas ACE y
hepatitis se actualizaron al mapa ya existente en la base; no se cambiaron esos
grupos clínicos.

Los artefactos agregados de trazabilidad no contienen identificadores ni datos
por paciente. Los outputs analíticos individuales históricos sólo se generaron
en directorios locales ignorados con pacientes sintéticos; no forman parte del
commit. Las instrucciones operativas y el orden de revisión están en el
[README de Graph](../src/studies/graph/README.md).
