# Step 08 — archivos de diagnóstico de medicación

Esta carpeta guarda **scripts auxiliares** de auditoría y revisión que sirvieron para investigar ambigüedades de identificación y superposición de episodios. **No son los puntos de entrada canónicos de Step 08.**

## Pipeline principal (sin cambios de ruta)

`src/block_A/08_treatment_audit_v2.py` → `08_build_medication_dataset.py` → `08_ctdb_visit_recovery.py` → `08_merge_medication_documentation.py` → `08_match_btris_episode_windows.py` → `08_build_medication_episode_links.py` → `10_build_integrated_longitudinal_dataset.py`.

`08_treatment_audit.py` (v1) y `_treatment_classification_v21.py` permanecen junto a v2 por sus dependencias. `_10_medication_documentation.py` es adaptador ACTIVO de 10_, no diagnóstico histórico.

## Diagnósticos trasladados

- `08_review_btris_episode_exceptions.py`: candidatos ambiguos y órdenes sin coincidencia.
- `08_btris_exception_sensitivity.py`: revisión temporal **sin** liberar las 15 órdenes QC-blocked.
- `08_audit_btris_exception_provenance.py`: revisión protegida de identificador y geometría de pares.
- `08_prepare_btris_review_packets.py`: paquetes de revisión humana protegidos.

Se han actualizado los imports y las rutas de pruebas para la nueva ubicación; las pruebas continúan en `tests/`.

## Reproducción opcional

Desde la raíz del repositorio, para ejecutar un diagnóstico concreto:

```bash
PY=/usr/local/Anaconda/envs/py3.10/bin/python
$PY src/block_A/medication_diagnostics/08_review_btris_episode_exceptions.py --help
$PY src/block_A/medication_diagnostics/08_btris_exception_sensitivity.py --help
$PY src/block_A/medication_diagnostics/08_audit_btris_exception_provenance.py --help
$PY src/block_A/medication_diagnostics/08_prepare_btris_review_packets.py --help
$PY -m pytest -q tests/test_08_review_btris_episode_exceptions.py tests/test_08_btris_exception_sensitivity.py tests/test_08_audit_btris_exception_provenance.py
```

## Resultados y seguridad

Los resultados ya ejecutados en Biowulf están bajo `outputs/audit/08_treatments_v2/restricted/` y en subcarpetas por estudio. **No mover, borrar ni subir a GitHub** los Parquet protegidos, la clave HMAC ni los cruces de identidad. La separación de scripts no altera los resultados, fingerprint ni la reconstrucción de datos. En cualquier nuevo run, los diagnósticos opcionales siguen escribiendo en sus subcarpetas protegidas originales.

El análisis principal conserva todas las asociaciones temporales many-to-many válidas; estos diagnósticos históricos no autorizan asignación clínica única ni equivalencia entre documentación y exposición.
