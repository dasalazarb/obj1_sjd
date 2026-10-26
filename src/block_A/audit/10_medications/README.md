# Historical Step 10 medication audits

`10_medication_visibility_audit.py` is a manual **requires old inputs** visibility audit of the historical catalogue/link/status products. Invoke its full path and original CLI; its historical counts and variables are not the new contract.

`_10_medication_documentation.py` is the **archived old adapter**, retained for historical regression tests in `tests/audit/10_medications/`. Active Step 10 never imports it. The productive reader is `_internal/step10_medications.py`, which reads only the published Step08 parquet/manifest and configured Step00 fingerprint. Neither audit validates or promotes a new scientific dataset.
