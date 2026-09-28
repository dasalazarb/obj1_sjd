# Block A story map

```text
clinical episode spine + validated domain products
                       |
                       v
10_build_integrated_longitudinal_dataset.py
             patient x clinical episode
                       |
          +------------+------------+
          |                         |
          v                         v
11_integrated_baseline_       12_followup_characterization.py
characterization.py           retention / gaps / episodes
patient x official baseline
Table 1 / availability / QC
```

Step 10 is the longitudinal source of truth. Step 11 consumes every available
column on the row explicitly marked `is_clinical_baseline`; it does not derive
scores, classifications, or select laboratory measurements. Step 12 uses the
same integrated input but never defines a replacement baseline.
