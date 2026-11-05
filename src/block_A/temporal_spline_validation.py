"""Aggregate-only global validation of frozen Step 17 V3 references.

No patient rows, omitted-patient effects, bootstrap draws or model objects are
persisted. Checkpoints contain disclosure-controlled task aggregates only.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd

from src.block_A.secondary_trajectory_analysis import ContractError, sha256
from src.block_A.secondary_comparison_models import PATTERNS, design_row
from src.block_A.temporal_spline_config import VERSION, STEM, C_SUPPORT_POLICY, BASIS_SPECIFICATIONS
from src.block_A.temporal_spline_contract import sample_fingerprint, fingerprints, verify_unchanged
from src.block_A.temporal_spline_outputs import directories as reference_directories, safe_table
from src.block_A.temporal_spline_contrasts import function_and_gradient
from src.block_A.temporal_spline_models import fit_fixed_family
from src.block_A.temporal_spline_inference import simultaneous_critical

VALIDATION_VERSION = 'validation_global_v2'
BOOTSTRAP_POLICY = 'strict_original_support'
SENSITIVITY_POLICY = 'fixed_original_grid_conditional'
CALIBRATION_DENOMINATOR = 'max(100, ceil(0.8*n_requested))'
EARLY = (0., .5)
LATE = (1.75, 2.5)
TABLES = ('07_outcome_validation_inventory', '07_patient_temporal_coverage',
          '07_within_patient_change', '07_distribution_diagnostics', '07_influence_summary',
          '07_fit_stability', '07_patient_bootstrap_qc', '07_bootstrap_failure_diagnostics', '07_scientific_validation_classification')
# Frozen descriptive thresholds, applied to every numeric outcome/model. These
# are not clinical MCIDs. All continuous diagnostics remain in the tables.
MATERIAL = dict(loo_standardized_shift_p95=1., loo_material_direction_fraction=.2,
                maximum_refit_failure_fraction=.2, minimum_longitudinal_overlap=.2,
                minimum_band_agreement=.8, maximum_critical_ratio=1.5,
                minimum_loo_support_fraction=.8)
KEYS = ['outcome_id', 'threshold_abs', 'time_axis', 'scenario', 'basis_specification']
BASE_COLUMNS = KEYS + ['domain', 'variable', 'outcome_type', 'unit', 'model',
    'validation_version', 'validation_depth', 'status', 'reason', 'sample_fingerprint',
    'n_patients_analytic', 'n_observations_analytic', 'seed']
REFERENCE_TABLES = ('00_variable_eligibility', '00_local_temporal_support', '00_group_support',
    '00_design_diagnostics', '01_model_comparison', '01_model_coefficients', '02_global_temporal_tests')


def json_safe(value):
    """Strict JSON normalization (no NaN/Infinity or numpy scalars)."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(json_safe(value), sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def directories(root, basis, threshold):
    suffix = Path(VERSION)/VALIDATION_VERSION/basis/'clinical_baseline'/f'delta{threshold}'/'main'
    return {area: Path(root)/'outputs'/area/'blockA'/STEM/suffix for area in ('tables', 'qc')}


def audit_payload(value, patient_ids=()):
    """Audit nested JSON as well as CSV, including IDs embedded in text."""
    ids = {str(p) for p in patient_ids}
    embedded = [re.escape(pid) for pid in ids if len(pid) >= 5 and not pid.isdigit()]
    embedded_ids = re.compile('|'.join(embedded)) if embedded else None
    forbidden = {'patient_id', 'clinical_episode_id', 'variable_measurement_date',
        'clinical_baseline_date', 'essdai_index_date', 'omitted_patient', 'bootstrap_id'}
    def check(node):
        if isinstance(node, dict):
            if forbidden.intersection(node):
                raise ContractError('identifying key in validation output')
            for key, val in node.items():
                check(str(key))
                check(val)
        elif isinstance(node, (list, tuple)):
            for val in node:
                check(val)
        elif isinstance(node, str):
            # Boundary matching catches nested JSON, decorated IDs and traces;
            # short numeric IDs must not match ordinary counts or hashes.
            tokens = re.findall(r'[\w.-]+', node)
            if node in ids or any(token in ids for token in tokens) or embedded_ids and embedded_ids.search(node):
                raise ContractError('patient identity in validation output')
    check(json_safe(value))


def suppress_table(table, patient_ids=(), sensitive_outcomes=()):
    """Conservative complementary suppression across partitions and tables.

    If ANY linked patient cell is small, hide the outcome's patient-derived
    descriptive metrics in ALL diagnostic tables (including totals/percentages).
    LOO tallies and reason histograms are suppressed together if any is small.
    """
    d = table.copy()
    sensitive = d.outcome_id.isin(sensitive_outcomes) if 'outcome_id' in d else pd.Series(False, index=d.index)
    tallies = [c for c in d if c.startswith('n_') and c not in
               ('n_supported_grid_times', 'n_supported_segments')]
    if 'calibration_valid' in d:
        # Bootstrap draws are not individuals. One failed replicate cannot
        # identify a patient; preserve its audit denominator and failure rate.
        tallies = [c for c in tallies if c not in ('n_requested', 'n_attempted', 'n_successful', 'n_failed')]
    small = pd.Series(False, index=d.index)
    for col in tallies:
        numeric = pd.to_numeric(d[col], errors='coerce')
        small |= numeric.between(1, 4)
    for col in ('refit_failure_category', 'support_loss_histogram'):
        if col in d:
            def has_small_tally(value):
                if not isinstance(value, str):
                    return False
                return any(0 < count < 5 for count in json.loads(value or '{}').values())
            small |= d[col].map(has_small_tally)
    hide = sensitive | small
    # Metadata numbers (seed, threshold, model dimensions, time grid) do not
    # describe a patient's measurement. All other numeric metrics are linked.
    keep = {'seed', 'threshold_abs', 'queue_position', 'reference_q_value', 'reference_family_n_tests',
            'reference_group_curvature_q', 'reference_group_curvature_family_n_tests',
            'n_supported_grid_times', 'n_supported_segments', 'supported_duration_years',
            'supported_grid_span_years', 'early_lower_years', 'early_upper_years',
            'late_lower_years', 'late_upper_years', 'duration_seconds', 'reference_comparable'}
    for col in d:
        if col not in keep and pd.api.types.is_numeric_dtype(d[col]) and not pd.api.types.is_bool_dtype(d[col]):
            d.loc[hide, col] = np.nan
    for col in ('failure_reasons', 'reference_tests'):
        if col in d and col == 'failure_reasons':
            d.loc[hide, col] = '{}'
    for col in ('refit_failure_category', 'support_loss_histogram'):
        if col in d:
            d.loc[hide, col] = '{}'
    for col in d:
        if pd.api.types.is_bool_dtype(d[col]) and col not in ('eligible_for_deep', 'reference_comparable'):
            d[col] = d[col].astype(object)
            d.loc[hide, col] = None
    for col in ('loo_stability_status', 'bootstrap_calibration_status', 'validation_reason_codes'):
        if col in d:
            d.loc[hide, col] = 'DISCLOSURE_SUPPRESSED'
    for col in ('scientific_validation_status', 'scientific_status'):
        if col in d:
            d.loc[hide, col] = 'INCONCLUSIVE'
    if 'calibration_valid' in d and 'status' in d:
        d.loc[hide & d.status.isin(['CALIBRATED', 'INCONCLUSIVE']), 'status'] = 'DISCLOSURE_SUPPRESSED'
    if 'scientific_validation_status' in d:
        d.loc[hide, 'status'] = 'INCONCLUSIVE'
    d['disclosure_status'] = np.where(hide, 'DISCLOSURE_SUPPRESSED', 'AGGREGATE')
    safe_table(d, patient_ids)
    audit_payload(d.to_dict('records'), patient_ids)
    return d


def authenticate_reference(root, basis, threshold, input_origin):
    """Authenticate the COMPLETE source bundle before any outcome selection."""
    paths = reference_directories(root, 'clinical_baseline', threshold, 'main', basis_specification=basis)
    path = paths['qc']/'run_manifest.json'
    manifest_hash = sha256(path)
    m = json.loads(path.read_text())
    expected = dict(version=VERSION, basis_specification=basis, threshold_abs=threshold,
        time_axis='clinical_baseline', scenario='main', analysis='all', input_origin=input_origin,
        c_support_policy=C_SUPPORT_POLICY, support_policy=C_SUPPORT_POLICY,
        qc_status='PASS', fdr_selection_scope='all_registered_variables')
    if any(m.get(k) != v for k, v in expected.items()):
        raise ContractError('NOT_COMPARABLE_REFERENCE: reference identity/policy/full FDR scope')
    if not m.get('inputs') or not m.get('code_hashes') or not m.get('data_cutoff'):
        raise ContractError('NOT_COMPARABLE_REFERENCE: missing provenance')
    for snapshot in (m['inputs'], m['code_hashes']):
        if any(not isinstance(item, dict) or not re.fullmatch(r'[0-9a-f]{64}', str(item.get('sha256', '')))
               for item in snapshot.values()):
            raise ContractError('NOT_COMPARABLE_REFERENCE: invalid provenance hashes')
    artifacts = m.get('artifacts', [])
    if len({a['path'] for a in artifacts}) != len(artifacts):
        raise ContractError('duplicate reference artifacts')
    snapshot = {'reference_manifest': dict(path=str(path), sha256=manifest_hash)}
    authenticated = set()
    for index, entry in enumerate(artifacts):
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ContractError('unsafe reference artifact path')
        area = 'figures' if relative.parts[0] == 'figures' else 'tables'
        item = paths[area]/(Path(*relative.parts[1:]) if area == 'figures' else relative)
        if item.is_symlink() or not item.resolve().is_relative_to(paths[area].resolve()):
            raise ContractError('unsafe reference artifact')
        if not item.is_file() or sha256(item) != entry['sha256']:
            raise ContractError('NOT_COMPARABLE_REFERENCE: artifact hash')
        snapshot[f'reference_artifact_{index}'] = dict(path=str(item), sha256=entry['sha256'])
        if area == 'tables':
            authenticated.add(relative.as_posix())
    if not {name+'.csv' for name in REFERENCE_TABLES}.issubset(authenticated):
        raise ContractError('NOT_COMPARABLE_REFERENCE: incomplete artifacts')
    tables = {name: pd.read_csv(paths['tables']/(name+'.csv'), keep_default_na=True)
              for name in REFERENCE_TABLES}
    for table in tables.values():
        for col in ('unit', 'model_family'):
            if col in table:
                table[col] = table[col].fillna('')
    inventory = tables['00_variable_eligibility']
    if inventory.empty or inventory.outcome_id.duplicated().any():
        raise ContractError('invalid complete reference inventory')
    for name, table in tables.items():
        if not set(KEYS).issubset(table):
            if len(table):
                raise ContractError('reference missing semantic keys: '+name)
            # Original V3's readable empty schema predates semantic-key columns
            # on wholly unestimated artifacts. Never synthesize nonempty rows.
            for key in KEYS:
                if key not in table:
                    table[key] = pd.Series(dtype=inventory[key].dtype)
        for key, val in expected.items():
            if key in table and not table[key].eq(val).all():
                raise ContractError('reference row identity mismatch')
        if not set(table.outcome_id).issubset(set(inventory.outcome_id)):
            raise ContractError('reference contains unregistered outcome')
    models = tables['01_model_comparison']
    if models.duplicated(KEYS+['model']).any() or set(models.model) != set('ABC'):
        raise ContractError('invalid model reference keys')
    if not models.groupby('outcome_id').model.nunique().eq(3).all() or set(models.outcome_id) != set(inventory.outcome_id):
        raise ContractError('incomplete ABC inventory')
    tests = tables['02_global_temporal_tests']
    if tests.duplicated(KEYS+['test']).any():
        raise ContractError('duplicate test reference keys')
    verify_unchanged(snapshot)
    return dict(manifest=m, tables=tables, snapshot=snapshot)


def _same_number(a, b):
    if pd.isna(a):  # upstream disclosure suppression; compare all other contracts
        return True
    return np.isclose(float(a), float(b), atol=1e-8, rtol=1e-8, equal_nan=True)


def reference_comparison(reference, evaluation, spec, meta, inputs):
    """Outcome-specific comparability, never a join on variable alone."""
    t, ev = reference['tables'], evaluation
    inv = t['00_variable_eligibility'].set_index('outcome_id')
    oid = meta['outcome_id']
    if oid not in inv.index:
        return False, 'outcome_not_in_reference'
    r, reasons = inv.loc[oid], []
    for key in ('domain', 'variable', 'unit', 'outcome_type', 'threshold_abs', 'time_axis', 'scenario', 'basis_specification'):
        if r[key] != meta[key]:
            reasons.append('identity_'+key)
    if r.sample_fingerprint != sample_fingerprint(ev['data']):
        reasons.append('sample_fingerprint')
    for field, actual in [('n_patients_analysis', ev['gate']['n_patients']),
                          ('n_observations', len(ev['data']))]:
        if field not in r or not _same_number(r[field], actual):
            reasons.append(field)
    actual_basis = ev['basis'].as_dict() if ev['basis'] else None
    reference_basis = json.loads(r.basis) if isinstance(r.basis, str) and r.basis else None
    if digest(actual_basis) != digest(reference_basis):
        reasons.append('frozen_basis')
    for model in 'ABC':
        if bool(r['eligible_'+model]) != ev['gate']['eligible_'+model]:
            reasons.append('eligibility_'+model)
    ref_inputs = reference['manifest']['inputs']
    for role, item in inputs.items():
        if role == 'config':
            continue
        if role not in ref_inputs or item['sha256'] != ref_inputs[role]['sha256']:
            reasons.append('upstream_hash')
    # Same matrices and same support are necessary even if patient totals were
    # suppressed in the reference. Do not compare disclosure-suppressed counts.
    for name, current, keys, columns in (
        ('00_design_diagnostics', pd.DataFrame(ev['designs']), ['model'], ['design_rank', 'n_columns', 'design_condition']),
        ('00_local_temporal_support', ev['support'], ['observed_pattern', 'time'], ['local_supported', 'four_group_supported', 'n_observed_local']),
        ('00_group_support', ev['counts'], ['observed_pattern'], ['n_patients', 'n_observations', 'n_three_distinct_dates'])):
        old = t[name].loc[t[name].outcome_id.eq(oid)]
        if len(old) != len(current):
            reasons.append(name)
            continue
        if not len(old):
            continue
        old, current = old.sort_values(keys).reset_index(drop=True), current.sort_values(keys).reset_index(drop=True)
        for col in keys+columns:
            if col not in old or col not in current:
                reasons.append(name)
            elif pd.api.types.is_numeric_dtype(current[col]):
                if not all(_same_number(a, b) for a, b in zip(old[col], current[col])):
                    reasons.append(name+'_'+col)
            elif not old[col].equals(current[col]):
                reasons.append(name+'_'+col)
    return not reasons, ';'.join(sorted(set(reasons)))


def diagnostic_type(spec):
    if (spec.type not in ('numeric', 'binary') or spec.unit in ('count', 'score', 'foci/4 mm²') or
            spec.is_essdai_derived or not spec.expected_longitudinal or
            spec.observation_semantics in ('historical_ever', 'cumulative_through_episode') or
            spec.variable_id.startswith(('ocular_oxford', 'ocular_van_bijsterveld', 'sgus_summary'))):
        return 'NOT_APPLICABLE_TYPE'
    return 'APPLICABLE'


def supported_times(data, times, cfg):
    """Intersect original grid with refit support; never fill temporal gaps."""
    supported = []
    for t in times:
        valid = True
        for g in PATTERNS:
            d = data.loc[data.observed_pattern.eq(g)]
            local = d.loc[d.time.sub(t).abs().le(cfg.prediction_window_years/2)]
            if (d.empty or not d.time.min() <= t <= d.time.max() or
                    local.patient_id.nunique() < cfg.minimum_prediction_patients):
                valid = False
                break
        if valid:
            supported.append(float(t))
    return supported


def original_times(evaluation):
    s = evaluation['support']
    return sorted(s.loc[s.four_group_supported, 'time'].unique())


def systematic(evaluation, groups, spec, meta, cfg):
    """Every outcome gets four coverage, change and distribution registry rows."""
    data, coverage, changes, distributions = evaluation['data'], [], [], []
    if data.empty:
        # Upstream returns a columnless frame for absent/unusable outcomes.
        data = pd.DataFrame(columns=['patient_id', 'observed_pattern', 'time',
                                     'variable_measurement_date', 'value'])
    applicable = diagnostic_type(spec) == 'APPLICABLE'
    times = original_times(evaluation)
    segments = 1+int(np.sum(np.diff(times) > cfg.grid_step*1.01)) if times else 0
    duration = float(np.sum([b-a for a, b in zip(times, times[1:]) if b-a <= cfg.grid_step*1.01]))
    sensitive = False
    for group in PATTERNS:
        d = data.loc[data.observed_pattern.eq(group)] if len(data) else data
        cohort = int(groups.observed_pattern.eq(group).sum())
        sizes = d.groupby('patient_id').variable_measurement_date.nunique() if len(d) else pd.Series(dtype=float)
        spans = d.groupby('patient_id').time.agg(lambda s: s.max()-s.min()) if len(d) else pd.Series(dtype=float)
        early = d.loc[d.time.between(*EARLY)] if len(d) else d
        late = d.loc[d.time.between(*LATE)] if len(d) else d
        e, l = set(early.patient_id), set(late.patient_id)
        both, only_e, only_l, neither = len(e & l), len(e-l), len(l-e), cohort-len(e | l)
        counts = [cohort, len(sizes), len(d), len(e), len(l), both, only_e, only_l, neither,
                  int(sizes.ge(2).sum()), int(sizes.ge(3).sum())]
        sensitive |= any(0 < n < 5 for n in counts)
        row = dict(meta, observed_pattern=group, status='SYSTEMATIC_COMPLETE' if len(d) else 'INSUFFICIENT_DATA',
            reason='' if len(d) else 'no_analytic_observations', n_group_denominator=cohort,
            n_unique_patients_analytic=len(sizes), n_observations_analytic=len(d),
            n_patients_ge2_dates=int(sizes.ge(2).sum()), n_patients_ge3_dates=int(sizes.ge(3).sum()),
            n_unique_early=len(e), n_unique_late=len(l), n_unique_both=both,
            n_only_early=only_e, n_only_late=only_l, n_neither=neither,
            pct_both_of_group=100*both/cohort if cohort else np.nan,
            pct_both_of_early=100*both/len(e) if e else np.nan,
            both_of_group_denominator='n_group_denominator', both_of_early_denominator='n_unique_early',
            n_supported_grid_times=len(times), n_supported_segments=segments,
            supported_duration_years=duration, supported_grid_span_years=float(np.ptp(times)) if times else 0.,
            early_lower_years=EARLY[0], early_upper_years=EARLY[1],
            late_lower_years=LATE[0], late_upper_years=LATE[1], horizon_years=3.,
            grid_interpretation='Overlapping local windows are not independent patients or continuous individual histories')
        for label, values in [('distinct_dates', sizes), ('followup_span_years', spans)]:
            for q, name in [(.25, 'p25'), (.5, 'median'), (.75, 'p75')]:
                row[label+'_'+name] = float(values.quantile(q)) if len(values) >= 5 else np.nan
        coverage.append(row)
        status = 'ESTIMATED' if applicable else 'NOT_APPLICABLE_TYPE'
        paired = pd.Series(dtype=float)
        if applicable and len(d):
            early_values = early.groupby('patient_id').value.mean()
            late_values = late.groupby('patient_id').value.mean()
            paired = (late_values-early_values).dropna()
        if applicable and len(paired) < 5:
            status = 'INSUFFICIENT_DATA'
        signs = [int(paired.gt(0).sum()), int(paired.lt(0).sum()), int(paired.eq(0).sum())]
        sensitive |= any(0 < n < 5 for n in signs+[len(paired)])
        changes.append(dict(meta, observed_pattern=group, status=status,
            reason='' if status == 'ESTIMATED' else 'paired_change_requires_valid_type_and_five_paired_patients',
            n_paired=len(paired), n_positive=signs[0], n_negative=signs[1], n_zero=signs[2],
            median_change=paired.median() if status == 'ESTIMATED' else np.nan,
            p25_change=paired.quantile(.25) if status == 'ESTIMATED' else np.nan,
            p75_change=paired.quantile(.75) if status == 'ESTIMATED' else np.nan,
            iqr_change=paired.quantile(.75)-paired.quantile(.25) if status == 'ESTIMATED' else np.nan,
            change_scale='late_minus_early_mean_binary_probability' if spec.type == 'binary' else spec.unit,
            aggregation='One arithmetic mean per patient per disjoint window; paired patient denominator'))
        for window, part in [('early', early), ('late', late), ('full_horizon', d)]:
            values = part.groupby('patient_id').value.mean() if applicable and len(part) else pd.Series(dtype=float)
            enough = applicable and len(values) >= 5
            median = float(values.median()) if enough else np.nan
            mad = float((values-median).abs().median()) if enough else np.nan
            tails = int((values-median).abs().gt(6*1.4826*mad).sum()) if enough and mad > 0 else 0
            sensitive |= 0 < tails < 5
            distributions.append(dict(meta, observed_pattern=group, window=window,
                status='ESTIMATED' if enough else 'INSUFFICIENT_DATA' if applicable else 'NOT_APPLICABLE_TYPE',
                reason='' if enough else 'distribution_requires_valid_type_and_five_patients',
                n_unique_patients=len(values), n_observations=len(part),
                mean=values.mean() if enough else np.nan, median=median, mad=mad,
                p25=values.quantile(.25) if enough else np.nan, p75=values.quantile(.75) if enough else np.nan,
                iqr=values.quantile(.75)-values.quantile(.25) if enough else np.nan,
                skewness=values.skew() if enough else np.nan,
                n_robust_tail_patients=tails,
                audit_signal='REVIEW_ASYMMETRY_OR_ROBUST_TAILS' if enough and (abs(values.skew()) > 2 or tails) else 'NO_AUTOMATIC_EXCLUSIONS',
                units_status='MATCHED_AUTHENTICATED_CATALOG',
                aggregation='One patient mean per window; descriptive audit only'))
    return dict(coverage=pd.DataFrame(coverage), changes=pd.DataFrame(changes),
                distributions=pd.DataFrame(distributions), sensitive=sensitive,
                minimum_continuity_fraction=min((
                    (r['n_unique_both'] if r['n_unique_early'] and r['n_unique_late'] else r['n_patients_ge2_dates']) /
                    r['n_group_denominator'] if r['n_group_denominator'] else 0.) for r in coverage),
                continuity_rule='Paired-window overlap when both windows observed; otherwise full-horizon repeated-patient fraction',
                minimum_overlap=min((r['n_unique_both']/r['n_group_denominator']
                    if r['n_group_denominator'] else 0.) for r in coverage))


def inventory_rows(reference, evaluation, spec, meta, comparable, reason, depth):
    rows = []
    tables = reference['tables']
    models = tables['01_model_comparison']
    tests = tables['02_global_temporal_tests']
    oid = meta['outcome_id']
    for model in 'ABC':
        ref = models.loc[models.outcome_id.eq(oid) & models.model.eq(model)]
        test_name = dict(A='linear_group_time', B='common_nonlinearity', C='global_temporal_heterogeneity')[model]
        tt = tests.loc[tests.outcome_id.eq(oid) & tests.test.eq(test_name)]
        curvature = tests.loc[tests.outcome_id.eq(oid) & tests.test.eq('group_specific_curvature')]
        ref = ref.iloc[0] if len(ref) else {}
        fitted = ref.get('fit_status') == 'ESTIMATED'
        applicable = diagnostic_type(spec) == 'APPLICABLE'
        status = ('REFERENCE_MISMATCH' if not comparable else 'NOT_APPLICABLE_TYPE' if not applicable else
                  'NOT_ELIGIBLE' if not evaluation['gate']['eligible_'+model] else
                  'NOT_ESTIMABLE' if not fitted else 'DEEP_PENDING' if depth == 'deep' else 'SYSTEMATIC_COMPLETE')
        eligible = bool(comparable and applicable and fitted)
        rows.append(dict(meta, model=model, status=status, reason=reason if not comparable else
            '' if eligible else evaluation['gate']['reason_'+model] or 'reference_model_not_fitted',
            eligible=evaluation['gate']['eligible_'+model], fit_status=ref.get('fit_status', 'NOT_ESTIMABLE'),
            test_status=tt.iloc[0].status if len(tt) else 'NOT_ESTIMABLE',
            model_family=ref.get('model_family', ''), reference_comparable=comparable,
            eligible_for_deep=eligible, deep_reason='FITTED_REFERENCE' if eligible else status,
            reference_q_value=tt.iloc[0].q_value if len(tt) else np.nan,
            reference_family_n_tests=tt.iloc[0].family_n_tests if len(tt) else 0,
            reference_group_curvature_q=curvature.iloc[0].q_value if model == 'C' and len(curvature) else np.nan,
            reference_group_curvature_family_n_tests=curvature.iloc[0].family_n_tests if model == 'C' and len(curvature) else 0,
            fdr_policy='Unchanged complete V3 BH family; descriptive reference only',
            validation_coverage=status))
    return rows


def frozen_plan(rows, priority_outcomes=(), seed=1701):
    """Stratified round-robin queue, independent of p/q and influence results.

    Within each model/type/threshold/basis/domain cell, clinical IDs (optional,
    supplied before fitting) precede seeded hash ties. One per nonempty stratum
    per round is the quota. All candidates stay in the queue.
    """
    priority = set(priority_outcomes)
    buckets = defaultdict(list)
    for row in rows:
        if not row['eligible_for_deep']:
            continue
        key = tuple(row[k] for k in ('model', 'outcome_type', 'threshold_abs', 'basis_specification', 'domain'))
        task = {k: json_safe(row[k]) for k in KEYS+['model', 'domain', 'variable', 'outcome_type', 'model_family']}
        task['task_id'] = digest({k: task[k] for k in KEYS+['model']})
        buckets[key].append(task)
    for key in buckets:
        buckets[key].sort(key=lambda t: (t['outcome_id'] not in priority, digest([seed, t['task_id']])))
    queue = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                queue.append(dict(buckets[key].pop(0), queue_position=len(queue)+1))
    return dict(selection='one_per_nonempty_model_type_threshold_basis_domain_stratum_per_round',
        priority_outcome_ids=sorted(priority), seed=seed, tie_break='sha256(seed, semantic task key)',
        significance_used=False, materiality=MATERIAL, tasks=queue)


def target_family(evaluation, fit, binary=False):
    """Frozen estimands: A slopes, B curvature/curves, C original pair sup-t."""
    times, basis = original_times(evaluation), evaluation['basis']
    if not times:
        return [], np.empty((0, len(fit.beta))), np.array([])
    rows, vectors = [], []
    def add(group, kind, time, gradient, contrast=''):
        rows.append(dict(observed_pattern=group, estimand=kind, time=float(time), contrast=contrast))
        vectors.append(gradient)
    if fit.model == 'A':
        slopes = {g: design_row(g, 1)-design_row(g, 0) for g in PATTERNS}
        for g in PATTERNS:
            add(g, 'slope_log_odds' if binary else 'slope', times[0], slopes[g])
        for a, b in combinations(PATTERNS, 2):
            add('', 'slope_contrast_log_odds' if binary else 'slope_contrast', times[0], slopes[b]-slopes[a], b+' - '+a)
    elif fit.model == 'B':
        for t in times:
            add('shared', 'shared_curvature', t, basis.row(PATTERNS[0], t, 'B', derivative=2))
            for g in PATTERNS:
                _, gradient = function_and_gradient(basis, fit, g, t, binary=binary)
                add(g, 'level', t, gradient)
    else:
        for kind in ('level', 'change', 'velocity'):
            for t in times:
                for a, b in combinations(PATTERNS, 2):
                    _, ga = function_and_gradient(basis, fit, a, t, kind, times[0], binary)
                    _, gb = function_and_gradient(basis, fit, b, t, kind, times[0], binary)
                    add('', kind, t, gb-ga, b+' - '+a)
    gradients = np.asarray(vectors)
    values = gradients@fit.beta
    if binary and fit.model != 'A':
        # Binary bootstrap is refused; LOO supports response-scale functions.
        values = target_values(rows, evaluation, fit, binary)
    return rows, gradients, values


def target_values(rows, evaluation, fit, binary=False):
    basis, values, times = evaluation['basis'], [], original_times(evaluation)
    if fit.model == 'A' or not binary:
        if fit.model == 'A':
            slopes = {g: design_row(g, 1)-design_row(g, 0) for g in PATTERNS}
            for row in rows:
                gradient = (slopes[row['observed_pattern']] if not row['contrast'] else
                    slopes[row['contrast'].split(' - ')[0]]-slopes[row['contrast'].split(' - ')[1]])
                values.append(float(gradient@fit.beta))
        else:
            for row in rows:
                if row['estimand'] == 'shared_curvature':
                    gradient = basis.row(PATTERNS[0], row['time'], 'B', derivative=2)
                elif row['contrast']:
                    b, a = row['contrast'].split(' - ')
                    _, gb = function_and_gradient(basis, fit, b, row['time'], row['estimand'], times[0])
                    _, ga = function_and_gradient(basis, fit, a, row['time'], row['estimand'], times[0])
                    gradient = gb-ga
                else:
                    _, gradient = function_and_gradient(basis, fit, row['observed_pattern'], row['time'])
                values.append(float(gradient@fit.beta))
        return np.asarray(values)
    for row in rows:
        if row['estimand'] == 'shared_curvature':
            # B's shared curvature is explicitly on the link scale for binary.
            values.append(float(basis.row(PATTERNS[0], row['time'], 'B', derivative=2)@fit.beta))
        elif row['contrast']:
            b, a = row['contrast'].split(' - ')
            vb, _ = function_and_gradient(basis, fit, b, row['time'], row['estimand'], times[0], True)
            va, _ = function_and_gradient(basis, fit, a, row['time'], row['estimand'], times[0], True)
            values.append(vb-va)
        else:
            v, _ = function_and_gradient(basis, fit, row['observed_pattern'], row['time'], binary=True)
            values.append(v)
    return np.asarray(values)


def family_name(model):
    return dict(A='four_group_slopes_and_six_slope_contrasts',
                B='shared_curvature_and_four_group_supported_curves',
                C='six_pairs_x_supported_grid_x_level_change_velocity')[model]


def loo_summary(evaluation, fit, spec, cfg, seed):
    """Whole-trajectory deletion, aggregated across deletions; never rank a person."""
    data = evaluation['data']
    rows, gradients, estimates = target_family(evaluation, fit, spec.type == 'binary')
    se = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', gradients, fit.covariance, gradients), 0)) if len(rows) else np.array([])
    failures, shifts, relative, signs, agreements, support_fractions = Counter(), [], [], [], [], []
    material_refits = 0
    by_estimand = defaultdict(list)
    requested, successful, noncomparable = data.patient_id.nunique(), 0, 0
    original_grid = original_times(evaluation)
    analytic = simultaneous_critical(gradients, fit.covariance, cfg.simulations, cfg.seed) if len(rows) else np.nan
    for pid in data.patient_id.unique():
        reduced = data.loc[data.patient_id.ne(pid)].copy()
        refit, reason = fit_fixed_family(reduced, evaluation['basis'], fit.model, fit.family, spec, cfg)
        if refit is None:
            failures[reason] += 1
            noncomparable += reason == 'NONCOMPARABLE_FAMILY'
            continue
        common = supported_times(reduced, original_grid, cfg)
        support_fractions.append(len(common)/len(original_grid) if original_grid else 0.)
        # Keep the frozen reference for change contrasts; it too must be supported.
        if not common or (fit.model == 'C' and original_grid[0] not in common):
            failures['NO_COMMON_SUPPORTED_REFERENCE'] += 1
            continue
        mask = np.array([r['time'] in common or fit.model == 'A' for r in rows]) & (se > 1e-12)
        if not mask.any():
            failures['NO_NONDEGENERATE_ESTIMANDS'] += 1
            continue
        values = target_values(rows, evaluation, refit, spec.type == 'binary')
        delta = values-estimates
        scaled = np.abs(delta[mask])/se[mask]
        material_refits += bool(np.any(scaled > MATERIAL['loo_standardized_shift_p95']))
        shifts.append(float(np.quantile(scaled, .95)))
        relative.append(float(np.quantile(np.abs(delta[mask])/np.maximum(np.abs(estimates[mask]), se[mask]), .95)))
        material_sign = (np.sign(values[mask]) != np.sign(estimates[mask])) & (np.abs(delta[mask]) > se[mask])
        signs.append(float(material_sign.mean()))
        # Band comparisons retain the original family and common grid. For C,
        # include all six pairs and all kinds, even across disconnected segments.
        ref_gradients = target_family(evaluation, refit, spec.type == 'binary')[1]
        ref_se = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', ref_gradients, refit.covariance, ref_gradients), 0))
        ref_critical = simultaneous_critical(ref_gradients, refit.covariance, cfg.simulations, cfg.seed)
        original_sign = np.where(estimates-analytic*se > 0, 1, np.where(estimates+analytic*se < 0, -1, 0))
        ref_sign = np.where(values-ref_critical*ref_se > 0, 1, np.where(values+ref_critical*ref_se < 0, -1, 0))
        agreements.append(float(np.mean(original_sign[mask] == ref_sign[mask])))
        for label in {(r['estimand'], r['observed_pattern'], r['contrast']) for r in rows}:
            selected = mask & np.array([(r['estimand'], r['observed_pattern'], r['contrast']) == label for r in rows])
            if selected.any():
                by_estimand[label].append(float(np.quantile(np.abs(delta[selected]), .95)))
        successful += 1
    summary = dict(model=fit.model, status='COMPLETE' if successful else 'INCONCLUSIVE', reason='',
        n_requested=requested, n_attempted=requested, n_successful=successful,
        n_failed=requested-successful, n_noncomparable=noncomparable,
        n_material_refits=material_refits, material_fragility_detected=bool(material_refits),
        failure_reasons=json.dumps(failures, sort_keys=True),
        standardized_shift_p95=float(np.quantile(shifts, .95)) if shifts else np.nan,
        relative_shift_p95=float(np.quantile(relative, .95)) if relative else np.nan,
        material_direction_fraction=float(np.mean(signs)) if signs else np.nan,
        band_agreement=float(np.mean(agreements)) if agreements else np.nan,
        support_fraction_p25=float(np.quantile(support_fractions, .25)) if support_fractions else 0.,
        estimand_family=family_name(fit.model), seed=seed, analytic_seed=cfg.seed,
        effect_scale='link_for_A_and_B_curvature_response_for_curves' if spec.type == 'binary' else spec.unit)
    influence = [dict(model=fit.model, estimand=k[0], observed_pattern=k[1], contrast=k[2],
        status='AGGREGATED', reason='', n_successful=len(v),
        absolute_shift_median=float(np.median(v)), absolute_shift_p75=float(np.quantile(v, .75)),
        absolute_shift_p95=float(np.quantile(v, .95)), seed=seed) for k, v in sorted(by_estimand.items())]
    return summary, influence


def bootstrap_summary(evaluation, fit, spec, cfg, seed=None):
    """Fixed-family stratified patient bootstrap, joint over model-specific family."""
    seed = cfg.seed if seed is None else seed
    requested = cfg.bootstrap_replicates
    base = dict(model=fit.model, n_requested=requested, n_attempted=0, n_successful=0, n_failed=0,
        seed=seed, estimand_family=family_name(fit.model), failure_reasons='{}',
        bootstrap_policy=BOOTSTRAP_POLICY, sensitivity_policy=SENSITIVITY_POLICY,
        calibration_denominator=CALIBRATION_DENOMINATOR,
        required_successes=max(100, math.ceil(.8*requested)), calibration_valid=False,
        calibration_valid_strict=False, calibration_valid_conditional=False)
    if spec.type == 'binary':
        return dict(base, status='NOT_ESTIMABLE_METHOD', reason='Binary response bootstrap calibration is not validated')
    rows, gradients, estimates = target_family(evaluation, fit)
    if not len(rows):
        return dict(base, status='NOT_ESTIMABLE', reason='No supported grid')
    se = np.sqrt(np.maximum(np.einsum('ij,jk,ik->i', gradients, fit.covariance, gradients), 0))
    valid = se > 1e-12
    if not valid.any() or not np.isfinite(se).all() or not np.isfinite(estimates).all():
        return dict(base, status='NOT_ESTIMABLE', reason='No finite nondegenerate estimands')
    analytic = simultaneous_critical(gradients, fit.covariance, cfg.simulations, cfg.seed)
    rng, maxima, conditional, failures = np.random.default_rng(seed), [], [], Counter()
    categories, primary = Counter(), Counter()
    losses, source_fractions, pseudo_fractions, distances = [], [], [], []
    grid = original_times(evaluation)
    for _ in range(requested):
        replicate, support = diagnostic_resample(evaluation['data'], rng, grid, cfg)
        refit, reason = fit_fixed_family(replicate, evaluation['basis'], fit.model, fit.family, spec, cfg)
        if refit is None:
            failures[reason] += 1
            categories[refit_failure_category(reason)] += 1
            primary['refit'] += 1
            continue
        if (refit.family != fit.family or refit.model != fit.model or
                not np.isfinite(refit.covariance).all() or
                np.linalg.eigvalsh(refit.covariance).min() < -1e-10):
            failures['NONCOMPARABLE_FAMILY_OR_COVARIANCE'] += 1
            categories['COVARIANCE_OR_FAMILY'] += 1
            primary['refit'] += 1
            continue
        common = supported_times(replicate, grid, cfg)
        lost = len(grid)-len(common)
        losses.append(lost)
        source_fractions.append(support[0])
        pseudo_fractions.append(support[1])
        distances.append(support[2])
        values = target_values(rows, evaluation, refit)
        maximum = float(np.max(np.abs(values[valid]-estimates[valid])/se[valid]))
        finite = np.isfinite(values).all() and np.isfinite(maximum)
        # Primary decomposition follows V1: refit, then support, then target.
        if lost:
            failures['ORIGINAL_SUPPORT_LOST'] += 1
            primary['support'] += 1
        elif not finite:
            failures['TARGET_NONFINITE_OR_DEGENERATE'] += 1
            primary['estimand'] += 1
        else:
            maxima.append(maximum)
        if finite:
            conditional.append(maximum)
    def calibration(samples):
        enough = len(samples) >= base['required_successes']
        critical = float(np.quantile(samples, .95)) if enough else np.nan
        original_sign = np.where(estimates-analytic*se > 0, 1, np.where(estimates+analytic*se < 0, -1, 0))
        calibrated_sign = np.where(estimates-critical*se > 0, 1, np.where(estimates+critical*se < 0, -1, 0))
        agreement = float(np.mean(original_sign[valid] == calibrated_sign[valid])) if enough else np.nan
        return enough, critical, agreement
    enough, critical, agreement = calibration(maxima)
    cond_valid, cond_critical, cond_agreement = calibration(conditional)
    return dict(base, status='CALIBRATED' if enough else 'INCONCLUSIVE',
        reason='' if enough else 'Insufficient comparable replicates',
        n_attempted=requested, n_successful=len(maxima), n_accepted=len(maxima), n_failed=requested-len(maxima),
        n_rejected_refit=primary['refit'], n_rejected_support=primary['support'],
        n_rejected_estimand=primary['estimand'], n_other=0,
        n_accepted_conditional=len(conditional), n_failed_conditional=requested-len(conditional),
        n_rejected_refit_conditional=primary['refit'], n_rejected_support_conditional=0,
        n_rejected_estimand_conditional=requested-primary['refit']-len(conditional),
        success_fraction_conditional=len(conditional)/requested if requested else 0.,
        bootstrap_accepted_strict=len(maxima), n_failed_strict=requested-len(maxima),
        failure_reasons=json.dumps(failures, sort_keys=True),
        refit_failure_category=json.dumps(categories, sort_keys=True),
        support_loss_histogram=json.dumps(Counter('1' if n == 1 else '2_or_more' if n > 1 else '0' for n in losses), sort_keys=True),
        support_loss_fraction=float(np.mean(np.array(losses)>0)) if losses else np.nan,
        support_loss_conditional=bool(len(conditional)>len(maxima)),
        support_loss_conditional_fraction=(len(conditional)-len(maxima))/len(conditional) if conditional else np.nan,
        source_unique_local_fraction_mean=float(np.mean(source_fractions)) if source_fractions and np.isfinite(source_fractions).all() else None,
        pseudo_local_fraction_mean=float(np.mean(pseudo_fractions)) if pseudo_fractions and np.isfinite(pseudo_fractions).all() else None,
        grid_distance_years_max=max(distances) if distances and np.isfinite(distances).all() else None,
        source_support_disclosure_status=('DISCLOSURE_SUPPRESSED' if source_fractions and not np.isfinite(source_fractions).all() else 'AGGREGATE'),
        success_fraction=len(maxima)/requested if requested else 0.,
        calibration_valid=enough, calibration_valid_strict=enough, calibration_valid_conditional=cond_valid,
        bootstrap_sup_t_critical=critical, bootstrap_sup_t_critical_conditional=cond_critical,
        analytic_sup_t_critical=analytic, analytic_seed=cfg.seed,
        critical_difference=critical-analytic, critical_ratio=critical/analytic,
        global_sup_t_p50=float(np.quantile(maxima, .5)) if enough else np.nan,
        global_sup_t_p90=float(np.quantile(maxima, .9)) if enough else np.nan,
        band_agreement=agreement, band_agreement_conditional=cond_agreement,
        inference='Exploratory original-SE standardized maximum deviation; not bootstrap-t or independent validation')


def refit_failure_category(reason):
    reason = reason.upper()
    for category in ('DESIGN', 'LONGITUDINAL_SUPPORT', 'NUMERICAL', 'CONVERGENCE',
                     'COVARIANCE', 'RESIDUAL', 'BINARY'):
        if category in reason:
            return 'RESIDUAL_QC' if category == 'RESIDUAL' else category
    return 'OTHER'


def diagnostic_resample(data, rng, times, cfg):
    """V1-identical draws; source identities exist only in this stack frame.

    Return the draw and three aggregate diagnostics, never a source-ID map.
    """
    if data.groupby('patient_id').observed_pattern.nunique().gt(1).any():
        raise ContractError('bootstrap requires fixed patient groups')
    by_patient = {pid: d for pid, d in data.groupby('patient_id')}
    parts, unique_fractions, pseudo_fractions, gaps = [], [], [], []
    source_small = False
    for group in PATTERNS:
        pool = data.loc[data.observed_pattern.eq(group), 'patient_id'].unique()
        if not len(pool):
            raise ContractError('bootstrap group missing')
        selected = rng.choice(pool, len(pool), replace=True)
        source = data.loc[data.patient_id.isin(set(selected))]
        for index, pid in enumerate(selected):
            part = by_patient[pid].copy()
            part['patient_id'] = group+'_bootstrap_'+str(index)
            parts.append(part)
        for t in times:
            local = source.loc[source.time.sub(t).abs().le(cfg.prediction_window_years/2)]
            unique_count = local.patient_id.nunique()
            source_small |= 0 < unique_count < 5
            unique_fractions.append(unique_count/len(pool))
            local_ids = set(local.patient_id)
            pseudo_fractions.append(sum(pid in local_ids for pid in selected)/len(pool))
            gaps.append(max(0., source.time.min()-t, t-source.time.max()))
    if source_small:
        # Never release a rate from which a small source-patient cell can be recovered.
        return pd.concat(parts, ignore_index=True), (np.nan, np.nan, np.nan)
    return pd.concat(parts, ignore_index=True), (float(np.mean(unique_fractions)),
        float(np.mean(pseudo_fractions)), max(gaps, default=0.))


def scientific_status(inventory, overlap, loo=None, bootstrap=None):
    """No p/q inputs are used for scientific classification."""
    if not inventory['reference_comparable']:
        return 'NOT_COMPARABLE_REFERENCE'
    if inventory['status'] == 'NOT_APPLICABLE_TYPE':
        return 'NOT_APPLICABLE_TYPE'
    if not inventory['eligible_for_deep']:
        return 'NOT_ESTIMABLE'
    if loo is None or bootstrap is None:
        return 'DEEP_PENDING_RESOURCE_BUDGET'
    if loo.get('status') == 'NOT_COMPARABLE_REFERENCE':
        return 'NOT_COMPARABLE_REFERENCE'
    if not loo.get('n_successful'):
        return 'INCONCLUSIVE'
    failed = loo['n_failed']/max(1, loo['n_requested'])
    if (failed > MATERIAL['maximum_refit_failure_fraction'] or
            loo.get('material_fragility_detected', False) or
            loo['standardized_shift_p95'] > MATERIAL['loo_standardized_shift_p95'] or
            loo['material_direction_fraction'] > MATERIAL['loo_material_direction_fraction']):
        return 'FRAGILE'
    if not bootstrap.get('calibration_valid'):
        return 'INCONCLUSIVE'
    if (overlap < MATERIAL['minimum_longitudinal_overlap'] or
            loo.get('support_fraction_p25', 1.) < MATERIAL['minimum_loo_support_fraction'] or
            loo['band_agreement'] < MATERIAL['minimum_band_agreement'] or
            bootstrap['band_agreement'] < MATERIAL['minimum_band_agreement'] or
            bootstrap['critical_ratio'] > MATERIAL['maximum_critical_ratio']):
        return 'INCONCLUSIVE'
    return 'SUPPORTED_EXPLORATORY'


class Checkpoints:
    """Atomic aggregate task cache with a frozen plan and content hashes.

    A model task is the interruption boundary. The refit budget never truncates
    a patient's trajectory or publishes a partly completed task as complete.
    """
    def __init__(self, path, context, plan, resume=False, patient_ids=()):
        self.path, self.ids = Path(path), tuple(patient_ids)
        self.context_hash = digest(context)
        self.plan_hash = digest(plan)
        if any(p.is_symlink() for p in [self.path, *self.path.parents]):
            raise ContractError('symlink in checkpoint path')
        self.path.mkdir(parents=True, exist_ok=True, mode=0o700)
        contract = dict(context_hash=self.context_hash, plan_hash=self.plan_hash, plan=plan)
        self.contract = contract
        target = self.path/'plan.json'
        if target.is_symlink():
            raise ContractError('symlink checkpoint plan')
        if target.exists():
            if not resume:
                raise FileExistsError('checkpoint exists; use --resume with identical analytical settings')
            if json.loads(target.read_text()) != json_safe(contract):
                raise ContractError('checkpoint plan/context hash mismatch')
        else:
            self._atomic(target, contract)

    def _atomic(self, target, payload):
        payload = json_safe(payload)
        audit_payload(payload, self.ids)
        fd, name = tempfile.mkstemp(prefix='.aggregate-', dir=self.path)
        try:
            with os.fdopen(fd, 'w') as handle:
                json.dump(payload, handle, sort_keys=True, allow_nan=False)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, target)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def load(self, task_id):
        if not re.fullmatch(r'[0-9a-f]{64}', task_id):
            raise ContractError('invalid task key')
        path = self.path/(task_id+'.json')
        if not path.exists():
            return None
        if path.is_symlink():
            raise ContractError('symlink checkpoint')
        record = json.loads(path.read_text())
        if (record.get('context_hash') != self.context_hash or record.get('task_id') != task_id or
                record.get('payload_hash') != digest(record.get('payload'))):
            raise ContractError('checkpoint aggregate hash mismatch')
        audit_payload(record, self.ids)
        return record['payload']

    def save(self, task_id, payload):
        if self.load(task_id) is not None:
            raise FileExistsError('completed task checkpoint is immutable')
        payload = json_safe(payload)
        self._atomic(self.path/(task_id+'.json'), dict(context_hash=self.context_hash, task_id=task_id,
            payload_hash=digest(payload), payload=payload))


def validation_dimensions(result, inventory=None):
    """Independent disclosure-controlled LOO and bootstrap dimensions."""
    if result is None:
        return dict(loo_stability_status=('NOT_APPLICABLE' if inventory and inventory['status'] == 'NOT_APPLICABLE_TYPE' else 'LOO_NOT_RUN'),
            bootstrap_calibration_status=('NOT_COMPARABLE_REFERENCE' if inventory and not inventory['reference_comparable'] else 'NOT_RUN'),
            loo_material_fragility_detected=None, bootstrap_strict_calibration_valid=None,
            bootstrap_conditional_calibration_valid=None, validation_reason_codes='DEEP_NOT_RUN')
    loo, boot = result['loo'][0], result['bootstrap'][0]
    hidden = any(r.get('disclosure_status') == 'DISCLOSURE_SUPPRESSED' for r in (loo, boot))
    if hidden:
        return dict(loo_stability_status='DISCLOSURE_SUPPRESSED',
            bootstrap_calibration_status='DISCLOSURE_SUPPRESSED', loo_material_fragility_detected=None,
            bootstrap_strict_calibration_valid=None, bootstrap_conditional_calibration_valid=None,
            validation_reason_codes='DISCLOSURE_SUPPRESSED')
    material = bool(loo.get('n_successful')) and (
        bool(loo.get('material_fragility_detected', False)) or
        loo.get('standardized_shift_p95', 0) > MATERIAL['loo_standardized_shift_p95'] or
        loo.get('material_direction_fraction', 0) > MATERIAL['loo_material_direction_fraction'] or
        loo.get('n_failed', 0)/max(1, loo.get('n_requested', 0)) > MATERIAL['maximum_refit_failure_fraction'])
    loo_status = ('MATERIAL_INFLUENCE' if material else 'LOO_INCONCLUSIVE' if not loo.get('n_successful') or
        loo.get('support_fraction_p25', 1.) < MATERIAL['minimum_loo_support_fraction'] or
        loo.get('band_agreement', 0) < MATERIAL['minimum_band_agreement'] else 'STABLE_WITHIN_DEFINED_LIMITS')
    strict = bool(boot.get('calibration_valid', False))
    conditional = bool(boot.get('calibration_valid_conditional', False))
    boot_status = (boot['status'] if boot.get('status') in ('NOT_ESTIMABLE_METHOD', 'NOT_COMPARABLE_REFERENCE') else
        'CALIBRATED_STRICT' if strict else 'CONDITIONAL_ONLY' if conditional else 'INCONCLUSIVE_STRICT')
    return dict(loo_stability_status=loo_status, bootstrap_calibration_status=boot_status,
        loo_material_fragility_detected=material, bootstrap_strict_calibration_valid=strict,
        bootstrap_conditional_calibration_valid=conditional,
        validation_reason_codes=loo_status+';'+boot_status)
