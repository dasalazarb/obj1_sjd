"""Run the non-defining Graph02 factorial on each frozen baseline panel."""
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, replace
import hashlib
from itertools import combinations
import logging
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from threadpoolctl import threadpool_limits

from .._pipeline import digest, save_table
from .s02_contracts import MANIFEST, validate_release
from .s02_factorial_imputation import (
    BENCHMARKS, CANDIDATES, MECHANISMS, complete_panel, factorial_metadata, select_candidate, validation,
)
from .s02_mapper_engine import analyze, frozen_membership
from .s02_missingness import administrative_frame, administrative_strata, compare_coverage, supported_hard
from .s02_missingness_registry import NotEstimable, classify_panel
from .s02_missingness_variants import factorial_mds, reconstructed_space, supported_mask
from .s02_representations import build_space
from .s02_resampling import draw_indices, match_sets, wilson


PREFIX = '02_graph_missingness_'
CONDITIONS = {'A': '', 'B': '-I-v3', 'C': '-M-v3', 'D': '-IM-v3'}


def checked_policy(cfg):
    historical = cfg.get('missingness_methods', {})
    policy = historical.get('factorial_missingness_v3', {})
    if not historical.get('enabled') or not policy.get('enabled'):
        return None
    if (set(policy) != {'enabled', 'version', 'defining_for_graph03', 'references', 'candidates',
                          'benchmarks', 'mask', 'validation', 'estimators', 'uncertainty', 'resampling'}
            or policy['version'] != 'graph02_factorial_v3' or policy['defining_for_graph03'] is not False
            or policy['references'] != ['R0', 'R2'] or tuple(policy['candidates']) != CANDIDATES
            or tuple(policy['benchmarks']) != BENCHMARKS):
        raise ValueError('INVALID_GRAPH02_FACTORIAL_PROTOCOL')
    expected = {
        'mask': {'primary_weight', 'sensitivity_weights', 'minimum_shared_domains', 'minimum_shared_weight_fraction',
                 'sensitivity_shared_weight_fraction'},
        'validation': {'repeats', 'tuning_repeats', 'masking_fraction', 'masking_mechanisms',
                       'minimum_successful_repeat_fraction', 'minimum_prediction_coverage', 'minimum_relative_improvement',
                       'maximum_domain_deterioration', 'minimum_paired_patients', 'patient_ci_replicates'},
        'estimators': {'candidate_rank', 'donor_count', 'chain_sweeps'},
        'uncertainty': {'draws', 'nested_draws_per_refit', 'delta'},
        'resampling': {'main_subsamples', 'main_bootstrap', 'parameter_subsamples'},
    }
    if any(set(policy[key])!=fields for key, fields in expected.items()):
        raise ValueError('UNKNOWN_OR_MISSING_FACTORIAL_SETTING')
    mask, vp, uncertainty = policy['mask'], policy['validation'], policy['uncertainty']
    if (not 0 < mask['primary_weight'] <= 1
            or any(not 0<=w<=1 for w in mask['sensitivity_weights'])
            or any(not 0<w<=1 for w in [mask['minimum_shared_weight_fraction'], *mask['sensitivity_shared_weight_fraction']])
            or not mask['minimum_shared_domains'] >= 1
            or not 0 in mask['sensitivity_weights']
            or set(vp['masking_mechanisms']) != set(MECHANISMS)
            or any(not 0 < f < 1 for f in vp['masking_fraction'])
            or any(not 0 < vp[k] <= 1 for k in ('minimum_successful_repeat_fraction', 'minimum_prediction_coverage'))
            or not 0 <= vp['minimum_relative_improvement'] < 1
            or vp['maximum_domain_deterioration'] < 0
            or any(vp[k]<1 for k in ('repeats', 'tuning_repeats', 'minimum_paired_patients', 'patient_ci_replicates'))
            or not policy['estimators']['candidate_rank']
            or any(r<1 for r in policy['estimators']['candidate_rank'])
            or not 0<uncertainty['nested_draws_per_refit']<=uncertainty['draws']
            or not 0 in uncertainty['delta'] or not np.isfinite(uncertainty['delta']).all()
            or any(v<1 for v in policy['resampling'].values())):
        raise ValueError('INVALID_FACTORIAL_BUDGET_OR_THRESHOLD')
    return policy


def historical_failure_audit(root):
    """Read prior real status only from a completed release, or cite the repository audit."""
    attempts = []
    paths = [root/MANIFEST]
    pointer = root/'latest_completed_02.json'
    if not paths[0].is_file() and pointer.is_file():
        import json
        try:
            prior = json.loads(pointer.read_text())
            candidate = (root/prior['manifest_path']).resolve()
            if candidate.is_relative_to(root.resolve()) and digest(candidate)==prior['manifest_sha256']:
                paths.append(candidate)
        except (OSError, ValueError, KeyError):
            pass
    for path in paths:
        if not path.is_file():
            continue
        try:
            release, verified = validate_release(path)
            run = verified.parent
            registry_path = run/(PREFIX+'method_registry.csv')
            registry = pd.read_csv(registry_path).fillna('')
            refits_path = run/(PREFIX+'resampling_replications.csv')
            refits = pd.read_csv(refits_path) if refits_path.is_file() and refits_path.stat().st_size>1 else pd.DataFrame()
            validation_path = run/(PREFIX+'imputation_attempts.csv')
            validation_rows = pd.read_csv(validation_path) if validation_path.is_file() and validation_path.stat().st_size>1 else pd.DataFrame()
            validation_sha = digest(validation_path) if not validation_rows.empty else ''
            qc_path = run/(PREFIX+'method_qc.csv')
            stability_path = run/(PREFIX+'method_stability.csv')
            for row in registry.loc[registry.experiment_id.isin(['R0-I','R2-I','R0-M','R2-M'])].itertuples():
                part = refits.loc[refits.representation_id.eq(row.experiment_id)] if 'representation_id' in refits else refits
                n = len(part)
                success = int(part.status.isin(['success','COMPLETE']).sum()) if n else 0
                reason = str(row.reason)
                category = ('REFIT_QC' if 'REFIT' in reason else
                            'SOFTWARE_OR_INPUT_FAILURE' if row.status=='FAILED' else
                            'STATISTICAL_SUPPORT' if row.status=='NOT_ESTIMABLE' else 'EXECUTED')
                attempts.append(dict(experiment_id=row.experiment_id, stage='prior_run', status=row.status,
                                     reason=reason, failure_class=category, n_attempted=n,
                                     n_success=success, n_failed=n-success, source=registry_path.name,
                                     source_sha256=digest(registry_path), release_sha256=digest(verified),
                                     previous_run_id=release.get('run_id', 'current_layout'),
                                     qc_source_sha256=digest(qc_path) if qc_path.is_file() else '',
                                     stability_source_sha256=digest(stability_path) if stability_path.is_file() else ''))
                if {'experiment_id','status'} <= set(validation_rows):
                    trial = validation_rows.loc[validation_rows.experiment_id.eq(row.experiment_id)]
                    keys = [key for key in ('validation_stage','repeat','rank') if key in trial]
                    for key, group in trial.groupby(keys, dropna=False) if keys else ():
                        statuses = group.status
                        ok = statuses.eq('COMPLETE').any()
                        notes = sorted(set(group['reason'].dropna())) if 'reason' in group else []
                        label = key if isinstance(key,tuple) else (key,)
                        attempts.append(dict(experiment_id=row.experiment_id, stage='historical_validation_attempt',
                            status='COMPLETE' if ok else 'NOT_ESTIMABLE', reason='|'.join(notes),
                            failure_class='EXECUTED' if ok else 'STATISTICAL_SUPPORT',
                            n_attempted=1, n_success=int(ok), n_failed=int(not ok), source=validation_path.name,
                            source_sha256=validation_sha, **dict(zip(keys,label))))
        except (OSError, ValueError, AttributeError, KeyError):
            # Invalid or incomplete prior outputs are never interpreted as a real outcome.
            continue
    historical = {'R0-I': 'NOMINAL_SOURCES_UNSUPPORTED', 'R2-I': 'NO_VALIDATED_RANK'}
    for reference in ('R0', 'R2'):
        for method in ('I', 'M'):
            identifier = f'{reference}-{method}'
            if any(row['experiment_id']==identifier for row in attempts):
                continue
            attempts.append(dict(experiment_id=identifier, stage='latest_clinical_artifacts', status='SOURCE_UNAVAILABLE',
                                 reason='PRIOR_CLINICAL_ARTIFACTS_UNAVAILABLE', failure_class='MISSING_SOURCE',
                                 n_attempted=None, n_success=None, n_failed=None, source=''))
            if identifier in historical:
                attempts.append(dict(experiment_id=identifier, stage='documented_prior_failure', status='NOT_ESTIMABLE',
                                     reason=historical[identifier], failure_class='STATISTICAL_SUPPORT', n_attempted=None,
                                     n_success=None, n_failed=None, source='docs/graph02_focal_fix_implementation.md'))
    return attempts


def comparison_registry():
    pairs = []
    for reference in ('R0', 'R2'):
        cells = {k: reference+v for k, v in CONDITIONS.items()}
        pairs += [(reference, cells[a], cells[b], 'factorial') for a, b in combinations(cells, 2)]
        pairs += [(reference, cells['B'], reference+'-I', 'historical_imputation'),
                  (reference, cells['C'], reference+'-M', 'historical_mask'),
                  (reference, cells['A'], reference+'-A-MDS-v3', 'lens_only'),
                  (reference, cells['B'], reference+'-B-MDS-v3', 'lens_only'),
                  (reference, reference+'-A-MDS-v3', cells['C'], 'common_lens'),
                  (reference, reference+'-B-MDS-v3', cells['D'], 'common_lens')]
        pairs += [(reference, cells[cell], reference+'-'+method,
                   'cross_algorithm_not_mapper' if method=='DG' else 'observed_pair_distance')
                  for cell in ('B','C','D') for method in ('D','DG')]
    return pairs + [('R0/R2', 'R0'+suffix, 'R2'+suffix, 'cross_panel_geometry') for suffix in CONDITIONS.values()]


def fingerprint(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def communities(mapped):
    return [members for members, ok in zip(mapped['community_sets'], mapped['supported']) if ok]


def comparison(left, right, minimum, n):
    from .s02_missingness_study import geometry_comparison
    if left is None or right is None:
        return dict(status='NOT_ESTIMABLE', reason='COMPARISON_ENDPOINT_UNAVAILABLE', n_total=n,
                    n_common=0, n_valid_distance_pairs=0)
    a, b = left['mapped'], right['mapped']
    row = (compare_coverage(a, b, minimum) if a is not None and b is not None else
           dict(status='NOT_ESTIMABLE', reason='TOPOLOGY_ENDPOINT_UNAVAILABLE', n_total=n, n_common=0))
    row.update(geometry_comparison(left['distance'], right['distance'], k=5))
    for k in (5, 10, 15):
        overlap = []
        for i in range(n):
            left_dist, right_dist = left['distance'][i], right['distance'][i]
            valid = np.isfinite(left_dist) & np.isfinite(right_dist)
            valid[i] = False
            positions = np.flatnonzero(valid)
            if len(positions)<=k:
                continue
            neighbors_a = positions[np.argsort(left_dist[positions], kind='stable')[:k]]
            neighbors_b = positions[np.argsort(right_dist[positions], kind='stable')[:k]]
            overlap.append(len(set(neighbors_a) & set(neighbors_b))/k)
        row[f'knn_overlap_{k}'] = float(np.mean(overlap)) if overlap else np.nan
        row[f'n_knn_evaluable_{k}'] = len(overlap)
    upper = np.triu_indices(n, 1)
    da, db = left['distance'][upper], right['distance'][upper]
    valid = np.isfinite(da) & np.isfinite(db)
    if valid.any():
        rms_a, rms_b = np.sqrt(np.mean(da[valid]**2)), np.sqrt(np.mean(db[valid]**2))
        if min(rms_a, rms_b)>0:
            row['distance_normalized_mae'] = float(np.mean(np.abs(da[valid]/rms_a-db[valid]/rms_b)))
    if a is None or b is None:
        return row
    row['n_covered_left_only'] = int((a['covered'] & ~b['covered']).sum())
    row['n_covered_right_only'] = int((b['covered'] & ~a['covered']).sum())
    row['n_communities_left'], row['n_communities_right'] = len(communities(a)), len(communities(b))
    hard_common = supported_hard(a) & supported_hard(b)
    row['n_hard_classes_left'] = np.unique(a['hard'][hard_common]).size
    row['n_hard_classes_right'] = np.unique(b['hard'][hard_common]).size
    if left.get('space') is not None and right.get('space') is not None:
        from scipy.spatial import procrustes
        da = left['space'].lens
        db = right['space'].lens
        if da.shape==db.shape and np.ptp(da)>0 and np.ptp(db)>0:
            _,_,row['lens_procrustes_disparity'] = procrustes(da, db)
    matched, _ = match_sets(communities(a), communities(b))
    row['jaccard_full_mean'] = np.mean([r['jaccard'] for r in matched]) if matched else np.nan
    shared = set(np.flatnonzero(hard_common))
    common, _ = match_sets([g & shared for g in communities(a) if g & shared],
                           [g & shared for g in communities(b) if g & shared])
    row['jaccard_common_mean'] = np.mean([r['jaccard'] for r in common]) if len(shared)>=minimum and common else np.nan
    row['n_matched_communities'] = len(matched)
    row['n_splits'] = sum(r['split_candidates']>1 for r in matched)
    row['n_merges'] = sum(r['merge_references']>1 for r in matched)
    return row


def unique_space(space, positions):
    rep = copy(space.legacy_rep)
    if rep is not None:
        rep.embedding = rep.embedding[positions]
        rep.raw = rep.raw.iloc[positions]
    return replace(space, raw=space.raw.iloc[positions], lens=space.lens[positions],
                   cluster=space.cluster[positions] if rep is not None or space.cluster.shape!=space.distance.shape
                   else space.cluster[np.ix_(positions,positions)],
                   distance=space.distance[np.ix_(positions,positions)], legacy_rep=rep)


def item(space, config, *, mapped=None, audit=False, **fields):
    if mapped is None:
        with threadpool_limits(limits=1):
            mapped = analyze(space, config, audit=audit)
    return dict(space=space, distance=space.distance, mapped=mapped, **fields)


@dataclass
class Accumulator:
    n: int

    def __post_init__(self):
        self.hard_denominator = np.zeros((self.n, self.n), dtype=np.int32)
        self.hard_same = np.zeros_like(self.hard_denominator)
        self.union_denominator = np.zeros_like(self.hard_denominator)
        self.union_same = np.zeros_like(self.hard_denominator)
        self.n_attempted = np.zeros(self.n, dtype=int)
        self.n_sampled = np.zeros(self.n, dtype=int)
        self.n_covered = np.zeros(self.n, dtype=int)
        self.assignment_counts = [defaultdict(int) for _ in range(self.n)]

    def collect(self, universe, mapped, aligned=None):
        self.n_sampled[universe] += 1
        self.n_covered[universe[mapped['covered']]] += 1
        valid = supported_hard(mapped)
        valid_ids = universe[valid]
        hard = mapped['hard'][valid].astype(int)
        ids = np.ix_(valid_ids, valid_ids)
        self.hard_denominator[ids] += 1
        self.hard_same[ids] += hard[:, None]==hard[None, :]
        overlap = np.zeros((len(universe), len(universe)), dtype=bool)
        eligible = np.zeros(len(universe), dtype=bool)
        for group in communities(mapped):
            ids = np.array(sorted(group), dtype=int)
            eligible[ids] = True
            overlap[np.ix_(ids, ids)] = True
        index = np.ix_(universe[eligible], universe[eligible])
        self.union_denominator[index] += 1
        self.union_same[index] += overlap[np.ix_(eligible, eligible)]
        if aligned is not None:
            for patient, label in zip(valid_ids, hard):
                if label in aligned:
                    self.assignment_counts[patient][aligned[label]] += 1

    def rows(self, experiment, source):
        pairs = []
        for i, j in zip(*np.triu_indices(self.n, 1)):
            hard_n, union_n = self.hard_denominator[i, j], self.union_denominator[i, j]
            pairs.append(dict(experiment_id=experiment, uncertainty_source=source, patient_index_a=i, patient_index_b=j,
                              n_evaluable=hard_n, n_same=self.hard_same[i,j],
                              coassignment=self.hard_same[i,j]/hard_n if hard_n else np.nan,
                              n_union_evaluable=union_n, n_same_union=self.union_same[i,j],
                              union_coassignment=self.union_same[i,j]/union_n if union_n else np.nan))
        patients = []
        for i, (n_sampled, n_covered) in enumerate(zip(self.n_sampled, self.n_covered)):
            aligned = np.array(list(self.assignment_counts[i].values()), dtype=float)
            probability = aligned/aligned.sum() if len(aligned) else aligned
            patients.append(dict(experiment_id=experiment, uncertainty_source=source, patient_index=i, n_sampled=n_sampled,
                                 n_covered=n_covered, coverage_frequency=n_covered/n_sampled if n_sampled else np.nan,
                                 n_attempted=int(self.n_attempted[i]),
                                 conservative_coverage_frequency=n_covered/self.n_attempted[i] if self.n_attempted[i] else np.nan,
                                 n_aligned_assignments=int(aligned.sum()),
                                 membership_consistency=probability.max() if len(probability) else np.nan,
                                 aligned_membership_entropy=-np.sum(probability*np.log(probability)) if len(probability) else np.nan))
        return pairs, patients


class Factorial:
    def __init__(self, reference, base, registry, manifest, baseline, families, config, cfg, lab_groups,
                 feature_to_lab, policy, tables):
        self.reference, self.base = reference, base
        self.raw = base['space'].raw
        if baseline.patient_id.isna().any() or baseline.patient_id.duplicated().any():
            raise ValueError('BASELINE_REQUIRES_UNIQUE_PATIENT_IDENTITIES')
        self.registry, self.manifest, self.baseline = registry, manifest, baseline
        self.families, self.config, self.cfg = families, config, cfg
        self.lab_groups, self.feature_to_lab = lab_groups, feature_to_lab
        self.policy, self.tables = policy, tables
        self.settings = cfg['experiments']
        self.date_column = cfg['validation']['date_column']
        self.seed = int(config['random_seed'])
        self.impute_policy = {**cfg['missingness_methods']['iterative_regularized'],
                              **policy['validation'], **policy['estimators']}
        self.meta = factorial_metadata(self.raw, registry, reference)
        self.cells = {k: reference+v for k, v in CONDITIONS.items()}
        self.reasons = classify_panel(self.raw, registry, reference, baseline, self.date_column)
        self.panel_fields = dict(reference_id=reference, panel_hash=self.reasons.observation.fingerprint,
                                 n_patients=len(self.raw))
        self.method = None
        self.reduced_sources = []
        self.selection = []

    def append(self, table, rows):
        self.tables[table].extend({**self.panel_fields, **row} for row in rows)

    def build(self, raw, baseline, completed=None, method=None):
        args = (self.families, self.config, self.registry, self.manifest, baseline, self.lab_groups, self.feature_to_lab)
        if completed is None:
            return build_space(self.reference, raw, *args)
        return reconstructed_space(self.reference, raw, completed, *args, method)

    def benchmark(self):
        cells = self.reasons.cells
        self.append('reason_cells_internal', cells.to_dict('records'))
        groups = ['source_column', 'family', 'domain', 'subdomain', 'reason_class', 'reason_confidence']
        summary = cells.groupby(groups, dropna=False).agg(
            n_cells=('patient_key','size'), n_eligible=('eligible_to_impute','sum'), n_valid=('observed_value_valid','sum')
        ).reset_index()
        summary['n_source_patients'] = summary.groupby('source_column').n_cells.transform('sum')
        summary['fraction_of_source'] = summary.n_cells/summary.n_source_patients
        self.append('reason_classification_summary', summary.assign(context='all_baseline').to_dict('records'))
        admin, _, _ = administrative_frame(self.baseline, self.cfg['validation']['confounders'], self.date_column)
        for column in admin:
            strata = administrative_strata(admin[column], column, self.settings['minimum_stratum_patients'])
            by_patient = pd.Series(strata.to_numpy(), index=self.baseline.patient_id)
            context = cells.assign(context_level=cells.patient_key.map(by_patient))
            counts = context.groupby([*groups, 'context_level'], dropna=True, observed=True).size().reset_index(name='n_cells')
            self.append('reason_classification_summary', counts.assign(context=column).to_dict('records'))
        if not self.raw.isna().any().any():
            self.append('candidate_selection', [dict(candidate=c, selected=False, status='NOT_ESTIMABLE',
                                                     reason='NO_ORIGINAL_MISSING_CELLS') for c in CANDIDATES])
            return
        ineligible = (self.raw.isna() & ~self.reasons.eligible) | (self.raw.notna() & ~self.reasons.valid)
        excluded = ineligible.any()
        # Exclude whole observation units to keep a genuinely comparable panel.
        for unit in self.reasons.observation.metadata.itertuples():
            if excluded.loc[list(unit.columns)].any():
                excluded.loc[list(unit.columns)] = True
        # Structural targets or unsafe derived groups cannot contribute to fitting
        # other sources. Exclusion is confined to target-level validation.
        for unit in self.reasons.observation.metadata.itertuples():
            if len(unit.sources)>1 and self.raw[list(unit.columns)].isna().any().any():
                excluded.loc[list(unit.columns)] = True
        for column in excluded.index[excluded]:
            self.append('target_support_audit', [dict(source_column=self.meta.loc[column,'source_column'],
                target_column=column, status='NOT_ESTIMABLE', reason='INELIGIBLE_FIXED_PANEL_SOURCE')])
        self.reduced_sources = list(excluded.index[excluded])
        valid = self.raw.mask(~self.reasons.valid).loc[:, ~excluded]
        if valid.shape[1]<2:
            self.append('candidate_selection', [dict(candidate=c, selected=False, status='NOT_ESTIMABLE',
                                                     reason='INSUFFICIENT_ELIGIBLE_VALIDATION_SOURCES') for c in CANDIDATES])
            return
        rows, support, attempts, paired = validation(valid, self.meta.loc[valid.columns],
            self.baseline.patient_id.to_numpy(), self.impute_policy, self.seed, self.reference)
        self.append('candidate_benchmark', rows)
        self.append('target_support_audit', support)
        self.append('candidate_fold_attempts', attempts)
        self.method, self.selection = select_candidate(attempts, paired, self.impute_policy, self.seed)
        self.append('candidate_selection', [{**row, 'validation_panel':'common' if self.reduced_sources else 'fixed'}
                                            for row in self.selection])

    def additional_candidates(self, full):
        fields = dict(seed=self.seed, n_patients=len(self.raw))
        holdout = {r['candidate']: r for r in self.selection}

        def candidate(raw, baseline, reason, method):
            meta = factorial_metadata(raw, self.registry, self.reference)
            value, _ = complete_panel(raw, reason, meta, baseline.patient_id.to_numpy(), method,
                                      self.impute_policy, self.seed, self.reference)
            return item(self.build(raw, baseline, value, method), self.config)

        if self.reduced_sources:
            statuses = {}
            short = self.raw.drop(columns=self.reduced_sources)
            common_fields = dict(**fields, alternate_panel_id=self.reference+'-common-v3',
                                 n_original_features=self.raw.shape[1], n_common_features=short.shape[1],
                                 excluded_features='|'.join(self.reduced_sources))
            def common_reference():
                if short.shape[1]<2:
                    raise NotEstimable('INSUFFICIENT_COMMON_PANEL_FEATURES')
                return item(self.build(short, self.baseline), self.config)
            base = self.attempt('common_A', common_reference, statuses)
            if base is None:
                self.append('common_panel_sensitivity', [dict(**common_fields,
                    status=statuses['common_A'][0], reason=statuses['common_A'][1])])
                return
            short = base['space'].raw
            reasons = classify_panel(short,self.registry,self.reference,self.baseline,self.date_column)
            common_fields['common_panel_hash'] = reasons.observation.fingerprint
            common_fields['alternate_reference'] = self.reference+'-common-median-v3'
            if not short.isna().any().any():
                self.append('common_panel_sensitivity', [dict(**common_fields, status='NOT_ESTIMABLE',
                                                             reason='COMMON_PANEL_HAS_NO_MISSING_CELLS')])
                return
        else:
            short, reasons, base = self.raw, self.reasons, full['A']
        for method in CANDIDATES:
            scores = holdout.get(method, {})
            suitable = scores.get('successful_repeat_fraction',0)>=self.impute_policy['minimum_successful_repeat_fraction']
            if not suitable:
                self.append('candidate_topology_sensitivity', [dict(**fields, candidate=method,
                    status='NOT_ESTIMABLE', reason='INSUFFICIENT_VALIDATION_SUCCESS')])
                continue
            statuses = {}
            fitted = full.get('B') if method==self.method and not self.reduced_sources else None
            if fitted is None:
                fitted = self.attempt('candidate', lambda: candidate(short, self.baseline, reasons, method), statuses)
            else:
                statuses['candidate'] = ('COMPLETE','')
            metrics = comparison(base,fitted,self.settings['minimum_comparison_patients'],len(short))
            result = dict(candidate=method, reconstruction_status=statuses['candidate'][0],
                          reconstruction_reason=statuses['candidate'][1], **metrics)
            if self.reduced_sources:
                self.append('common_panel_sensitivity', [dict(**common_fields, **result)])
            else:
                self.append('candidate_topology_sensitivity', [dict(**fields, **result)])

    def attempt(self, cell, build, attempts):
        try:
            result = build()
            attempts[cell] = ('COMPLETE', '')
            return result
        except NotEstimable as exc:
            attempts[cell] = ('NOT_ESTIMABLE', str(exc))
            self.append('candidate_benchmark', getattr(exc, 'tuning', []))
        except (ValueError, np.linalg.LinAlgError, MemoryError) as exc:
            logging.warning('Graph02 factorial experiment failed: %s %s (%s)', self.reference, cell, type(exc).__name__)
            attempts[cell] = ('FAILED', type(exc).__name__+':'+str(exc).split(':')[0])
        return None

    def mask(self, reasons, layer, fields, *, record_pairs=False, threshold=None, states=None):
        params = self.policy['mask']
        if threshold is not None:
            params = {**params, 'minimum_shared_weight_fraction': threshold}
        distance, denominator, domains, audit = supported_mask(reasons, self.reference, layer, params, states=states)
        if record_pairs:
            a, b = np.triu_indices(len(distance), 1)
            self.append('mask_pair_support_internal', [
                dict(mask_layer=layer, patient_index_a=i, patient_index_b=j, support_weight=denominator[i,j],
                     shared_domains=domains[i,j], disagreement=distance[i,j]) for i,j in zip(a,b)])
        self.append('mask_support_and_contribution', [{**fields, **audit, 'mask_hash': reasons.fingerprint}])
        if audit['status'] != 'COMPLETE':
            raise NotEstimable(audit['reason'])
        return distance

    def panel_spaces(self, raw, baseline, reasons, seed, report, *, base=None, weight=None, only=None):
        desired = set(CONDITIONS) if only is None else only
        fields = dict(seed=seed, replicate_id=report, mask_hash=reasons.fingerprint)
        weight = self.policy['mask']['primary_weight'] if weight is None else weight
        spaces, statuses = {}, {}

        def required(dependency):
            if dependency not in spaces:
                status, reason = statuses[dependency]
                if status == 'FAILED':
                    raise ValueError(reason)
                raise NotEstimable(reason)
            return spaces[dependency]

        def a():
            return base or self.build(raw, baseline)

        def b():
            if (raw.notna() & ~reasons.valid).any().any():
                raise NotEstimable('OBSERVED_VALUE_CONTRADICTS_PROVENANCE')
            if (raw.isna() & ~reasons.eligible).any().any():
                raise NotEstimable('FIXED_PANEL_HAS_INELIGIBLE_MISSING_CELLS')
            if self.method is None:
                raise NotEstimable('NO_CANDIDATE_PASSED_RECONSTRUCTION_GATE')
            value, tuning = complete_panel(raw, reasons, self.meta, baseline.patient_id.to_numpy(),
                self.method, self.impute_policy, seed, self.reference)
            self.append('candidate_benchmark', [{**r, 'stage':'refit_rank_tuning', 'replicate_id': report} for r in tuning])
            return self.build(raw, baseline, value, self.method), fingerprint(value.to_numpy())

        def mixed(cell, dependency):
            base_space = required(dependency)
            if weight and mask is None:
                if statuses['mask'][0] == 'FAILED':
                    raise ValueError(statuses['mask'][1])
                raise NotEstimable(statuses['mask'][1])
            space, audit = factorial_mds(base_space, mask=mask, weight=weight)
            self.append('mask_support_and_contribution', [{**fields, **audit, 'condition':cell, 'status':'COMPLETE',
                'completed_hash': completed_hash if dependency=='B' else '', 'mask_layer':'clinical_documented'}])
            return space

        sample = self.attempt('A', a, statuses)
        if sample is not None:
            spaces['A'] = sample
        fitted = self.attempt('B', b, statuses) if desired & {'B','D'} else None
        completed_hash = ''
        if fitted is not None:
            spaces['B'], completed_hash = fitted
        mask = None
        if desired & {'C','D'} and weight:
            mask = self.attempt('mask', lambda: self.mask(reasons, 'clinical_documented', fields,
                                                          record_pairs=report=='full'), statuses)
        for cell, dependency in [('C','A'), ('D','B')]:
            if cell not in desired:
                continue
            candidate = self.attempt(cell, lambda: mixed(cell, dependency), statuses)
            if candidate is not None:
                spaces[cell] = candidate
        return spaces, statuses, completed_hash

    def full(self):
        spaces, statuses, completed_hash = self.panel_spaces(
            self.raw, self.baseline, self.reasons, self.seed, 'full', base=self.base['space'])
        full = {}
        for cell, space in spaces.items():
            mapped = self.base['mapped'] if cell=='A' else None
            result = self.attempt(cell, lambda: item(space, self.config, mapped=mapped, audit=True,
                                  completed_hash=completed_hash if cell in ('B','D') else '',
                                  mask_hash=self.reasons.fingerprint if cell in ('C','D') else '',
                                  panel_hash=self.panel_fields['panel_hash']), statuses)
            if result:
                full[cell] = result
        rows = []
        for cell, identifier in self.cells.items():
            status, reason = statuses[cell]
            result = full.get(cell)
            topology = {}
            if result:
                mapped = result['mapped']
                topology = dict(n_covered=int(mapped['covered'].sum()), n_uncovered=int((~mapped['covered']).sum()),
                                n_nodes=len(mapped['nodes']), n_edges=mapped['nerve'].number_of_edges(),
                                n_components=len(mapped['components']), n_communities=len(mapped['community_sets']),
                                n_supported_communities=len(communities(mapped)),
                                component_sizes='|'.join(str(len(c)) for c in mapped['components']),
                                node_sizes='|'.join(str(len(c)) for c in mapped['nodes'].values()),
                                positive_lens_rank=int(np.linalg.matrix_rank(result['space'].lens)),
                                completed_hash=result['completed_hash'],
                                mask_hash=result['mask_hash'])
                audit = mapped.get('coverage_audit')
                if audit is None:
                    audit = analyze(result['space'], self.config, audit=True)['coverage_audit']
                counts = pd.Series([r['coverage_reason'] for r in audit], dtype=object).value_counts()
                self.append('factorial_coverage', [dict(experiment_id=identifier, condition=cell, reason=reason, count=int(count))
                                                   for reason,count in counts.items()])
                if cell != 'A':
                    self.append('factorial_membership_internal', frozen_membership(identifier, self.baseline, mapped, {}))
            rows.append(dict(experiment_id=identifier, condition=cell, status=status, reason=reason, seed=self.seed,
                             method=(self.method or 'unselected') if cell in ('B','D') else 'historical_median',
                             mask_layer='clinical_documented' if cell in ('C','D') else 'none',
                             graph03_defining=False, **topology))
        self.append('factorial_registry', rows)
        lenses = {}
        for cell in ('A','B'):
            identifier = self.reference+f'-{cell}-MDS-v3'
            if cell not in full:
                lenses[identifier] = None
                continue
            try:
                space, audit = factorial_mds(full[cell]['space'], lens_only=True)
                lenses[identifier] = item(space, self.config, panel_hash=self.panel_fields['panel_hash'])
                self.append('mask_support_and_contribution', [dict(condition=cell, experiment_id=identifier, status='COMPLETE', **audit)])
            except (NotEstimable, ValueError, np.linalg.LinAlgError) as exc:
                lenses[identifier] = None
                self.append('mask_support_and_contribution', [dict(condition=cell, experiment_id=identifier,
                    status='NOT_ESTIMABLE' if isinstance(exc, NotEstimable) else 'FAILED', reason=str(exc))])
        return full, lenses

    def resample(self, full):
        n = len(self.raw)
        minimum = self.settings['minimum_comparison_patients']
        for mode, budget in [('repeated_subsampling', self.policy['resampling']['main_subsamples']),
                             ('classic_bootstrap', self.policy['resampling']['main_bootstrap'])]:
            counters = {c: np.zeros((2, len(communities(ref['mapped']))), int) for c, ref in full.items()}
            accumulators = {c: Accumulator(n) for c in full}
            successes = dict.fromkeys(CONDITIONS, 0)
            for repeat in range(budget):
                seed = self.seed+repeat
                draw = draw_indices(n, self.settings['patient_fraction'], seed, mode)
                raw = self.raw.iloc[draw].reset_index(drop=True)
                baseline = self.baseline.iloc[draw].reset_index(drop=True)
                reasons = classify_panel(raw, self.registry, self.reference, baseline, self.date_column)
                unit = f'{mode}/{repeat}'
                spaces, statuses, completed_hash = self.panel_spaces(raw, baseline, reasons, seed, unit, only=set(full))
                unique, positions = np.unique(draw, return_index=True) if mode=='classic_bootstrap' else (draw, np.arange(len(draw)))
                rows = dict(replicate_id=repeat, resampling=mode, seed=seed, n_draws=len(draw), n_unique=len(unique),
                            split_hash=fingerprint(draw), mask_hash=reasons.fingerprint)
                mapped_items = {}
                for accumulator in accumulators.values():
                    accumulator.n_attempted[unique] += 1
                for cell in CONDITIONS:
                    identifier = self.cells[cell]
                    if cell in spaces and cell in full:
                        space = spaces[cell]
                        if mode == 'classic_bootstrap':
                            space = unique_space(space, positions)
                        mapped = self.attempt(cell, lambda: item(space, self.config), statuses)
                        if mapped:
                            successes[cell] += 1
                            mapped_items[cell] = mapped
                            match_rows, aligned = self.stability_match(full[cell], mapped, unique, counters[cell])
                            self.append('factorial_matching', [dict(**rows, experiment_id=identifier, **match) for match in match_rows])
                            accumulators[cell].collect(unique, mapped['mapped'], aligned)
                    status, reason = statuses.get(cell, ('NOT_ESTIMABLE','FULL_PANEL_CONDITION_UNAVAILABLE'))
                    self.append('factorial_refit_attempts', [dict(**rows, experiment_id=identifier, condition=cell,
                        completed_hash=completed_hash if cell in ('B','D') else '', status=status, reason=reason)])
                for left, right in combinations(CONDITIONS, 2):
                    metrics = comparison(mapped_items.get(left), mapped_items.get(right), minimum, len(unique))
                    self.append('factorial_paired_refit_comparisons', [dict(**rows, left=self.cells[left], right=self.cells[right], **metrics)])
                if len(mapped_items)==4:
                    coverage = {c: mapped_items[c]['mapped']['covered'].mean() for c in CONDITIONS}
                    fields = dict(interaction=coverage['D']-coverage['B']-coverage['C']+coverage['A'],
                                  a_to_b=coverage['B']-coverage['A'], a_to_c=coverage['C']-coverage['A'],
                                  b_to_d=coverage['D']-coverage['B'], c_to_d=coverage['D']-coverage['C'])
                    self.append('factorial_paired_effects', [dict(**rows, metric='coverage', status='COMPLETE', **fields)])
                else:
                    self.append('factorial_paired_effects', [dict(**rows, metric='coverage', status='NOT_ESTIMABLE',
                                                                  reason='FACTORIAL_CELL_UNAVAILABLE')])
                if self.method == 'mice_pmm_regularized' and 'B' in mapped_items:
                    self.nested_draws(raw, baseline, reasons, unique, positions, seed, rows, mapped_items)
            for cell, (evaluable, matched) in counters.items():
                identifier = self.cells[cell]
                self.record_stability(identifier, mode, budget, successes[cell], evaluable, matched)
                pairs, patients = accumulators[cell].rows(identifier, mode)
                self.append('factorial_coassignment_internal', pairs)
                self.append('factorial_coverage_internal', patients)
            for cell in set(CONDITIONS)-set(full):
                self.record_stability(self.cells[cell], mode, budget, 0, [], [])

    def stability_match(self, full, candidate, unique, counts):
        a = communities(full['mapped'])
        b = [set(unique[list(group)]) for group in communities(candidate['mapped'])]
        universe = set(unique)
        eligible = [k for k, group in enumerate(a) if len(group & universe)>=self.settings.get('minimum_match_patients',3)]
        counts[0, eligible] += 1
        matches, _ = match_sets([a[k] & universe for k in eligible], b)
        candidate_labels = np.flatnonzero(candidate['mapped']['supported'])
        aligned = {}
        for match in matches:
            match['reference'] = eligible[match['reference']]
            same = match['jaccard']>=self.settings['match_jaccard']
            counts[1, match['reference']] += same
            if same:
                aligned[candidate_labels[match['candidate']]] = match['reference']
        return matches, aligned

    def record_stability(self, identifier, mode, budget, successes, evaluable, matched):
        qc = successes/budget>=self.settings['minimum_success_fraction']
        rows = []
        for label, (n, hit) in enumerate(zip(evaluable, matched)):
            low, high = wilson(hit, n)
            rows.append(dict(community_id=label, n_evaluable=n, n_matched=hit,
                             stability=hit/n if n and qc else np.nan, conditional_stability=hit/n if n else np.nan,
                             conservative_stability=hit/budget, ci_low=low, ci_high=high,
                             stable=bool(qc and n and hit/n>=self.settings['stable_threshold']),
                             status='COMPLETE' if qc and n else 'NOT_ESTIMABLE',
                             reason='' if qc and n else 'REFIT_QC_OR_MATCH_SUPPORT_INSUFFICIENT'))
        self.append('factorial_stability', [dict(experiment_id=identifier, resampling=mode, n_requested=budget,
            n_success=successes, n_failed=budget-successes, qc_pass=qc, **row) for row in (rows or [
                dict(status='NOT_ESTIMABLE', reason='NO_SUPPORTED_REFERENCE_COMMUNITIES', n_evaluable=0)])])

    def nested_draws(self, raw, baseline, reasons, unique, positions, seed, fields, mapped_items):
        for draw in range(self.policy['uncertainty']['nested_draws_per_refit']):
            draw_seed = seed + 100000 + draw*1000
            states = {}
            candidate = self.attempt('B', lambda: self.imputed_draw(raw, baseline, reasons, draw_seed), states)
            if candidate is None:
                self.append('imputation_uncertainty', [dict(**fields, draw=draw, draw_seed=draw_seed,
                    uncertainty_source='one_or_more_completed_draws_within_refit', condition='B',
                    status=states['B'][0], reason=states['B'][1])])
                self.append('imputation_uncertainty', [dict(**fields, draw=draw, draw_seed=draw_seed,
                    uncertainty_source='one_or_more_completed_draws_within_refit', condition='D',
                    status='NOT_ESTIMABLE', reason='PAIRED_B_DRAW_UNAVAILABLE')])
                continue
            variants = {'B': candidate}
            if 'D' in mapped_items:
                mask = self.attempt('mask', lambda: self.mask(reasons, 'clinical_documented', fields), states)
                if mask is not None:
                    variants['D'] = self.attempt('D', lambda: factorial_mds(
                        candidate, mask=mask, weight=self.policy['mask']['primary_weight'])[0], states)
            for cell in ['B','D']:
                space = variants.get(cell)
                if space is not None and len(positions)!=len(raw):
                    space = unique_space(space, positions)
                mapped = self.attempt(cell, lambda: item(space, self.config), states) if space is not None else None
                values = comparison(mapped_items.get(cell), mapped, self.settings['minimum_comparison_patients'], len(unique))
                self.append('imputation_uncertainty', [dict(**fields, draw=draw, draw_seed=draw_seed,
                    uncertainty_source='one_or_more_completed_draws_within_refit', condition=cell, **values)])

    def imputed_draw(self, raw, baseline, reasons, seed, *, delta=0., delta_reason=None, stochastic=True):
        completed, _ = complete_panel(raw, reasons, self.meta, baseline.patient_id.to_numpy(), self.method,
            self.impute_policy, seed, self.reference, stochastic=stochastic, delta=delta, delta_reason=delta_reason)
        return self.build(raw, baseline, completed, self.method)

    def uncertainty(self, full):
        n = len(self.raw)
        mask = None
        if 'D' in full:
            mask = self.mask(self.reasons, 'clinical_documented', dict(replicate_id='uncertainty', seed=self.seed))
        if self.method != 'mice_pmm_regularized' or 'B' not in full:
            self.append('imputation_uncertainty', [dict(status='NOT_ESTIMABLE', reason='NO_SELECTED_STOCHASTIC_PMM_PANEL',
                n_requested=self.policy['uncertainty']['draws'], n_success=0, uncertainty_source='between_completed_draws')])
        else:
            accumulators = {c:Accumulator(n) for c in ('B','D') if c in full}
            for draw in range(self.policy['uncertainty']['draws']):
                for accumulator in accumulators.values():
                    accumulator.n_attempted += 1
                seed = self.seed+50000+draw
                statuses = {}
                b = self.attempt('B', lambda: self.imputed_draw(self.raw, self.baseline, self.reasons, seed), statuses)
                for cell in ('B','D'):
                    def with_map():
                        if b is None:
                            raise NotEstimable('PAIRED_B_DRAW_UNAVAILABLE')
                        if cell=='D' and mask is None:
                            raise NotEstimable('CLINICAL_MASK_UNAVAILABLE')
                        return item(b if cell=='B' else factorial_mds(
                            b, mask=mask, weight=self.policy['mask']['primary_weight'])[0], self.config)
                    mapped = self.attempt(cell, with_map, statuses)
                    status, reason = statuses[cell]
                    row = dict(condition=cell, experiment_id=self.cells[cell], draw=draw, seed=seed,
                               uncertainty_source='between_completed_draws', draw_scheme='patient_bootstrap_PMM_and_categorical_draws')
                    if mapped is not None:
                        metrics = comparison(full[cell], mapped, self.settings['minimum_comparison_patients'], n)
                        row.update(metrics, draw_status='COMPLETE')
                        counters = np.zeros((2,len(communities(full[cell]['mapped']))), dtype=int)
                        _, aligned = self.stability_match(full[cell], mapped, np.arange(n), counters)
                        accumulators[cell].collect(np.arange(n), mapped['mapped'], aligned)
                    else:
                        row.update(status=status, reason=reason, draw_status=status)
                    self.append('imputation_uncertainty', [row])
            for cell, accumulator in accumulators.items():
                pairs, patients = accumulator.rows(self.cells[cell], 'between_completed_draws')
                self.append('factorial_coassignment_internal', pairs)
                self.append('factorial_coverage_internal', patients)
                evaluable = [p for p in pairs if p['n_union_evaluable']>0]
                self.append('imputation_uncertainty', [dict(condition=cell, experiment_id=self.cells[cell], status='COMPLETE' if evaluable else 'NOT_ESTIMABLE',
                    reason='' if evaluable else 'NO_EVALUABLE_PAIRS', uncertainty_source='between_completed_draws_summary',
                    n_requested=self.policy['uncertainty']['draws'], n_success=int(accumulator.n_sampled.max()),
                    n_pairs_evaluable=len(evaluable), mean_union_coassignment=np.mean([r['union_coassignment'] for r in evaluable]) if evaluable else np.nan)])
        ordered_sources = self.meta.loc[~self.meta.encoding.str.contains('nominal|one_hot'), 'source_column']
        for reason in ('unknown','clinical_documented'):
            cells = self.reasons.cells
            n_affected = int((cells.reason_class.eq(reason) & cells.eligible_to_impute
                              & cells.source_column.isin(ordered_sources)).sum())
            for delta in self.policy['uncertainty']['delta']:
                fields = dict(delta=delta, reason_class=reason, n_eligible_targets=n_affected, seed=self.seed,
                              assumption='training_scale_shift_or_ordered_log_odds_tilt',
                              uncertainty_source='fixed_deterministic_completion')
                statuses = {}
                if not n_affected or 'B' not in full:
                    self.append('mnar_delta_sensitivity', [dict(**fields, condition=c, status='NOT_ESTIMABLE',
                        reason='NO_ELIGIBLE_BASELINE_COMPLETION_FOR_DELTA') for c in ('B','D')])
                    continue
                space = full['B']['space'] if not delta else self.attempt('delta', lambda: self.imputed_draw(
                    self.raw, self.baseline, self.reasons, self.seed, delta=delta, delta_reason=reason, stochastic=False), statuses)
                for cell in ('B','D'):
                    if cell not in full or space is None:
                        state, why = statuses.get('delta', ('NOT_ESTIMABLE','FACTORIAL_CELL_UNAVAILABLE'))
                        self.append('mnar_delta_sensitivity',[dict(**fields, condition=cell, status=state, reason=why)])
                        continue
                    def delta_map():
                        if not delta:
                            return full[cell]
                        candidate = space if cell=='B' else factorial_mds(space, mask=mask, weight=self.policy['mask']['primary_weight'])[0]
                        return item(candidate, self.config)
                    candidate = self.attempt(cell, delta_map, statuses)
                    metrics = comparison(full[cell], candidate, self.settings['minimum_comparison_patients'], n)
                    self.append('mnar_delta_sensitivity', [dict(**fields, condition=cell, **metrics)])

    def controls(self, full, legacy_experiments):
        n = len(self.raw)
        minimum = self.settings['minimum_comparison_patients']
        statuses = {}
        fields = dict(replicate_id='controls', seed=self.seed)
        base = full['A']['space']
        mask_policy = self.policy['mask']
        clinical = self.attempt('clinical_documented', lambda: self.mask(self.reasons, 'clinical_documented', fields), statuses)
        for layer in ('clinical_documented','administrative_only','clinical_plus_unknown','legacy_states','unknown_only'):
            candidate = None
            if layer=='clinical_documented':
                distance = clinical
            else:
                distance = self.attempt(layer, lambda: self.mask(self.reasons, layer, fields), statuses)
            if distance is not None:
                weight = 1. if layer in ('clinical_documented','administrative_only','unknown_only') else mask_policy['primary_weight']
                def control_map():
                    space, audit = factorial_mds(base, mask=distance, weight=weight)
                    self.append('mask_support_and_contribution', [dict(**fields, mask_layer=layer, status='COMPLETE', **audit)])
                    return item(space, self.config)
                candidate = self.attempt(layer, control_map, statuses)
            status, reason = statuses[layer]
            for cell in ('A','B','C'):
                values = comparison(full.get(cell), candidate, minimum, n)
                self.append('confounding_controls', [dict(control=layer, comparison_cell=cell,
                    control_status=status, control_reason=reason, **values)])
        for row in legacy_experiments:
            if row['experiment_id'].startswith(('N2a/','N2b/','N2c/','N2d/','M1/','N5/','N6/')):
                self.append('confounding_controls', [dict(control=row['experiment_id'], comparison_cell='historical_R0',
                    status=row['status'], reason=row['reason'], control_source='02_graph_experiment_registry.csv')])
        for weight in mask_policy['sensitivity_weights']:
            for cell, dependency in [('C','A'), ('D','B')]:
                def weight_map():
                    if dependency not in full or weight and clinical is None:
                        raise NotEstimable('VALUES_OR_CLINICAL_MASK_UNAVAILABLE')
                    space, audit = factorial_mds(full[dependency]['space'], mask=clinical, weight=weight)
                    if audit.get('reference_identity'):
                        return full[dependency]
                    return item(space, self.config)
                candidate = self.attempt(cell, weight_map, statuses)
                values = comparison(full.get(dependency), candidate, minimum, n)
                self.append('factorial_weight_sensitivity', [dict(condition=cell, mask_weight=weight,
                    variant_status=statuses[cell][0], variant_reason=statuses[cell][1], **values)])
        for threshold in mask_policy['sensitivity_shared_weight_fraction']:
            mask = self.attempt('support_threshold', lambda: self.mask(
                self.reasons, 'clinical_documented', fields, threshold=threshold), statuses)
            for cell, dependency in [('C','A'), ('D','B')]:
                def threshold_map():
                    if dependency not in full or mask is None:
                        raise NotEstimable('VALUES_OR_MASK_UNAVAILABLE_AT_SUPPORT_THRESHOLD')
                    return item(factorial_mds(full[dependency]['space'], mask=mask,
                                             weight=mask_policy['primary_weight'])[0], self.config)
                candidate = self.attempt(cell, threshold_map, statuses)
                values = comparison(full.get(dependency),candidate,minimum,n)
                self.append('factorial_support_threshold_sensitivity', [dict(condition=cell, support_threshold=threshold,
                    variant_status=statuses[cell][0], variant_reason=statuses[cell][1], **values)])
        self.parameter_refits(full)
        self.permutations(full, clinical)

    def parameter_refits(self, full):
        budget = self.policy['resampling']['parameter_subsamples']
        minimum = self.settings['minimum_comparison_patients']
        for repeat in range(budget):
            seed = self.seed+repeat
            draw = draw_indices(len(self.raw), self.settings['patient_fraction'], seed, 'repeated_subsampling')
            raw = self.raw.iloc[draw].reset_index(drop=True)
            baseline = self.baseline.iloc[draw].reset_index(drop=True)
            reasons = classify_panel(raw, self.registry, self.reference, baseline, self.date_column)
            wanted = {'A','B'} if 'B' in full else {'A'}
            spaces, statuses, completed_hash = self.panel_spaces(raw, baseline, reasons, seed,
                f'parameter_subsamples/{repeat}', only=wanted)
            fields = dict(replicate_id=repeat, seed=seed, n_requested=budget, split_hash=fingerprint(draw),
                          completed_hash=completed_hash, mask_hash=reasons.fingerprint)
            mask = self.attempt('mask', lambda: self.mask(reasons, 'clinical_documented', fields), statuses)
            for cell, dependency in [('C','A'), ('D','B')]:
                base = self.attempt(dependency, lambda: item(spaces[dependency], self.config), statuses) if dependency in spaces else None
                for weight in self.policy['mask']['sensitivity_weights']:
                    candidate = None
                    if base is not None and (not weight or mask is not None):
                        candidate = self.attempt(cell, lambda: base if not weight else item(
                            factorial_mds(base['space'], mask=mask, weight=weight)[0], self.config), statuses)
                    status, reason = statuses.get(cell, ('NOT_ESTIMABLE','VALUES_OR_CLINICAL_MASK_UNAVAILABLE'))
                    values = comparison(base, candidate, minimum, len(draw))
                    self.append('factorial_weight_refits', [dict(**fields, condition=cell, mask_weight=weight,
                        variant_status=status if candidate is not None else 'NOT_ESTIMABLE',
                        variant_reason=reason if candidate is not None else 'VALUES_OR_CLINICAL_MASK_UNAVAILABLE', **values)])

    def permutations(self, full, clinical):
        admin, _, audit = administrative_frame(self.baseline, self.cfg['validation']['confounders'], self.date_column)
        self.append('confounding_controls', [dict(control='administrative_source_audit', **row) for row in audit.to_dict('records')])
        minimum = self.settings['minimum_stratum_patients']
        strata = {column: administrative_strata(admin[column], column, minimum) for column in admin}
        strata['measurement_intensity'] = pd.Series(pd.qcut(self.reasons.observation.mask.mean(axis=1),
                                                          4, duplicates='drop')).astype('string')
        for family, part in self.reasons.observation.metadata.groupby('family'):
            intensity = self.reasons.observation.mask[part.index].mean(axis=1)
            strata['availability_'+family] = pd.Series(pd.qcut(intensity, 4, duplicates='drop')).astype('string')
        replicates = self.settings['observation_permutations']
        for cell, candidate in full.items():
            mapped = candidate['mapped']
            for name, labels in strata.items():
                mask = supported_hard(mapped) & labels.notna().to_numpy()
                hard = mapped['hard'][mask]
                group = labels[mask].to_numpy(dtype=str)
                values = dict(control='association_with_observation_context', context=name, condition=cell,
                              n_common=len(hard), n_requested=replicates, seed=self.seed)
                if len(hard)<minimum or len(np.unique(hard))<2 or len(np.unique(group))<2:
                    self.append('confounding_controls', [dict(**values, status='NOT_ESTIMABLE', reason='INSUFFICIENT_STRATUM_SUPPORT')])
                    continue
                rng = np.random.default_rng(self.seed)
                observed = normalized_mutual_info_score(hard, group)
                permuted = [normalized_mutual_info_score(hard, rng.permutation(group)) for _ in range(replicates)]
                p = (1+sum(v>=observed for v in permuted))/(1+replicates)
                self.append('confounding_controls', [dict(**values, status='COMPLETE', reason='', nmi=observed,
                    null_median=np.median(permuted), permutation_p=p, n_success=replicates)])
        if clinical is None or 'C' not in full:
            self.append('confounding_controls', [dict(control=mode, status='NOT_ESTIMABLE',
                reason='NO_ESTIMABLE_CLINICAL_MASK', n_requested=replicates, n_success=0)
                for mode in ('whole_mask_rows_permuted','whole_mask_rows_permuted_within_admin')])
            return
        states = self.reasons.masks['clinical_documented']
        n = len(states)
        eligible = []
        if len(admin.columns):
            joint = pd.DataFrame({c:strata[c] for c in admin}).dropna()
            eligible = [g.index.to_numpy() for _,g in joint.groupby(list(joint.columns), sort=True)
                        if len(g)>=minimum and g.index.isin(states.index).all()]
        modes = [('whole_mask_rows_permuted', [np.arange(n)]), ('whole_mask_rows_permuted_within_admin', eligible)]
        upper = np.triu_indices(n, 1)
        from scipy.stats import spearmanr
        base_values = full['A']['distance'][upper]
        observed = spearmanr(base_values, full['C']['distance'][upper]).statistic
        for mode, blocks in modes:
            statistics = []
            for repeat in range(replicates if blocks else 0):
                rng = np.random.default_rng(self.seed+repeat)
                order = np.arange(n)
                for block in blocks:
                    order[block] = rng.permutation(block)
                # Reordering all units jointly preserves intra-mask correlations.
                distance = clinical[np.ix_(order, order)]
                weight = self.policy['mask']['primary_weight']
                values = np.sqrt((1-weight)*(base_values/np.sqrt(np.mean(base_values**2)))**2 + weight*distance[upper])
                statistic = spearmanr(base_values, values).statistic
                if np.isfinite(statistic):
                    statistics.append(statistic)
                self.append('factorial_permutation_attempts', [dict(control=mode, replicate_id=repeat, seed=self.seed+repeat,
                    n_permutable=sum(map(len,blocks)), distance_spearman=statistic,
                    status='COMPLETE' if np.isfinite(statistic) else 'NOT_ESTIMABLE')])
            p = (1+sum(v<=observed for v in statistics))/(1+len(statistics)) if statistics else np.nan
            self.append('confounding_controls', [dict(control=mode, seed=self.seed, status='COMPLETE' if statistics else 'NOT_ESTIMABLE',
                reason='' if statistics else 'NO_SUPPORTED_PERMUTATIONS', n_requested=replicates, n_success=len(statistics),
                n_permutable=sum(map(len,blocks)), observed_distance_spearman=observed,
                null_median=np.median(statistics) if statistics else np.nan, permutation_p=p)])

    def decision(self, rows):
        by_cell = {r['condition']:r for r in rows}
        incomplete = [r for r in rows if r['status']!='COMPLETE']
        failed = any(r['status']=='FAILED' for r in incomplete)
        refits = [r for r in self.tables['factorial_stability'] if r['reference_id']==self.reference]
        supported = {r['experiment_id'] for r in refits if r.get('stable')}
        qc_pass = bool(refits) and all(r['qc_pass'] for r in refits)
        technical_state = 'FAILED' if failed else 'NOT_ESTIMABLE' if incomplete or not qc_pass else 'COMPLETE'
        if by_cell['B']['reason'] == 'NO_CANDIDATE_PASSED_RECONSTRUCTION_GATE':
            review = 'RETAIN_REFERENCE'
        elif incomplete:
            review = 'NOT_ESTIMABLE'
        elif not qc_pass or self.cells['A'] not in supported:
            review = 'RETAIN_REFERENCE'
        elif not set(self.cells.values()) <= supported:
            review = 'MISSINGNESS_SENSITIVE'
        else:
            # Crossing the reconstruction and stability gates only earns review.
            review = 'CANDIDATE_FOR_REVIEW'
        self.append('decision_summary', [dict(technical_state=technical_state, scientific_decision=review,
            selected_candidate=self.method or '', reason='|'.join(sorted({r['reason'] for r in incomplete})),
            graph03_defining=False, clinical_interpretation='EXPLORATORY_REQUIRES_METHODOLOGIC_AND_CLINICAL_REVIEW')])


def run_factorial(results, registry, manifest, baseline, families, config, cfg, run_dir, lab_groups, feature_to_lab,
                  historical, legacy_spaces, legacy_experiments, figure_dir, figure_prefix, source_hashes):
    policy = checked_policy(cfg)
    if policy is None:
        return None
    tables = defaultdict(list)
    tables['historical_failure_audit'] = historical
    all_items = dict(legacy_spaces)
    factor_items = {}
    n = len(baseline)
    for reference in policy['references']:
        raw = results.get(reference)
        try:
            if raw is None:
                raise NotEstimable('REFERENCE_NOT_ESTIMABLE')
            study = Factorial(reference, raw, registry, manifest, baseline, families, config, cfg, lab_groups,
                              feature_to_lab, policy, tables)
        except (ValueError, np.linalg.LinAlgError, MemoryError) as exc:
            status = 'NOT_ESTIMABLE' if isinstance(exc, NotEstimable) else 'FAILED'
            reason = str(exc).split(':')[0]
            for condition, suffix in CONDITIONS.items():
                built = condition=='A' and raw is not None
                tables['factorial_registry'].append(dict(reference_id=reference, experiment_id=reference+suffix,
                    condition=condition, status='COMPLETE' if built else status,
                    reason='' if built else reason, graph03_defining=False, n_patients=n))
            tables['decision_summary'].append(dict(reference_id=reference, technical_state=status,
                scientific_decision='NOT_ESTIMABLE', reason=reason, graph03_defining=False))
            if raw is not None:
                all_items[reference] = item(raw['space'], config, mapped=raw['mapped'])
            continue
        with threadpool_limits(limits=1):
            validation_status = {}
            study.attempt('benchmark', study.benchmark, validation_status)
            status, reason = validation_status['benchmark']
            if status!='COMPLETE':
                study.append('candidate_selection', [dict(candidate=method, selected=False, status=status, reason=reason)
                                                     for method in CANDIDATES])
            full, lenses = study.full()
            all_items.update({study.cells[c]:result for c,result in full.items()})
            all_items.update(lenses)
            factor_items.update({study.cells[c]:result for c,result in full.items()})
            study.additional_candidates(full)
            study.resample(full)
            study.uncertainty(full)
            study.controls(full, legacy_experiments)
            study.decision([r for r in tables['factorial_registry'] if r['reference_id']==reference])
    minimum = cfg['experiments']['minimum_comparison_patients']
    for reference, left, right, contrast in comparison_registry():
        a, b = all_items.get(left), all_items.get(right)
        values = comparison(a, b, minimum, n)
        tables['factorial_comparisons'].append(dict(reference_id=reference, left=left, right=right,
            contrast=contrast, left_panel_hash=a.get('panel_hash','') if a else '',
            right_panel_hash=b.get('panel_hash','') if b else '', **values))
        if a and b and a['mapped'] is not None and b['mapped'] is not None:
            full_matches, _ = match_sets(communities(a['mapped']), communities(b['mapped']))
            for match in full_matches:
                tables['factorial_community_matching'].append(dict(reference_id=reference, left=left, right=right,
                    contrast=contrast, n_total=n, **match))

    version = policy['version']
    run_id = uuid4().hex
    context = dict(run_id=run_id, estimator_version=version, config_sha256=cfg['_config_sha256'],
                   source_sha256=source_hashes.get('integrated',''), seed=int(config['random_seed']), graph03_defining=False)
    inventory = ['historical_failure_audit','reason_classification_summary','target_support_audit','candidate_benchmark',
                 'candidate_selection','factorial_comparisons','factorial_stability','mask_support_and_contribution',
                 'imputation_uncertainty','mnar_delta_sensitivity','confounding_controls','decision_summary',
                 'factorial_coverage','factorial_refit_attempts','factorial_paired_refit_comparisons',
                 'factorial_paired_effects','factorial_community_matching','factorial_matching','factorial_weight_sensitivity',
                 'factorial_support_threshold_sensitivity',
                 'factorial_weight_refits','candidate_fold_attempts','factorial_permutation_attempts',
                 'common_panel_sensitivity','candidate_topology_sensitivity',
                 'factorial_coassignment_internal','factorial_coverage_internal','factorial_membership_internal',
                 'mask_pair_support_internal','reason_cells_internal','factorial_registry']
    from .s02_reporting import factorial_figures, suppress_factorial_aggregates
    minimum_cell = cfg['experiments']['minimum_publication_cell']
    for name in ('candidate_benchmark','target_support_audit'):
        internal = name+'_internal'
        tables[internal] = tables[name]
        inventory.append(internal)
    for name in inventory:
        rows = tables[name] or [dict(status='NOT_ESTIMABLE', reason='NO_EVALUABLE_ROWS')]
        frame = pd.DataFrame([{**context, **row} for row in rows])
        save_table(run_dir/(PREFIX+name+'.csv'), suppress_factorial_aggregates(name, frame, minimum_cell))
    figures, figure_inventory = factorial_figures(figure_dir, tables, factor_items, minimum_cell, context)
    summary = [
        '# Graph02 missingness factorial',
        'The factorial changes values and the baseline observation metric independently for R0 and R2. '
        'Both contrasts use the frozen patients and encoded panel of their reference.',
        'Reconstruction selection uses patient-disjoint holdouts with paired median/mode losses. '
        'The clinical mask requires an explicit upstream reason. Unsupported pairs remain undefined.',
        'PMM draws use patient bootstrap and conditional donor or categorical sampling. '
        'One or more draws per resample provide a reduced nested sensitivity. '
        'Delta values are assumptions about missing values. The data do not identify MNAR.',
        'Each variant remains exploratory. The technical status and review decision are recorded separately.',
        '| Reference | Technical state | Review decision | Candidate |',
        '| --- | --- | --- | --- |',
        *[f"| {row['reference_id']} | {row['technical_state']} | {row['scientific_decision']} | {row.get('selected_candidate') or '—'} |"
          for row in tables['decision_summary']],
        '',
        *[f'- [{name}]({PREFIX}{name}.csv)' for name in inventory if not name.endswith('_internal')],
    ]
    name = PREFIX+'factorial_summary.md'
    (run_dir/name).write_text('\n'.join(summary)+'\n')
    table_names = [PREFIX+entry+'.csv' for entry in inventory] + [name]
    figure_names = {(figure_prefix / entry).as_posix(): figure_dir/entry for entry in figures}
    file_hashes = {**{entry: digest(run_dir/entry) for entry in table_names},
                   **{entry: digest(path) for entry, path in figure_names.items()}}
    artifacts = [*table_names, *figure_names]
    statuses = {r['experiment_id']:r['status'] for r in tables['factorial_registry'] if r['condition']!='A'}
    return dict(version=version, analysis_role='sensitivity', graph03_defining=False, run_id=run_id,
                policy=policy, statuses=statuses, artifacts=artifacts, sha256_by_artifact=file_hashes, source_hashes=source_hashes,
                decisions=tables['decision_summary'], figure_inventory=figure_inventory,
                experiments=[dict(**r, kind='missingness_factorial_v3') for r in tables['factorial_registry'] if r['condition']!='A'])
