"""Training-only reconstruction and paired validation for the Graph02 factorial."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from threadpoolctl import threadpool_limits

from .s02_feature_registry import RESERVED
from .s02_imputation import LowRankImputer, _guard, balanced_accuracy, source_groups
from .s02_missingness_registry import NotEstimable, panel_metadata


CANDIDATES = ('low_rank_regularized', 'mixed_regularized', 'mice_pmm_regularized')
BENCHMARKS = ('median_legacy', 'median_mode')
MECHANISMS = ('random_source', 'instrument_block', 'domain_block', 'observed_context', 'simulated_mnar')


def factorial_metadata(raw, registry, reference):
    meta = panel_metadata(raw, registry, reference)
    omitted = registry.loc[~registry.feature.isin(raw)
                           & registry.encoding.astype(str).str.contains('nominal|one_hot'), 'source_column']
    return meta.assign(nominal_levels_omitted=meta.source_column.isin(omitted))


def grouped_split(identities, seed):
    groups = pd.unique(identities)
    if len(groups) < 10:
        raise NotEstimable('INSUFFICIENT_DISTINCT_PATIENTS_FOR_VALIDATION')
    rng = np.random.default_rng(seed)
    held = rng.choice(groups, size=max(2, len(groups)//5), replace=False)
    test = np.isin(identities, held)
    # Fit with bootstrap multiplicities, evaluate each identity only once.
    unique = ~pd.Index(identities).duplicated()
    return np.flatnonzero(~test), np.flatnonzero(test & unique)


def artificial_mask(frame, meta, fraction, mechanism, seed):
    rng = np.random.default_rng(seed)
    hidden = np.zeros(frame.shape, dtype=bool)
    grouping = {'instrument_block': 'instrument_id', 'domain_block': 'clinical_domain'}.get(mechanism, 'source_column')
    parts = list(meta.groupby(grouping, sort=True, dropna=False))
    for _, part in parts:
        columns = frame.columns.get_indexer(part.index)
        support = frame.iloc[:, columns].notna().any(axis=1).to_numpy()
        rows = np.flatnonzero(support)
        count = int(np.ceil(fraction*len(rows)))
        if not count:
            continue
        probability = None
        if mechanism in ('observed_context', 'simulated_mnar'):
            context = frame.iloc[:, np.setdiff1d(np.arange(frame.shape[1]), columns)].notna().mean(axis=1)
            rank = context.rank(pct=True).fillna(.5)
            if mechanism == 'observed_context':
                available = [c for c in frame.columns if c not in part.index and frame[c].notna().any()]
                if available:
                    rank = frame[available[0]].rank(pct=True).fillna(rank)
            else:
                rank = frame.iloc[:, columns[0]].rank(pct=True).fillna(.5)
            weights = .15 + rank.iloc[rows].to_numpy()**2
            probability = weights/weights.sum()
        chosen = rng.choice(rows, size=count, replace=False, p=probability)
        hidden[np.ix_(chosen, columns)] = True
    return hidden & frame.notna().to_numpy()


@dataclass
class Target:
    source: str
    columns: list
    kind: str
    observed: np.ndarray
    values: np.ndarray
    counts: pd.Series
    scale: float
    reason: str
    nominal_levels_omitted: bool

    @property
    def n_categories(self):
        return len(self.columns)+int(self.nominal_levels_omitted)

    def nominal_codes(self, block):
        labels = block.argmax(axis=1)
        return np.where(block.sum(axis=1)==0, len(self.columns), labels) if self.nominal_levels_omitted else labels


def validate_nominal(block, allows_zero):
    seen = np.isfinite(block)
    complete = seen.all(axis=1)
    total = np.nansum(block, axis=1)
    if not np.isin(block[seen], [0,1]).all() or (total>1).any() or ((total[complete]!=1).any() and not allows_zero):
        raise NotEstimable('INVALID_NOMINAL_ENCODING')


class CompletionModel:
    """One categorical target per source. Missing training inputs stay within the fold."""

    def __init__(self, train, meta, method, settings, identities, needed, seed, reference, rank=None):
        _guard(train, meta)
        if np.isinf(train.to_numpy(float)).any():
            raise ValueError('INFINITE_IMPUTATION_INPUT')
        self.method, self.train, self.settings = method, train, settings
        self.identities = identities
        reserved = meta.role.isin(RESERVED) | meta.clinical_domain.isin(['D08', 'D09'])
        self.usable_predictors = set(meta.index[~reserved])
        self.targets = []
        self.reasons = {}
        self.models = []
        self.low_rank = None
        self.fitted_columns = []
        minimum = settings['minimum_category_support']
        for source, columns, kind in source_groups(meta):
            block = train[columns].to_numpy(float)
            observed = np.isfinite(block).all(axis=1)
            x = block[observed]
            projected = bool(kind=='nominal' and 'nominal_levels_omitted' in meta
                             and meta.loc[columns[0], 'nominal_levels_omitted'])
            if kind=='nominal':
                validate_nominal(block, projected)
                x = np.where(x.sum(axis=1)==0, len(columns), x.argmax(axis=1)) if projected else x.argmax(axis=1)
            else:
                x = x[:, 0]
            unique = ~pd.Index(identities[observed]).duplicated()
            counts = pd.Series(x[unique]).value_counts().sort_index()
            reason = ''
            if kind == 'binary' and not np.isin(x, [0, 1]).all():
                raise NotEstimable('INVALID_BINARY_ENCODING')
            if kind == 'ordinal' and not columns[0].endswith('__ordinal'):
                raise NotEstimable('UNVERIFIED_ORDINAL_ORDER')
            if not len(x):
                reason = 'ALL_MISSING_TRAINING_TARGET'
            elif method in CANDIDATES and any(c in needed for c in columns):
                if not set(columns) <= self.usable_predictors:
                    reason = 'RESERVED_SOURCE_NOT_RECONSTRUCTED'
                elif unique.sum() < max(minimum, settings['donor_count'] if method=='mice_pmm_regularized' and kind=='continuous' else 0):
                    reason = 'INSUFFICIENT_DISTINCT_TRAINING_PATIENTS'
                elif kind != 'continuous' and (len(counts) < 2 or counts.min() < minimum
                        or (kind == 'nominal' and len(counts) != len(columns)+int(projected))):
                    reason = 'INSUFFICIENT_CATEGORY_SUPPORT'
                elif kind == 'continuous' and len(counts) < 2:
                    reason = 'CONSTANT_TARGET_WITH_MISSING_VALUES'
                elif method == 'low_rank_regularized' and kind != 'continuous':
                    reason = 'LOW_RANK_CONTINUOUS_TARGETS_ONLY'
            scale = np.subtract(*np.percentile(x, [75, 25])) if len(x) else np.nan
            scale = scale if scale > 0 else np.std(x) if len(x) else np.nan
            target = Target(source, columns, kind, observed, x, counts, float(scale), reason, projected)
            self.targets.append(target)
            self.reasons[columns[0]] = reason
        self.fitted_columns = [c for target in self.targets if len(target.values) and not target.reason
                               for c in target.columns if c in self.usable_predictors]
        if method == 'low_rank_regularized':
            columns = [t.columns[0] for t in self.targets if t.kind == 'continuous' and t.columns[0] in self.usable_predictors
                       and not t.reason and np.isfinite(t.scale) and t.scale > 0]
            try:
                self.low_rank = LowRankImputer(rank, settings, reference).fit(train[columns], meta.loc[columns])
                self.fitted_columns = columns
            except (NotEstimable, ValueError, np.linalg.LinAlgError) as exc:
                self.low_rank = None
                for column in columns:
                    self.reasons[column] = str(exc)
        if method in ('mixed_regularized', 'mice_pmm_regularized'):
            self.fit_chain(needed, seed)

    @staticmethod
    def allowed_categories(target, block):
        valid = block != 0
        positive = block == 1
        has_known_category = positive.any(axis=1)[:, None]
        valid = np.where(has_known_category, positive, valid)
        return np.column_stack([valid, ~has_known_category]) if target.nominal_levels_omitted else valid

    @classmethod
    def assign(cls, filled, raw, target, prediction):
        original = raw[target.columns].to_numpy(float)
        observed = np.isfinite(original)
        if target.kind == 'nominal':
            validate_nominal(original, target.nominal_levels_omitted)
            allowed = cls.allowed_categories(target, original)
            if not allowed.any(axis=1).all():
                raise NotEstimable('NOMINAL_PARTIAL_ENCODING_INCONSISTENT')
            choices = np.argmax(np.where(allowed, prediction, -np.inf), axis=1)
            prediction = np.eye(target.n_categories)[choices, :len(target.columns)]
        else:
            prediction = np.asarray(prediction).reshape(-1, 1)
        filled[target.columns] = np.where(observed, original, prediction)

    def baseline(self, raw):
        filled = raw.copy().astype(float)
        if self.method == 'median_legacy':
            return filled.fillna(self.train.median())
        for target in self.targets:
            if not len(target.values):
                continue
            if target.kind == 'nominal':
                marginal = np.bincount(target.values, minlength=target.n_categories)
                prediction = np.broadcast_to(marginal, (len(raw), len(marginal)))
            elif target.kind in ('binary', 'ordinal'):
                prediction = np.full(len(raw), pd.Series(target.values).mode().iloc[0])
            else:
                prediction = np.full(len(raw), np.median(target.values))
            self.assign(filled, raw, target, prediction)
        return filled

    def fit_chain(self, needed, seed):
        filled = self.baseline(self.train)
        for sweep in range(self.settings['chain_sweeps']):
            models = []
            for target in self.targets:
                key = target.columns[0]
                if self.reasons[key] or not set(target.columns).intersection(needed):
                    continue
                # Sources with only unsupported levels cannot supply inferred predictors.
                predictors = [c for c in self.fitted_columns if c not in target.columns]
                if not predictors or not self.train.loc[target.observed, predictors].notna().any().any():
                    self.reasons[key] = 'NO_COOBSERVED_PREDICTORS'
                    continue
                x = filled.loc[target.observed, predictors].to_numpy()
                center, scale = x.mean(axis=0), x.std(axis=0)
                scale = np.where(scale > 0, scale, 1.)
                x = (x-center)/scale
                if target.kind == 'continuous':
                    model = Ridge(alpha=self.settings['mixed_ridge'])
                else:
                    model = LogisticRegression(C=self.settings['mixed_logistic_c'], max_iter=500, random_state=seed)
                model.fit(x, target.values)
                if target.kind != 'continuous' and np.max(model.n_iter_) >= 500:
                    self.reasons[key] = 'CLASSIFIER_NONCONVERGENCE'
                    continue
                donors = None
                if target.kind == 'continuous' and self.method == 'mice_pmm_regularized':
                    unique = ~pd.Index(self.identities[target.observed]).duplicated()
                    donors = model.predict(x[unique]), target.values[unique]
                step = (target, predictors, center, scale, model, donors)
                models.append(step)
                self.apply_step(filled, self.train, step)
            self.models.append(models)

    def apply_step(self, filled, original, step, rng=None):
        target, predictors, center, scale, model, donors = step
        x = (filled[predictors].to_numpy()-center)/scale
        missing = original[target.columns].isna().any(axis=1).to_numpy()
        probabilities = None
        if target.kind == 'continuous':
            pred = model.predict(x)
            if self.method == 'mice_pmm_regularized':
                scores, donor_values = donors
                chosen = np.flatnonzero(missing)
                distance = np.abs(pred[chosen, None]-scores[None, :])
                k = min(len(scores), self.settings['donor_count'])
                nearest = np.argsort(distance, axis=1, kind='stable')[:, :k]
                donors = nearest[:, 0] if rng is None else nearest[np.arange(len(chosen)), rng.integers(k, size=len(chosen))]
                pred[chosen] = donor_values[donors]
        else:
            probabilities = model.predict_proba(x)
            if target.kind == 'nominal':
                observed = original[target.columns].to_numpy(float)
                allowed = self.allowed_categories(target, observed)
                probabilities *= allowed[:, model.classes_.astype(int)]
            mass = probabilities.sum(axis=1)
            if (mass[missing]<=0).any():
                raise NotEstimable('NO_SUPPORTED_CATEGORY_COMPATIBLE_WITH_OBSERVATIONS')
            probabilities = np.divide(probabilities, mass[:, None], out=np.zeros_like(probabilities), where=mass[:, None]>0)
            score = probabilities if rng is None else np.log(np.maximum(probabilities, 1e-300))+rng.gumbel(size=probabilities.shape)
            if target.kind == 'nominal':
                pred = np.full((len(x), target.n_categories), -np.inf)
                pred[:, model.classes_.astype(int)] = score
            else:
                pred = model.classes_[score.argmax(axis=1)]
        self.assign(filled, original, target, pred)
        return probabilities, model.classes_ if probabilities is not None else None

    def predict(self, raw, *, seed=None, delta=0., delta_rows=None):
        filled = self.baseline(raw)
        probabilities = {}
        if self.method == 'low_rank_regularized' and self.low_rank is not None:
            columns = self.fitted_columns
            estimable = raw[columns].notna().any(axis=1)
            if estimable.any():
                filled.loc[estimable, columns] = self.low_rank.transform(raw.loc[estimable, columns])
            filled.loc[~estimable, columns] = np.nan
        elif self.method in ('mixed_regularized', 'mice_pmm_regularized'):
            rng = None if seed is None else np.random.default_rng(seed)
            for steps in self.models:
                probabilities = {}
                for step in steps:
                    target = step[0]
                    probabilities[target.columns[0]] = self.apply_step(filled, raw, step, rng)
        if self.method in CANDIDATES:
            for target in self.targets:
                columns = target.columns
                if self.reasons[columns[0]] or (self.method == 'low_rank_regularized'
                                              and (self.low_rank is None or not set(columns).issubset(self.fitted_columns))):
                    filled[columns] = raw[columns]
            # Baseline initialization of a patient with no observed predictors
            # cannot count as multivariable reconstruction.
            filled.loc[~raw.notna().any(axis=1)] = np.nan
        if delta:
            for target in self.targets:
                column = target.columns[0]
                if target.kind=='nominal' or self.reasons[column]:
                    continue
                affected = raw[column].isna() & delta_rows[column] & filled[column].notna()
                if not affected.any():
                    continue
                if target.kind=='continuous':
                    filled.loc[affected, column] += delta*target.scale
                elif column in probabilities:
                    prob, levels = probabilities[column]
                    tilt = np.exp(delta*np.linspace(-.5,.5,len(levels)))
                    filled.loc[affected, column] = levels[(prob[affected]*tilt).argmax(axis=1)]
        return filled, probabilities


def target_needed(raw):
    return set(raw.columns[raw.isna().any()])


def tune_rank(train, meta, ids, policy, seed, reference):
    rows = []
    for repeat in range(policy['tuning_repeats']):
        fold_seed = seed+repeat
        try:
            fit, held = grouped_split(ids, fold_seed)
        except NotEstimable as exc:
            rows.extend(dict(stage='inner_rank_tuning', candidate=CANDIDATES[0], repeat=repeat, seed=fold_seed,
                             rank=rank, n_predictions=0, n_hidden=0, normalized_mae=np.nan,
                             status='NOT_ESTIMABLE', reason=str(exc)) for rank in policy['candidate_rank'])
            continue
        test = train.iloc[held]
        hidden = artificial_mask(test, meta, policy['masking_fraction'][0], 'random_source', fold_seed)
        masked = test.mask(hidden)
        for rank in policy['candidate_rank']:
            model = CompletionModel(train.iloc[fit], meta, CANDIDATES[0], policy, ids[fit], target_needed(masked), fold_seed, reference, rank)
            prediction, _ = model.predict(masked)
            errors = []
            n_hidden = 0
            for target in model.targets:
                if target.kind != 'continuous' or target.scale <= 0:
                    continue
                column = target.columns[0]
                hidden_target = hidden[:, test.columns.get_loc(column)]
                n_hidden += int(hidden_target.sum())
                cells = hidden_target & prediction[column].notna().to_numpy()
                errors.extend((abs(prediction.loc[cells, column]-test.loc[cells, column])/target.scale).tolist())
            enough = n_hidden and len(errors)/n_hidden>=policy['minimum_prediction_coverage']
            reasons = '|'.join(sorted({r for r in model.reasons.values() if r}))
            rows.append(dict(stage='inner_rank_tuning', candidate=CANDIDATES[0], repeat=repeat, seed=fold_seed,
                             rank=rank, n_predictions=len(errors), n_hidden=n_hidden, n_train=len(fit), n_test=len(held),
                             normalized_mae=np.mean(errors) if enough else np.nan,
                             status='COMPLETE' if enough else 'NOT_ESTIMABLE',
                             reason='' if enough else reasons or 'INSUFFICIENT_HIDDEN_CELL_COVERAGE'))
    frame = pd.DataFrame(rows)
    grouped = frame.groupby('rank').normalized_mae.agg(['mean', 'count'])
    eligible = grouped.loc[grouped['count'] >= policy['tuning_repeats']*policy['minimum_successful_repeat_fraction']]
    return (int(eligible['mean'].idxmin()) if len(eligible) else None), rows


def evaluate_target(target, actual, predicted, hidden, probabilities, baseline, patient_ids):
    cols = target.columns
    was_hidden = hidden[:, actual.columns.get_indexer(cols)].all(axis=1)
    complete = was_hidden & actual[cols].notna().all(axis=1).to_numpy()
    valid = complete & predicted[cols].notna().all(axis=1).to_numpy()
    common = valid & baseline[cols].notna().all(axis=1).to_numpy()
    n_valid, n_hidden = int(valid.sum()), int(complete.sum())
    categorical = target.kind != 'continuous'
    metrics = dict(n_hidden=n_hidden, n_predictions=n_valid, n_paired=int(common.sum()),
                   n_patient_predictions=len(pd.unique(patient_ids[valid])))
    if categorical:
        truth = actual.loc[valid, cols].to_numpy()
        guess = predicted.loc[valid, cols].to_numpy()
        ref = baseline.loc[valid, cols].to_numpy()
        if target.kind == 'nominal':
            has_legal_sum = np.isin(guess.sum(axis=1), [0,1] if target.nominal_levels_omitted else [1])
            legal = np.isin(guess, [0, 1]).all(axis=1) & has_legal_sum
            metrics['n_illegal_predictions'] = int((~legal).sum())
            truth, guess, ref = map(target.nominal_codes, (truth, guess, ref))
        else:
            truth, guess, ref = truth[:, 0], guess[:, 0], ref[:, 0]
        if n_valid:
            metrics.update(accuracy=float(np.mean(truth==guess)), balanced_accuracy=balanced_accuracy(truth, guess),
                           class_support=json.dumps(pd.Series(truth).value_counts().to_dict(), sort_keys=True),
                           confusion_counts=json.dumps(pd.crosstab(truth, guess).to_dict(), sort_keys=True))
            if target.kind == 'ordinal':
                metrics['ordinal_mae'] = float(np.mean(np.abs(truth-guess)))
        if n_valid and probabilities and probabilities[0] is not None:
            prob, levels = probabilities
            actual_classes = truth[:, None]==levels[None, :]
            p = prob[valid]
            metrics.update(brier=float(np.mean(np.sum((p-actual_classes)**2, axis=1))),
                           log_loss=float(-np.mean(np.log(np.maximum(np.sum(p*actual_classes, axis=1), 1e-15)))))
        loss, ref_loss = (truth!=guess).astype(float), (truth!=ref).astype(float)
    else:
        truth = actual.loc[valid, cols[0]].to_numpy()
        guess = predicted.loc[valid, cols[0]].to_numpy()
        ref = baseline.loc[valid, cols[0]].to_numpy()
        error = np.abs(truth-guess)
        if n_valid:
            train_min, train_max = np.min(target.values), np.max(target.values)
            metrics.update(mae=float(error.mean()), rmse=float(np.sqrt(np.mean(error**2))),
                           normalized_mae=float(error.mean()/target.scale) if target.scale > 0 else np.nan,
                           observed_mean=truth.mean(), predicted_mean=guess.mean(),
                           observed_sd=truth.std(), predicted_sd=guess.std(),
                           observed_median=np.median(truth), predicted_median=np.median(guess),
                           predictions_outside_training_range=int(((guess<train_min)|(guess>train_max)).sum()))
        loss = error/target.scale if target.scale > 0 else np.full(n_valid, np.nan)
        ref_loss = np.abs(truth-ref)/target.scale if target.scale > 0 else np.full(n_valid, np.nan)
    paired = common[valid] & np.isfinite(loss) & np.isfinite(ref_loss)
    metrics['paired_loss'] = np.mean(loss[paired]) if paired.any() else np.nan
    metrics['paired_median_mode_loss'] = np.mean(ref_loss[paired]) if paired.any() else np.nan
    return metrics, zip(patient_ids[valid][paired], loss[paired], ref_loss[paired])


def validation(raw, meta, identities, policy, seed, reference):
    rows, support, paired, outcomes = [], [], [], []
    methods = [*BENCHMARKS, *CANDIDATES]
    for repeat in range(policy['repeats']):
        fold_seed = seed + 1000*repeat
        try:
            fit, held = grouped_split(identities, fold_seed)
        except NotEstimable as exc:
            for method in methods:
                outcomes.append(dict(candidate=method, repeat=repeat, status='NOT_ESTIMABLE', reason=str(exc),
                                     n_hidden=0, n_predictions=0, n_train=0, n_test=0, seed=fold_seed))
            continue
        train, test = raw.iloc[fit], raw.iloc[held]
        rank, tuning = tune_rank(train, meta, identities[fit], policy, fold_seed+1, reference)
        rows.extend({**row, 'outer_repeat': repeat} for row in tuning)
        split_hash = hashlib.sha256(np.asarray(held, dtype=np.int64).tobytes()).hexdigest()
        hidden_scenarios = []
        for mechanism in policy['masking_mechanisms']:
            for fraction in policy['masking_fraction']:
                hidden = artificial_mask(test, meta, fraction, mechanism, fold_seed+2)
                hidden_scenarios.append((mechanism, fraction, hidden))
        # All targets can occur in artificial masking. Their model is fitted once
        # per development fold and reused for every held-out masking mechanism.
        with threadpool_limits(limits=1):
            models = {
                method: CompletionModel(train, meta, method, policy, identities[fit], set(train.columns),
                                        fold_seed, reference, rank)
                for method in methods if method != CANDIDATES[0] or rank is not None
            }
        for mechanism, fraction, hidden in hidden_scenarios:
            masked = test.mask(hidden)
            baseline = models['median_mode'].predict(masked)[0]
            common_fields = dict(repeat=repeat, mechanism=mechanism, masking_fraction=fraction, seed=fold_seed,
                                 split_sha256=split_hash, n_train=len(fit), n_test=len(held),
                                 n_train_unique=len(pd.unique(identities[fit])))
            for method in methods:
                model = models.get(method)
                if model is None:
                    prediction, probabilities, failure = masked, {}, 'NO_VALIDATED_LOW_RANK'
                else:
                    try:
                        prediction, probabilities = model.predict(masked)
                        failure = ''
                    except NotEstimable as exc:
                        prediction, probabilities, failure = masked, {}, str(exc)
                hidden_n = predicted_n = 0
                critical_failure = False
                for target in (model or models['median_mode']).targets:
                    column = target.columns[0]
                    domain = meta.loc[column, 'clinical_domain']
                    family = meta.loc[column, 'technical_family'] if 'technical_family' in meta else meta.loc[column, 'family']
                    numbers, cells = evaluate_target(target, test, prediction, hidden, probabilities.get(column),
                                                     baseline, identities[held])
                    reason = failure or ((model.reasons[column] if model else '') if numbers['n_hidden'] else 'NO_HIDDEN_CELLS')
                    hidden_n += numbers['n_hidden']
                    predicted_n += numbers['n_predictions']
                    critical_failure |= numbers.get('n_illegal_predictions', 0)>0
                    rows.append(dict(**common_fields, stage='outer_evaluation', candidate=method, rank=rank if method==CANDIDATES[0] else None,
                                     source_column=target.source, target_column=column, variable_type=target.kind, domain=domain, family=family,
                                     status='COMPLETE' if numbers['n_predictions'] else 'NOT_ESTIMABLE',
                                     reason=reason or ('' if numbers['n_predictions'] else 'NO_PREDICTIONS'), **numbers))
                    paired.extend((method, domain, target.kind, patient, loss, ref) for patient, loss, ref in cells)
                outcomes.append(dict(**common_fields, candidate=method, n_hidden=hidden_n, n_predictions=predicted_n,
                                     status='COMPLETE' if hidden_n and predicted_n/hidden_n>=policy['minimum_prediction_coverage']
                                     and not critical_failure else 'NOT_ESTIMABLE',
                                     reason=failure or ('ILLEGAL_CATEGORIES' if critical_failure else
                                         'INSUFFICIENT_PREDICTION_COVERAGE' if not hidden_n or
                                         predicted_n/hidden_n<policy['minimum_prediction_coverage'] else '')))
        for method, model in models.items():
            for target in model.targets:
                key = target.columns[0]
                support.append(dict(candidate=method, repeat=repeat, seed=fold_seed, source_column=target.source, target_column=key,
                    domain=meta.loc[key, 'clinical_domain'], variable_type=target.kind,
                    n_observed=int(target.observed.sum()), n_observed_unique=int(target.counts.sum()), n_train=len(train),
                    n_missing=int((~target.observed).sum()), nominal_levels_omitted=target.nominal_levels_omitted,
                    training_variance=float(np.var(target.values)) if len(target.values) else np.nan,
                    target_scale=target.scale, n_levels=len(target.counts) if target.kind!='continuous' else np.nan,
                    category_counts=json.dumps(target.counts.to_dict(), sort_keys=True) if target.kind!='continuous' else '',
                    n_coobserved_predictor_rows=int(train.drop(columns=target.columns).notna().any(axis=1).to_numpy()[target.observed].sum()),
                    status='NOT_ESTIMABLE' if model.reasons[key] else 'COMPLETE', reason=model.reasons[key]))
    return rows, support, outcomes, paired


def select_candidate(outcomes, paired, policy, seed):
    scores = pd.DataFrame(paired, columns=['candidate', 'domain', 'kind', 'patient', 'loss', 'ref_loss'])
    attempts = pd.DataFrame(outcomes)
    decisions = []
    for method in CANDIDATES:
        folds = attempts.loc[attempts.candidate.eq(method)]
        success = float(folds.groupby('repeat').status.apply(lambda s: s.eq('COMPLETE').all()).mean())
        coverage = folds.n_predictions.sum()/max(1, folds.n_hidden.sum())
        candidate = scores.loc[scores.candidate.eq(method)]
        # Weight clinical domains and kinds equally per patient, then resample patients.
        balanced = candidate.groupby(['patient', 'domain', 'kind'])[['loss', 'ref_loss']].mean()
        patients = balanced.groupby('patient').mean().to_numpy()
        domain = balanced.groupby(['domain', 'kind']).mean()
        mean = patients.mean(axis=0) if len(patients) else [np.nan, np.nan]
        improvement = 1-mean[0]/mean[1] if mean[1]>0 else np.nan
        worst = (domain.loss-domain.ref_loss).max() if len(domain) else np.nan
        low = high = np.nan
        if len(patients) >= policy['minimum_paired_patients']:
            rng = np.random.default_rng(seed)
            sampled = patients[rng.integers(len(patients), size=(policy['patient_ci_replicates'], len(patients)))].mean(axis=1)
            supported = sampled[:, 1]>0
            if supported.any():
                low, high = np.quantile(1-sampled[supported, 0]/sampled[supported, 1], [.025, .975])
        passed = (success >= policy['minimum_successful_repeat_fraction'] and coverage >= policy['minimum_prediction_coverage']
                  and improvement >= policy['minimum_relative_improvement'] and low > 0
                  and worst <= policy['maximum_domain_deterioration'])
        decisions.append(dict(candidate=method, successful_repeat_fraction=success, historical_all_repeats_pass=success==1,
                              prediction_coverage=coverage, n_paired_patients=len(patients), relative_improvement=improvement,
                              ci_low=low, ci_high=high, maximum_domain_deterioration=worst,
                              status='COMPLETE' if passed else 'NOT_ESTIMABLE',
                              reason='' if passed else 'RECONSTRUCTION_GATE_NOT_MET'))
    passed = [d for d in decisions if d['status']=='COMPLETE']
    selected = max(passed, key=lambda d: d['relative_improvement'])['candidate'] if passed else None
    for decision in decisions:
        decision['selected'] = decision['candidate']==selected
    return selected, decisions


def complete_panel(raw, reasons, meta, identities, method, policy, seed, reference,
                   *, stochastic=False, delta=0., delta_reason=None):
    if (raw.notna() & ~reasons.valid).any().any():
        raise NotEstimable('OBSERVED_VALUE_CONTRADICTS_PROVENANCE')
    if (raw.isna() & ~reasons.eligible).any().any():
        raise NotEstimable('FIXED_PANEL_HAS_INELIGIBLE_MISSING_CELLS')
    for unit in reasons.observation.metadata.itertuples():
        if len(unit.sources)>1 and raw[list(unit.columns)].isna().any().any():
            raise NotEstimable('MULTISOURCE_DERIVATION_WITHOUT_RECONSTRUCTION_RULE')
    needed = target_needed(raw)
    if not needed:
        return raw.copy(), []
    rank, tuning = (None, [])
    if method==CANDIDATES[0]:
        rank, tuning = tune_rank(raw, meta, identities, policy, seed, reference)
        if rank is None:
            exc = NotEstimable('NO_RANK_VALIDATED_IN_REFIT')
            exc.tuning = tuning
            raise exc
    positions = np.arange(len(raw))
    if stochastic:
        rng = np.random.default_rng(seed)
        groups = pd.unique(identities)
        positions = np.concatenate([np.flatnonzero(identities==p) for p in rng.choice(groups, size=len(groups), replace=True)])
    train = raw.iloc[positions].reset_index(drop=True)
    delta_rows = reasons.eligible.copy()
    if delta_reason is not None:
        sources = reasons.cells.drop_duplicates(['patient_key','source_column']).pivot(
            index='patient_key', columns='source_column', values='reason_class')
        for column in raw:
            delta_rows[column] &= sources.loc[identities, meta.loc[column, 'source_column']].eq(delta_reason).to_numpy()
    with threadpool_limits(limits=1):
        model = CompletionModel(train, meta, method, policy, identities[positions], needed, seed, reference, rank)
        prediction_seed = seed if stochastic else None
        completed, _ = model.predict(raw, seed=prediction_seed, delta=delta, delta_rows=delta_rows)
    missing = completed.columns[completed.isna().any()]
    if len(missing):
        failures = sorted({model.reasons[t.columns[0]] or 'UNPREDICTED_TARGET'
                           for t in model.targets if set(missing).intersection(t.columns)})
        exc = NotEstimable('|'.join(failures))
        exc.tuning = tuning
        raise exc
    observed = raw.notna().to_numpy()
    if not np.array_equal(raw.to_numpy()[observed], completed.to_numpy()[observed]) or not np.isfinite(completed.to_numpy()).all():
        raise ValueError('INVALID_COMPLETED_PANEL')
    return completed, tuning
