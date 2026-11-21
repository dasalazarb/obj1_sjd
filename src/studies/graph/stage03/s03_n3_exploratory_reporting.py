"""Disclosure-controlled tables, figures, and release text for exploratory 03e."""

from collections import Counter
import json

import numpy as np
import pandas as pd

from .s03_n3_exploratory_models import ESTIMATED
from .s03_baseline_window import AFTER_BASELINE_WINDOW, FROZEN_SNAPSHOT, PREBASELINE, PRIMARY, WARNING
from .s03_n3_exploratory_source import DAY0, DAY30, EXACT_WINDOW, LATEST
from .s03_profiles import observed_values

FROZEN = FROZEN_SNAPSHOT
SCOPE = {
    PRIMARY: 'RETROSPECTIVE_BASELINE_CHARACTERIZATION',
    PREBASELINE: 'PREBASELINE_AUXILIARY_DESCRIPTION',
    AFTER_BASELINE_WINDOW: 'AFTER_CHARACTERIZATION_DESCRIPTION',
    DAY0: 'DAY0_DESCRIPTIVE_SENSITIVITY',
    DAY30: 'DAY0_TO_30D_DESCRIPTIVE_SENSITIVITY',
    LATEST: 'LATEST_IN_WINDOW_DESCRIPTIVE_SENSITIVITY',
    EXACT_WINDOW: 'EXACT_EPISODE_DESCRIPTIVE_SENSITIVITY',
}
LAB_FIELDS = ('representation_id outcome stratum provenance interpretation finding_class topology_status '
              'definition_id definition_role unit')
COUNTS = 'status n_total n_measured n_missing measurement_fraction'
PROFILE = (LAB_FIELDS + ' community_id ' + COUNTS
           + ' statistic_status n_positive n_negative positive_fraction median q1 q3')
MODEL = (LAB_FIELDS + ' hypothesis_family model_a model_b status reason not_estimable_reason n covariates family '
         'p_incremental q_bh fdr_reject_05 fdr_test_count fdr_families r2_a r2_b effects_json diagnostics observed_skewness '
         'n_parameters_a n_parameters_b residual_df_b')
TABLE_COLUMNS = {
    'coverage': LAB_FIELDS + ' ' + COUNTS + ' missing_reasons_json',
    'community_profiles': PROFILE,
    'frozen_baseline_profiles': ('representation_id source_column community_id topology_status interpretation '
                                 'panel tier dependence phase_a_denominator_verified status n_total n_measured '
                                 'n_missing measurement_fraction statistic_status median q1 q3 categories_json '
                                 'n_positive n_negative positive_fraction n_covered n_hard n_stable_hard n_provisional_hard'),
    'cohort_flow': (LAB_FIELDS + ' stage count_unit order status n_previous n_remaining n_excluded '
                    'all_removed_reason'),
    'source_linkage_audit': ('outcome source_role dimension category status count_unit n_source_files '
                             'collection_column original_discriminators n_total n_count fraction'),
    'exclusions': ('outcome stratum provenance audit_kind reason status denominator_reason '
                   'n_rows n_patients denominator_rows denominator_patients'),
    'independence_audit': 'representation_id outcome classification reason defining_sources blocking_sources',
    'models': MODEL,
    'sensitivity_models': MODEL,
    'linkage_sensitivity_models': MODEL,
    'followup_profiles': PROFILE,
    'baseline_eligibility': (LAB_FIELDS + ' community_id stage count_unit order status '
                             'n_previous n_remaining n_excluded all_removed_reason'),
    'measurement_delay': (LAB_FIELDS + ' community_id window_bin count_unit status n_patients n_total'),
    'baseline_inference': MODEL,

}


def _small(count, cell):
    return 0 < count < cell


def _coverage_counts(total, measured, cell):
    missing = total - measured
    if missing < 0:
        raise ValueError('EXPLORATORY_MEASUREMENT_EXCEEDS_DENOMINATOR')
    if any(_small(count, cell) for count in (total, measured, missing)):
        return dict(status='SUPPRESSED')
    return dict(status='DESCRIPTIVE', n_total=total, n_measured=measured, n_missing=missing,
                measurement_fraction=measured / total if total else None)


def _stats(values, numeric, cell):
    if len(values) < cell:
        return dict(statistic_status='SUPPRESSED' if len(values) else 'NO_OBSERVATIONS')
    if numeric:
        q1, median, q3 = values.quantile([.25, .5, .75])
        return dict(statistic_status='DESCRIPTIVE', median=float(median), q1=float(q1), q3=float(q3))
    positive = int(values.eq(1).sum())
    negative = len(values) - positive
    if _small(positive, cell) or _small(negative, cell):
        return dict(statistic_status='SUPPRESSED')
    return dict(statistic_status='DESCRIPTIVE', n_positive=positive, n_negative=negative,
                positive_fraction=positive / len(values))


def _profiles(frame, label, numeric, cell):
    profiles, withheld = [], False
    for community, group in frame.groupby('community_id', sort=True):
        if len(group) < cell:
            withheld = True
            continue
        values = group.value.dropna()
        counts = _coverage_counts(len(group), len(values), cell)
        row = dict(**label, community_id=community, **counts)
        if counts['status'] == 'DESCRIPTIVE':
            row.update(_stats(values, numeric, cell))
        profiles.append(row)
    if withheld:
        return [dict(**label, status='SUPPRESSED')]
    if any(profile['status'] == 'SUPPRESSED' for profile in profiles):
        # A shared frozen denominator or another representation can expose a
        # suppressed community through the other communities in this panel.
        return [dict(**label, community_id=profile['community_id'], status='SUPPRESSED')
                for profile in profiles]
    return profiles


def _partition(counts, cell):
    """Return a full partition only when every cell is releasable."""
    return dict(counts) if not any(_small(int(count), cell) for count in counts.values()) else None


def _coverage(frame, label, profiles, cell, disposition=None):
    if any(profile['status'] == 'SUPPRESSED' for profile in profiles):
        return dict(**label, status='SUPPRESSED')
    result = dict(**label, **_coverage_counts(len(frame), int(frame.value.notna().sum()), cell))
    if disposition is not None and result['status'] == 'DESCRIPTIVE':
        counts = _partition(disposition.reindex(frame.patient_id).value_counts().to_dict(), cell)
        if counts is not None:
            result['missing_reasons_json'] = json.dumps(counts, sort_keys=True)
    return result


def _audit_partition(label, dimension, series, cell, count_unit='patient'):
    counts = series.value_counts().to_dict()
    total = sum(counts.values())
    if _partition(counts, cell) is None:
        return [dict(**label, dimension=dimension, count_unit=count_unit, status='SUPPRESSED')]
    if not counts:
        return [dict(**label, dimension=dimension, count_unit=count_unit, status='DESCRIPTIVE',
                     category='NO_SOURCE_PATIENTS', n_total=0, n_count=0)]
    return [dict(**label, dimension=dimension, category=category, status='DESCRIPTIVE',
                 count_unit=count_unit, n_total=total, n_count=count, fraction=count / total)
            for category, count in sorted(counts.items())]


def _source_audit(inventory, outcome, cell):
    label = {'outcome': outcome, 'source_role': 'eda_long'}
    result = []
    identified = inventory.loc[inventory._patient_match.notna()]
    for field in ('identity_status', 'date_evidence'):
        # Repeated source rows can disagree for a patient. A single explicit
        # mixed class makes this a partition of distinct source patients.
        patient_states = identified.groupby('_patient_match')[field].agg(
            lambda values: values.iloc[0] if values.nunique() == 1 else 'MULTIPLE_' + field.upper())
        result.extend(_audit_partition(label, field, patient_states, cell))
    flags = {
        'BTRIS_CLUSTER_OBSERVATION_UNRECONCILED': inventory.btris_discriminator_unmatched,
        'STEP01_CONTEXT_SELECTION_DATE_DISAGREES': inventory.step01_context_date_disagreement,
        'ORIGINAL_SOURCE_ID_MISSING': inventory.source_patient_id.isna(),
        'PATIENT_NORMALIZATION_UNRESOLVED': inventory._patient_match.isna(),
    }
    for dimension, mask in flags.items():
        n = int(mask.sum())
        universe = len(inventory)
        unique = inventory.loc[mask, '_patient_match'].nunique()
        supported = n == 0 or unique >= cell
        # Identifier-less rows have no patient denominator and stay suppressed.
        counts = _coverage_counts(universe, n, cell) if supported else dict(status='SUPPRESSED')
        result.append(dict(**label, dimension=dimension, status=counts['status'], count_unit='source_row',
                           n_total=counts.get('n_total'), n_count=counts.get('n_measured'),
                           fraction=counts.get('measurement_fraction')))
    return result


def _exclusions(inventory, label, cell):
    excluded = inventory.loc[inventory.first_reason.ne('')]
    if excluded.empty:
        return []
    groups = list(excluded.groupby('first_reason', sort=True))
    if (any(group._patient_match.nunique() < cell or group._patient_match.isna().any() for _, group in groups)
            or _small(inventory._patient_match.nunique() - excluded._patient_match.nunique(), cell)
            or _small(len(inventory) - len(excluded), cell)):
        return [dict(**label, audit_kind='FIRST_CAUSE_SOURCE_ROW', status='SUPPRESSED',
                     denominator_reason='SOURCE_ROWS_IN_OUTCOME_STRATUM')]
    return [dict(**label, audit_kind='FIRST_CAUSE_SOURCE_ROW', reason=reason, status='DESCRIPTIVE',
                 denominator_reason='SOURCE_ROWS_IN_OUTCOME_STRATUM',
                 n_rows=len(group), n_patients=group._patient_match.nunique(),
                 denominator_rows=len(inventory), denominator_patients=inventory._patient_match.nunique())
            for reason, group in groups]


def _chain(label, counts, cell, unit='patient'):
    """Suppress adjacent cumulative counts around a small loss, including ties."""
    values = list(counts.values())
    if any(a < b for a, b in zip(values, values[1:])):
        raise ValueError('EXPLORATORY_NONMONOTONE_COHORT_FLOW')
    hidden_values = {n for n in values if _small(n, cell)}
    for before, remaining in zip(values, values[1:]):
        if _small(before - remaining, cell):
            hidden_values.update((before, remaining))
    rows = []
    for i, (stage, remaining) in enumerate(counts.items()):
        before = values[i - 1] if i else None
        safe = remaining not in hidden_values
        pair_safe = safe and before is not None and before not in hidden_values
        rows.append(dict(**label, stage=stage, count_unit=unit, order=i, status='DESCRIPTIVE' if safe else 'SUPPRESSED',
                         n_previous=before if pair_safe else None, n_remaining=remaining if safe else None,
                         n_excluded=before - remaining if pair_safe else None,
                         all_removed_reason=stage if before and remaining == 0 and pair_safe else None))
    return rows


def _baseline_statistics(values, cell, binary=False):
    if binary:
        return _stats(values, False, cell)
    if pd.api.types.is_numeric_dtype(values.dtype):
        return _stats(values, True, cell)
    labels = values.astype('string').str.strip().str.casefold()
    # Only controlled clinical vocabulary can enter a category label.
    safe = {'female', 'male', 'f', 'm', 'positive', 'negative', 'present', 'absent', 'yes', 'no',
            'normal', 'abnormal', 'true', 'false', 'trace', 'borderline', 'detected', 'not detected'}
    counts = _partition(labels.value_counts().to_dict(), cell)
    if counts is None or not set(counts) <= safe:
        return dict(statistic_status='CATEGORIES_WITHHELD')
    return dict(statistic_status='DESCRIPTIVE', categories_json=json.dumps(counts, sort_keys=True))


def _baseline_profiles(baseline, members, membership, roles, features, reps, topology, cell, provenance_verified):
    rows = []
    clinical = {'demo__age_at_baseline', 'demo__sex'}
    clinical.update(name for name in baseline if name.startswith(
        ('essdai__', 'esspri__', 'ocular__', 'serology__', 'lab__', 'ext__ocular', 'ext__biopsy')))
    clinical = {name for name in clinical if not name.endswith(('_date', '_datetime', '__unit', '_id'))}
    for rep in ('ALL_FROZEN_PATIENTS', *reps):
        common = dict(representation_id=rep, topology_status=topology.get(rep), interpretation=FROZEN,
                      panel='FROZEN_BASELINE_DESCRIPTIVE', phase_a_denominator_verified=provenance_verified)
        if rep != 'ALL_FROZEN_PATIENTS' and rep not in membership:
            rows.append(dict(**common, source_column='COHORT', status='REPRESENTATION_UNAVAILABLE'))
            continue
        frame = baseline if rep == 'ALL_FROZEN_PATIENTS' else membership[rep].merge(
            baseline, on='patient_id', validate='one_to_one')
        covered = len(baseline) if rep == 'ALL_FROZEN_PATIENTS' else members.loc[
            members.representation_id.eq(rep) & members.mapper_covered, 'patient_id'].nunique()
        stable = int(frame.structural_stability_status.eq('stable').sum()) if rep in membership else 0
        strata = [len(baseline) - covered, covered - len(frame), stable, len(frame) - stable]
        counts = _coverage_counts(len(baseline), len(frame), cell)
        if any(_small(n, cell) for n in strata):
            counts = dict(status='SUPPRESSED')
        rows.append(dict(**common, source_column='COHORT', **counts))
        if counts['status'] == 'DESCRIPTIVE' and rep in membership:
            rows[-1].update(n_covered=covered, n_hard=len(frame), n_stable_hard=stable,
                            n_provisional_hard=len(frame) - stable)
        selected = features.loc[features.representation_id.eq(rep) & features.included.eq(True)]
        defining = set(selected.source_column.dropna())
        n_roles = {name for (r, name), role in roles.items() if r == rep and str(role.get('tier', '')).startswith(
            ('N1', 'N2', 'N3'))}
        variables = sorted((defining | n_roles | clinical) & set(baseline))
        if rep in membership and (_small(len(baseline) - len(frame), cell) or
                                  frame.community_id.value_counts().lt(cell).any()):
            rows.append(dict(**common, source_column='COMMUNITY_PROFILES', status='SUPPRESSED'))
            continue
        groups = [('ALL', frame)] if rep == 'ALL_FROZEN_PATIENTS' else frame.groupby('community_id', sort=True)
        for community, group in groups:
            if len(group) < cell:
                continue
            for source in variables:
                values = observed_values(group[source]).dropna()
                tier = roles.get((rep, source), {}).get('tier',
                    'N1_DEFINING' if source in defining else 'BASELINE_CLINICAL')
                binary = pd.api.types.is_bool_dtype(values.dtype)
                counts = _coverage_counts(len(group), len(values), cell)
                stats = _baseline_statistics(values, cell, binary) if counts['status'] == 'DESCRIPTIVE' else {}
                rows.append(dict(**common, source_column=source, community_id=community, tier=tier,
                                 dependence='POTENTIALLY_CIRCULAR', **counts, **stats))
    return rows


def _figures(folder, frames):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    names = []

    def plot(filename, title, ylabel, samples):
        filename = '03e_' + filename
        if not samples:
            return
        fig = Figure(figsize=(9, max(2.8, len(samples) * .32 + 1.5)), layout='constrained')
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        labels, values = zip(*samples)
        ax.barh(range(len(values)), values, color='#296580')
        ax.set_yticks(range(len(values)), labels)
        ax.invert_yaxis()
        ax.set(title=title, xlabel=ylabel)
        fig.savefig(folder / filename, dpi=130)
        names.append(filename)

    baseline = frames['frozen_baseline_profiles']
    cohort = baseline.loc[baseline.source_column.eq('COHORT') & baseline.n_measured.notna()]
    plot('baseline_congelado.png', 'Cobertura del baseline congelado', 'Pacientes',
         list(zip(cohort.representation_id, cohort.n_measured)))

    flow = frames['cohort_flow']
    stages = flow.loc[flow.stratum.eq('ALL') & flow.n_remaining.notna() & flow.count_unit.eq('patient')]
    for outcome, section in stages.groupby('outcome'):
        if section.loc[section.stage.eq('n_distinct_source_patients'), 'n_remaining'].gt(0).any():
            plot('flujo_' + outcome + '.png', 'Flujo de selección · ' + outcome, 'Pacientes',
                 list(zip(section.stage, section.n_remaining)))
    coverage = frames['coverage']
    visible = coverage.loc[coverage.representation_id.eq('R0') & coverage.definition_role.eq('PRIMARY') &
                           coverage.measurement_fraction.notna()]
    for outcome, slices in visible.groupby('outcome'):
        if slices.measurement_fraction.gt(0).any():
            labels = slices.stratum + ' / ' + slices.provenance
            plot('cobertura_' + outcome + '.png', 'Cobertura R0 · ' + outcome, 'Fracción con medición',
                 list(zip(labels, slices.measurement_fraction)))
    dates = frames['source_linkage_audit']
    for outcome, slices in dates.loc[dates.dimension.eq('date_evidence') & dates.n_count.notna()].groupby('outcome'):
        if slices.n_count.gt(0).any():
            plot('procedencia_' + outcome + '.png', 'Procedencia de fecha · ' + outcome, 'Pacientes',
                 list(zip(slices.category, slices.n_count)))
    profiles = frames['community_profiles']
    profiles = profiles.loc[profiles.representation_id.eq('R0') & profiles.definition_role.eq('PRIMARY')]
    for outcome, slices in profiles.groupby('outcome'):
        for metric, axis in [('median', 'Mediana en la unidad original'),
                             ('positive_fraction', 'Fracción positiva')]:
            visible = slices.loc[slices[metric].notna()]
            labels = visible.stratum + ' / ' + visible.provenance + ' / ' + visible.community_id.astype(str)
            plot('comunidades_' + outcome + '_' + metric + '.png', 'Perfiles R0 · ' + outcome, axis,
                 list(zip(labels, visible[metric])))
    for table in ('models', 'sensitivity_models', 'linkage_sensitivity_models'):
        models = frames[table]
        estimated = models.loc[models.status.eq(ESTIMATED)]
        for scale in ('mean_difference', 'odds_ratio'):
            effects = [(f'{row.representation_id} / {row.outcome} / {effect["community_id"]} vs {effect["reference_community_id"]}',
                        effect) for row in estimated.itertuples() for effect in json.loads(row.effects_json)
                       if effect['scale'] == scale]
            if not effects:
                continue
            fig = Figure(figsize=(9, max(3, len(effects) * .35 + 1.5)), layout='constrained')
            FigureCanvasAgg(fig)
            ax = fig.subplots()
            for i, (_, effect) in enumerate(effects):
                center = effect['effect']
                ax.errorbar(center, i, xerr=[[center - effect['ci95_lower']], [effect['ci95_upper'] - center]],
                            fmt='o', capsize=3, color='#296580')
            ax.set_yticks(range(len(effects)), [label for label, _ in effects])
            ax.axvline(0 if scale == 'mean_difference' else 1, color='grey', linestyle='--')
            ax.invert_yaxis()
            ax.set(xlabel='Diferencia de medias (IC95%)' if scale == 'mean_difference' else 'OR (IC95%)',
                   xscale='linear' if scale == 'mean_difference' else 'log')
            filename = '03e_efectos_' + table + '_' + scale + '.png'
            fig.savefig(folder / filename, dpi=130)
            names.append(filename)
    return names


def _markdown(summary, frames):
    n = summary['n_frozen_cohort']
    n_label = str(n) if n is not None else 'suprimido o no disponible'
    lines = ['# Resultados exploratorios de Graph03 N3', '',
             f'**Estado:** `{summary["status"]}`. **Pacientes en el baseline congelado:** {n_label}.',
             f'Graph02 `{summary["graph02_input"] or "no disponible"}` · '
             f'Graph03 `{summary["graph03_input"] or "no disponible"}`.', '',
             '## Baseline y cobertura', '',
             'Los perfiles congelados describen las comunidades R0–R4 a partir de las variables observadas. '
             'Los perfiles de variables definidoras o relacionadas pueden ser circulares.', '',
             '| Resultado | Ventana | Procedencia | Medidos R0 |', '|---|---|---|---:|']
    coverage = frames['coverage']
    primary = coverage.loc[coverage.representation_id.eq('R0') & coverage.definition_role.eq('PRIMARY')]
    for row in primary.itertuples():
        count = 'SUPRIMIDO' if row.status == 'SUPPRESSED' else (
            'NA' if pd.isna(row.n_measured) else str(int(row.n_measured)))
        lines.append(f'| {row.outcome} | {row.stratum} | {row.provenance} | {count} |')
    if primary.empty:
        lines.append('| — | — | — | NA |')
    frozen = frames['frozen_baseline_profiles']
    r0 = frozen.loc[frozen.representation_id.eq('R0') & frozen.n_hard.notna()]
    if not r0.empty:
        row = r0.iloc[0]
        lines += ['', f'R0 incluyó {int(row.n_hard)} asignaciones hard con soporte. '
                  f'{int(row.n_stable_hard)} tenían etiqueta estructural estable y '
                  f'{int(row.n_provisional_hard)} eran provisionales.']
    profiles = frames['community_profiles']
    observed = profiles.loc[profiles.representation_id.eq('R0') & profiles.definition_role.eq('PRIMARY') &
                            (profiles['median'].notna() | profiles.positive_fraction.notna())]
    if not observed.empty:
        lines += ['', '| Resultado | Ventana | Procedencia | Comunidad R0 | Medidos | Perfil |',
                  '|---|---|---|---|---:|---|']
        for row in observed.itertuples():
            statistic = (f'{row.median:.3g} [{row.q1:.3g}, {row.q3:.3g}] {row.unit}'
                         if pd.notna(row.median) else f'{row.positive_fraction:.1%} positivos')
            lines.append(f'| {row.outcome} | {row.stratum} | {row.provenance} | {row.community_id} | '
                         f'{int(row.n_measured)} | {statistic} |')
    for table, title, intro in [
        ('models', 'Asociaciones exploratorias de día 0 a seis meses', WARNING),
        ('sensitivity_models', 'Sensibilidad temporal', 'Ventana basal con fecha de EDA sin corroboración original.'),
        ('linkage_sensitivity_models', 'Sensibilidad de vinculación', 'Ventana basal corroborada con enlace exacto por episodio.')]:
        models = frames[table]
        estimated = models.loc[models.status.eq(ESTIMATED)]
        lines += ['', '## ' + title, '', intro + f' Se estimaron {len(estimated)} pruebas globales.', '']
        if not estimated.empty:
            lines += ['| Representación | Resultado | Familia | p | q BH |',
                      '|---|---|---|---:|---:|']
            for row in estimated.itertuples():
                lines.append(f'| {row.representation_id} | {row.outcome} | {row.family} | '
                             f'{row.p_incremental:.4g} | {row.q_bh:.4g} |')
            r0 = estimated.loc[estimated.representation_id.eq('R0')]
            for row in r0.itertuples():
                for effect in json.loads(row.effects_json):
                    scale = 'diferencia de medias' if effect['scale'] == 'mean_difference' else 'OR'
                    lines.append(f'- R0 · {row.outcome}, comunidad {effect["community_id"]} frente a '
                                 f'{effect["reference_community_id"]}, {scale} {effect["effect"]:.3g} '
                                 f'(IC95% {effect["ci95_lower"]:.3g} a {effect["ci95_upper"]:.3g}).')
        reasons = Counter(models.loc[models.status.ne(ESTIMATED), 'not_estimable_reason'].dropna())
        if reasons:
            lines += ['', 'Motivos registrados por par de representación y resultado.', '']
            lines += [f'- `{reason}` ({count} pares)' for reason, count in sorted(reasons.items())]
    lines += ['', '## Día basal y seguimiento', '',
              WARNING,
              'El orden intradía del día 0 es desconocido. Los resultados prebasales y posteriores al '
              'límite calendario se describen por separado. La disponibilidad de mediciones y su demora '
              'pueden depender de la indicación clínica.', '', '## Límites', '']
    lines += [f'- {item}' for item in summary['limitations']]
    if summary['data_issues']:
        lines += ['', 'Insumos con incidencias.', '']
        lines += [f'- `{row["artifact"]}` — `{row.get("reason", row["status"])}`'
                  for row in summary['data_issues']]
    removed = frames['cohort_flow'].dropna(subset=['all_removed_reason'])
    if not removed.empty:
        lines += ['', 'Filtros que agotaron poblaciones con denominadores divulgables.', '']
        lines += [f'- `{row.outcome}` / `{row.stratum}` / `{row.stage}`'
                  for row in removed[['outcome', 'stratum', 'stage']].drop_duplicates().itertuples(index=False)]
    return '\n'.join(lines) + '\n'
