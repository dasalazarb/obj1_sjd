"""Presentation metadata and integrity checks; never authorize disclosure."""
from io import StringIO
import json
import pandas as pd

DISCLOSURE_POLICY_VERSION = 'conservative_whole_source_partition_v1'
TIERS = {'N1': ['N1_DEFINING'], 'N2': ['N2_RELATED'],
         'N3': ['N3_CANDIDATE', 'N3_REVIEWED']}
WITHHELD_MESSAGE = (
    'WITHHELD_DISCLOSURE_CONTROL\nData exist in frozen clinical characterization,\n'
    'but this panel is withheld under disclosure control.\n'
    'No clinical absence or lack of estimability is implied.')


def validate_figure_policy(validation):
    policy = validation.get('characterization', {}).get('figure_policy', {})
    if policy.get('disclosure_mode', DISCLOSURE_POLICY_VERSION) != DISCLOSURE_POLICY_VERSION:
        raise ValueError('UNSUPPORTED_DISCLOSURE_POLICY')
    restricted = policy.get('restricted_view', {})
    if restricted.get('enabled'):
        # This implementation has no alternate renderer or packaging route.
        if not restricted.get('authorization_reference') or not restricted.get('approved_output_root'):
            raise ValueError('RESTRICTED_VIEW_REQUIRES_DOCUMENTED_AUTHORIZATION_AND_DESTINATION')
        raise ValueError('RESTRICTED_VIEW_NOT_IMPLEMENTED_USE_PROTECTED_OUTPUTS')
    if restricted.get('automatic_packaging') or restricted.get('publication_eligible'):
        raise ValueError('RESTRICTED_VIEW_EXPORT_FORBIDDEN')


def observed_panel(matrix):
    """Do not count unknown/not tested as usable measurements."""
    return matrix.loc[pd.to_numeric(matrix.n_observed, errors='coerce').gt(0)].copy()


def drawable_panel(matrix):
    part = observed_panel(matrix)
    if part.empty:
        return part
    numeric = part.data_type.eq('numeric')
    usable = (~numeric & part.proportion.notna()) | (
        numeric & part[['median', 'q25', 'q75']].notna().all(axis=1))
    return part.loc[usable].copy()


def evaluate_characterization_panel(raw_matrix, safe_matrix, roles, *, representation_id,
                                    tier, selected_sources, minimum_cell, eligible_sources=None):
    """Return source metadata only, never counts/values of withheld patients.

    safe_matrix must come from reporting.safe_characterization_panels BEFORE
    selecting sources or communities. No disclosure rules are duplicated here.
    """
    reps = representation_id.split('|')
    selected = list(dict.fromkeys(selected_sources))
    role_part = roles.loc[roles.representation_id.isin(reps) & roles.source_column.isin(selected)]
    if tier in TIERS:
        role_part = role_part.loc[role_part.tier.isin(TIERS[tier])]
    matching = set(role_part.source_column)
    eligible = matching if eligible_sources is None else matching & set(eligible_sources)
    raw = observed_panel(raw_matrix.loc[raw_matrix.representation_id.isin(reps)
        & raw_matrix.source_column.isin(matching)])
    safe = drawable_panel(safe_matrix.loc[safe_matrix.representation_id.isin(reps)
        & safe_matrix.source_column.isin(eligible)])
    raw_keys = set(map(tuple, raw[['representation_id', 'source_column']].to_numpy()))
    safe_keys = set(map(tuple, safe_matrix[['representation_id', 'source_column']].to_numpy()))
    withheld = {key for key in raw_keys - safe_keys if key[1] in eligible}
    if not selected or not matching:
        status, reason = 'NO_PRESPECIFIED_TIER_SOURCES', 'No selected sources match this tier for this representation.'
    elif not eligible:
        status, reason = 'SOURCE_SELECTION_REJECTED', 'Selected sources failed documented relationship or interpretation checks.'
    elif raw.empty:
        status, reason = 'SOURCE_UNAVAILABLE', 'Selected sources are absent or have no usable observed values in frozen hard assignments.'
    elif safe.empty and withheld:
        status, reason = 'WITHHELD_DISCLOSURE_CONTROL', WITHHELD_MESSAGE.split('\n', 1)[1].replace('\n', ' ')
    elif safe.empty:
        status, reason = 'NOT_INTERPRETABLE_UNITS_OR_LEVELS', 'Observed values lack drawable frozen summaries; no recoding was applied.'
    else:
        status = 'DISPLAYED_DESCRIPTIVE'
        reason = ('Safe descriptive panels displayed; some source partitions remain withheld.' if withheld
                  else 'Safe descriptive panels displayed.')
    inference = ('REVIEW_PENDING' if role_part.tier.eq('N3_CANDIDATE').any() else
                 'DESCRIPTIVE_DEPENDENT' if tier == 'N2' else
                 'DESCRIPTIVE_DEFINING' if tier == 'N1' else
                 'EXISTING_STATISTICAL_GATES_REQUIRED' if tier == 'N3' else 'DESCRIPTIVE_ONLY')
    return dict(representation_id=representation_id, tier=tier,
        n_prespecified_sources=len(selected), n_sources_matching_tier=len(matching),
        n_sources_with_observations=raw.source_column.nunique(),
        n_displayable_sources=safe.source_column.nunique(),
        n_source_partitions_withheld=len(withheld),
        n_communities_displayed=len(safe[['representation_id', 'community_id']].drop_duplicates()),
        status=status, display_status=status, reason=reason,
        data_availability_status='OBSERVED' if not raw.empty else 'SOURCE_UNAVAILABLE',
        inference_status=inference, clinical_inference_status=inference,
        disclosure_policy_version=DISCLOSURE_POLICY_VERSION, minimum_publication_cell=minimum_cell,
        visibility='INTERNAL_RESTRICTED')


def select_n2_sources(matrix, safe, roles, validation, minimum_cell, provenance=None):
    """Validate each configured source against immutable representation roles."""
    specs = validation.get('characterization', {}).get('figure_sources_N2', [])
    required = ['source_column', 'label', 'related_to', 'expected_relationship',
                'clinical_domain', 'selection_reason', 'review_status']
    names = [s.get('source_column') for s in specs]
    rows = []
    for rep in ['R0', 'R2']:
        for spec in specs:
            source = spec.get('source_column', '')
            match = roles.loc[roles.representation_id.eq(rep) & roles.source_column.eq(source)]
            target = roles.loc[roles.representation_id.eq(rep) & roles.source_column.eq(spec.get('related_to'))]
            reason = ''
            if any(not isinstance(spec.get(k), str) or not spec[k].strip() for k in required):
                reason = 'INVALID_SELECTION_METADATA'
            elif names.count(source) != 1:
                reason = 'DUPLICATE_SOURCE_SELECTION'
            elif spec['clinical_domain'] not in {f'D{i:02d}' for i in range(1, 8)}:
                reason = 'INVALID_CLINICAL_DOMAIN'
            elif len(match) != 1 or match.iloc[0].tier != 'N2_RELATED':
                reason = 'SOURCE_NOT_N2_FOR_REPRESENTATION'
            else:
                role = match.iloc[0]
                if spec['expected_relationship'] not in str(role.relationship_type).split('|'):
                    reason = 'EXPECTED_RELATIONSHIP_NOT_DOCUMENTED'
                elif len(target) != 1 or target.iloc[0].tier != 'N1_DEFINING':
                    reason = 'RELATED_TO_NOT_DEFINING_FOR_REPRESENTATION'
                elif spec['clinical_domain'] != role.clinical_domain:
                    reason = 'CLINICAL_DOMAIN_MISMATCH'
                else:
                    conflicts = str(role.conflicting_defining_sources).split('|')
                    try:
                        lineage = json.loads(role.lineage_evidence)
                    except (ValueError, TypeError):
                        lineage = []
                    documented = spec['related_to'] in conflicts or any(
                        item.get('defining_source') == spec['related_to'] and any(
                            edge.get('relationship_type') == spec['expected_relationship'] and edge.get('source_reference')
                            for edge in item.get('path', [])) for item in lineage)
                    if not documented:
                        reason = 'RELATED_TO_NOT_IN_FROZEN_LINEAGE'
            part = matrix.loc[matrix.representation_id.eq(rep) & matrix.source_column.eq(source)]
            has_observed = not observed_panel(part).empty
            selection_status = 'REJECTED' if reason else 'SELECTED_DESCRIPTIVE'
            if not reason and source.startswith('lab__') and source.endswith('__text') and has_observed:
                # Existing binary assay semantics are usable; arbitrary free text
                # or numeric-looking laboratory text is never standardized here.
                if not observed_panel(part).data_type.eq('binary').all():
                    reason = 'NOT_INTERPRETABLE_UNITS_OR_LEVELS'
                    selection_status = reason
            if selection_status == 'SELECTED_DESCRIPTIVE':
                diagnostic = evaluate_characterization_panel(matrix, safe, roles,
                    representation_id=rep, tier='N2', selected_sources=[source], minimum_cell=minimum_cell)
                display = diagnostic['status']
                reason = diagnostic['reason']
            else:
                display = ('NOT_INTERPRETABLE_UNITS_OR_LEVELS' if selection_status != 'REJECTED'
                           else 'SOURCE_SELECTION_REJECTED')
            rows.append(dict(representation_id=rep, **{k: spec.get(k, '') for k in required},
                relationship_type=match.iloc[0].relationship_type if len(match) == 1 else '',
                selection_status=selection_status, has_observed_values=has_observed,
                display_status=display, reason=reason, **(provenance or {})))
    return pd.DataFrame(rows, columns=['representation_id', *required, 'relationship_type',
        'selection_status', 'has_observed_values', 'display_status', 'reason',
        'graph02_run_id', 'graph02_release_sha256'])


def figure_qc(folder, expected_sources, provenance=None, *, protected_matrix=None):
    """Check registered artifacts against the exact protected rendering inputs.

    Hashes have one owner: 03_graph_artifact_inventory.csv. This QC references
    those entries rather than maintaining a second digest inventory.
    """
    errors = []
    path = folder / '03_graph_figure_status.csv'
    inventory = pd.read_csv(path).fillna('') if path.exists() else pd.DataFrame(columns=['figure'])
    if not path.exists(): errors.append('MISSING_FIGURE_INVENTORY')
    if inventory.figure.duplicated().any(): errors.append('DUPLICATE_FIGURE_INVENTORY')
    if set(inventory.figure) != set(expected_sources): errors.append('FIGURE_INVENTORY_MISMATCH')
    expected_tiers = {f'03_graph_characterization_{tier}_{rep}' for tier in TIERS for rep in ['R0', 'R2']}
    expected_tiers |= {'03_graph_characterization_role_crosswalk', '03_graph_characterization_availability'}
    if not expected_tiers <= set(expected_sources): errors.append('MISSING_CHARACTERIZATION_FIGURES')
    items = []; generated = 0
    for name, source in expected_sources.items():
        item = dict(figure=name)
        available = {}
        for ext in ['png', 'pdf', 'csv']:
            artifact = folder / f'{name}.{ext}'
            available[ext] = artifact.is_file() and artifact.stat().st_size > 0
            if not available[ext]: errors.append(f'MISSING_ARTIFACT:{name}.{ext}')
            elif ext in {'png', 'pdf'}:
                with artifact.open('rb') as stream:
                    header = stream.read(8)
                if not header.startswith(b'\x89PNG\r\n\x1a\n' if ext == 'png' else b'%PDF-'):
                    errors.append(f'INVALID_ARTIFACT:{name}.{ext}')
        verified = False
        if available['csv']:
            # Match the serialization used by save_figure; catches extra rows,
            # columns, counts and values, not just membership in a source set.
            canonical = source if len(source.columns) else pd.DataFrame(columns=['status', 'reason'])
            try:
                expected = pd.read_csv(StringIO(canonical.to_csv(index=False)))
                actual = pd.read_csv(folder / f'{name}.csv')
                pd.testing.assert_frame_equal(actual, expected)
                verified = True
            except (AssertionError, ValueError, pd.errors.ParserError):
                errors.append(f'FIGURE_CSV_PROVENANCE_MISMATCH:{name}')
        clinical_source = (name.startswith('03_graph_characterization_N') or
            name == '03_graph_characterization_availability' or name.startswith('03_graph_clinical_marker_panels_'))
        if protected_matrix is not None and clinical_source and not source.empty:
            candidate = source.copy()
            if 'n_patients' in candidate and 'n_group' not in candidate:
                candidate = candidate.rename(columns={'n_patients': 'n_group'})
            keys = ['representation_id', 'source_column', 'community_id']
            if not set(keys) <= set(candidate):
                errors.append(f'CLINICAL_FIGURE_MISSING_SOURCE_KEYS:{name}')
            else:
                columns = keys + [c for c in ['category', 'n_group', 'n_observed', 'n_missing', 'n_pos', 'n_neg',
                    'n_tested', 'n_unknown', 'n_not_tested', 'n_category', 'median', 'q25', 'q75', 'proportion']
                    if c in candidate and c in protected_matrix]
                # Normalize CSV nulls/types on both sides without changing any values.
                left = pd.read_csv(StringIO(candidate[columns].to_csv(index=False)))
                right = pd.read_csv(StringIO(protected_matrix[columns].to_csv(index=False)))
                if left.merge(right.drop_duplicates(), on=columns, how='left', indicator=True)['_merge'].ne('both').any():
                    errors.append(f'CLINICAL_FIGURE_OUTSIDE_PROTECTED_VIEW:{name}')
        matches = inventory.loc[inventory.figure.eq(name)]
        if len(matches) == 1:
            item.update(matches.iloc[0].to_dict())
            if name in expected_tiers:
                required = ['status', 'reason', 'tier', 'representation_id', 'display_status',
                            'disclosure_policy_version', 'visibility', 'clinical_inference_status']
                if any(not item.get(k) for k in required): errors.append(f'INCOMPLETE_FIGURE_STATUS:{name}')
                displayed = item.get('status') == 'DISPLAYED_DESCRIPTIVE'
                if displayed != (not source.empty): errors.append(f'FIGURE_CONTENT_STATUS_MISMATCH:{name}')
                if item.get('visibility') != 'INTERNAL_RESTRICTED': errors.append(f'INVALID_VISIBILITY:{name}')
                if item.get('status') != item.get('display_status'):
                    errors.append(f'FIGURE_DISPLAY_STATUS_MISMATCH:{name}')
                if item.get('disclosure_policy_version') != DISCLOSURE_POLICY_VERSION:
                    errors.append(f'FIGURE_POLICY_MISMATCH:{name}')
                communities = (len(source[['representation_id', 'community_id']].drop_duplicates())
                    if {'representation_id', 'community_id'} <= set(source) else 0)
                try:
                    if (float(item['n_displayable_sources']) != source.source_column.nunique()
                            or float(item['n_communities_displayed']) != communities):
                        errors.append(f'FIGURE_METADATA_COUNT_MISMATCH:{name}')
                    if item['status'] == 'WITHHELD_DISCLOSURE_CONTROL' and (
                            float(item['n_sources_with_observations']) <= 0
                            or float(item['n_source_partitions_withheld']) <= 0):
                        errors.append(f'FIGURE_WITHHOLDING_WITHOUT_OBSERVATIONS:{name}')
                except (KeyError, TypeError, ValueError, AttributeError):
                    errors.append(f'INVALID_FIGURE_METADATA_COUNTS:{name}')
        item.update(files=available, protected_source_csv_verified=verified,
            artifact_hash_inventory={ext: f'figures/{name}.{ext}' for ext in available})
        generated += int(all(available.values()))
        items.append(item)
    clinical = [item for item in items if item['figure'] in expected_tiers and item.get('tier') in TIERS]
    displayed = sum(item.get('status') == 'DISPLAYED_DESCRIPTIVE' for item in clinical)
    limited = displayed != 6 or any(item.get('status') != 'DISPLAYED_DESCRIPTIVE'
        for item in items if item['figure'] in expected_tiers)
    limited |= any(item.get('n_source_partitions_withheld', 0) for item in clinical)
    return dict(**(provenance or {}), schema_version='graph03_figure_qc_v1',
        status='FAILED' if errors else 'PASS_WITH_PRESENTATION_LIMITATIONS' if limited else 'PASS',
        errors=errors, n_figures_expected=len(expected_sources), n_figures_generated=generated,
        n_descriptive_displayed=sum(i.get('status') in {'DISPLAYED_DESCRIPTIVE', 'DESCRIPTIVE'} for i in items),
        n_withheld=sum(i.get('status') == 'WITHHELD_DISCLOSURE_CONTROL' for i in items),
        n_no_prespecified_sources=sum(i.get('status') == 'NO_PRESPECIFIED_TIER_SOURCES' for i in items),
        n_unavailable=sum(i.get('status') == 'SOURCE_UNAVAILABLE' for i in items),
        n_r0_r2_clinical_figures_displayed=displayed, r0_r2_display_limitation=bool(limited),
        disclosure_policy_version=DISCLOSURE_POLICY_VERSION, figures=items,
        hash_inventory='03_graph_artifact_inventory.csv', publication_review_required=True,
        privacy_class='INTERNAL_RESTRICTED', shareable_export_allowed=False)
