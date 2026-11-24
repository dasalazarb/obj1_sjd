"""Graph03g reports built exclusively from disclosure-reviewed rows."""

from .s03_n3_exploratory_models import DIRECT, PROXY, UNRESOLVED
from .s03_n3_exploratory_source import RF_TEXT, RF_VALUE

ANA = 'lab__ana_status__text'
HEP2 = 'lab__ana_hep2_status__text'
URINE = 'lab__urine_protein_qualitative__text'
QUESTIONS = {
    RF_VALUE: ('¿El RF cuantitativo se usó en Mapper, en una recodificación de RF o en la clasificación serológica? '
               '¿Qué método, unidad y procedencia clínica documentan la comparabilidad?'),
    ANA: ('¿El ANA cualitativo corresponde a una prueba distinta de HEp-2 y de la serología de Mapper? '
          '¿Qué evidencia permite identificar la prueba y descartar nombres equivalentes?'),
    URINE: ('¿La proteinuria forma parte de ESSDAI renal u otro feature de Mapper? '
            '¿Qué método y episodio basal verificable respaldan la interpretación cualitativa?'),
    RF_TEXT: ('¿Hay observaciones cualitativas de RF reconocidas, válidas y comparables dentro de la ventana? '
              'RF cualitativo es la codificación alternativa de la misma hipótesis.'),
    HEP2: ('¿Qué documentación del ensayo distingue categorías válidas de títulos, patrones, censura y '
           'resultados administrativos, y permite diferenciar HEp-2 de ANA?'),
}
PRIORITY = {
    'INTEGRITY_BLOCKED': 1, 'DATA_UNAVAILABLE': 2, 'CLINICAL_PROXY_REVIEW_REQUIRED': 3,
    'SOURCE_OR_ASSAY_REVIEW_REQUIRED': 4, 'TECHNICAL_EVIDENCE_COMPLETE_REVIEW_REQUIRED': 5,
    'AUTHORIZED_STATISTICAL_REVIEW_REQUIRED': 6, 'NO_ELIGIBLE_MEASUREMENTS': 7,
    'TOPOLOGY_NOT_ELIGIBLE': 8, 'APPROVED_ONLY_IF_EXISTING_SIGNED_RECORD_VERIFIED': 9,
    'READY_FOR_03E': 10,
}
EXPLANATIONS = {
    'AUDIT_COMPLETE': 'Se completaron las verificaciones previstas.',
    'AUDIT_COMPLETE_WITH_LIMITATIONS': 'La auditoría conserva las limitaciones en las tablas de control.',
    'AUDIT_FAILED': 'Una comprobación de integridad impide usar la auditoría para decidir el avance a Graph03e.',
    'INTEGRITY_BLOCKED': 'Un contrato de procedencia, esquema o integridad impide usar el resultado actual.',
    'DATA_UNAVAILABLE': 'Falta un insumo o su esquema es ilegible. La cobertura publicada es evidencia histórica.',
    'REPRESENTATION_UNAVAILABLE': 'El release no dispone de membresías elegibles para esta representación.',
    'TOPOLOGY_NOT_STABLE': 'El estado topológico congelado limita el uso a la descripción.',
    'TOPOLOGY_UNVERIFIED': 'La publicación no establece el estado topológico para este par.',
    'COVERAGE_SUPPRESSED': 'La cobertura publicada está protegida y su magnitud queda indeterminada.',
    'ZERO_ELIGIBLE_MEASUREMENTS': 'La cobertura publicada indica cero mediciones que pasaron sus filtros.',
    'INDEPENDENCE_REVIEW_PENDING': 'Falta una revisión humana o una preespecificación con alcance compatible.',
    'HUMAN_REVIEW_REQUIRED': 'Las decisiones pendientes constan en el expediente de revisión.',
    'AUTHORIZED_STATISTICAL_REVIEW_REQUIRED': 'El procedimiento institucional regula el acceso al soporte recalculado.',
    'PATIENT_LEVEL_SELECTION_NOT_IN_PUBLISHED_CONTRACT': 'Los agregados permiten conciliar cobertura y definiciones.',
    'LONGITUDINAL_REUSE_EVIDENCE_UNAVAILABLE': 'El contrato no aporta evidencia divulgable para el desenlace longitudinal.',
    'DISCLOSURE_RESTRICTED': 'El resultado interno requiere revisión estadística por personal autorizado.',
    'INSUFFICIENT_COMPLETE_CASE_COMMUNITY_SUPPORT': 'Graph03e informó insuficiencia de casos completos o comunidades.',
    'INSUFFICIENT_BINARY_COMMUNITY_SUPPORT': 'Graph03e informó insuficiencia de eventos o no eventos por comunidad.',
    'INSUFFICIENT_EVENTS_PER_PARAMETER': 'Graph03e informó insuficiencia de eventos por parámetro.',
    'INSUFFICIENT_RESIDUAL_DF': 'El diseño carece de los grados de libertad residuales requeridos.',
    'RANK_DEFICIENT_DESIGN': 'El diseño carece de rango completo.',
    'DEGENERATE_OUTCOME': 'El desenlace carece de la variación necesaria para el diseño.',
    'FROZEN_SOURCE_OR_DERIVATION_DEFINES_MAPPER': 'La fuente o un derivado participa en la definición de Mapper.',
    'DOCUMENTED_LINEAGE_OR_CLINICAL_PROXY': 'El linaje documenta un constructo relacionado o un proxy.',
    'FROZEN_DEFINING_FEATURE_LINEAGE_INCOMPLETE': 'La representación contiene features sin linaje completo.',
    'OUTCOME_CONSTRUCT_LINEAGE_MISSING_OR_AMBIGUOUS': 'El constructo del desenlace carece de linaje inequívoco.',
    'DEFINING_SOURCE_REGISTRY_LINEAGE_MISSING': 'El registro congelado no documenta todas las fuentes definitorias.',
    'OUTCOME_ROLE_UNAVAILABLE': 'El contrato de caracterización no registra un rol para el desenlace.',
    'NO_VALID_MEASUREMENTS_IN_SCOPE': 'Graph03e informó que ninguna medición cumplió los filtros del análisis.',
    'SEPARATION': 'Graph03e detectó separación para un diseño binario.',
    'ALTERNATIVE_RF_ENCODING_DESCRIPTIVE_ONLY': 'Graph03e conservó la otra codificación de RF para la misma hipótesis.',
    'INSUFFICIENT_COVARIATE_COMPLETE_PATIENTS': 'Graph03e informó casos completos insuficientes.',
    'INSUFFICIENT_COMMUNITIES': 'Graph03e informó comunidades insuficientes.',
    'LOW_COMMUNITY_SUPPORT': 'Graph03e informó comunidades con soporte insuficiente.',
    'MODEL_DIAGNOSTIC_FAILURE': 'Graph03e informó un problema en los diagnósticos del ajuste previo.',
    'PUBLISHED_SELECTION_OR_COVERAGE_DRIFT': 'La selección o cobertura recalculada difiere de un agregado verificable.',
}
COMMAND = 'python src/studies/graph/03g_graph_n3_model_readiness_audit.py'

REVIEW_COLUMNS = [
    'priority', 'representation_id', 'outcome', 'clinical_question', 'structural_class',
    'source_code_evidence_status', 'measurement_provenance_status', 'time_policy_version',
    'assay_comparability_status', 'topology_status', 'statistical_support_public_status',
    'required_human_decision', 'review_status', 'candidate_role', 'interpretation',
    'missing_evidence', 'controlled_review_evidence', 'reviewer', 'decision', 'source_reference', 'signature', 'actions',
]


def review_queue(rows, version):
    queue = []
    for row in rows:
        source = row['outcome']
        structural = row['structural_class']
        actions = []
        if row['readiness_status'] == 'INTEGRITY_BLOCKED':
            actions.append('INTEGRITY_BLOCKED')
        if row['data_availability_status'] == 'DATA_UNAVAILABLE':
            actions.append('DATA_UNAVAILABLE')
        if structural in {PROXY, UNRESOLVED}:
            actions.append('CLINICAL_PROXY_REVIEW_REQUIRED')
        if row['topology_use'] == 'DESCRIPTIVE_ONLY':
            actions.append('TOPOLOGY_NOT_ELIGIBLE')
        if row['primary_measurement_status'] == 'ZERO_ELIGIBLE_MEASUREMENTS':
            actions.append('NO_ELIGIBLE_MEASUREMENTS')
        if (source in {RF_TEXT, HEP2} or row['assay_comparability_status'] != 'COMPARABLE_PRIMARY_DEFINITION'):
            actions.append('SOURCE_OR_ASSAY_REVIEW_REQUIRED')
        if row['model_A_support_status'] == 'SUPPRESSED':
            actions.append('AUTHORIZED_STATISTICAL_REVIEW_REQUIRED')
        candidate = row['readiness_status'].startswith('N3_')
        if candidate:
            actions.append('TECHNICAL_EVIDENCE_COMPLETE_REVIEW_REQUIRED' if row['review_status'] == 'PENDING'
                           else 'APPROVED_ONLY_IF_EXISTING_SIGNED_RECORD_VERIFIED')
        actions.sort(key=PRIORITY.__getitem__)
        if not actions and not candidate:
            continue
        queue.append(dict(
            priority=PRIORITY[actions[0]], representation_id=row['representation_id'], outcome=source,
            clinical_question=QUESTIONS[source], structural_class=structural,
            source_code_evidence_status=row['structural_reason'],
            measurement_provenance_status=row['time_eligibility_status'], time_policy_version=version,
            assay_comparability_status=row['assay_comparability_status'], topology_status=row['topology_status'],
            statistical_support_public_status=row['model_A_support_status'],
            required_human_decision='APPROVE|REJECT|REQUEST_EVIDENCE' if candidate or structural == PROXY else
            'REQUEST_EVIDENCE', review_status=row['review_status'],
            candidate_role=row['readiness_status'], interpretation=row['interpretation'],
            controlled_review_evidence=row.get('controlled_review_evidence', 'PENDING'),
            missing_evidence=row['missing_review_requirements'], reviewer='', decision='', source_reference='',
            signature='', actions='|'.join(actions)))
    return sorted(queue, key=lambda row: (row['priority'], row['representation_id'], row['outcome']))


def _matrix(rows, fields, headings):
    lines = ['| ' + ' | '.join(headings) + ' |', '|' + '---|' * len(fields)]
    lines.extend('| ' + ' | '.join(row.get(field, '').replace('|', ' / ') for field in fields) + ' |' for row in rows)
    return lines


def report(rows, summary, queue):
    lines = [
        '# Auditoría integral de Graph03g', '',
        f"**{summary['status']}**. {EXPLANATIONS[summary['status']]}", '',
        'Se usa la primera medición comparable de procedencia original verificada desde el día basal '
        'hasta seis meses calendario después, inclusive. La cobertura primaria se refiere a pacientes '
        'con asignación de comunidad hard. Los modelos A y B comparten la máscara de casos completos de Graph03e.', '',
        f"Representaciones solicitadas. {', '.join(summary['representations'])}.", '',
        f"Desenlaces configurados. {', '.join(summary['outcomes'])}.", '',
        '## Resultados por representación y desenlace', '',
    ]
    actions = {(row['representation_id'], row['outcome']): row['actions'].split('|')[0] for row in queue}
    table = [dict(row, next_action=actions.get((row['representation_id'], row['outcome']), ''))
             for row in rows]
    lines.extend(_matrix(table, [
        'representation_id', 'outcome', 'structural_class', 'review_status', 'topology_status',
        'primary_measurement_status', 'source_funnel_status', 'model_A_support_status', 'model_B_support_status',
        'disclosure_status', 'readiness_status', 'next_action'], [
        'Representación', 'Desenlace', 'Linaje', 'Revisión', 'Topología', 'Cobertura', 'Embudo',
        'A', 'B', 'Divulgación', 'Disposición', 'Acción']))
    codes = {code for row in rows for field in ('primary_blocker', 'additional_blockers', 'graph03e_reason')
             for code in row.get(field, '').split('|') if code}
    codes.update(summary['limitations'])
    lines.extend(['', '## Motivos y límites', ''])
    lines.extend(f'- `{code}`. {EXPLANATIONS.get(code, "Consulte el control o la evidencia de Graph03e correspondiente.")}'
                 for code in sorted(codes))
    lines.extend(['', '## Revisión pendiente', '',
                  '- El código verifica contratos, linaje, fechas, selección, comparabilidad y diseños A/B.',
                  '- El equipo clínico adjudica la independencia y documenta las decisiones en el procedimiento vigente.',
                  '- El personal estadístico autorizado revisa los diagnósticos protegidos. '
                  '`SUPPRESSED` deja indeterminado el soporte.',
                  '- RF cuantitativo tiene prioridad. RF cualitativo es una codificación alternativa. '
                  'ANA y HEp-2 comparten una familia de hipótesis. La ficha de proteinuria incluye el constructo renal.',
                  '', 'La interpretación es retrospectiva y exploratoria, condicionada a comunidades congeladas. '
                  'La medición podía estar indisponible en t0. Graph03g ejecutó cero estimadores.', '',
                  '## Ejecución y salidas', '', 'Desde la raíz del repositorio, con Python 3.10.', '',
                  '```bash', COMMAND, '```', '',
                  'Directorio `outputs/tables/studies/graph/03_graph_n3_model_readiness_audit/`.', ''])
    return '\n'.join(lines)


def review_packet(queue, references, manifest_sha):
    lines = [
        '# Expediente de revisión de Graph03g', '',
        'La cola contiene una decisión clínica por par candidato. Registre la decisión, justificación, '
        'fuente, rol e identidad del revisor y firma por el procedimiento institucional.', '',
        '## Evidencia reproducible', '',
    ]
    if manifest_sha:
        lines.append(f'- Descriptor Graph02 `{manifest_sha}` (SHA-256).')
    lines.extend(f'- `{path}` · `{sha}` (SHA-256).' for path, sha in sorted(references.items()))
    lines.extend(['', '## Fichas', ''])
    for row in queue:
        lines.extend([f"### {row['representation_id']} · {row['outcome']}", '',
                      row['clinical_question'], '',
                      f"Linaje `{row['structural_class']}`. Evidencia `{row['source_code_evidence_status']}`.",
                      f"Procedencia `{row['measurement_provenance_status']}`. "
                      f"Comparabilidad `{row['assay_comparability_status']}`. "
                      f"Soporte divulgado `{row['statistical_support_public_status']}`.",
                      f"Interpretación `{row['interpretation']}`. Estado `{row['review_status']}`. "
                      f"Evidencia de la ruta controlada `{row['controlled_review_evidence']}`.", '',
                      f"Decisión requerida `{row['required_human_decision']}`. "
                      f"Acciones `{row['actions'].replace('|', ', ')}`.", '',
                      f"Documentación pendiente. {row['missing_evidence'].replace('|', ', ') or 'Véanse las acciones.'}", ''])
    return '\n'.join(lines)
