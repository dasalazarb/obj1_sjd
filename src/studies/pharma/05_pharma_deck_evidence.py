"""Publish the fixed, aggregate Pharma 05 clinical and translational evidence."""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.studies._shared import sha256_file
from src.studies.pharma.stage05.absolute_risk import absolute_risk
from src.studies.pharma.stage05.biomarker_correlations import correlations
from src.studies.pharma.stage05.clinical_profiles import clinical_profiles
from src.studies.pharma.stage05.contracts import load_inputs, load_settings
from src.studies.pharma.stage05.eligibility import eligibility
from src.studies.pharma.stage05.patient_trajectories import (
    converter_profiles, marker_trends, sequence_history, trajectory_summary,
)
from src.studies.pharma.stage05.progression import risk_tables
from src.studies.pharma.stage05.quality_control import (
    baseline_reconciliation, coverage, evidence_by_transition, pro_slopes_fdr,
)
from src.studies.pharma.stage05.reporting import publish
from src.studies.pharma.stage05.serological_profiles import serological_profiles
from src.studies.pharma.stage05.specimen_feasibility import specimen_availability
from src.studies.pharma.stage05.transition_mechanisms import mechanisms
from src.studies.pharma.stage05.treatment_context import medication_summary, treatment_confounding


def run():
    directory = Path(__file__).resolve().parent
    settings, mapping = load_settings(directory)
    inputs = load_inputs(settings, mapping)
    master, intervals, features = inputs.master, inputs.intervals, inputs.features
    inputs.preflight['config_sha256'] = {
        name: sha256_file(directory / name)
        for name in ('config.yaml', 'config_stage05.yaml', '05_pharma_mapping.json')
    }
    projection, projection_support, bootstrap_qc = absolute_risk(inputs, settings)
    ever, risk, curve = risk_tables(master, settings)
    history = sequence_history(master, settings['sustained_visits'])
    trajectories, sustained = trajectory_summary(history, settings)
    clinical = clinical_profiles(intervals, features)
    serology = serological_profiles(intervals, features)
    domains, cooccurrence, pro, decomposition, leave_out = mechanisms(intervals, features)
    correlation, longitudinal = correlations(master, features, settings)
    biomarker_coverage = coverage(master, intervals, features)
    summary = evidence_by_transition(intervals, clinical, serology, biomarker_coverage, domains,
                                     projection_support, inputs.tables['models'])
    tables = {
        'ever_pop1_by_baseline_pop': ever,
        'pop1_cumulative_risk': risk,
        'pop1_risk_curve_source': curve,
        'patient_profiles_by_transition': clinical,
        'serology_profiles_by_transition': serology,
        'pop2_converter_profiles': pd.concat(
            [table.loc[table.from_pop.eq('Pop2')] for table in (clinical, serology)], ignore_index=True),
        'domain_drivers_by_transition': domains,
        'domain_cooccurrence': cooccurrence,
        'pro_changes_by_transition': pro,
        'essdai_decomposition': decomposition,
        'leave_domain_out_sensitivity': leave_out,
        'biomarker_correlations_by_pop': correlation,
        'pop2_biomarker_correlations': correlation.loc[correlation['pop'].eq('Pop2')],
        'longitudinal_biomarker_correlations': longitudinal,
        'absolute_risk_by_biomarker': projection,
        'absolute_risk_support': projection_support,
        'medication_by_transition': medication_summary(master, intervals),
        'treatment_confounding': treatment_confounding(intervals, settings, features),
        'trial_eligibility_proxy': eligibility(master, features),
        'patient_trajectory_summary': trajectories,
        'sustained_conversion': sustained,
        'patient_converter_profiles': converter_profiles(master, history, features, settings),
        'pretransition_biomarker_trends': marker_trends(master, history, features, settings),
        'biomarker_coverage': biomarker_coverage,
        'baseline_reconciliation': baseline_reconciliation(master, intervals, inputs.tables.get('baseline_reference')),
        'transition_evidence_summary': summary,
        'specimen_availability': specimen_availability(inputs.tables.get('specimens'), master, history, settings),
        'pro_slopes_fdr': pro_slopes_fdr(inputs.tables.get('pro_models')),
    }
    optional_status = {
        role: {'status': 'found' if role in inputs.tables else 'unavailable',
               'n_rows': len(inputs.tables[role]) if role in inputs.tables else None}
        for role in ('history_feasibility', 'history_models')
    }
    publish(tables, settings, inputs.preflight, inputs.inventory, bootstrap_qc, optional_status)


if __name__ == '__main__':
    if len(sys.argv) != 1:
        raise SystemExit('Pharma 05 uses the fixed repository configuration and takes no arguments.')
    try:
        run()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f'Pharma 05 failed. Previous outputs, if present, were preserved.\n{exc}') from None
    print('Pharma 05 completed. See the evidence summary and manifest in the fixed outputs directory.')
