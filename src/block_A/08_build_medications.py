#!/usr/bin/env python3
"""Build documented medication names from full BTRIS and original CTDB visits.

Orders, end dates and consent never establish administration or active exposure.
Consent-only matches are chart-review candidates, never medication dates.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import common
from src.block_A._internal.medications import contracts, source_inputs
from src.block_A._internal.medications.episode_linkage import match
from src.block_A._internal.medications.episode_features import build
from src.block_A._internal.medications.persistence import publish, sha256, manifest_path
from src.block_A._internal.step10_medications import load_contract


def run(args):
    if args.output.resolve().is_relative_to(args.eda_root.resolve()) or args.audit_root.resolve().is_relative_to(args.eda_root.resolve()):
        raise ValueError('Cannot write medication outputs back into the upstream EDA tree')
    if args.output.resolve() in {args.spine.resolve(),args.all_spine.resolve(),args.cohort.resolve()}:
        raise ValueError('Cannot overwrite a source or authoritative Step00 spine')
    key_path=args.pseudonym_key or args.audit_root/'restricted/.pseudonym_key'
    # This producer NEVER generates or silently replaces the established key.
    if not key_path.is_file(): raise FileNotFoundError('Existing compatible HMAC key required: --pseudonym-key')
    secret=key_path.read_bytes()
    if len(secret)<32: raise ValueError('HMAC key must contain at least 32 bytes')
    from src.block_A._internal.medications.documentation import pseudonym
    key_check=args.audit_root/'restricted/08_key_compatibility.json'
    if key_check.is_file() and json.loads(key_check.read_text()).get('scope')!=pseudonym(secret,'compatibility','08-medications'):
        raise ValueError('pseudonym key changed; use the original compatible key or a separate reviewed audit root')
    spine=pd.read_parquet(args.spine); all_spine=pd.read_parquet(args.all_spine)
    contracts.spine_signature(spine)
    paths={'spine':args.spine,'all_spine':args.all_spine,'cohort':args.cohort,
        'ctdb':args.eda_root/'data_analytic/visits_long.parquet',
        'row_map':args.eda_root/'data_intermediate/clinical_episode_row_map.parquet',
        **{protocol:args.eda_root/relative for protocol,relative in source_inputs.SOURCES.items()}}
    paths.update({f'code_{p.stem}':p for p in Path(source_inputs.__file__).parent.glob('*.py')})
    paths['producer']=Path(__file__)
    fingerprints={label:sha256(path) for label,path in paths.items()}
    # Reuse only a COMPLETE validated publication, never old truncated parquets.
    if args.reuse_validated_inputs:
        features, _=load_contract(args.output,spine,args.spine)
        if not args.output.with_suffix('.csv').is_file():
            raise ValueError('validated reuse requires medication CSV; run --force-rebuild')
        prior=json.loads(manifest_path(args.output).read_text())
        key_check=args.audit_root/'restricted/08_key_compatibility.json'
        from src.block_A._internal.medications.documentation import pseudonym
        if (prior.get('inputs_sha256')!=fingerprints or not key_check.is_file()
                or json.loads(key_check.read_text()).get('scope')!=pseudonym(secret,'compatibility','08-medications')):
            raise ValueError('validated inputs or pseudonym key changed; force rebuild after review')
        return {'status':'validated_reuse','n_rows':len(features)}
    if args.output.exists() and not args.force_rebuild and not args.dry_run:
        existing_manifest=manifest_path(args.output)
        if not existing_manifest.is_file():
            raise ValueError('Existing output is missing its manifest; manual review required')
        prior=json.loads(existing_manifest.read_text())
        if (prior.get('sha256')!=sha256(args.output)
            or prior.get('spine_sha256')!=fingerprints['spine']
            or prior.get('spine_signature')!=contracts.spine_signature(spine)
            or prior.get('keys')!=contracts.KEYS
            or prior.get('confirmed_medication_exposure') is not False
            or prior.get('consent_is_not_medication_date') is not True
            or prior.get('temporal_policy')!='one_calendar_year'
            or prior.get('contract_version') not in ('08-medication-episode-features-1.0',contracts.VERSION)):
            raise ValueError('Existing medication publication failed validation')
        if (not key_check.is_file() or
            json.loads(key_check.read_text()).get('scope') !=
                pseudonym(secret,'compatibility','08-medications')):
            raise ValueError('Existing Step08 publication has unverified HMAC identity')
        if prior.get('contract_version')==contracts.VERSION:
            load_contract(args.output,spine,args.spine)
            if prior.get('inputs_sha256')==fingerprints and args.output.with_suffix('.csv').is_file() and (args.audit_root/'08_medication_publication_qc.csv').is_file():
                return {'status':'validated_reuse','n_rows':len(spine)}
        else:
            # Older contract uses the same schema but may contain review names.
            # It must still validate before a canonical-only rebuild.
            contracts.validate(pd.read_parquet(args.output),spine)
        # Existing version or input has changed: proceed to rebuilt publication.
    catalog,qc=source_inputs.load(args.eda_root,args.cohort,spine,all_spine,secret)
    links=match(catalog,spine,secret); features=build(spine,links)
    unresolved=links.loc[links.name_requires_review].copy()
    qc.update(n_rows=len(features),n_patients=int(features.patient_id.nunique()),
        n_order_relations=int(links.evidence_level.eq('order_date_proximity').sum()),
        n_consent_only_relations=int(links.evidence_level.eq('consent_only_review').sum()),
        n_unassigned=int((~catalog.documentation_record_id.isin(links.documentation_record_id)).sum()),
        n_post_anchor_relations=int(links.post_anchor.sum()),
        confirmed_medication_exposure=False,consent_is_not_medication_date=True,
        n_names_present_by_category={c:int(features[c].notna().sum()) for c in contracts.COLUMNS},
        n_unresolved_episode_candidate_relations=len(unresolved),
        n_unresolved_candidate_documents=int(unresolved.documentation_record_id.nunique()),
        analytic_names_only_validated_canonical=True)
    from src.block_A._internal.medications.dates import exact_day
    unassigned=catalog.loc[~catalog.documentation_record_id.isin(links.documentation_record_id)].copy()
    unassigned['unassigned_reason']=[
        'original_nonclinical_episode' if r.source_kind=='CTDB' else
        ('no_exact_source_date' if all(pd.isna(exact_day(getattr(r,k+'_raw'))) for k in source_inputs.DATE_FIELDS)
         else 'outside_clinical_episode_windows') for r in unassigned.itertuples()]
    btris=catalog.loc[catalog.source_kind.eq('BTRIS')]
    no_order_date=[all(pd.isna(exact_day(getattr(r,k+'_raw'))) for k in ('order_start','order_end')) for r in btris.itertuples()]
    ritux=catalog.medication_name.str.contains(r'rituximab|rituxan|mabthera|truxima|ruxience|riabni',case=False,regex=True,na=False)
    ritux_links=links.loc[links.documentation_record_id.isin(catalog.loc[ritux,'documentation_record_id'])]
    qc.update(n_patient_anytime_documents=len(catalog),n_btris_without_exact_order_date=sum(no_order_date),
        unassigned_reasons={str(k):int(v) for k,v in unassigned.unassigned_reason.value_counts().items()},
        rituximab_documentation={
            'n_documents':int(ritux.sum()),
            'n_patients':int(catalog.loc[ritux,'linked_patient_pseudonymous'].nunique()),
            'n_order_date_relations':int(ritux_links.evidence_level.eq('order_date_proximity').sum()),
            'n_consent_only_relations':int(ritux_links.evidence_level.eq('consent_only_review').sum()),
            'n_candidate_episodes':int(ritux_links[contracts.KEYS].drop_duplicates().shape[0]),
            'not_administrations':True})
    if args.dry_run: return dict(qc,status='validated_dry_run_no_publication')
    tables={
        '08_medication_documentation_long.parquet':catalog,
        '08_medication_candidate_episode_links.parquet':links,
        '08_medication_order_date_details.parquet':catalog.loc[catalog.source_kind.eq('BTRIS')],
        '08_medication_patient_anytime.parquet':catalog,
        '08_medication_unassigned.parquet':unassigned,
        '08_medication_review_queue.parquet':links,
        '08_medication_unresolved_name_review.parquet':unresolved}
    sha=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
    manifest={'contract_version':contracts.VERSION,'producer_version':contracts.PRODUCER_VERSION,
        'run_at':datetime.now(timezone.utc).isoformat(),'code_sha':sha,
        'code_dirty':bool(subprocess.run(['git','status','--porcelain'],cwd=ROOT,text=True,capture_output=True,check=True).stdout),
        'inputs_sha256':fingerprints,'spine_sha256':fingerprints['spine'],
        'spine_signature':contracts.spine_signature(spine),'keys':contracts.KEYS,
        'n_rows':len(features),'n_patients':int(features.patient_id.nunique()),
        'n_episodes':len(features),'temporal_policy':'one_calendar_year',
        'confirmed_medication_exposure':False,'consent_is_not_medication_date':True,
        'qc':qc}
    publish(features,manifest,args.output,tables,qc,args.audit_root)
    from src.block_A._internal.medications.persistence import private_json
    from src.block_A._internal.medications.documentation import pseudonym
    private_json(args.audit_root/'restricted/08_key_compatibility.json',
                 {'scope':pseudonym(secret,'compatibility','08-medications')})
    return dict(qc,status='published')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eda-root',type=Path,default=common.EDA_SJD_ROOT)
    p.add_argument('--audit-root',type=Path,default=ROOT/'outputs/audit/08_medications')
    p.add_argument('--spine',type=Path,default=common.CLINICAL_VISIT_SPINE_PARQUET)
    p.add_argument('--all-spine',type=Path,default=common.EPISODE_SPINE_PARQUET)
    p.add_argument('--cohort',type=Path,default=ROOT/'data/raw/list_ids_longitudinal.csv')
    p.add_argument('--output',type=Path,default=common.MEDICATION_EPISODE_FEATURES_PARQUET)
    p.add_argument('--pseudonym-key',type=Path,help='Existing NIH protected key; never generated or exported')
    group=p.add_mutually_exclusive_group()
    group.add_argument('--force-rebuild',action='store_true')
    group.add_argument('--reuse-validated-inputs',action='store_true',help='Reuse complete publication only if source/code hashes and key scope match')
    p.add_argument('--dry-run',action='store_true',help='Validate all inputs and build in memory without writing')
    args=p.parse_args(argv)
    print(json.dumps(run(args),indent=2,sort_keys=True))

if __name__=='__main__': main()
