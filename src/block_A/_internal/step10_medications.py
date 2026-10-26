"""Read only the published Step08 table and manifest; never rebuild or read audits."""
import json
from pathlib import Path
import pandas as pd
from .medications.contracts import VERSION, KEYS, validate, spine_signature
from .medications.persistence import sha256, manifest_path, publication_lock


def load_contract(path, spine, clinical_spine_path):
    path=Path(path)
    if not path.is_file() or not manifest_path(path).is_file():
        raise FileNotFoundError('Step08 medication contract is required; run 08_build_medications.py or use --no-medications')
    with publication_lock(path):
        manifest=json.loads(manifest_path(path).read_text())
        if manifest.get('contract_version')!=VERSION:
            raise ValueError('incompatible medication contract version')
        if manifest.get('sha256')!=sha256(path):
            raise ValueError('medication contract hash mismatch')
        if manifest.get('spine_sha256')!=sha256(clinical_spine_path) or manifest.get('spine_signature')!=spine_signature(spine):
            raise ValueError('medication contract Step00 spine mismatch')
        if manifest.get('confirmed_medication_exposure') is not False or manifest.get('consent_is_not_medication_date') is not True or manifest.get('temporal_policy')!='one_calendar_year':
            raise ValueError('incompatible medication evidence policy')
        features=pd.read_parquet(path)
        validate(features,spine)
        if (manifest.get('keys')!=KEYS or manifest.get('n_episodes')!=len(features) or manifest.get('columns')!=list(features) or manifest.get('dtypes')!={c:str(features[c].dtype) for c in features}
            or manifest.get('n_rows')!=len(features) or manifest.get('n_patients')!=features.patient_id.nunique()):
            raise ValueError('medication manifest schema/count mismatch')
    return features, {'contract_version':VERSION,'sha256':manifest['sha256'],'n_rows':len(features),
        'confirmed_medication_exposure':False,'pre_anchor_treatment_exposure_established':False}
