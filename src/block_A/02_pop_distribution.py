#!/usr/bin/env python3
"""SECTION 2 — ESSDAI/ESSPRI-defined phenotypic subpopulations.

Builds the visit-level Pop1/Pop2/Pop3/Unclassifiable master file using the
same conceptual logic as src/block_A/01_pop_distribution.py, then produces
baseline/longitudinal distributions, trajectory figures, and QC outputs.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))
import common  # noqa:E402

PATIENT_ID_COL = "ids__patient_record_number"
VISIT_DATE_COL = "ids__visit_date"
ESSDAI_TOTAL_CANDIDATES = ["essdai__essdai_total_score"]
ESSPRI_COMPONENTS = ["esspri_questionnaire__dryness","esspri_questionnaire__fatigue","esspri_questionnaire__pain"]
REQUIRED_COLUMNS = [PATIENT_ID_COL, VISIT_DATE_COL, *ESSDAI_TOTAL_CANDIDATES, *ESSPRI_COMPONENTS]
POP_ORDER = ["Pop1", "Pop2", "Pop3", "Unclassifiable"]
POP_COLORS = {"Pop1":"#d95f02","Pop2":"#7570b3","Pop3":"#1b9e77","Unclassifiable":"#9e9e9e"}
MISSING_STRINGS = {"", "na", "n/a", "nan", "none", "unknown", "unk", "-99"}
INPUT_PATH = Path(getattr(common, "DEFAULT_POP_DISTRIBUTION_INPUT", common.DEFAULT_ANALYTIC_DATASET))
INTERMEDIATE_DIR = Path(getattr(common, 'INTERMEDIATE_DATA_DIR', PROJECT_ROOT/'data'/'intermediate'))
OUTPUTS_DIR = Path(getattr(common, 'OUTPUTS_DIR', PROJECT_ROOT/'outputs'))
TABLES_DIR = Path(getattr(common, 'BLOCKA_TABLES_DIR', OUTPUTS_DIR/'tables'/'blockA'))
FIGURES_DIR = Path(getattr(common, 'BLOCKA_FIGURES_DIR', OUTPUTS_DIR/'figures'/'blockA'))
QC_DIR = OUTPUTS_DIR/'qc'/'blockA'
SCRIPT_OUTPUT_PREFIX = Path(__file__).stem.replace('_distribution', '')
DISPLAY = {"Unclassifiable":"Unclassified", "Pop1":"Pop1", "Pop2":"Pop2", "Pop3":"Pop3", "Overall":"Overall"}
MASTER = INTERMEDIATE_DIR/f'{SCRIPT_OUTPUT_PREFIX}_visit_level_classification.parquet'


def is_missing(value: object) -> bool:
    if pd.isna(value): return True
    return str(value).strip().lower() in MISSING_STRINGS

def parse_visit_dates(value: object) -> list[pd.Timestamp]:
    if is_missing(value): return []
    dates=[]
    for fragment in str(value).split('|'):
        parsed = pd.to_datetime(fragment.strip(), errors='coerce')
        if pd.notna(parsed): dates.append(pd.Timestamp(parsed).normalize())
    return dates

def numeric_from_first_number(series: pd.Series) -> pd.Series:
    extracted = series.astype('string').str.extract(r"([-+]?\d*\.?\d+)", expand=False)
    return pd.to_numeric(extracted, errors='coerce')

def coalesce_essdai(df: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=df.index, dtype='float64')
    for col in ESSDAI_TOTAL_CANDIDATES:
        if col in df.columns: result = result.combine_first(numeric_from_first_number(df[col]))
    return result

def compute_esspri_from_components(dry: pd.Series, fat: pd.Series, pain: pd.Series) -> pd.Series:
    comp = pd.concat([dry, fat, pain], axis=1); comp.columns=['dryness','fatigue','pain']
    return comp.mean(axis=1).where(comp.notna().all(axis=1), np.nan)

def compute_esspri(df: pd.DataFrame) -> pd.Series:
    return compute_esspri_from_components(*(numeric_from_first_number(df[c]) for c in ESSPRI_COMPONENTS))

def classify_pop(essdai_total: object, esspri_total: object) -> str:
    essdai_missing = pd.isna(essdai_total); esspri_missing = pd.isna(esspri_total)
    if not essdai_missing and float(essdai_total) >= 5: return 'Pop1'
    if not essdai_missing and float(essdai_total) < 5 and not esspri_missing and float(esspri_total) >= 5: return 'Pop2'
    if not essdai_missing and float(essdai_total) < 5 and not esspri_missing and float(esspri_total) < 5: return 'Pop3'
    return 'Unclassifiable'

def first_nonmissing(s: pd.Series) -> Any:
    x = s.dropna(); return x.iloc[0] if len(x) else np.nan

def validate_columns(df: pd.DataFrame) -> list[str]:
    missing=[c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing: raise ValueError(f"Missing critical columns for Section 2: {missing}")
    return missing

def label_visit(v:int)->str: return 'Baseline' if int(v)==0 else f'Visit {int(v)}'

def reason(row: pd.Series) -> str:
    e_m=pd.isna(row.essdai_total); p_m=pd.isna(row.esspri_total)
    if e_m and p_m: return 'missing ESSDAI and ESSPRI'
    if e_m: return 'missing ESSDAI'
    if (not e_m) and float(row.essdai_total) < 5 and p_m: return 'missing ESSPRI with ESSDAI <5'
    return 'not classifiable by ESSDAI/ESSPRI rule'

def q(s, p): return s.quantile(p) if len(s.dropna()) else np.nan

def normalize_visit_level_dtypes(vis: pd.DataFrame) -> pd.DataFrame:
    """Use concrete dtypes before writing with parquet engines.

    Some pandas groupby aggregations that return a mix of floats and missing
    values can leave numeric columns as ``object`` dtype.  CSV output tolerates
    that, but fastparquet cannot infer an object encoding for columns such as
    ``esspri_total`` when their values are actually numeric floats/NaNs.
    """
    out = vis.copy()
    numeric_cols = [
        'time_since_baseline_days',
        'time_since_baseline_years',
        'visit_number',
        'essdai_total',
        'esspri_dryness',
        'esspri_fatigue',
        'esspri_pain',
        'esspri_total',
    ]
    datetime_cols = ['row_date_min', 'row_date_max', 'visit_date_clean', 'baseline_date', 'event_date']
    string_cols = [
        'patient_id',
        'row_date_original',
        'pop_status',
        'pop_status_display',
        'baseline_pop_status',
        'baseline_pop_status_display',
    ]
    for col in numeric_cols:
        if col in out.columns: out[col] = pd.to_numeric(out[col], errors='coerce')
    for col in datetime_cols:
        if col in out.columns: out[col] = pd.to_datetime(out[col], errors='coerce')
    for col in string_cols:
        if col in out.columns: out[col] = out[col].astype('string')
    return out

def build_visit_level(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    validate_columns(df); qc={}; warnings=[]
    qc['n_input_rows']=len(df)
    work=df.copy(); work['patient_id']=work[PATIENT_ID_COL].astype('string').str.strip()
    valid_pid=~work['patient_id'].map(is_missing); qc['n_rows_with_valid_patient_id']=int(valid_pid.sum()); qc['n_rows_excluded_missing_patient_id']=int((~valid_pid).sum())
    work['row_date_original']=work[VISIT_DATE_COL]
    parsed=work[VISIT_DATE_COL].map(parse_visit_dates)
    work['row_date_min']=parsed.map(lambda x: min(x) if x else pd.NaT); work['row_date_max']=parsed.map(lambda x: max(x) if x else pd.NaT); work['visit_date_clean']=work['row_date_min']
    valid_date=work['visit_date_clean'].notna(); qc['n_rows_with_valid_visit_date']=int(valid_date.sum()); qc['n_rows_excluded_missing_visit_date']=int((~valid_date).sum())
    work['essdai_total']=coalesce_essdai(work)
    work['esspri_dryness']=numeric_from_first_number(work[ESSPRI_COMPONENTS[0]])
    work['esspri_fatigue']=numeric_from_first_number(work[ESSPRI_COMPONENTS[1]])
    work['esspri_pain']=numeric_from_first_number(work[ESSPRI_COMPONENTS[2]])
    qc['n_invalid_essdai_values']=int(((work.essdai_total<0)|(work.essdai_total>123)).sum())
    comp_invalid=((work[['esspri_dryness','esspri_fatigue','esspri_pain']]<0)|(work[['esspri_dryness','esspri_fatigue','esspri_pain']]>10)).sum().sum(); qc['n_invalid_esspri_component_values']=int(comp_invalid)
    if qc['n_invalid_essdai_values']: raise ValueError(f"ESSDAI values outside plausible range 0-123: {qc['n_invalid_essdai_values']}")
    if qc['n_invalid_esspri_component_values']: raise ValueError(f"ESSPRI component values outside plausible range 0-10: {qc['n_invalid_esspri_component_values']}")
    work=work[valid_pid & valid_date].copy(); qc['n_unique_patients']=int(work.patient_id.nunique())
    qc['n_patient_visit_rows_before_collapse']=len(work); qc['n_duplicate_patient_visit_rows_before_collapse']=int(work.duplicated(['patient_id','visit_date_clean']).sum())
    agg=work.groupby(['patient_id','visit_date_clean'], as_index=False).agg(row_date_original=('row_date_original', lambda s:' | '.join(pd.Series(s).dropna().astype(str).unique())), row_date_min=('row_date_min','min'), row_date_max=('row_date_max','max'), essdai_total=('essdai_total',first_nonmissing), esspri_dryness=('esspri_dryness',first_nonmissing), esspri_fatigue=('esspri_fatigue',first_nonmissing), esspri_pain=('esspri_pain',first_nonmissing))
    agg['esspri_total']=compute_esspri_from_components(agg.esspri_dryness, agg.esspri_fatigue, agg.esspri_pain)
    if int(((agg.esspri_total<0)|(agg.esspri_total>10)).sum()): raise ValueError('ESSPRI total values outside plausible range 0-10 after collapse')
    agg['pop_status']=[classify_pop(e,p) for e,p in zip(agg.essdai_total, agg.esspri_total)]; agg['pop_status_display']=agg.pop_status.map(DISPLAY)
    agg=agg.sort_values(['patient_id','visit_date_clean']).reset_index(drop=True); agg['baseline_date']=agg.groupby('patient_id').visit_date_clean.transform('min'); agg['event_date']=agg.visit_date_clean
    agg['time_since_baseline_days']=(agg.visit_date_clean-agg.baseline_date).dt.days; agg['time_since_baseline_years']=agg.time_since_baseline_days/365.25; agg['visit_number']=agg.groupby('patient_id').cumcount()
    base=agg.loc[agg.visit_number.eq(0), ['patient_id','pop_status','pop_status_display']].rename(columns={'pop_status':'baseline_pop_status','pop_status_display':'baseline_pop_status_display'})
    agg=agg.merge(base,on='patient_id',how='left')
    qc['n_patient_visit_rows_after_collapse']=len(agg)
    if agg.patient_id.isna().any() or agg.visit_date_clean.isna().any() or (agg.time_since_baseline_years<0).any(): raise ValueError('Final visit-level validation failed')
    if not (agg.groupby('patient_id').visit_number.apply(lambda x:(x==0).sum()).eq(1).all()): raise ValueError('Each patient must have exactly one baseline visit')
    cols=['patient_id','row_date_original','row_date_min','row_date_max','visit_date_clean','baseline_date','event_date','time_since_baseline_days','time_since_baseline_years','visit_number','essdai_total','esspri_dryness','esspri_fatigue','esspri_pain','esspri_total','pop_status','pop_status_display','baseline_pop_status','baseline_pop_status_display']
    return normalize_visit_level_dtypes(agg[cols]), qc, {'warnings':warnings}

def baseline_distribution(vis: pd.DataFrame) -> pd.DataFrame:
    base=vis[vis.visit_number.eq(0)].copy(); follow=vis.groupby('patient_id').agg(followup=('time_since_baseline_years','max'), n_visits=('visit_number','count')).reset_index(); base=base.merge(follow,on='patient_id')
    total=len(base); class_n=int(base.pop_status.isin(POP_ORDER[:3]).sum()); rows=[]
    for pop in POP_ORDER:
        g=base[base.pop_status.eq(pop)]; rows.append({'pop_status':pop,'pop_status_display':DISPLAY[pop],'n_patients':len(g),'pct_of_total_baseline':len(g)/total*100 if total else np.nan,'pct_of_classifiable_baseline':(len(g)/class_n*100 if pop!='Unclassifiable' and class_n else np.nan),'median_followup_yrs':g.followup.median(),'q1_followup_yrs':q(g.followup,.25),'q3_followup_yrs':q(g.followup,.75),'median_n_visits':g.n_visits.median(),'q1_n_visits':q(g.n_visits,.25),'q3_n_visits':q(g.n_visits,.75),'n_essdai_available_baseline':int(g.essdai_total.notna().sum()),'n_esspri_available_baseline':int(g.esspri_total.notna().sum()),'n_essdai_esspri_available_baseline':int((g.essdai_total.notna()&g.esspri_total.notna()).sum())})
    rows.append({'pop_status':'Overall','pop_status_display':'Overall','n_patients':total,'pct_of_total_baseline':100.0 if total else np.nan,'pct_of_classifiable_baseline':100.0 if class_n else np.nan,'median_followup_yrs':base.followup.median(),'q1_followup_yrs':q(base.followup,.25),'q3_followup_yrs':q(base.followup,.75),'median_n_visits':base.n_visits.median(),'q1_n_visits':q(base.n_visits,.25),'q3_n_visits':q(base.n_visits,.75),'n_essdai_available_baseline':int(base.essdai_total.notna().sum()),'n_esspri_available_baseline':int(base.esspri_total.notna().sum()),'n_essdai_esspri_available_baseline':int((base.essdai_total.notna()&base.esspri_total.notna()).sum())})
    return pd.DataFrame(rows)

def distribution_by_visit(vis: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for vn,g in vis.groupby('visit_number'):
        denom=len(g)
        for pop in POP_ORDER:
            gp=g[g.pop_status.eq(pop)]
            rows.append({'visit_number':int(vn),'time_point_label':label_visit(vn),'median_time_since_baseline_yrs':g.time_since_baseline_years.median(),'q1_time_since_baseline_yrs':q(g.time_since_baseline_years,.25),'q3_time_since_baseline_yrs':q(g.time_since_baseline_years,.75),'pop_status':pop,'pop_status_display':DISPLAY[pop],'n_patients':len(gp),'n_patients_evaluable_at_visit':denom,'pct_patients_at_visit':len(gp)/denom*100 if denom else np.nan,'n_essdai_available':int(gp.essdai_total.notna().sum()),'n_esspri_available':int(gp.esspri_total.notna().sum()),'n_essdai_esspri_available':int((gp.essdai_total.notna()&gp.esspri_total.notna()).sum()),'n_unclassifiable':int((g.pop_status=='Unclassifiable').sum())})
    return pd.DataFrame(rows)

def unclass_reasons(vis: pd.DataFrame) -> pd.DataFrame:
    u=vis[vis.pop_status.eq('Unclassifiable')].copy(); u['unclassifiable_reason']=u.apply(reason, axis=1)
    rows=[]
    for (vn,rsn),g in u.groupby(['visit_number','unclassifiable_reason']):
        denom=int((u.visit_number==vn).sum()); rows.append({'visit_number':int(vn),'time_point_label':label_visit(vn),'unclassifiable_reason':rsn,'n_visits':len(g),'pct_unclassifiable_visits':len(g)/denom*100 if denom else np.nan})
    return pd.DataFrame(rows), u

def plot_one(vis: pd.DataFrame, pop: str, path: Path|None=None, pdf: PdfPages|None=None) -> int:
    g=vis[vis.baseline_pop_status.eq(pop)].copy(); n=g.patient_id.nunique()
    if n==0: fig,ax=plt.subplots(figsize=(8,3)); ax.text(.5,.5,f'No baseline {pop} patients',ha='center'); ax.axis('off')
    else:
        order=g.groupby('patient_id').agg(mx=('time_since_baseline_years','max'), nv=('visit_number','count')).reset_index().sort_values(['mx','nv','patient_id'], ascending=[False,False,True])
        mapper={pid:i for i,pid in enumerate(order.patient_id)}; g['y']=g.patient_id.map(mapper)
        height=max(4,min(60,1.8+n*.12)); size=max(8, min(28, 900/max(n,1)))
        fig,ax=plt.subplots(figsize=(11,height));
        for x in [0,.5,1,2,4,6,8,10]: ax.axvline(x,color='#dddddd',lw=.7,zorder=0)
        for st,c in POP_COLORS.items():
            gg=g[g.pop_status.eq(st)]; ax.scatter(gg.time_since_baseline_years, gg.y, c=c, s=size, label=DISPLAY[st], edgecolor='none')
        ax.set_yticks([]); ax.set_xlabel('Time since baseline (years)'); ax.set_ylabel('Patients (anonymized)')
        ax.set_title(f'Longitudinal classification for baseline {pop} patients (n={n})\nTime since first recorded visit; points colored by ESSDAI/ESSPRI-defined population')
        ax.set_xticks([0,.5,1,2,4,6,8,10], ['baseline','6 mo','1y','2y','4y','6y','8y','10y'], rotation=0)
        ax.legend(loc='upper right'); fig.text(.01,.01,'Pop1 = ESSDAI ≥5; Pop2 = ESSDAI <5 and ESSPRI ≥5; Pop3 = ESSDAI <5 and ESSPRI <5; grey = insufficient data.', fontsize=8)
        fig.tight_layout(rect=(0, .03, 1, 1))
    if path: fig.savefig(path)
    if pdf: pdf.savefig(fig)
    plt.close(fig); return 0

def write_json(obj, path): path.write_text(json.dumps(obj, indent=2, default=str))

def write_parquet_with_csv(df: pd.DataFrame, parquet_path: Path) -> None:
    """Write a parquet artifact and a human-readable CSV beside it."""
    df.to_parquet(parquet_path, index=False)
    df.to_csv(parquet_path.with_suffix('.csv'), index=False)


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--input', type=Path, default=INPUT_PATH); args=ap.parse_args()
    for d in [INTERMEDIATE_DIR,TABLES_DIR,FIGURES_DIR,QC_DIR]: d.mkdir(parents=True, exist_ok=True)
    df=pd.read_parquet(args.input) if args.input.suffix=='.parquet' else pd.read_csv(args.input, low_memory=False)
    vis, qc0, extra=build_visit_level(df)
    write_parquet_with_csv(vis, MASTER)
    base=vis[vis.visit_number.eq(0)].copy(); write_parquet_with_csv(base, INTERMEDIATE_DIR/f'{SCRIPT_OUTPUT_PREFIX}_baseline_classification.parquet')
    bdist=baseline_distribution(vis); bdist.to_csv(TABLES_DIR/f'{SCRIPT_OUTPUT_PREFIX}_distribution_baseline.csv',index=False)
    vdist=distribution_by_visit(vis); vdist.to_csv(TABLES_DIR/f'{SCRIPT_OUTPUT_PREFIX}_distribution_by_visit.csv',index=False); write_parquet_with_csv(vdist, INTERMEDIATE_DIR/f'{SCRIPT_OUTPUT_PREFIX}_distribution_by_visit.parquet')
    reasons,u=unclass_reasons(vis); reasons.to_csv(TABLES_DIR/f'{SCRIPT_OUTPUT_PREFIX}_unclassifiable_reason_counts_by_visit.csv',index=False); reasons[reasons.visit_number.eq(0)].to_csv(TABLES_DIR/f'{SCRIPT_OUTPUT_PREFIX}_unclassifiable_baseline_reason_counts.csv',index=False); write_parquet_with_csv(u, INTERMEDIATE_DIR/f'{SCRIPT_OUTPUT_PREFIX}_unclassifiable_reasons_visit_level.parquet')
    with PdfPages(FIGURES_DIR/f'{SCRIPT_OUTPUT_PREFIX}_trajectory_over_time.pdf') as pdf:
        for pop in POP_ORDER:
            plot_one(vis,pop,FIGURES_DIR/f'{SCRIPT_OUTPUT_PREFIX}_trajectory_over_time_baseline_{pop.lower()}.pdf',pdf)
    counts=base.pop_status.value_counts().reindex(POP_ORDER, fill_value=0); total=len(base)
    qc={**qc0,'input_path':str(args.input),'classification_logic_source':Path(__file__).name,'columns_used':{'patient_id':PATIENT_ID_COL,'visit_date':VISIT_DATE_COL,'essdai_total':ESSDAI_TOTAL_CANDIDATES,'esspri_components':ESSPRI_COMPONENTS},'missing_required_columns':[],'pop_labels_used':POP_ORDER,'pop1_rule':'ESSDAI >= 5 regardless of ESSPRI availability','pop2_rule':'ESSDAI < 5 and ESSPRI >= 5','pop3_rule':'ESSDAI < 5 and ESSPRI < 5','unclassifiable_rule':'Not classifiable by Pop1/Pop2/Pop3 rules','date_logic':{'row_date_original':'original ids__visit_date','row_date_min':'minimum parsed date from ids__visit_date','row_date_max':'maximum parsed date from ids__visit_date','visit_date_clean':'row_date_min','event_date':'row_date_min'},'n_baseline_patients':total,'baseline_pop_counts':counts.to_dict(),'baseline_pop_percentages_total':(counts/total*100).to_dict() if total else {},'baseline_sum_equals_total':bool(counts.sum()==total),'n_patients_with_baseline_unclassifiable':int(counts['Unclassifiable']),'unclassifiable_baseline_reason_counts':reasons[reasons.visit_number.eq(0)].set_index('unclassifiable_reason')['n_visits'].to_dict() if not reasons.empty else {},'traceability_intermediate_file':str(MASTER),'warnings':extra['warnings']}
    if not qc['baseline_sum_equals_total']: raise ValueError('Baseline Pop counts do not sum to total')
    write_json(qc, QC_DIR/f'{SCRIPT_OUTPUT_PREFIX}_distribution_qc.json')
    den=vis.groupby('visit_number').patient_id.count(); byvisit={'n_visits_max':int(vis.visit_number.max()),'denominators_by_visit_number':{str(k):int(v) for k,v in den.items()},'pop_counts_by_visit_number':{str(k):v.value_counts().reindex(POP_ORDER,fill_value=0).to_dict() for k,v in vis.groupby('visit_number').pop_status},'unclassifiable_counts_by_visit_number':{str(k):int((g.pop_status=='Unclassifiable').sum()) for k,g in vis.groupby('visit_number')},'essdai_missing_by_visit_number':{str(k):int(g.essdai_total.isna().sum()) for k,g in vis.groupby('visit_number')},'esspri_missing_by_visit_number':{str(k):int(g.esspri_total.isna().sum()) for k,g in vis.groupby('visit_number')},'late_followup_sparse_flags':{str(k):bool(v<10) for k,v in den.items()},'n_plot_points_outside_xlim':0,'warnings':[]}
    write_json(byvisit, QC_DIR/f'{SCRIPT_OUTPUT_PREFIX}_distribution_by_visit_qc.json')
    print(f'Wrote {MASTER} and Section 2 distribution outputs')
if __name__=='__main__': main()
