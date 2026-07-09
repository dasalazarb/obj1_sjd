#!/usr/bin/env python3
"""SECTION 2 — consecutive-visit Pop transition analyses."""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
import common  # noqa:E402
POP_ORDER=["Pop1","Pop2","Pop3","Unclassifiable"]
DISPLAY={"Unclassifiable":"Unclassified","Pop1":"Pop1","Pop2":"Pop2","Pop3":"Pop3"}
COLORS={"Pop1":"#d95f02","Pop2":"#7570b3","Pop3":"#1b9e77","Unclassifiable":"#9e9e9e"}
INTERMEDIATE_DIR=Path('/data/salazarda/data/obj1_sjd/data/intermediate')
MASTER=INTERMEDIATE_DIR/'02_pop_visit_level_classification.parquet'
INTERVALS=INTERMEDIATE_DIR/'10_pop_transition_intervals.parquet'
OUTPUTS_DIR=Path(getattr(common,'OUTPUTS_DIR',PROJECT_ROOT/'outputs'))
TABLES_DIR=Path(getattr(common,'BLOCKA_TABLES_DIR',OUTPUTS_DIR/'tables'/'blockA'))
FIGURES_DIR=Path(getattr(common,'BLOCKA_FIGURES_DIR',OUTPUTS_DIR/'figures'/'blockA'))
QC_DIR=OUTPUTS_DIR/'qc'/'blockA'

def write_json(obj,path): path.write_text(json.dumps(obj, indent=2, default=str))
def pct(n,d): return n/d*100 if d else np.nan

def load_classification(input_path: Path|None) -> tuple[pd.DataFrame,bool]:
    if MASTER.exists(): return pd.read_parquet(MASTER), False
    # fallback: import and run exact builder from 02 script
    spec=importlib.util.spec_from_file_location('popdist', Path(__file__).with_name('02_pop_distribution.py'))
    mod=importlib.util.module_from_spec(spec); assert spec.loader; spec.loader.exec_module(mod)
    path=input_path or mod.INPUT_PATH
    df=pd.read_parquet(path) if Path(path).suffix=='.parquet' else pd.read_csv(path, low_memory=False)
    vis,_,_=mod.build_visit_level(df)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True); vis.to_parquet(MASTER,index=False)
    return vis, True

def build_intervals(vis: pd.DataFrame) -> tuple[pd.DataFrame,int]:
    rows=[]
    for pid,g in vis.sort_values(['patient_id','visit_date_clean','visit_number']).groupby('patient_id'):
        if len(g)<2: continue
        rec=g.to_dict('records')
        for a,b in zip(rec[:-1], rec[1:]):
            days=(pd.Timestamp(b['visit_date_clean'])-pd.Timestamp(a['visit_date_clean'])).days
            rows.append({'patient_id':pid,'from_visit_number':a['visit_number'],'to_visit_number':b['visit_number'],'from_date':a['visit_date_clean'],'to_date':b['visit_date_clean'],'from_time_since_baseline_years':a['time_since_baseline_years'],'to_time_since_baseline_years':b['time_since_baseline_years'],'from_pop':a['pop_status'],'to_pop':b['pop_status'],'from_pop_display':a.get('pop_status_display',DISPLAY.get(a['pop_status'],a['pop_status'])),'to_pop_display':b.get('pop_status_display',DISPLAY.get(b['pop_status'],b['pop_status'])),'interval_days':days,'interval_years':days/365.25,'changed_state':a['pop_status']!=b['pop_status'],'transition_pair':f"{a['pop_status']} -> {b['pop_status']}"})
    out=pd.DataFrame(rows)
    nonpos=int((out.interval_days<=0).sum()) if not out.empty else 0
    if not out.empty: out=out[out.interval_days>0].copy()
    return out, nonpos

def transition_matrix(intervals: pd.DataFrame) -> pd.DataFrame:
    idx=pd.MultiIndex.from_product([POP_ORDER,POP_ORDER], names=['from_pop','to_pop'])
    counts=intervals.groupby(['from_pop','to_pop']).size().reindex(idx, fill_value=0).rename('n_intervals').reset_index()
    stats=intervals.groupby(['from_pop','to_pop']).interval_years.agg(median_interval_years='median', q1_interval_years=lambda s:s.quantile(.25), q3_interval_years=lambda s:s.quantile(.75)).reindex(idx).reset_index()
    m=counts.merge(stats,on=['from_pop','to_pop']); total=int(m.n_intervals.sum()); rowtot=m.groupby('from_pop').n_intervals.transform('sum')
    m['row_total_intervals']=rowtot; m['row_pct']=np.where(rowtot>0, m.n_intervals/rowtot*100, np.nan); m['overall_pct']=m.n_intervals/total*100 if total else np.nan
    m['from_pop_display']=m.from_pop.map(DISPLAY); m['to_pop_display']=m.to_pop.map(DISPLAY)
    return m[['from_pop','to_pop','from_pop_display','to_pop_display','n_intervals','row_total_intervals','row_pct','overall_pct','median_interval_years','q1_interval_years','q3_interval_years']]

def poisson_ci(n:int, pt:float):
    if pt<=0 or pd.isna(pt): return (np.nan,np.nan,np.nan)
    rate=n/pt
    if n==0: return (0.0,0.0,3.69/pt)
    try:
        from scipy.stats import chi2
        return rate, 0.5*chi2.ppf(.025,2*n)/pt, 0.5*chi2.ppf(.975,2*(n+1))/pt
    except Exception:
        se=np.sqrt(n)/pt; return rate, max(0, rate-1.96*se), rate+1.96*se

def rates(intervals: pd.DataFrame) -> pd.DataFrame:
    pt=intervals.groupby('from_pop').interval_years.sum().reindex(POP_ORDER, fill_value=0.0)
    rows=[]
    for f in POP_ORDER:
        for t in POP_ORDER:
            if f==t: continue
            n=int(((intervals.from_pop==f)&(intervals.to_pop==t)).sum()); r,lo,hi=poisson_ci(n,float(pt[f])); sparse=bool(n<5 or pt[f]<5)
            rows.append({'from_pop':f,'to_pop':t,'from_pop_display':DISPLAY[f],'to_pop_display':DISPLAY[t],'n_transitions':n,'person_time_from_state_yrs':float(pt[f]),'rate_per_person_year':r,'ci95_low':lo,'ci95_high':hi,'sparse_flag':sparse,'interpretation_note':'Sparse transition; interpret descriptively.' if sparse else 'Descriptive transition intensity; interpret exploratorily.'})
    return pd.DataFrame(rows)

def plot_heatmap(m,path):
    piv=m.pivot(index='from_pop',columns='to_pop',values='row_pct').reindex(index=POP_ORDER,columns=POP_ORDER)
    counts=m.pivot(index='from_pop',columns='to_pop',values='n_intervals').reindex(index=POP_ORDER,columns=POP_ORDER)
    fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(piv.fillna(0), cmap='Blues', vmin=0, vmax=np.nanmax(piv.values) if np.isfinite(piv.values).any() else 1)
    ax.set_xticks(range(4), [DISPLAY[x] for x in POP_ORDER]); ax.set_yticks(range(4), [DISPLAY[x] for x in POP_ORDER]); ax.set_xlabel('To population'); ax.set_ylabel('From population'); ax.set_title('Consecutive-visit transition matrix')
    for i in range(4):
        for j in range(4): ax.text(j,i,f"{int(counts.iloc[i,j])}\n{piv.iloc[i,j]:.1f}%" if pd.notna(piv.iloc[i,j]) else f"{int(counts.iloc[i,j])}\nNA",ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=ax,label='Row %'); fig.text(.01,.01,'Rows sum to 100% within each starting population. Transitions involving Unclassifiable may reflect missing ESSDAI/ESSPRI data rather than true clinical change.',fontsize=8); fig.tight_layout(rect=(0,.05,1,1)); fig.savefig(path); plt.close(fig)

def plot_sankey(intervals,path):
    counts=intervals.groupby(['from_pop','to_pop']).size().reset_index(name='n')
    try:
        import plotly.graph_objects as go
        nodes=[f'From {DISPLAY[x]}' for x in POP_ORDER]+[f'To {DISPLAY[x]}' for x in POP_ORDER]
        fig=go.Figure(go.Sankey(node={'label':nodes,'color':[COLORS[x] for x in POP_ORDER]*2}, link={'source':[POP_ORDER.index(x) for x in counts.from_pop], 'target':[4+POP_ORDER.index(x) for x in counts.to_pop], 'value':counts.n}))
        fig.update_layout(title_text='Aggregated consecutive-visit transitions<br><sup>Transitions involving Unclassifiable may reflect missing data rather than true clinical change.</sup>')
        fig.write_image(str(path))
    except Exception:
        fig,ax=plt.subplots(figsize=(8,5)); y_from=np.linspace(.85,.15,4); y_to=np.linspace(.85,.15,4)
        maxn=counts.n.max() if len(counts) else 1
        for _,r in counts.iterrows(): ax.plot([0,1],[y_from[POP_ORDER.index(r.from_pop)],y_to[POP_ORDER.index(r.to_pop)]], lw=0.5+5*r.n/maxn, color=COLORS[r.from_pop], alpha=.45)
        for i,p in enumerate(POP_ORDER): ax.text(-.03,y_from[i],DISPLAY[p],ha='right',va='center'); ax.text(1.03,y_to[i],DISPLAY[p],ha='left',va='center')
        ax.text(0,.95,'From previous visit',ha='center'); ax.text(1,.95,'To next visit',ha='center'); ax.set_axis_off(); ax.set_title('Aggregated consecutive-visit transitions'); fig.text(.01,.01,'Transitions involving Unclassifiable may reflect missing data rather than true clinical change.',fontsize=8); fig.savefig(path); plt.close(fig)

def plot_diagram(rates_df,path):
    pos={'Pop1':(0,.7),'Pop2':(.8,.7),'Pop3':(.8,.1),'Unclassifiable':(0,.1)}; fig,ax=plt.subplots(figsize=(7,5))
    show=rates_df[rates_df.n_transitions>=3].copy(); maxn=show.n_transitions.max() if len(show) else 1
    for _,r in show.iterrows():
        ax.annotate('', xy=pos[r.to_pop], xytext=pos[r.from_pop], arrowprops=dict(arrowstyle='->', lw=0.5+4*r.n_transitions/maxn, color='0.35', alpha=.65, connectionstyle='arc3,rad=0.15'))
    for p,(x,y) in pos.items(): ax.scatter([x],[y],s=1800,c=COLORS[p]); ax.text(x,y,DISPLAY[p],ha='center',va='center',color='white',weight='bold')
    ax.set_axis_off(); ax.set_title('Descriptive multi-state transition diagram'); fig.text(.01,.01,'Rates are descriptive transition intensities estimated from consecutive observed visit intervals. Transitions involving Unclassifiable may reflect missing data.',fontsize=8); fig.savefig(path); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input', type=Path, default=None); args=ap.parse_args()
    for d in [INTERMEDIATE_DIR,TABLES_DIR,FIGURES_DIR,QC_DIR]: d.mkdir(parents=True, exist_ok=True)
    vis,fallback=load_classification(args.input); intervals,nonpos=build_intervals(vis); intervals.to_parquet(INTERVALS,index=False)
    m=transition_matrix(intervals); m.to_csv(TABLES_DIR/'10_transition_matrix_consecutive_visits.csv',index=False)
    r=rates(intervals); r.to_csv(TABLES_DIR/'10_multistate_transition_rates.csv',index=False)
    plot_heatmap(m, FIGURES_DIR/'10_transition_heatmap.pdf'); plot_sankey(intervals, FIGURES_DIR/'10_transition_sankey.pdf'); plot_diagram(r, FIGURES_DIR/'10_multistate_transition_diagram.pdf')
    total_int=len(intervals); stable=int((~intervals.changed_state).sum()) if total_int else 0; changed=int(intervals.changed_state.sum()) if total_int else 0
    pats_ge2=int(vis.groupby('patient_id').size().ge(2).sum()); anychg=intervals.groupby('patient_id').changed_state.any() if total_int else pd.Series(dtype=bool)
    involved=int(((intervals.from_pop=='Unclassifiable')|(intervals.to_pop=='Unclassifiable')).sum()) if total_int else 0
    row_sums=m.groupby('from_pop').row_pct.sum(min_count=1).reindex(POP_ORDER); row_tot=m.groupby('from_pop').n_intervals.sum().reindex(POP_ORDER,fill_value=0)
    common_pair=intervals.transition_pair.value_counts().head(1)
    mc=common_pair.index[0] if len(common_pair) else None; mcn=int(common_pair.iloc[0]) if len(common_pair) else 0; mcr=np.nan
    if mc: mcr=float(m.loc[(m.from_pop==mc.split(' -> ')[0])&(m.to_pop==mc.split(' -> ')[1]),'row_pct'].iloc[0])
    tqc={'classification_file_used':str(MASTER),'classification_recomputed_fallback':fallback,'n_patients_total_in_classification_file':int(vis.patient_id.nunique()),'n_patients_with_ge2_visits':pats_ge2,'n_transition_intervals':total_int,'state_order':POP_ORDER,'row_totals':{k:int(v) for k,v in row_tot.items()},'row_percent_sums':{k:(None if pd.isna(v) else float(v)) for k,v in row_sums.items()},'row_percent_sums_close_to_100':bool(all((row_tot[p]==0) or np.isclose(row_sums[p],100) for p in POP_ORDER)),'n_intervals_with_nonpositive_time':nonpos,'n_intervals_excluded_nonpositive_time':nonpos,'n_transitions_involving_unclassifiable':involved,'pct_transitions_involving_unclassifiable':pct(involved,total_int),'n_stable_intervals':stable,'pct_stable_intervals':pct(stable,total_int),'n_changed_intervals':changed,'pct_changed_intervals':pct(changed,total_int),'n_patients_with_any_transition':int(anychg.sum()) if len(anychg) else 0,'pct_patients_with_any_transition':pct(int(anychg.sum()) if len(anychg) else 0,pats_ge2),'median_time_between_all_consecutive_visits_yrs':float(intervals.interval_years.median()) if total_int else np.nan,'median_time_between_changed_transitions_yrs':float(intervals.loc[intervals.changed_state,'interval_years'].median()) if changed else np.nan,'most_common_transition':mc,'most_common_transition_n':mcn,'most_common_transition_row_pct':mcr,'warnings':[]}
    if len(m)!=16: raise ValueError('Transition matrix does not contain 16 combinations')
    write_json(tqc, QC_DIR/'10_transition_matrix_qc.json')
    pt=intervals.groupby('from_pop').interval_years.sum().reindex(POP_ORDER, fill_value=0.0)
    mqc={'model_type':'descriptive transition intensity per person-year from consecutive observed intervals','exploratory_flag':True,'person_time_by_state':{k:float(v) for k,v in pt.items()},'n_transitions_by_pair':{f"{x.from_pop} -> {x.to_pop}":int(x.n_transitions) for x in r.itertuples()},'sparse_transition_pairs':[f"{x.from_pop} -> {x.to_pop}" for x in r.itertuples() if x.sparse_flag],'states_with_zero_person_time':[k for k,v in pt.items() if v==0],'transitions_involving_unclassifiable_note':'May reflect missing ESSDAI/ESSPRI data rather than true clinical change.','warnings':[]}
    write_json(mqc, QC_DIR/'10_multistate_transition_qc.json')
    print(f'Wrote {INTERVALS} and transition outputs')
if __name__=='__main__': main()
