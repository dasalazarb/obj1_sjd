#!/usr/bin/env python3
"""Pop transitions between consecutive canonical clinical episodes."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.stats import norm

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
import common  # noqa:E402
POP_ORDER=["Pop1","Pop2","Pop3","Unclassifiable"]
DISPLAY={"Unclassifiable":"Unclassified","Pop1":"Pop1","Pop2":"Pop2","Pop3":"Pop3"}
COLORS={"Pop1":"#d95f02","Pop2":"#7570b3","Pop3":"#1b9e77","Unclassifiable":"#9e9e9e"}
INTERMEDIATE_DIR=Path(getattr(common,'INTERMEDIATE_DATA_DIR',PROJECT_ROOT/'data'/'intermediate'))/'10_pop_transitions'
MASTER=common.POP_LONGITUDINAL_PARQUET
INTERVALS=common.POP_TRANSITION_INTERVALS_PARQUET
OUTPUTS_DIR=Path(getattr(common,'OUTPUTS_DIR',PROJECT_ROOT/'outputs'))
TABLES_DIR=Path(getattr(common,'BLOCKA_TABLES_DIR',OUTPUTS_DIR/'tables'/'blockA'))/'10_pop_transitions'
FIGURES_DIR=Path(getattr(common,'BLOCKA_FIGURES_DIR',OUTPUTS_DIR/'figures'/'blockA'))/'10_pop_transitions'
QC_DIR=OUTPUTS_DIR/'qc'/'blockA'/'10_pop_transitions'
MODEL_STATES=["Pop1","Pop2","Pop3"]
SPARSE_THRESHOLD=5
REQUIRED_COLUMNS={
    'patient_id', 'clinical_episode_id', 'clinical_anchor_date',
    'clinical_visit_number', 'pop_status',
}
ORDER_COLUMNS=[
    'patient_id', 'clinical_anchor_date', 'clinical_visit_number',
    'clinical_episode_id',
]

def write_json(obj,path): path.write_text(json.dumps(obj, indent=2, default=str))
def pct(n,d): return n/d*100 if d else np.nan

def load_classification(input_path: Path|None) -> pd.DataFrame:
    path=input_path or common.POP_LONGITUDINAL_PARQUET
    if not path.exists():
        raise FileNotFoundError(f'Required upstream Pop dataset not found: {path}')
    return pd.read_parquet(path)

def validate_input(vis: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the canonical episode-level Pop contract."""
    missing=REQUIRED_COLUMNS-set(vis.columns)
    if missing: raise ValueError(f'Missing required Pop episode columns: {sorted(missing)}')
    out=vis.copy()
    duplicates=out.duplicated(['patient_id','clinical_episode_id'], keep=False)
    if duplicates.any():
        raise ValueError(f'Duplicate patient_id + clinical_episode_id keys: {int(duplicates.sum())} rows')
    out['clinical_anchor_date']=pd.to_datetime(out.clinical_anchor_date, errors='coerce')
    if out.clinical_anchor_date.isna().any(): raise ValueError('clinical_anchor_date contains missing or invalid values')
    visit_number=pd.to_numeric(out.clinical_visit_number, errors='coerce')
    if visit_number.isna().any() or (visit_number<1).any(): raise ValueError('clinical_visit_number must be >= 1')
    out['clinical_visit_number']=visit_number
    if out.pop_status.isna().any(): raise ValueError('pop_status must be present for every clinical episode')
    unknown=set(out.pop_status.unique())-set(POP_ORDER)
    if unknown: raise ValueError(f'Unexpected pop_status values: {sorted(unknown)}')
    return out.sort_values(ORDER_COLUMNS).reset_index(drop=True)

def build_intervals(vis: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    vis=validate_input(vis)
    rows=[]
    for pid,g in vis.groupby('patient_id', sort=False):
        if len(g)<2: continue
        rec=g.to_dict('records')
        for a,b in zip(rec[:-1], rec[1:]):
            days=(b['clinical_anchor_date']-a['clinical_anchor_date']).days
            rows.append({'patient_id':pid,
                'from_clinical_episode_id':a['clinical_episode_id'],'to_clinical_episode_id':b['clinical_episode_id'],
                'from_clinical_visit_number':a['clinical_visit_number'],'to_clinical_visit_number':b['clinical_visit_number'],
                'from_clinical_anchor_date':a['clinical_anchor_date'],'to_clinical_anchor_date':b['clinical_anchor_date'],
                'from_pop':a['pop_status'],'to_pop':b['pop_status'],
                'from_pop_display':a.get('pop_status_display',DISPLAY.get(a['pop_status'],a['pop_status'])),
                'to_pop_display':b.get('pop_status_display',DISPLAY.get(b['pop_status'],b['pop_status'])),
                'interval_days':days,'interval_years':days/365.25,
                'changed_state':a['pop_status']!=b['pop_status'],'transition_pair':f"{a['pop_status']} -> {b['pop_status']}"})
    columns=['patient_id','from_clinical_episode_id','to_clinical_episode_id','from_clinical_visit_number','to_clinical_visit_number','from_clinical_anchor_date','to_clinical_anchor_date','from_pop','to_pop','from_pop_display','to_pop_display','interval_days','interval_years','changed_state','transition_pair']
    out=pd.DataFrame(rows, columns=columns)
    expected=int(vis.groupby('patient_id').size().sub(1).clip(lower=0).sum())
    if len(out)!=expected: raise AssertionError(f'Created {len(out)} adjacent intervals; expected {expected}')
    nonpos=int((out.interval_days<=0).sum()) if not out.empty else 0
    if not out.empty: out=out[out.interval_days>0].copy()
    qc={'n_input_episodes':int(len(vis)),'n_input_patients':int(vis.patient_id.nunique()),
        'n_patients_with_ge2_clinical_episodes':int(vis.groupby('patient_id').size().ge(2).sum()),
        'n_adjacent_intervals_expected':expected,
        'n_adjacent_intervals_created_before_time_filter':expected,
        'n_intervals_nonpositive_time':nonpos,'n_intervals_retained':int(len(out))}
    return out.reset_index(drop=True),qc

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
    ax.set_xticks(range(4), [DISPLAY[x] for x in POP_ORDER]); ax.set_yticks(range(4), [DISPLAY[x] for x in POP_ORDER]); ax.set_xlabel('To population'); ax.set_ylabel('From population'); ax.set_title('Consecutive clinical-episode transition matrix')
    for i in range(4):
        for j in range(4): ax.text(j,i,f"{int(counts.iloc[i,j])}\n{piv.iloc[i,j]:.1f}%" if pd.notna(piv.iloc[i,j]) else f"{int(counts.iloc[i,j])}\nNA",ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=ax,label='Row %'); fig.text(.01,.01,'Rows sum to 100% within each starting population. Transitions involving Unclassifiable may reflect missing ESSDAI/ESSPRI data rather than true clinical change.',fontsize=8); fig.tight_layout(rect=(0,.05,1,1)); fig.savefig(path); plt.close(fig)

def plot_sankey(intervals,path):
    counts=intervals.groupby(['from_pop','to_pop']).size().reset_index(name='n')
    try:
        import plotly.graph_objects as go
        nodes=[f'From {DISPLAY[x]}' for x in POP_ORDER]+[f'To {DISPLAY[x]}' for x in POP_ORDER]
        labels=[f'{a} → {b}: n={n}' for a,b,n in counts[['from_pop','to_pop','n']].itertuples(index=False,name=None)]
        fig=go.Figure(go.Sankey(node={'label':nodes,'color':[COLORS[x] for x in POP_ORDER]*2}, link={'source':[POP_ORDER.index(x) for x in counts.from_pop], 'target':[4+POP_ORDER.index(x) for x in counts.to_pop], 'value':counts.n,'customdata':labels,'hovertemplate':'%{customdata}<extra></extra>'}))
        fig.update_layout(title_text='Observed consecutive clinical-episode transitions<br><sup>Link width represents interval count n; hover shows the exact count. Unclassifiable may reflect missing data.</sup>')
        fig.write_image(str(path))
    except Exception:
        fig,ax=plt.subplots(figsize=(10,6)); y_from=np.linspace(.85,.15,4); y_to=np.linspace(.85,.15,4)
        maxn=counts.n.max() if len(counts) else 1
        for _,r in counts.iterrows():
            y0=y_from[POP_ORDER.index(r.from_pop)]; y1=y_to[POP_ORDER.index(r.to_pop)]
            width=1.0+8*np.sqrt(r.n/maxn)
            ax.plot([0,1],[y0,y1],lw=width,color=COLORS[r.from_pop],alpha=.55,solid_capstyle='round')
            ax.text(.5,(y0+y1)/2,f'n={int(r.n)}',ha='center',va='center',fontsize=8,
                    bbox={'boxstyle':'round,pad=.2','fc':'white','ec':COLORS[r.from_pop],'alpha':.9})
        for i,p in enumerate(POP_ORDER): ax.text(-.03,y_from[i],DISPLAY[p],ha='right',va='center'); ax.text(1.03,y_to[i],DISPLAY[p],ha='left',va='center')
        ax.text(0,.95,'From previous clinical episode',ha='center',weight='bold'); ax.text(1,.95,'To next clinical episode',ha='center',weight='bold'); ax.set_axis_off(); ax.set_title('Observed consecutive clinical-episode transitions\nLine width scales with $\\sqrt{n}$; labels show interval counts',weight='bold'); fig.text(.01,.01,'Transitions involving Unclassifiable may reflect missing data rather than true clinical change.',fontsize=8); fig.tight_layout(rect=(0,.04,1,1)); fig.savefig(path,bbox_inches='tight'); plt.close(fig)

def plot_diagram(rates_df,path):
    pos={'Pop1':(0,.72),'Pop2':(.85,.72),'Pop3':(.85,.08),'Unclassifiable':(0,.08)}; fig,ax=plt.subplots(figsize=(9,7))
    show=rates_df[rates_df.n_transitions>=3].copy(); maxn=show.n_transitions.max() if len(show) else 1
    for _,r in show.iterrows():
        rad=.18 if POP_ORDER.index(r.from_pop)<POP_ORDER.index(r.to_pop) else -.18
        width=1.2+6*np.sqrt(r.n_transitions/maxn)
        ax.annotate('',xy=pos[r.to_pop],xytext=pos[r.from_pop],arrowprops=dict(arrowstyle='-|>',mutation_scale=14,lw=width,color=COLORS[r.from_pop],alpha=.65,connectionstyle=f'arc3,rad={rad}',shrinkA=34,shrinkB=34))
        x=(pos[r.from_pop][0]+pos[r.to_pop][0])/2; y=(pos[r.from_pop][1]+pos[r.to_pop][1])/2+rad*.24
        ax.text(x,y,f'{r.from_pop}→{r.to_pop}\nn={int(r.n_transitions)}; rate={r.rate_per_person_year:.3f}/y',ha='center',va='center',fontsize=7.5,
                bbox={'boxstyle':'round,pad=.25','fc':'white','ec':COLORS[r.from_pop],'alpha':.94})
    for p,(x,y) in pos.items(): ax.scatter([x],[y],s=2200,c=COLORS[p],edgecolors='white',linewidths=2,zorder=5); ax.text(x,y,DISPLAY[p],ha='center',va='center',color='white',weight='bold',zorder=6)
    ax.set(xlim=(-.18,1.03),ylim=(-.1,.9)); ax.set_axis_off(); ax.set_title('Observed consecutive clinical-episode transitions\nArrow width scales with $\\sqrt{n}$',weight='bold'); fig.text(.01,.01,'Descriptive rates per person-year, not model-estimated probabilities. Unclassifiable may reflect missing data. Arrows with n<3 are omitted for readability.',fontsize=8); fig.tight_layout(rect=(0,.04,1,1)); fig.savefig(path,bbox_inches='tight'); plt.close(fig)

def prepare_multistate_data(vis: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    """Create model intervals after canonical adjacent episodes are paired.

    Pairing is performed *before* removing Unclassifiable episodes. Consequently,
    Pop1 -> Unclassifiable -> Pop2 contributes no Pop1 -> Pop2 interval.
    """
    ordered=validate_input(vis)
    adjacent,adjacent_qc=build_intervals(ordered)
    excluded=(~adjacent.from_pop.isin(MODEL_STATES)|~adjacent.to_pop.isin(MODEL_STATES))
    out=adjacent.loc[~excluded, ['patient_id','from_clinical_episode_id','to_clinical_episode_id',
        'from_pop','to_pop','interval_years','from_clinical_visit_number','to_clinical_visit_number']].copy()
    counts={(f'{a} -> {b}'):int(((out.from_pop==a)&(out.to_pop==b)).sum()) for a in MODEL_STATES for b in MODEL_STATES if a!=b}
    state_counts=ordered.loc[ordered.pop_status.isin(MODEL_STATES),'pop_status'].value_counts().reindex(MODEL_STATES,fill_value=0)
    used_episodes=(set(zip(out.patient_id,out.from_clinical_episode_id))|set(zip(out.patient_id,out.to_clinical_episode_id))) if len(out) else set()
    info={'n_patients_total':int(ordered.patient_id.nunique()),
          'n_patients_with_ge2_clinical_episodes':int(ordered.groupby('patient_id').size().ge(2).sum()),
          'n_patients_used':int(out.patient_id.nunique()),'n_observations_used':len(used_episodes),
          'n_model_intervals':int(len(out)),
          'n_intervals_excluded_nonpositive_time':adjacent_qc['n_intervals_nonpositive_time'],
          'n_adjacent_intervals_excluded_unclassifiable':int(excluded.sum()),
          'observations_by_state':{k:int(v) for k,v in state_counts.items()},'transitions_by_pair':counts,
          'median_time_between_clinical_episodes_years':float(out.interval_years.median()) if len(out) else None}
    return out,info

def build_q_matrix(theta: np.ndarray, n_states: int=3) -> np.ndarray:
    q=np.zeros((n_states,n_states)); k=0
    for i in range(n_states):
        for j in range(n_states):
            if i!=j: q[i,j]=np.exp(np.clip(theta[k],-30,30)); k+=1
        q[i,i]=-q[i].sum()
    return q

def multistate_loglik(theta: np.ndarray, trajectories: list[tuple[np.ndarray,np.ndarray]]) -> float:
    """Negative panel likelihood, summed as complete patient trajectories."""
    q=build_q_matrix(theta); total=0.0
    for times,pairs in trajectories:
        for dt,(r,s) in zip(times,pairs):
            probability=expm(q*dt)[r,s]
            if not np.isfinite(probability) or probability<=0: return 1e100
            total+=np.log(max(probability,1e-300))
    return -total

def _trajectories(intervals: pd.DataFrame):
    index={s:i for i,s in enumerate(MODEL_STATES)}
    return [(g.interval_years.to_numpy(float),np.array([(index[a],index[b]) for a,b in zip(g.from_pop,g.to_pop)],int))
            for _,g in intervals.groupby('patient_id',sort=False)]

def fit_multistate_model(intervals: pd.DataFrame) -> dict:
    if intervals.empty: raise ValueError('No usable intervals for the continuous-time multi-state model')
    exposure=intervals.groupby('from_pop').interval_years.sum().reindex(MODEL_STATES,fill_value=0.0)
    theta=[]
    for a in MODEL_STATES:
        for b in MODEL_STATES:
            if a!=b:
                n=((intervals.from_pop==a)&(intervals.to_pop==b)).sum()
                theta.append(np.log(max(n,0.25)/max(float(exposure[a]),0.25)))
    trajectories=_trajectories(intervals)
    result=minimize(multistate_loglik,np.array(theta),args=(trajectories,),method='L-BFGS-B',bounds=[(-20,10)]*6,
                    options={'maxiter':2000,'ftol':1e-12,'gtol':1e-7})
    q=build_q_matrix(result.x); warnings=[]; ci_method='inverse Hessian (delta method)'
    try:
        covariance=np.asarray(result.hess_inv.todense())
        if covariance.shape!=(6,6) or not np.isfinite(covariance).all() or np.any(np.diag(covariance)<=0) or np.linalg.cond(covariance)>1e12:
            raise ValueError('unstable inverse Hessian')
        se=np.sqrt(np.diag(covariance)); lo=np.exp(np.clip(result.x-norm.ppf(.975)*se,-30,30)); hi=np.exp(np.clip(result.x+norm.ppf(.975)*se,-30,30))
    except Exception as exc:
        # A cluster bootstrap is deliberately patient-level: every sampled unit
        # contains all of that patient's adjacent intervals.
        ci_method='patient-level bootstrap'; warnings.append(f'Hessian unreliable ({exc}); patient bootstrap used.')
        rng=np.random.default_rng(20260818); patients=list(intervals.patient_id.unique()); estimates=[]
        for _ in range(200):
            sampled=rng.choice(patients,len(patients),replace=True); pieces=[]
            for sample_id,pid in enumerate(sampled):
                piece=intervals[intervals.patient_id==pid].copy(); piece['patient_id']=sample_id; pieces.append(piece)
            boot=pd.concat(pieces,ignore_index=True); br=minimize(multistate_loglik,result.x,args=(_trajectories(boot),),method='L-BFGS-B',bounds=[(-20,10)]*6)
            if br.success and np.isfinite(br.x).all(): estimates.append(np.exp(br.x))
        if len(estimates)<100: warnings.append(f'Only {len(estimates)}/200 bootstrap fits converged; confidence intervals are unreliable.')
        arr=np.asarray(estimates); lo=np.quantile(arr,.025,axis=0) if len(arr) else np.full(6,np.nan); hi=np.quantile(arr,.975,axis=0) if len(arr) else np.full(6,np.nan)
    if not result.success: warnings.append(f'Optimizer did not converge: {result.message}. Estimates must not be treated as valid.')
    if np.any(np.abs(result.x)>15): warnings.append('One or more log-intensities are extreme (absolute value >15).')
    return {'result':result,'Q':q,'ci_low':lo,'ci_high':hi,'ci_method':ci_method,'warnings':warnings}

def transition_probability_matrix(q: np.ndarray,t: float) -> pd.DataFrame:
    return pd.DataFrame(expm(q*t),index=MODEL_STATES,columns=MODEL_STATES).rename_axis('From')

def multistate_sojourn_times(q: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({'pop':MODEL_STATES,'mean_sojourn_time_years':[-1/q[i,i] for i in range(3)]})

def plot_multistate_model(
    q: np.ndarray,
    path: Path,
    observed_counts: dict[str, int] | None = None,
) -> None:
    """Plot directed intensities with unambiguous arrows, widths and labels.

    ``path`` deliberately remains the second argument to preserve the public
    two-argument calling convention used by earlier pipeline versions.
    Observed counts are optional visual context and must therefore be supplied
    by keyword; they are not part of the fitted continuous-time model.
    """
    observed_counts = observed_counts or {}
    pos={'Pop1':(0.5,.84),'Pop2':(.12,.18),'Pop3':(.88,.18)}
    off_diagonal=np.array([q[i,j] for i in range(3) for j in range(3) if i!=j]); maximum=float(off_diagonal.max())
    fig,ax=plt.subplots(figsize=(8.5,7))
    for i,a in enumerate(MODEL_STATES):
        for j,b in enumerate(MODEL_STATES):
            if i==j: continue
            rad=.22 if i<j else -.22
            relative=np.sqrt(q[i,j]/maximum) if maximum>0 else 0
            width=1.2+7*relative
            ax.annotate('',xy=pos[b],xytext=pos[a],zorder=2,
                        arrowprops={'arrowstyle':'-|>','mutation_scale':18,'lw':width,'color':COLORS[a],
                                    'alpha':.72,'connectionstyle':f'arc3,rad={rad}','shrinkA':45,'shrinkB':45})
            x=(pos[a][0]+pos[b][0])/2; y=(pos[a][1]+pos[b][1])/2+rad*.30
            n=observed_counts.get(f'{a} -> {b}',0)
            ax.text(x,y,f'{a} → {b}\nq = {q[i,j]:.3f}/y  |  n = {n}',ha='center',va='center',fontsize=9,weight='semibold',zorder=7,
                    bbox={'boxstyle':'round,pad=.3','fc':'white','ec':COLORS[a],'lw':1.1,'alpha':.96})
    for s,(x,y) in pos.items():
        ax.scatter(x,y,s=3000,color=COLORS[s],edgecolors='white',linewidths=2.5,zorder=8)
        ax.text(x,y,s,color='white',weight='bold',fontsize=14,ha='center',va='center',zorder=9)
    # Reference widths make the visual encoding interpretable without comparing
    # nearly overlapping reciprocal arrows by eye.
    refs=[maximum*.25,maximum*.5,maximum] if maximum>0 else [0,0,0]
    handles=[plt.Line2D([0],[0],color='#555',lw=1.2+7*np.sqrt(v/maximum),label=f'q = {v:.3f}/y') for v in refs] if maximum>0 else []
    if handles: ax.legend(handles=handles,title='Arrow-width scale',loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.02))
    ax.set(xlim=(-.08,1.08),ylim=(-.02,1.02)); ax.axis('off')
    ax.set_title('Continuous-time multi-state model\nDirected intensity q (per year) and observed transition count n',weight='bold',pad=12)
    fig.text(.5,.025,'Arrow color identifies the origin state. Width uses a square-root scale to improve contrast.',ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.06,1,1)); fig.savefig(path,bbox_inches='tight'); plt.close(fig)

def plot_probability_heatmap(p: pd.DataFrame,t: float,path: Path):
    fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(p.values,cmap='Blues',vmin=0,vmax=1)
    ax.set_xticks(range(3),MODEL_STATES); ax.set_yticks(range(3),MODEL_STATES); ax.set_xlabel('To state'); ax.set_ylabel('From state'); ax.set_title(f'Estimated transition probabilities at {t:g} years')
    for i in range(3):
        for j in range(3): ax.text(j,i,f'{p.iloc[i,j]:.3f}\n({p.iloc[i,j]:.1%})',ha='center',va='center')
    fig.colorbar(im,ax=ax,label='Probability'); fig.tight_layout(); fig.savefig(path); plt.close(fig)

def multistate_qc(prep: dict,fit: dict,intervals: pd.DataFrame) -> dict:
    q=fit['Q']; p1=expm(q); res=fit['result']; warnings=list(fit['warnings'])
    checks={'intensities_finite':bool(np.isfinite(q).all()),'off_diagonal_nonnegative':bool(np.all(q[~np.eye(3,dtype=bool)]>=0)),
            'diagonal_negative':bool(np.all(np.diag(q)<0)),'q_rows_sum_to_zero':bool(np.allclose(q.sum(1),0,atol=1e-8)),
            'p_1y_rows_sum_to_one':bool(np.allclose(p1.sum(1),1,atol=1e-8)),'p_1y_finite':bool(np.isfinite(p1).all())}
    sparse=[pair for pair,n in prep['transitions_by_pair'].items() if n<SPARSE_THRESHOLD]
    if sparse: warnings.append(f'Sparse observed transitions (<{SPARSE_THRESHOLD}): {", ".join(sparse)}.')
    return {**prep,'model_type':'Continuous-time homogeneous Markov multi-state panel model (state ~ time; no covariates)',
            'unclassifiable_treatment':'Excluded as potentially missing; intervals adjacent to it are excluded and clinical episodes on either side are not joined.',
            'Q':q.tolist(),'convergence':bool(res.success),'optimizer_message':str(res.message),'log_likelihood':float(-res.fun),
            'AIC':float(2*len(res.x)+2*res.fun),'n_parameters':len(res.x),'ci_method':fit['ci_method'],
            'sparse_transitions':sparse,'checks':checks,'warnings':warnings}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input', type=Path, default=None); args=ap.parse_args()
    for d in [INTERMEDIATE_DIR,TABLES_DIR,FIGURES_DIR,QC_DIR]: d.mkdir(parents=True, exist_ok=True)
    input_path=args.input or common.POP_LONGITUDINAL_PARQUET
    vis=validate_input(load_classification(args.input)); intervals,interval_qc=build_intervals(vis); intervals.to_parquet(INTERVALS,index=False)
    m=transition_matrix(intervals); m.to_csv(TABLES_DIR/'10_transition_matrix_consecutive_episodes.csv',index=False)
    r=rates(intervals); r.to_csv(TABLES_DIR/'10_multistate_transition_rates.csv',index=False)
    plot_heatmap(m, FIGURES_DIR/'10_transition_heatmap.pdf'); plot_sankey(intervals, FIGURES_DIR/'10_transition_sankey.pdf'); plot_diagram(r, FIGURES_DIR/'10_multistate_transition_diagram.pdf')
    total_int=len(intervals); stable=int((~intervals.changed_state).sum()) if total_int else 0; changed=int(intervals.changed_state.sum()) if total_int else 0
    pats_ge2=int(vis.groupby('patient_id').size().ge(2).sum()); anychg=intervals.groupby('patient_id').changed_state.any() if total_int else pd.Series(dtype=bool)
    involved=int(((intervals.from_pop=='Unclassifiable')|(intervals.to_pop=='Unclassifiable')).sum()) if total_int else 0
    row_sums=m.groupby('from_pop').row_pct.sum(min_count=1).reindex(POP_ORDER); row_tot=m.groupby('from_pop').n_intervals.sum().reindex(POP_ORDER,fill_value=0)
    common_pair=intervals.transition_pair.value_counts().head(1)
    mc=common_pair.index[0] if len(common_pair) else None; mcn=int(common_pair.iloc[0]) if len(common_pair) else 0; mcr=np.nan
    if mc: mcr=float(m.loc[(m.from_pop==mc.split(' -> ')[0])&(m.to_pop==mc.split(' -> ')[1]),'row_pct'].iloc[0])
    tqc={'classification_file_used':str(input_path),'upstream_recomputed':False,
        'n_input_episodes':interval_qc['n_input_episodes'],'n_patients_total':interval_qc['n_input_patients'],
        'n_patients_with_ge2_clinical_episodes':pats_ge2,
        'n_adjacent_intervals_expected':interval_qc['n_adjacent_intervals_expected'],
        'n_adjacent_intervals_created':interval_qc['n_adjacent_intervals_created_before_time_filter'],
        'n_transition_intervals_retained':total_int,'n_intervals_nonpositive_time':interval_qc['n_intervals_nonpositive_time'],
        'state_order':POP_ORDER,'row_totals':{k:int(v) for k,v in row_tot.items()},'row_percent_sums':{k:(None if pd.isna(v) else float(v)) for k,v in row_sums.items()},'row_percent_sums_close_to_100':bool(all((row_tot[p]==0) or np.isclose(row_sums[p],100) for p in POP_ORDER)),'n_transitions_involving_unclassifiable':involved,'pct_transitions_involving_unclassifiable':pct(involved,total_int),'n_stable_intervals':stable,'pct_stable_intervals':pct(stable,total_int),'n_changed_intervals':changed,'pct_changed_intervals':pct(changed,total_int),'n_patients_with_any_transition':int(anychg.sum()) if len(anychg) else 0,'pct_patients_with_any_transition':pct(int(anychg.sum()) if len(anychg) else 0,pats_ge2),'median_time_between_consecutive_episodes_yrs':float(intervals.interval_years.median()) if total_int else np.nan,'median_time_between_changed_transitions_yrs':float(intervals.loc[intervals.changed_state,'interval_years'].median()) if changed else np.nan,'most_common_transition':mc,'most_common_transition_n':mcn,'most_common_transition_row_pct':mcr,'warnings':[]}
    if len(m)!=16: raise ValueError('Transition matrix does not contain 16 combinations')
    write_json(tqc, QC_DIR/'10_transition_matrix_qc.json')
    pt=intervals.groupby('from_pop').interval_years.sum().reindex(POP_ORDER, fill_value=0.0)
    mqc={'model_type':'descriptive transition intensity per person-year from consecutive observed intervals','exploratory_flag':True,'person_time_by_state':{k:float(v) for k,v in pt.items()},'n_transitions_by_pair':{f"{x.from_pop} -> {x.to_pop}":int(x.n_transitions) for x in r.itertuples()},'sparse_transition_pairs':[f"{x.from_pop} -> {x.to_pop}" for x in r.itertuples() if x.sparse_flag],'states_with_zero_person_time':[k for k,v in pt.items() if v==0],'transitions_involving_unclassifiable_note':'May reflect missing ESSDAI/ESSPRI data rather than true clinical change.','warnings':[]}
    write_json(mqc, QC_DIR/'10_multistate_transition_qc.json')
    # Complementary analysis: panel-observed continuous-time Markov model.  This
    # is intentionally separate from observed consecutive-episode summaries.
    model_intervals,prep=prepare_multistate_data(vis); fit=fit_multistate_model(model_intervals); q=fit['Q']
    intensity_rows=[]; k=0
    for i,a in enumerate(MODEL_STATES):
        for j,b in enumerate(MODEL_STATES):
            if i==j: continue
            n=prep['transitions_by_pair'][f'{a} -> {b}']
            intensity_rows.append({'from_pop':a,'to_pop':b,'n_observed_transitions':n,'intensity':q[i,j],
                                   'ci95_low':fit['ci_low'][k],'ci95_high':fit['ci_high'][k],
                                   'sparse_flag':n<SPARSE_THRESHOLD}); k+=1
    pd.DataFrame(intensity_rows).to_csv(TABLES_DIR/'10_multistate_intensity_matrix.csv',index=False)
    probabilities={}
    for horizon,label in [(.5,'0.5'),(1,'1'),(2,'2'),(5,'5')]:
        probabilities[horizon]=transition_probability_matrix(q,horizon)
        probabilities[horizon].to_csv(TABLES_DIR/f'10_multistate_transition_probabilities_{label}y.csv')
    multistate_sojourn_times(q).to_csv(TABLES_DIR/'10_multistate_sojourn_times.csv',index=False)
    plot_multistate_model(
        q,
        FIGURES_DIR/'10_multistate_model_diagram.pdf',
        observed_counts=prep['transitions_by_pair'],
    )
    plot_probability_heatmap(probabilities[1],1,FIGURES_DIR/'10_multistate_probability_heatmap_1y.pdf')
    plot_probability_heatmap(probabilities[2],2,FIGURES_DIR/'10_multistate_probability_heatmap_2y.pdf')
    model_qc=multistate_qc(prep,fit,model_intervals); write_json(model_qc,QC_DIR/'10_multistate_model_qc.json')
    print(f'Wrote {INTERVALS} and transition outputs')
    print(f"Continuous-time multi-state model converged: {fit['result'].success}")
    print('Q =\n',q); print('P(1 year) =\n',probabilities[1].to_string())
    print('Sparse transitions:',', '.join(model_qc['sparse_transitions']) or 'none')
    if model_qc['warnings']: print('Warnings:',*model_qc['warnings'],sep='\n- ')
if __name__=='__main__': main()
