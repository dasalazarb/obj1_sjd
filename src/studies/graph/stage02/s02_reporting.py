"""Source-backed technical figures and summaries, including negative findings."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_figure(fig,folder,name):
    for ext in ('png','pdf'):
        fig.savefig(folder/f'{name}.{ext}',bbox_inches='tight',dpi=180)
    plt.close(fig)


def figures(results,stability,folder,registry,coverage,alignment=None):
    count=max(1,len(results))
    fig,axes=plt.subplots(1,count,figsize=(4*count,4),squeeze=False)
    for ax,(name,item) in zip(axes[0],results.items()):
        mapped=item['mapped']; lens=item['space'].lens
        reference_colors={c:plt.get_cmap('tab10')(c%10) for c in range(max(10,len(results.get('R0',item)['mapped']['community_sets'])))}
        matched={c:c for c in range(len(mapped['community_sets']))} if name=='R0' else {
            row['candidate']:row['reference'] for row in (alignment or [])
            if row['representation_id']==name and row['jaccard']>=.5}
        colors=[reference_colors.get(matched.get(int(h)),(.55,.55,.55,1.)) if np.isfinite(h) else (.75,.75,.75,1.) for h in mapped['hard']]
        ax.scatter(lens[:,0],lens[:,1],c=colors,s=14,alpha=.4,zorder=1)
        positions={node:lens[sorted(patients)].mean(axis=0) for node,patients in mapped['nodes'].items()}
        for left,right in mapped['nerve'].edges():
            if left in positions and right in positions:
                a,b=positions[left],positions[right]
                ax.plot([a[0],b[0]],[a[1],b[1]],color='#777777',lw=.8,alpha=.6,zorder=2)
        for community,group in enumerate(mapped['communities']):
            for node in group:
                x,y=positions[node]
                ax.scatter(x,y,s=12+8*len(mapped['nodes'][node]),
                    color=reference_colors.get(matched.get(community),(.55,.55,.55,1.)),
                    edgecolors='white',linewidths=.6,alpha=.85,zorder=3)
        uncovered=~mapped['covered']
        if uncovered.any(): ax.scatter(lens[uncovered,0],lens[uncovered,1],marker='x',s=22,color='#333333',zorder=4)
        ax.set(title=f'{name}: coverage {mapped["covered"].mean():.1%}',xlabel='Lens 1',ylabel='Lens 2')
    fig.suptitle('Exploratory Mapper: node sizes show support; edges show shared patients')
    fig.text(.5,-.02,'Colors match R0 communities at Jaccard >= 0.5; gray = unmatched; x = uncovered. Communities are not progression branches.',ha='center',fontsize=9)
    save_figure(fig,folder,'02_graph_topologies')
    fig,ax=plt.subplots(figsize=(9,4))
    if not stability.empty:
        part=stability.loc[(stability.method=='repeated_subsampling') & (stability.phase=='final')]
        x=np.arange(len(part)); score=pd.to_numeric(part.stability,errors='coerce').to_numpy(float)
        valid=np.isfinite(score)
        ax.errorbar(x[valid],score[valid],yerr=np.vstack([np.maximum(0,score[valid]-part.ci_low.to_numpy(float)[valid]),np.maximum(0,part.ci_high.to_numpy(float)[valid]-score[valid])]),fmt='o')
        ax.set(xticks=x,xticklabels=[f'{r.representation_id}/C{r.community_id}' for r in part.itertuples()],ylim=(0,1.05))
        ax.tick_params(axis='x',rotation=90)
    ax.set(title='Repeated subsampling: community matching and 95% Wilson intervals',ylabel='Matching fraction')
    save_figure(fig,folder,'02_graph_stability')
    fig,ax=plt.subplots(figsize=(9,4))
    if coverage:
        data=pd.DataFrame(coverage)
        data=data.loc[data.representation_id.eq('R0')]
        ax.bar(data.clinical_domain,data.observed_fraction)
    ax.set(title='Primary representation: clinical-domain observation coverage',ylabel='Observed fraction',ylim=(0,1))
    save_figure(fig,folder,'02_graph_domain_coverage')
    fig,ax=plt.subplots(figsize=(8,4))
    weights=[]
    for identifier,item in results.items():
        for row in item['space'].weights.to_dict('records'):
            weights.append({'representation_id':identifier,**row})
    data=pd.DataFrame(weights)
    if not data.empty:
        selected=data.loc[data.representation_id.isin(['R0','R2'])]
        if not selected.empty:
            ax.bar(np.arange(len(selected)),selected.weight_applied)
            ax.set(xticks=np.arange(len(selected)),xticklabels=selected.block)
            ax.tick_params(axis='x',rotation=90)
    ax.set(title='Estimated block normalization weights; no clinical selection',ylabel='Weight')
    save_figure(fig,folder,'02_graph_domain_weights')
