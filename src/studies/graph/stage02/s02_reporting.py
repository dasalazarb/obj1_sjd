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
        pending=name=='R2' and not registry.loc[registry.included_R2,'clinically_approved'].all()
        title=f'{name}: coverage {mapped["covered"].mean():.1%}'
        if pending: title+='\nEXPLORATORY / mapping pending clinical approval'
        ax.set(title=title,xlabel='Lens 1',ylabel='Lens 2')
    fig.suptitle('Exploratory Mapper: node sizes show support; edges show shared patients')
    fig.text(.5,-.02,'Colors match R0 communities at Jaccard >= 0.5; gray = unmatched; x = uncovered. Communities are not progression branches.',ha='center',fontsize=9)
    save_figure(fig,folder,'02_graph_topologies')
    fig,ax=plt.subplots(figsize=(9,4))
    if not stability.empty:
        part=stability.loc[(stability.phase=='final') & stability.representation_id.isin(results)]
        x=np.arange(len(part)); score=pd.to_numeric(part.stability,errors='coerce').to_numpy(float)
        valid=np.isfinite(score)
        ax.errorbar(x[valid],score[valid],yerr=np.vstack([np.maximum(0,score[valid]-part.ci_low.to_numpy(float)[valid]),np.maximum(0,part.ci_high.to_numpy(float)[valid]-score[valid])]),fmt='o')
        ax.set(xticks=x,xticklabels=[f'{r.representation_id}/C{r.community_id}/{r.method}' for r in part.itertuples()],ylim=(0,1.05))
        ax.tick_params(axis='x',rotation=90)
    ax.set(title='Subsampling and classic bootstrap: separate matching estimates, 95% Wilson intervals',ylabel='Matching fraction')
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
    title='Estimated block normalization weights; no clinical selection'
    if 'R2' in results and not registry.loc[registry.included_R2,'clinically_approved'].all(): title+='\nR2: EXPLORATORY / mapping pending clinical approval'
    ax.set(title=title,ylabel='Weight')
    save_figure(fig,folder,'02_graph_domain_weights')


def diagnostic_figures(observation, comparisons, experiments, folder, registry, minimum=5):
    """Aggregate figures only. Unestimable figures get an inventory row."""
    inventory=[]
    def finish(fig,name):
        save_figure(fig,folder,'02_graph_'+name)
        for ext in ('png','pdf'): inventory.append(dict(artifact='figures/02_graph_'+name+'.'+ext,status='COMPLETE',reason=''))
    def unavailable(name,reason):
        for ext in ('png','pdf'): inventory.append(dict(artifact='figures/02_graph_'+name+'.'+ext,status='NOT_ESTIMABLE',reason=reason))
    features=observation['features']; labels=observation['community_labels']
    domain=[c for c in features if c.startswith('clinical_domain__')]
    groups=[c for c,n in labels.value_counts().items() if n>=minimum]
    if domain and groups:
        matrix=np.array([features.loc[labels.eq(c),domain].mean().to_numpy() for c in groups]).T
        fig,ax=plt.subplots(figsize=(max(6,len(groups)*1.5),5))
        im=ax.imshow(matrix,vmin=0,vmax=1,aspect='auto',cmap='viridis')
        ax.set(yticks=np.arange(len(domain)),yticklabels=[c.split('__',1)[1] for c in domain],xticks=np.arange(len(groups)),
            xticklabels=[f'{c} (n={int(labels.eq(c).sum())})' for c in groups],title='R0 fixed panel: observed fraction by domain and assignment')
        ax.tick_params(axis='x',rotation=45); fig.colorbar(im,ax=ax)
        finish(fig,'missingness_domain_heatmap')
    else: unavailable('missingness_domain_heatmap','INSUFFICIENT_AGGREGATE_DOMAIN_SUPPORT')
    coverage=observation['coverage']; eligible=coverage.groupby('mapper_covered').filter(lambda g:len(g)>=minimum)
    if not eligible.empty:
        means=eligible.groupby('mapper_covered').observed_fraction.mean()
        fig,ax=plt.subplots(figsize=(6,4)); ax.bar([str(c) for c in means.index],means)
        ax.set(title=f'R0 observed fraction by Mapper coverage (eligible n={len(eligible)})',xlabel='Mapper covered',ylabel='Fraction observed',ylim=(0,1))
        finish(fig,'missingness_coverage')
    else: unavailable('missingness_coverage','INSUFFICIENT_COVERAGE_GROUP_SUPPORT')
    controls=observation['controls']
    if controls:
        fig,axes=plt.subplots(1,len(controls),figsize=(4*len(controls),4),squeeze=False)
        for ax,(mode,item) in zip(axes[0],controls.items()):
            mapped=item['mapped']; lens=item['space'].lens
            for patients in mapped['nodes'].values():
                if len(patients)<minimum: continue
                x,y=lens[sorted(patients)].mean(axis=0); ax.scatter(x,y,s=10*len(patients),alpha=.5)
            ax.set(title=f'{mode} ({item["space"].method}): n={len(lens)}\ncoverage={mapped["covered"].mean():.1%}',xlabel='Fitted lens 1',ylabel='Fitted lens 2')
        fig.suptitle('Observation sensitivities: node aggregates; small nodes suppressed')
        finish(fig,'missingness_topologies')
    else: unavailable('missingness_topologies','NO_ESTIMABLE_OBSERVATION_CONTROLS')
    data=pd.DataFrame(comparisons)
    part=data.loc[data.experiment_id.isin(['N2a/missingness_only','N2b/administrative_only','N2c/missingness_plus_admin','N2d/domain_missingness_only','M1/value_plus_mask_sensitivity'])].copy()
    if 'ari_primary' in part and part.ari_primary.notna().any():
        part=part.loc[part.ari_primary.notna()]
        fig,ax=plt.subplots(figsize=(8,4)); ax.bar(part.experiment_id,part.ari_primary)
        ax.set(xticks=np.arange(len(part)),xticklabels=[f'{r.experiment_id.split("/")[0]} (n={r.n_common})' for r in part.itertuples()],ylabel='ARI on common supported hard assignments',title='Observation controls versus R0; full cohort and common denominators in CSV')
        finish(fig,'missingness_vs_reference')
    else: unavailable('missingness_vs_reference','INSUFFICIENT_COMMON_SUPPORTED_ASSIGNMENTS_OR_CLASSES')
    part=observation['group_comparison']
    part=part.loc[part.status.eq('COMPLETE') & part.metric.isin(['observed_fraction','knn_mean_distance','nearest_neighbor_distance','local_density_percentile'])]
    if not part.empty:
        fig,axes=plt.subplots(1,len(part),figsize=(4*len(part),4),squeeze=False)
        for ax,row in zip(axes[0],part.itertuples()):
            ax.errorbar([0],[row.effect],yerr=[[max(0,row.effect-row.ci_low)],[max(0,row.ci_high-row.effect)]],fmt='o')
            ax.axhline(0,color='gray'); ax.set(title=f'{row.metric}\ncovered n={row.n_covered}; uncovered n={row.n_uncovered}',xticks=[],ylabel='Uncovered − covered, patient bootstrap 95% interval')
        finish(fig,'uncovered_diagnostics')
    else: unavailable('uncovered_diagnostics','INSUFFICIENT_COVERAGE_GROUP_SUPPORT')
    data=pd.DataFrame(experiments)
    part=data.loc[data.experiment_id.str.startswith(('A1/','A2/','A6/')) & data.status.eq('COMPLETE')]
    if 'ari_primary' in part and part.ari_primary.notna().any():
        part=part.loc[part.ari_primary.notna()]
        fig,ax=plt.subplots(figsize=(10,4)); ax.bar(part.experiment_id,part.ari_primary)
        ax.set(ylabel='ARI versus R0 on supported common patients',title='Prespecified ablative refits; no clinical outcome selection')
        ax.tick_params(axis='x',rotation=70); finish(fig,'domain_ablation')
    else: unavailable('domain_ablation','NO_ESTIMABLE_ABLATION_COMPARISON')
    return inventory


def results_summary(path,results,perturbations,experiments,comparisons,registry,observation,stability,qc):
    """Execution-derived answers, including negative findings and interpretation limits."""
    def table(frame,columns):
        if frame.empty: return 'No estimable rows.\n'
        data=frame.reindex(columns=columns).fillna('NA')
        return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+''.join('| '+' | '.join(str(v).replace('|',';') for v in row)+' |\n' for row in data.itertuples(index=False,name=None))
    mapped=results['R0']['mapped']; n=len(mapped['covered'])
    text=['# Graph02: análisis exploratorio técnico',
        'La validez clínica no está establecida. No se usaron Pop, ESSDAI, ESSPRI ni otros desenlaces para seleccionar una representación.',
        f'Estado: {qc["release_status"]}; ejecución: {qc["execution_status"]}; QC primario: {qc["primary_qc"]}.',
        f'Cohorte: {n}; columnas R0: {len(results["R0"]["space"].features)}; cobertura: {int(mapped["covered"].sum())}/{n} ({mapped["covered"].mean():.2%}); no cubiertos: {int((~mapped["covered"]).sum())}.',
        '## Registro clínico y R2',
        f'Fuentes seleccionadas sin mapear: {registry.loc[registry.selected_in_primary & registry.clinical_domain.eq("UNMAPPED"),"source_column"].nunique()}. La auditoría por fuente está en 02_graph_clinical_feature_mapping_audit.csv. D01–D09 son grupos de integración, no dominios ESSDAI.',
        table(registry.loc[registry.selected_in_primary].groupby('clinical_domain').agg(n_original_sources=('source_column','nunique'),n_encoded_features=('feature','size'),n_unique_constructs=('derivation_group','nunique'),n_approved_encoded=('clinically_approved','sum'),n_technical_encoded=('exploratory_technical_eligible','sum')).reset_index(),['clinical_domain','n_original_sources','n_encoded_features','n_unique_constructs','n_approved_encoded','n_technical_encoded']),
        table(pd.DataFrame(experiments).loc[lambda d:d.experiment_id.isin(['R0','R2','A5/context_excluded'])],['experiment_id','status','reason','evidence_level','n_patients','n_features','n_domains_effective','mapper_coverage']),
        ('R2: '+next((r.get('evidence_level','NOT_ESTIMABLE') for r in experiments if r['experiment_id']=='R2'),'NOT_ESTIMABLE')+'. EXPLORATORY / mapping pending clinical approval cuando la evidencia sea provisional. Los pesos singulares, constructos y dominios efectivos están en 02_graph_domain_contributions.csv. Las exclusiones técnicas y las revisiones de unidades/tiempo pendientes son explícitas en la auditoría; estandarizar dentro de cada fuente no armoniza unidades incompatibles.'),
        '## Contrastes y denominadores',table(pd.DataFrame(comparisons).loc[lambda d:d.experiment_id.isin(['R2','A5/context_excluded']) | d.experiment_id.str.startswith(('N2','M1/'))],['experiment_id','n_total','n_covered_reference','n_covered_control','n_covered_both','n_common','fraction_of_total','ari_primary','nmi_primary','jaccard_full_mean','comparison_status','comparison_reason']),
        'El control N2 legacy mezcla ausencia y administración; un ARI próximo a cero sobre una intersección pequeña no descarta estructura del proceso de observación. N2a usa solo máscaras, N2b solo administración, N2c bloques balanceados y N2d dominio/instrumento. M1 incorpora valores e indicadores con refit.',
        '## Estabilidad',table(stability.loc[stability.representation_id.isin(['R0','R2','A5/context_excluded']) & stability.phase.eq('final')],['representation_id','community_id','method','n_requested','n_success','n_evaluable','stability','ci_low','ci_high','qc_status']),
        'Submuestreo sin reemplazo y bootstrap con reemplazo son estimaciones distintas; los intervalos Wilson usan refits evaluables y no establecen estabilidad incondicional.',
        '## Proceso de observación',table(observation['summary'],['panel','group_type','group','n_patients','n_observation_units','n_missing','missing_fraction']),
        'N5/N6 conservan filas completas de máscaras y contrastan su asociación con etiquetas; p=(1+extremos)/(B+1). Son análisis exploratorios múltiples, no identifican MCAR/MAR/MNAR ni etiología. N6 requiere estratos de protocolo/era verificados y suficientes.',
        table(observation['diagnostic'].drop_duplicates('model'),['model','status','reason','n_patients','n_covered','n_uncovered','auc_mean','auc_p025','auc_p975']),
        '## Pacientes sin cobertura',table(observation['reason_summary'],['coverage_reason','n_patients','n_total','fraction_of_total','status']),
        table(observation['group_comparison'].loc[lambda d:d.metric.isin(['observed_fraction','knn_mean_distance','nearest_neighbor_distance'])],['metric','status','reason','n_covered','n_uncovered','mean_covered','mean_uncovered','effect','ci_low','ci_high']),
        'La completitud y el soporte geométrico se describen por separado; no se reasignan pacientes ni se optimiza Mapper para aumentar cobertura. Las máscaras R0 deduplican codificaciones, por lo que sus porcentajes no equivalen al porcentaje de celdas codificadas ausentes del release histórico.',
        '## Inventario de experimentos',table(pd.DataFrame(experiments),['experiment_id','status','reason','n_patients','n_common']),
        '## Límites y Graph03',
        'Graph03 puede consumir el contrato congelado tras validar hashes y QC; no recompone Mapper/PCA. Una comunidad estable no implica fenotipo validado, progresión temporal ni subtipo etiológico. Siguen pendientes la revisión clínica humana del mapa, temporalidad, unidades y anticircularidad. Las fuentes usadas para definir el grafo no son validadores independientes.',
        'Limitaciones: '+ '; '.join(qc['limitations']),
        'artifact_risk=false es una regla automática limitada a los controles efectivamente estimados. No establece ausencia de sesgo administrativo. Los archivos por paciente son internos/restringidos; el inventario de privacidad excluye todos los artefactos de exportación automática hasta revisión NIH.']
    path.write_text('\n\n'.join(text)+'\n')


def privacy_inventory(root):
    """Deny by default: an inventory is not authorization for public export."""
    from .._pipeline import digest
    return [dict(artifact=p.relative_to(root).as_posix(),sha256=digest(p),classification='INTERNAL_RESTRICTED',
        shareable_export_allowed=False,reason='NIH_publication_review_required;patient_artifacts_never_auto_exported')
        for p in sorted(root.rglob('*')) if p.is_file()]
