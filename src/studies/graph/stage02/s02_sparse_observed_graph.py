"""Observed-pair kNN sensitivity, explicitly a graph and never Mapper.

Unknown pairs remain unknown. Similarity 1/(1+d) is strictly decreasing and
preserves the observed-distance neighbor ordering. No shortest-path completion.
"""
from dataclasses import dataclass
import hashlib
import math
import networkx as nx
import numpy as np
from .s02_missingness_registry import NotEstimable, build_panel
from .s02_pairwise_distance import observed_distance
from .s02_resampling import match_sets, wilson


@dataclass
class SparseObservedGraph:
    graph: nx.Graph
    mapped: dict
    summary: dict


def build_sparse_graph(evidence,patients,policy,*,k,seed,panel_hash=''):
    d=np.asarray(evidence.distance,float); n=len(patients)
    if d.shape!=(n,n) or len(set(patients))!=n: raise ValueError('AMBIGUOUS_GRAPH_PATIENT_UNIVERSE')
    if not np.allclose(d,d.T,equal_nan=True) or not np.all(np.diag(d)==0): raise ValueError('INVALID_PAIRWISE_GEOMETRY')
    if k<1 or policy['symmetrization'] not in {'union','mutual'}: raise ValueError('INVALID_SPARSE_GRAPH_POLICY')
    eligible=np.zeros((n,n),bool)
    for row in evidence.support.itertuples():
        a,b=int(row.patient_index_a),int(row.patient_index_b)
        if not 0<=a<b<n: raise ValueError('INVALID_SUPPORT_PAIR')
        ok=(row.distance_estimable and np.isfinite(d[a,b]) and row.n_shared_constructs>0
            and row.n_shared_domains>=int(policy['minimum_shared_domains'])
            and row.shared_weight_fraction>=float(policy['minimum_shared_weight_fraction']))
        eligible[a,b]=eligible[b,a]=ok
    directed=np.zeros((n,n),bool)
    enough=eligible.sum(axis=1)>=int(policy['minimum_neighbors'])
    for i in np.flatnonzero(enough):
        ids=np.flatnonzero(eligible[i]&enough)
        if len(ids)<int(policy['minimum_neighbors']): continue
        neighbors=ids[np.argsort(d[i,ids],kind='stable')[:k]]
        directed[i,neighbors]=True
    chosen=(directed|directed.T) if policy['symmetrization']=='union' else (directed&directed.T)
    graph=nx.Graph(); graph.add_nodes_from(range(n))
    for a,b in zip(*np.where(np.triu(chosen,1))):
        graph.add_edge(int(a),int(b),distance=float(d[a,b]),weight=1/(1+float(d[a,b])))
    eligibility=nx.from_numpy_array(eligible)
    components=sorted(nx.connected_components(graph),key=lambda c:(-len(c),min(c)))
    connected=[i for i in graph if graph.degree(i)>0]
    largest=len(components[0]) if components else 0
    status='COMPLETE'; reason=''
    if (len(connected)<int(policy['minimum_connected_patients']) or len(connected)/max(1,n)<float(policy['minimum_connected_fraction'])
            or largest/max(1,n)<float(policy['minimum_largest_component_fraction'])):
        status='NOT_ESTIMABLE'; reason='INSUFFICIENT_SPARSE_GRAPH_CONNECTIVITY'
    groups=[]
    if graph.number_of_edges() and status=='COMPLETE':
        found=nx.community.louvain_communities(graph,weight='weight',resolution=float(policy['resolution']),seed=int(seed))
        groups=sorted([set(g) for g in found if len(g)>=int(policy['minimum_community_patients']) and any(graph.degree(i)>0 for i in g)],
            key=lambda c:(-len(c),min(c)))
        if not groups: status='NOT_ESTIMABLE'; reason='NO_SUPPORTED_SPARSE_GRAPH_GROUPS'
    hard=np.full(n,np.nan)
    if status=='COMPLETE':
        for label,group in enumerate(groups): hard[list(group)]=label
    else: groups=[]
    degrees=np.array([graph.degree(i) for i in range(n)])
    pairs=n*(n-1)//2; eligible_pairs=int(np.triu(eligible,1).sum())
    summary=dict(method_type='sparse_observed_graph',is_mapper=False,graph03_defining=False,status=status,reason=reason,
        estimator='sparse_observed_graph_v1',n_patients=n,n_pairs=pairs,n_valid_pair_distances=eligible_pairs,n_estimable_pairs=eligible_pairs,
        n_unevaluated_pairs=pairs-eligible_pairs,n_edges=graph.number_of_edges(),n_connected=len(connected),
        n_isolated=int((degrees==0).sum()),n_unsupported=int((~enough).sum()),n_components=len(components),
        component_sizes='|'.join(str(len(c)) for c in components),n_eligibility_components=nx.number_connected_components(eligibility),
        n_eligibility_isolated=len(list(nx.isolates(eligibility))),largest_component_fraction=largest/max(1,n),
        degree_min=int(degrees.min()) if n else 0,degree_max=int(degrees.max()) if n else 0,degree_mean=float(degrees.mean()) if n else np.nan,
        edge_fraction_of_eligible=graph.number_of_edges()/eligible_pairs if eligible_pairs else np.nan,
        density=nx.density(graph),n_covered=int(np.isfinite(hard).sum()),n_communities=len(groups),
        modularity=float(nx.community.modularity(graph,found,weight='weight',resolution=float(policy['resolution']))) if groups else np.nan,
        k=int(k),symmetrization=policy['symmetrization'],algorithm='networkx_louvain',algorithm_version=nx.__version__,
        resolution=float(policy['resolution']),seed=int(seed),panel_hash=panel_hash,
        similarity='1/(1+observed_distance);strictly_decreasing;no_path_completion',
        absence_of_edge='not_evidence_of_large_distance',comparison_warning='cross_algorithm_not_mapper')
    mapped=dict(hard=hard,covered=np.isfinite(hard),ties=np.zeros(n,bool),supported=np.ones(len(groups),bool),community_sets=groups)
    return SparseObservedGraph(graph,mapped,summary)


def graph_resampling(raw,registry,baseline,reference,policy,distance_policy,settings,*,k,seed,replicates,reference_graph):
    """Recompute scales, pair eligibility, neighbors and groups within every draw."""
    attempts=[]; matches=np.zeros(len(reference_graph.mapped['community_sets']),int); evaluable=np.zeros_like(matches)
    success=0; n=len(raw)
    for repeat in range(int(replicates)):
        replicate_seed=int(seed)+repeat
        draw=np.random.default_rng(replicate_seed).choice(n,size=min(n,max(3,math.floor(n*settings['patient_fraction']))),replace=False)
        sample=raw.iloc[draw].reset_index(drop=True); sample_baseline=baseline.iloc[draw].reset_index(drop=True)
        row=dict(replicate_id=repeat,seed=replicate_seed,n_draws=len(draw),k=k,
            sample_sha256=hashlib.sha256(draw.tobytes()).hexdigest(),
            mask_sha256=hashlib.sha256(sample.notna().to_numpy().tobytes()).hexdigest(),
            refit='scales_distances_eligibility_neighbors_groups',cache_used=False,estimator_version='sparse_observed_graph_v1')
        try:
            panel=build_panel(sample,registry,reference,sample_baseline)
            if panel.fingerprint!=reference_graph.summary['panel_hash']: raise ValueError('FIXED_PANEL_MISMATCH')
            row['panel_hash']=panel.fingerprint
            evidence=observed_distance(sample,registry,reference,distance_policy,sample_baseline)
            result=build_sparse_graph(evidence,list(range(len(draw))),policy,k=k,seed=replicate_seed,panel_hash=panel.fingerprint)
            row.update(status=result.summary['status'],reason=result.summary['reason'],n_connected=result.summary['n_connected'])
            if row['status']=='COMPLETE':
                success+=1
                candidates=[set(draw[list(g)].tolist()) for g in result.mapped['community_sets']]
                sampled=set(draw.tolist()); refs=[g&sampled for g in reference_graph.mapped['community_sets']]
                ids=[i for i,g in enumerate(refs) if len(g)>=int(settings.get('minimum_match_patients',3))]
                evaluable[ids]+=1
                matched,_=match_sets([refs[i] for i in ids],candidates)
                for match in matched: matches[ids[match['reference']]]+=int(match['jaccard']>=settings['match_jaccard'])
        except NotEstimable as exc: row.update(status='NOT_ESTIMABLE',reason=str(exc))
        except (ValueError,np.linalg.LinAlgError) as exc: row.update(status='FAILED',reason=str(exc).split(':')[0])
        attempts.append(row)
    qc=success/max(1,int(replicates))>=settings['minimum_success_fraction']
    rows=[]
    for group,(count,denom) in enumerate(zip(matches,evaluable)):
        low,high=wilson(count,denom)
        rows.append(dict(group_id=group,method='repeated_subsampling',n_requested=int(replicates),n_successful_refits=success,
            n_evaluable=int(denom),matched_successes=int(count),stability=count/denom if denom and qc else np.nan,
            ci_low=low,ci_high=high,status='COMPLETE' if qc and denom else 'NOT_ESTIMABLE',
            reason='' if qc and denom else 'INSUFFICIENT_SUCCESSFUL_OR_EVALUABLE_GRAPH_REFITS',
            matching='one_to_one_jaccard_on_unique_sampled_patients',k=k))
    if not rows: rows=[dict(status='NOT_ESTIMABLE',reason='NO_SUPPORTED_REFERENCE_GRAPH_GROUPS',n_requested=int(replicates),
        n_successful_refits=success,n_evaluable=0,stability=np.nan,k=k)]
    return rows,attempts,success
