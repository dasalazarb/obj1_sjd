"""Mapper with explicit lens and valid precomputed local distances."""
import numpy as np
from sklearn.cluster import DBSCAN
import kmapper as km
from . import s02_legacy as legacy


def analyze(space, config, *, lens=None, distance=None):
    if space.legacy_rep is not None and lens is None and distance is None:
        return legacy.analyze_mapper(space.legacy_rep,config)
    settings=config['mapper']
    d=space.distance if distance is None else distance
    if d.shape!=(len(space.raw),len(space.raw)) or not np.isfinite(d).all() or not np.allclose(d,d.T) or not np.allclose(np.diag(d),0):
        raise ValueError('Invalid precomputed distance')
    k=min(int(settings['eps_k_neighbors']),len(d)-1)
    if k<1: raise ValueError('Insufficient Mapper patients')
    eps=float(np.percentile(np.sort(d,axis=1)[:,k],float(settings['eps_percentile'])))
    if eps<=0:
        positive=d[d>0]
        if not len(positive): raise ValueError('Degenerate Mapper distance')
        eps=float(np.median(positive))
    graph=km.KeplerMapper(verbose=0).map(
        space.lens if lens is None else lens,d,precomputed=True,
        cover=km.Cover(n_cubes=int(settings['n_cubes']),perc_overlap=float(settings['perc_overlap'])),
        clusterer=DBSCAN(eps=eps,min_samples=int(settings['min_samples']),metric='precomputed'))
    counts_components,components,nodes,_=legacy.CANON.branches_from_nerve(graph,len(d),int(settings['minimum_node_support']),0)
    nerve,communities,counts,modularity=legacy.CANON.mapper_communities(graph,nodes,len(d))
    scores,covered,hard=legacy.CANON.soft_membership(counts,float(settings['soft_membership_tau']))
    sets=[set().union(*(nodes[node] for node in group)) for group in communities]
    return {'graph':graph,'nerve':nerve,'nodes':nodes,'components':components,'communities':communities,
            'community_sets':sets,'modularity':modularity,'covered':covered,'hard':hard,'probabilities':scores,
            'ties':covered & np.isnan(hard),'supported':np.array([len(s)>=int(config['community_detection']['minimum_patients']) for s in sets]),
            'component_counts':counts_components}


def frozen_membership(identifier,baseline,mapped,stability):
    rows=[]
    for i,patient in enumerate(baseline.patient_id):
        hard=mapped['hard'][i]
        for c in range(len(mapped['community_sets'])):
            rows.append({'patient_id':str(patient),'representation_id':identifier,'community_id':c,
                'mapper_covered':bool(mapped['covered'][i]),'hard_assignment':bool(np.isfinite(hard) and hard==c),
                'membership_tie':bool(mapped['ties'][i]),'supported':bool(mapped['supported'][c]),
                'structural_stability_status':stability.get(c,'not_evaluated'),
                'membership_score':mapped['probabilities'][i,c] if mapped['covered'][i] else np.nan,
                'membership_score_type':'mapper_node_count_softmax_descriptive',
                'community_membership':i in mapped['community_sets'][c]})
        if not mapped['community_sets']:
            rows.append({'patient_id':str(patient),'representation_id':identifier,'community_id':np.nan,
                'mapper_covered':False,'hard_assignment':False,'membership_tie':False,'supported':False,
                'structural_stability_status':'no_communities','membership_score':np.nan,
                'membership_score_type':'mapper_node_count_softmax_descriptive','community_membership':False})
    return rows
