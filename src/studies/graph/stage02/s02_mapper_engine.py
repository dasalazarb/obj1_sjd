"""Mapper with explicit lens and valid precomputed local distances."""
import numpy as np
from sklearn.cluster import DBSCAN
import kmapper as km
from . import s02_legacy as legacy


def analyze(space, config, *, lens=None, distance=None, audit=False):
    if space.legacy_rep is not None and lens is None and distance is None:
        result=legacy.analyze_mapper(space.legacy_rep,config)
        if audit:
            eps=legacy.CANON.eps_kdist(space.legacy_rep.embedding,int(config['mapper']['eps_k_neighbors']),float(config['mapper']['eps_percentile']))
            result['coverage_audit']=coverage_audit(space,result,config,eps)
        return result
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
    result={'graph':graph,'nerve':nerve,'nodes':nodes,'components':components,'communities':communities,
            'community_sets':sets,'modularity':modularity,'covered':covered,'hard':hard,'probabilities':scores,
            'ties':covered & np.isnan(hard),'supported':np.array([len(s)>=int(config['community_detection']['minimum_patients']) for s in sets]),
            'component_counts':counts_components}
    if audit: result['coverage_audit']=coverage_audit(space,result,config,eps,lens=lens,distance=d)
    return result


def coverage_audit(space,mapped,config,eps,*,lens=None,distance=None):
    """Audit the exact cover and emitted DBSCAN nodes, without changing Mapper.

    KeplerMapper numbers nonempty cover cubes in transform order. Raw graph
    nodes retain clusters later rejected by minimum_node_support.
    """
    n=len(space.raw); settings=config['mapper']
    projection=space.lens if lens is None else lens
    d=space.distance if distance is None else distance
    cover=km.Cover(n_cubes=int(settings['n_cubes']),perc_overlap=float(settings['perc_overlap']))
    indexed=np.column_stack([np.arange(n),projection]); cover.fit(indexed)
    counts=np.zeros((n,5),int); border=np.full(n,np.inf)
    for cube_id,cube in enumerate(cover.transform(indexed)):
        ids=cube[:,0].astype(int); counts[ids,0]+=1
        prefix=f'cube{cube_id}_cluster'
        emitted=[set(members) for name,members in mapped['graph']['nodes'].items() if name.startswith(prefix)]
        emitted_union=set().union(*emitted) if emitted else set()
        for i in ids:
            if len(ids)<int(settings['min_samples']): counts[i,1]+=1
            elif i not in emitted_union: counts[i,2]+=1
            if any(i in group and len(group)<int(settings['minimum_node_support']) for group in emitted): counts[i,3]+=1
            counts[i,4]=max(counts[i,4],int(np.sum(d[i,ids]<=eps)))
    for center in cover.centers_:
        cube=cover.transform_single(indexed,center)
        if not len(cube): continue
        ids=cube[:,0].astype(int)
        radius=np.asarray(cover.radius_)
        delta=np.abs(projection[ids]-center)
        margin=np.min(np.divide(radius-delta,radius,out=np.full_like(delta,np.inf),where=radius>0),axis=1)
        border[ids]=np.minimum(border[ids],margin)
    rows=[]
    for i in range(n):
        if mapped['covered'][i]: reason='covered'
        elif counts[i,0]==0: reason='outside_lens_cover'
        elif counts[i,1]==counts[i,0]: reason='all_cubes_below_min_samples'
        elif counts[i,3]>0: reason='cluster_below_minimum_node_support'
        elif counts[i,2]+counts[i,1]==counts[i,0]: reason='dbscan_noise_in_all_eligible_cubes'
        else: reason='reason_unresolved'
        rows.append(dict(patient_index=i,n_cubes=int(counts[i,0]),n_cubes_below_min_samples=int(counts[i,1]),
            n_cubes_noise=int(counts[i,2]),n_clusters_rejected_support=int(counts[i,3]),local_dbscan_neighbors=int(counts[i,4]),
            nearest_cube_border_fraction=float(border[i]) if np.isfinite(border[i]) else np.nan,eps=float(eps),coverage_reason=reason))
    return rows


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
