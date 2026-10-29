"""Small prespecified clustering comparators on the same patient distance."""
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def kmedoids(distance,k,seed=0,max_iter=100):
    n=len(distance)
    if k<2 or k>=n: raise ValueError('Comparator K outside estimable range')
    medoids=np.sort(np.random.default_rng(seed).choice(n,k,replace=False))
    for _ in range(max_iter):
        labels=np.argmin(distance[:,medoids],axis=1)
        new=medoids.copy()
        for c in range(k):
            group=np.flatnonzero(labels==c)
            if not len(group): raise ValueError('Empty medoid cluster')
            new[c]=group[np.argmin(distance[np.ix_(group,group)].sum(axis=1))]
        if np.array_equal(new,medoids): break
        medoids=new
    return np.argmin(distance[:,medoids],axis=1)


def cluster(distance,k,method,seed):
    if method=='kmedoids': return kmedoids(distance,k,seed)
    return AgglomerativeClustering(n_clusters=k,metric='precomputed',linkage='average').fit_predict(distance)


def comparators(space,mapped,settings,seed,refit):
    rows=[]
    valid=np.isfinite(mapped['hard']) & ~mapped['ties']
    for method in ('kmedoids','agglomerative'):
        for k in settings['comparator_k']:
            row={'method':method,'k':k,'n_patients':len(space.raw),'metric':'same_patient_distance',
                 'status':'COMPLETE','reason':''}
            try:
                labels=cluster(space.distance,int(k),method,seed)
                valid_supported=valid & np.array([mapped['supported'][int(h)] if np.isfinite(h) else False for h in mapped['hard']])
                row.update(n_common=int(valid_supported.sum()),coverage=1.,min_support=int(np.bincount(labels).min()),
                    silhouette=float(silhouette_score(space.distance,labels,metric='precomputed')),
                    ari_mapper=float(adjusted_rand_score(mapped['hard'][valid_supported],labels[valid_supported])) if valid_supported.sum()>1 else np.nan,
                    nmi_mapper=float(normalized_mutual_info_score(mapped['hard'][valid_supported],labels[valid_supported])) if valid_supported.sum()>1 else np.nan)
                scores=[]; failures=0
                for r in range(int(settings['comparator_replicates'])):
                    draw=np.random.default_rng(seed+r).choice(len(space.raw),max(k+1,int(.8*len(space.raw))),replace=False)
                    try:
                        subspace=refit(draw)
                        sublabels=cluster(subspace.distance,int(k),method,seed+r)
                        scores.append(adjusted_rand_score(labels[draw],sublabels))
                    except (ValueError,np.linalg.LinAlgError): failures+=1
                row.update(resampling_method='repeated_subsampling_refit',stability_mean_ari=float(np.mean(scores)) if scores else np.nan,
                    n_replicates=int(settings['comparator_replicates']),n_success=len(scores),n_failed=failures,
                    qc_status='PASS' if len(scores)/int(settings['comparator_replicates'])>=settings['minimum_success_fraction'] else 'QC_FAILED')
            except ValueError as exc:
                row.update(status='NOT_ESTIMABLE',reason=str(exc))
            rows.append(row)
    return rows
