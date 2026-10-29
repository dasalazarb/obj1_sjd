"""Seeded refitting, one-to-one matching, and explicit resampling denominators."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm
from .._pipeline import write_json, digest
from .s02_representations import Space
from .s02_missingness_registry import NotEstimable
from . import s02_legacy as legacy


def match_sets(reference, candidates):
    scores=np.array([[len(a&b)/len(a|b) if a|b else 0. for b in candidates] for a in reference])
    if not reference or not candidates:
        return [],scores.reshape((len(reference),len(candidates)))
    rows,cols=linear_sum_assignment(-scores)
    matches=[{'reference':int(a),'candidate':int(b),'jaccard':float(scores[a,b]),
        'split_candidates':int(np.count_nonzero(scores[a]>0)),
        'merge_references':int(np.count_nonzero(scores[:,b]>0))} for a,b in zip(rows,cols)]
    return matches,scores


def wilson(success,n):
    if not n: return np.nan,np.nan
    z=norm.ppf(.975); p=success/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,center-half),min(1.,center+half)


def draw_indices(n,fraction,seed,method):
    rng=np.random.default_rng(seed)
    if method=='repeated_subsampling':
        return rng.choice(n,size=min(n,max(3,math.floor(n*fraction))),replace=False)
    if method=='classic_bootstrap':
        return rng.choice(n,size=n,replace=True)
    raise ValueError('Unknown resampling method')


def collapse_bootstrap(space,draw):
    """Fit with multiplicities, then map each unique patient once."""
    unique,positions=np.unique(draw,return_index=True)
    return Space(space.raw.iloc[positions].reset_index(drop=True),space.lens[positions],
                 space.distance[np.ix_(positions,positions)],space.distance[np.ix_(positions,positions)],
                 space.weights,space.method,space.features),unique


def resample(reference, n, fit, map_space, settings, *, seed, identifier, method='repeated_subsampling',
             replicates=None, cache_dir=None, cache_key='', preliminary=False):
    requested=int(replicates if replicates is not None else settings['final_subsampling_replicates'])
    if requested<1 or not 0<float(settings['patient_fraction'])<=1:
        raise ValueError('Invalid resampling budget')
    attempts=[]; matches=np.zeros(len(reference),int); evaluable=np.zeros(len(reference),int)
    pair_denom=np.zeros((n,n),int); pair_same=np.zeros((n,n),int); success=0; alignments=[]; assignments=[]
    for r in range(requested):
        replicate_seed=int(seed)+r
        draw=draw_indices(n,float(settings['patient_fraction']),replicate_seed,method)
        key=hashlib.sha256((cache_key+identifier+method+str(replicate_seed)+draw.tobytes().hex()).encode()).hexdigest()
        cache=Path(cache_dir)/f'{key}.json' if cache_dir else None
        result=None
        if cache and cache.is_file():
            envelope=json.loads(cache.read_text())
            payload=json.dumps(envelope['payload'],sort_keys=True,separators=(',',':'))
            if envelope.get('sha256')!=hashlib.sha256(payload.encode()).hexdigest():
                raise ValueError('Resampling cache integrity failure')
            result=envelope['payload']
        if result is None:
            try:
                space=fit(draw)
                unique=draw
                if method=='classic_bootstrap': space,unique=collapse_bootstrap(space,draw)
                mapped=map_space(space)
                candidate_ids=[c for c in range(len(mapped['community_sets'])) if mapped['supported'][c]]
                sets=[set(unique[list(mapped['community_sets'][c])].tolist()) for c in candidate_ids]
                hard=mapped['hard']
                valid=[i for i,h in enumerate(hard) if np.isfinite(h) and not mapped['ties'][i] and mapped['supported'][int(h)]]
                result={'status':'success','sets':[sorted(s) for s in sets],
                    'valid_patients':[int(unique[i]) for i in valid], 'hard':[int(hard[i]) for i in valid],
                    'candidate_ids':candidate_ids}
            except (legacy.MappingError,legacy.LAB.LabContractError):
                raise
            except NotEstimable as exc:
                result={'status':'NOT_ESTIMABLE','reason':str(exc)}
            except (ValueError,np.linalg.LinAlgError) as exc:
                result={'status':'failed','reason':str(exc).split(':')[0]}
            if cache:
                cache.parent.mkdir(parents=True,exist_ok=True)
                payload=json.dumps(result,sort_keys=True,separators=(',',':'))
                write_json(cache,{'payload':result,'sha256':hashlib.sha256(payload.encode()).hexdigest()},atomic=True)
        row={'representation_id':identifier,'method':method,'phase':'preliminary' if preliminary else 'final',
             'replicate_id':r,'seed':replicate_seed,'n_draws':len(draw),'n_unique_patients':len(np.unique(draw)),
             'sample_sha256':hashlib.sha256(draw.tobytes()).hexdigest(),'cache_key':key,
             'status':result['status'],'reason':result.get('reason','')}
        attempts.append(row)
        if result['status']!='success': continue
        success+=1
        sampled=set(draw.tolist())
        ref=[s&sampled for s in reference]
        candidates=[set(s) for s in result['sets']]
        minimum=int(settings.get('minimum_match_patients',3))
        eligible=[i for i,s in enumerate(ref) if len(s)>=minimum]
        evaluable[eligible]+=1
        paired,_=match_sets([ref[i] for i in eligible],candidates)
        mapping={}
        for match in paired:
            c=eligible[match['reference']]
            matches[c]+=int(match['jaccard']>=float(settings['match_jaccard']))
            alignments.append({**row,**match,'reference':c})
            original_id=result['candidate_ids'][match['candidate']]
            mapping[original_id]=(c,match['jaccard'])
        if not preliminary and method=='repeated_subsampling':
            for patient,label in zip(result['valid_patients'],result['hard']):
                matched,jaccard=mapping.get(label,(None,None))
                assignments.append({'representation_id':identifier,'replicate_id':r,'seed':replicate_seed,
                    'patient_index':patient,'candidate_community_id':label,'reference_community_id':matched,
                    'matching_jaccard':jaccard})
        ids=np.array(result['valid_patients'],int); labels=np.array(result['hard'],int)
        pair_denom[np.ix_(ids,ids)]+=1
        pair_same[np.ix_(ids,ids)]+=(labels[:,None]==labels[None,:])
    qc=success/requested>=float(settings['minimum_success_fraction'])
    rows=[]
    for c in range(len(reference)):
        lo,hi=wilson(int(matches[c]),int(evaluable[c]))
        score=matches[c]/evaluable[c] if evaluable[c] else np.nan
        rows.append({'representation_id':identifier,'community_id':c,'method':method,'phase':'preliminary' if preliminary else 'final',
            'n_requested':requested,'n_success':success,'n_evaluable':int(evaluable[c]),'matched_successes':int(matches[c]),
            'success_fraction':success/requested,'stability':score if qc else np.nan,'stability_conditional':score,
            'stability_conservative':matches[c]/requested,'ci_low':lo,'ci_high':hi,
            'qc_status':'PASS' if qc else 'QC_FAILED','stable':bool(qc and not preliminary and score>=float(settings['stable_threshold'])),
            'matching_object':'mapper_node_union_on_unique_sampled_patients','primary_denominator':'n_evaluable_successful_refits'})
    return {'assignments':assignments,'stability':rows,'attempts':attempts,'alignments':alignments,'pair_denom':pair_denom,'pair_same':pair_same,
            'qc_pass':qc,'n_success':success,'n_requested':requested}
