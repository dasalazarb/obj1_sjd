"""Optional exact-mapped rare events; no name inference or overfit models."""
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

def wilson(success,n):
    return proportion_confint(success,n,alpha=.05,method="wilson")
from .s03_profiles import numeric


def rare_event_table(baseline,config):
    rows=[]
    if not config['rare_events']:
        return pd.DataFrame([{'event':name,'status':'NOT_ESTIMABLE','reason':'NO_APPROVED_EXACT_SOURCE_AND_EVENT_DATE_MAP'} for name in ['MGUS','LYMPHOMA','HLA']])
    for event in config['rare_events']:
        column=event['source_column']; row={'event':event['id'],'source_column':column,'status':'NOT_ESTIMABLE'}
        if column not in baseline: row['reason']='SOURCE_UNAVAILABLE'
        elif event.get('approval')!='approved': row['reason']='CLINICAL_MAP_NOT_APPROVED'
        else:
            s=numeric(baseline[column]).dropna()
            if not set(s)<={0.,1.}: row['reason']='NOT_BINARY_EXACT_EVENT_CODING'
            elif len(s):
                count=int(s.sum()); lo,hi=wilson(count,len(s))
                row.update(status='DESCRIPTIVE',reason='',n_observed=len(s),n_events=count,ci_low=lo,ci_high=hi,
                           inferential_model='not_attempted_rare_events')
            else: row['reason']='NO_OBSERVED_EVENT_STATUS'
        rows.append(row)
    return pd.DataFrame(rows)
