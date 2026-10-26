import pandas as pd
import pytest
from src.block_A._internal.medications.dates import parse_date, exact_day
from src.block_A._internal.medications.episode_linkage import match
from src.block_A._internal.medications.documentation import pseudonym

@pytest.mark.parametrize('value',['2020-02-29','2020-02-29 12:30','2020-02-29T12:30:01.123Z',
    '02/29/2020','02/29/2020 12:30','02/29/2020 12:30:01.125 PM +00:00'])
def test_source_dates_retain_recorded_day(value):
    assert exact_day(value)==pd.Timestamp('2020-02-29')

@pytest.mark.parametrize('value',['2020','2020-02','02/2020','02/30/2020',r'\\d{4}-\\d{2}-\\d{2}','2020-01-01 junk'])
def test_partial_invalid_and_double_escaped_dates_cannot_match(value):
    assert pd.isna(exact_day(value))


def test_leap_year_boundaries_priority_and_patient_isolation():
    secret=b'x'*32
    spine=pd.DataFrame({'patient_id':['A','B'],'clinical_episode_id':['a','b'],
        'clinical_anchor_date':['2020-02-29']*2,'episode_start_date':['2020-02-29']*2,
        'episode_end_date':['2020-02-29']*2,'clinical_visit':[True]*2})
    docs=[]
    for i,date in enumerate(['2019-02-28','2021-02-28','2019-02-27','2021-03-01']):
        docs.append(dict(documentation_record_id=str(i),linked_patient_pseudonymous=pseudonym(secret,'patient','A'),
            source_kind='BTRIS',medication_name='synthetic drug',order_start_raw=date,
            order_end_raw='',initial_consent_raw='',qc_flags=''))
    links=match(pd.DataFrame(docs),spine,secret)
    assert links.documentation_record_id.tolist()==['0','1']
    assert links.patient_id.eq('A').all()
    assert links.relation_to_episode.tolist()==['before_episode','after_episode']
    assert links.post_anchor.tolist()==[False,True]


def test_partial_date_preserves_uncertainty():
    result=parse_date('2020-02')
    assert result['precision']=='month' and result['parsed']==''
    assert result['lower_bound']=='2020-02-01' and result['upper_bound']=='2020-02-29'
