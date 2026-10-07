from decimal import Decimal
import pytest
from shared.loaders.parsers import parse_cell,ParseFailure
from shared.utils.formatting import format_number


@pytest.mark.parametrize('raw,kind,field,expected',[
 ('7.5 (6.0–13.2); n=32','median_n','median','7.5'),
 ('4.0 (1.9–6.1)','median','q3','6.1'),
 ('49/62 (79.0%)','fraction','N','62'),
 ('155 (97.5%)','count_pct','pct','97.5'),
 ('394.2–765.0','range','range_lo','394.2'),
 ('-1.5','number','number','-1.5'),
])
def test_strict_patterns(raw,kind,field,expected):
    result=parse_cell(raw); assert result.kind==kind; assert result.fields[field]==Decimal(expected)


@pytest.mark.parametrize('raw',['7.5 (6.0-13.2); n=32',' 49/62 (79.0%)','7.5 (6.0–13.2); n = 32','nan','Infinity'])
def test_no_repair(raw): assert isinstance(parse_cell(raw),ParseFailure)


def test_empty_and_na_are_distinct():
    assert parse_cell('').kind=='empty'; assert parse_cell('NA').kind=='na'


def test_half_up(): assert format_number(Decimal('1996.5'),0)=='1,997'
