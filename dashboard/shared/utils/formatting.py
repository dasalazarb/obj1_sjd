from decimal import Decimal, ROUND_HALF_UP


def format_number(number, decimals=1):
    value = Decimal(number).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)
    return f"{value:,.{decimals}f}"
