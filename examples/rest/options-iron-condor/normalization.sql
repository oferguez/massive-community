create table underlying_eod as
select
  'AAPL' as symbol,
  QUOTE_DATE as quote_date,
  max(UNDERLYING_LAST) as close,
  max(QUOTE_READTIME) as as_of
from raw.aapl_16_23_options
where QUOTE_DATE is not null
  and UNDERLYING_LAST is not null
group by QUOTE_DATE;
