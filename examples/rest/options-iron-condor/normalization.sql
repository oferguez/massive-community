ATTACH 'aapl_options.duckdb' AS raw;

create table underlying_eod asing_eod as
select
  'AAPL' as symbol,
  "[QUOTE_DATE]" as quote_date,e_date,
  max("[UNDERLYING_LAST]") as close,T) as close,
  max("[QUOTE_READTIME]") as as_of) as as_of
from raw.aapl_16_23_optionsoptions
where "[QUOTE_DATE]" is not nullnot null
  and "[UNDERLYING_LAST]" is not nullT is not null
group by "[QUOTE_DATE]";

-- two steps, as creating at once proved too stressful for duckdb
create or replace table option_quotes_stage as
with base as (
  select
    'AAPL' as symbol,
    "[QUOTE_DATE]" as quote_date,
    "[QUOTE_READTIME]" as as_of,
    "[EXPIRE_DATE]" as expiration,
    cast("[DTE]" as integer) as dte,
    "[STRIKE]" as strike,

    "[C_BID]" as c_bid, "[C_ASK]" as c_ask, "[C_LAST]" as c_last, "[C_IV]" as c_iv, "[C_VOLUME]" as c_volume, "[C_SIZE]" as c_size,
    "[C_DELTA]" as c_delta, "[C_GAMMA]" as c_gamma, "[C_VEGA]" as c_vega, "[C_THETA]" as c_theta, "[C_RHO]" as c_rho,

    "[P_BID]" as p_bid, "[P_ASK]" as p_ask, "[P_LAST]" as p_last, "[P_IV]" as p_iv, "[P_VOLUME]" as p_volume, "[P_SIZE]" as p_size,
    "[P_DELTA]" as p_delta, "[P_GAMMA]" as p_gamma, "[P_VEGA]" as p_vega, "[P_THETA]" as p_theta, "[P_RHO]" as p_rho
  from raw.aapl_16_23_options
  where "[QUOTE_DATE]" is not null
    and "[EXPIRE_DATE]" is not null
    and "[STRIKE]" is not null
)
select
  symbol, quote_date, as_of, expiration, dte, 'C' as right, strike,
  try_cast(nullif(trim(c_bid), '') as double) as bid,
  try_cast(nullif(trim(c_ask), '') as double) as ask,
  try_cast(nullif(trim(c_last), '') as double) as last,
  try_cast(nullif(trim(c_iv),  '') as double) as iv,
  try_cast(nullif(trim(c_volume), '') as bigint) as volume,
  c_size as size_raw,
  c_delta as delta, c_gamma as gamma, c_vega as vega, c_theta as theta, c_rho as rho
from base
union all
select
  symbol, quote_date, as_of, expiration, dte, 'P' as right, strike,
  try_cast(nullif(trim(p_bid), '') as double) as bid,
  try_cast(nullif(trim(p_ask), '') as double) as ask,
  try_cast(nullif(trim(p_last), '') as double) as last,
  try_cast(nullif(trim(p_iv),  '') as double) as iv,
  try_cast(nullif(trim(p_volume), '') as bigint) as volume,
  p_size as size_raw,
  p_delta as delta, p_gamma as gamma, p_vega as vega, p_theta as theta, p_rho as rho
from base;

--Join view for “chain browsing”
create or replace view v_chain as
select
  q.*,
  u.close as underlying_close
from option_quotes q
join underlying_eod u
  on u.symbol = q.symbol
 and u.quote_date = q.quote_date;

-- index (prbly not really necessary due to duckdb columnar mode?)
create index idx_option_quotes_chain
on option_quotes(symbol, quote_date, expiration, "right", strike);

-- inspections

select count(*) as raw_rows from raw.aapl_16_23_options;
select count(*) as quote_rows from option_quotes;
select count(*) as days from underlying_eod;

select
    "right",
    avg(is_missing_market::int) as missing_rate,
    avg(is_crossed::int) as crossed_rate,
    avg((iv is null)::int) as iv_null_rate
from option_quotes
group by "right";

select
  min(quote_date) as min_quote_date,
  max(quote_date) as max_quote_date,
  min(expiration) as min_exp,
  max(expiration) as max_exp
from option_quotes;
