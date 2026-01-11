# Options Iron Condor (OG) Workspace

This folder is the active research workspace for the options iron condor pipeline. It is intentionally separated from the `examples/rest/` demo content, and focuses on local DuckDB data, data-fetch abstractions, and snapshot assembly that will feed the portfolio/rebalancing model.

## What Data Exists Today
- DuckDB files live in `examples/rest/options-iron-condor/og/data/`.
- The current dataset was imported from Kaggle and includes option metadata and end-of-day quotes for a limited time range.
- Schema reference lives in `examples/rest/options-iron-condor/og/data/schemas_desc.md` with tables like `option_quotes`, `underlying_eod`, and the `v_chain` view.

## Core Modules
- `models/` contains dataclasses for contracts, prices, option quotes, and snapshots.
  - `InstrumentPriceRow` is the underlying price record.
  - `OptionQuoteRow` represents a single options quote row.
  - `MarketSnapshot` / `InstrumentSnapshot` / `OptionChain` capture the date → instrument → price + chain hierarchy.
- `fetchers/` defines DI-friendly fetcher interfaces and implementations.
  - DuckDB-backed fetchers read from `underlying_eod` and `option_quotes`.
  - Massive-backed fetchers remain available for live data when needed.
- `utils/market_snapshot_builder.py` assembles a `MarketSnapshot` from any injected fetchers.

## Current Entry Point
`main.py` builds a snapshot from DuckDB and logs the symbol list, price, quote counts, and expiration counts.

Example:
```bash
python examples/rest/options-iron-condor/og/main.py --quote-date 2020-01-02 --symbols AAPL --min-dte 1 --max-dte 7
```

## Data Access
- Default DuckDB file: `examples/rest/options-iron-condor/og/data/aapl_options_norm.duckdb`.
- Tables used today: `underlying_eod` for prices and `option_quotes` for options quotes.
- Swap in another DuckDB file by updating `db_path` in `examples/rest/options-iron-condor/og/main.py`.

## Workflow Direction (Planned)
The higher-level model will run for a target date and rebalance target (cash to add/withdraw). It will load a portfolio and positions from the database (initially just `AAPL` with multiple legs is expected), scan quotes/greeks, and propose rebalancing choices based on market outlook. When a user selects a recommendation, the model will assume execution at current quotes, rebalance the portfolio, and persist the updated state. Execution commands and portfolio evolution should be tracked over time; via dependency injection, these can be stored in DuckDB, logs, or test doubles.

## Iron Condor Selection Logic (From `screener.py`)
The current screener (`examples/rest/options-iron-condor/screener.py`) builds and ranks iron condors with the following rules:

- **Expiration window**: only expirations between `min_days` and `max_days` from today.
- **Liquidity filter**: keep options with `volume >= min_vol` and `open_interest >= min_oi`. 
   * Volume measures the total number of contracts that changed hands during a specific period (usually one trading day).
   * Open Interest represents the total number of options contracts that are currently "live" or outstanding in the marke. **NOTE** currently its missing from the model 
   
- **Strike window**: consider strikes within ±`STRIKE_DISTANCE_PCT` of spot.
  - Calls: spot → spot × (1 + `STRIKE_DISTANCE_PCT`) and keep at most `MAX_OPTIONS_PER_SIDE`.
  - Puts: spot × (1 - `STRIKE_DISTANCE_PCT`) → spot and keep at most `MAX_OPTIONS_PER_SIDE` closest to spot.
- **Spread width limits**: each spread must be positive and ≤ `max(MIN_SPREAD_ABS, min(MAX_SPREAD_ABS, spot × MAX_SPREAD_PCT))`.
- **Leg filters**: optional ranges for delta/theta on short vs long legs, plus optional IV range per leg.
- **Profit zone**: short put strike < short call strike.
- **Credit & risk**:
  - Use mid prices (`(bid + ask) / 2`) for each leg.
  - Net credit must be > 0.
  - `max_loss = widest_spread - net_credit` and must be > 0.
  - `credit_ratio = net_credit / widest_spread`.
- **Probability of profit**: estimated via a Black-Scholes-based probability of expiring between the short strikes (mean IV from legs).
- **Filtering & ranking** (in `find_best_iron_condors`): apply minimum net credit, max risk, minimum probability, optional credit ratio, and optional capital limit; then rank by highest net credit (default display order).

## Portfolio State & Executions (Planned)
- Track portfolio state changes over time in internal models.
- Persist execution commands and resulting portfolio snapshots for auditability.
- Use dependency injection to swap storage targets (DuckDB tables, logs, or in-memory fakes for unit tests).

## TODO
- Decide whether DuckDB fetchers reuse a shared connection (hot) or open per query; prefer shared connections for repeated reads.

## Tables / Views

|      name      |
|----------------|
| option_quotes  |
| underlying_eod |
| v_chain        |

## option_quotes

|    column_name    | column_type | null | key | default | extra |
|-------------------|-------------|------|-----|---------|-------|
| symbol            | VARCHAR     | YES  |     |         |       |
| quote_date        | DATE        | YES  |     |         |       |
| as_of             | TIMESTAMP   | YES  |     |         |       |
| expiration        | DATE        | YES  |     |         |       |
| dte               | INTEGER     | YES  |     |         |       |
| right             | VARCHAR     | YES  |     |         |       |
| strike            | DOUBLE      | YES  |     |         |       |
| bid               | DOUBLE      | YES  |     |         |       |
| ask               | DOUBLE      | YES  |     |         |       |
| last              | DOUBLE      | YES  |     |         |       |
| iv                | DOUBLE      | YES  |     |         |       |
| volume            | BIGINT      | YES  |     |         |       |
| size_raw          | VARCHAR     | YES  |     |         |       |
| delta             | DOUBLE      | YES  |     |         |       |
| gamma             | DOUBLE      | YES  |     |         |       |
| vega              | DOUBLE      | YES  |     |         |       |
| theta             | DOUBLE      | YES  |     |         |       |
| rho               | DOUBLE      | YES  |     |         |       |
| bid_size          | BIGINT      | YES  |     |         |       |
| ask_size          | BIGINT      | YES  |     |         |       |
| mid               | DOUBLE      | YES  |     |         |       |
| spread            | DOUBLE      | YES  |     |         |       |
| spread_pct        | DOUBLE      | YES  |     |         |       |
| is_missing_market | BOOLEAN     | YES  |     |         |       |
| is_crossed        | BOOLEAN     | YES  |     |         |       |

## underlying_eod   

| column_name | column_type | null | key | default | extra |
|-------------|-------------|------|-----|---------|-------|
| symbol      | VARCHAR     | YES  |     |         |       |
| quote_date  | DATE        | YES  |     |         |       |
| close       | DOUBLE      | YES  |     |         |       |
| as_of       | TIMESTAMP   | YES  |     |         |       |

## v_chain

|    column_name    | column_type | null | key | default | extra |
|-------------------|-------------|------|-----|---------|-------|
| symbol            | VARCHAR     | YES  |     |         |       |
| quote_date        | DATE        | YES  |     |         |       |
| as_of             | TIMESTAMP   | YES  |     |         |       |
| expiration        | DATE        | YES  |     |         |       |
| dte               | INTEGER     | YES  |     |         |       |
| right             | VARCHAR     | YES  |     |         |       |
| strike            | DOUBLE      | YES  |     |         |       |
| bid               | DOUBLE      | YES  |     |         |       |
| ask               | DOUBLE      | YES  |     |         |       |
| last              | DOUBLE      | YES  |     |         |       |
| iv                | DOUBLE      | YES  |     |         |       |
| volume            | BIGINT      | YES  |     |         |       |
| size_raw          | VARCHAR     | YES  |     |         |       |
| delta             | DOUBLE      | YES  |     |         |       |
| gamma             | DOUBLE      | YES  |     |         |       |
| vega              | DOUBLE      | YES  |     |         |       |
| theta             | DOUBLE      | YES  |     |         |       |
| rho               | DOUBLE      | YES  |     |         |       |
| bid_size          | BIGINT      | YES  |     |         |       |
| ask_size          | BIGINT      | YES  |     |         |       |
| mid               | DOUBLE      | YES  |     |         |       |
| spread            | DOUBLE      | YES  |     |         |       |
| spread_pct        | DOUBLE      | YES  |     |         |       |
| is_missing_market | BOOLEAN     | YES  |     |         |       |
| is_crossed        | BOOLEAN     | YES  |     |         |       |
| underlying_close  | DOUBLE      | YES  |     |         |       |
