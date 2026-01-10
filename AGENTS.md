# Repository Guidelines

## Project Structure & Module Organization
- `examples/` contains API demo projects; each subfolder has its own README and tooling.
- `examples/rest/options-iron-condor/og/` is the active research workspace, with fetchers, models, and utilities.
- `examples/rest/options-iron-condor/og/data/` stores local DuckDB files and schema docs (see `examples/rest/options-iron-condor/og/data/schemas_desc.md`).
- `images/` holds assets; `logs/` and `obs/` store experiment outputs.
- `examples/rest/options-iron-condor/og/models/` contains the generic models used for trading logic.


## Build, Test, and Development Commands
- Most examples are standalone; follow each `README.md` under `examples/` for setup.
- For the `og` workspace, run scripts directly, e.g. `python examples/rest/options-iron-condor/og/main.py`.
- DuckDB exploration happens via `examples/rest/options-iron-condor/og/data/` and helper scripts in `og/data`.

## Coding Style & Naming Conventions
- Python is the primary language; use 4-space indentation and keep modules small and composable.
- Prefer explicit, descriptive names (`ContractFetcher`, `PriceFetcher`) and keep file names snake_case.
- Use dependency injection throughout the project; all external services or storage targets should be passed in rather than instantiated globally.
- Linting is optional; a minimal `pylint` config exists in `examples/rest/options-iron-condor/og/pyproject.toml`. Keep suppressions minimal and revisit once `og` is extracted.
- Use Python >= v3.13 features such as: 
    - Structural Pattern Matching (match-case statements)  
    - Generics such as in: `def safe_enum[E: Enum](enum_cls: type[E], val: Any, fallback: E | None = None) -> E | None:`


## Testing Guidelines
- Some examples use `unittest` (see `examples/rest/options-iron-condor/tests/`).
- For `og`, keep tests in `examples/rest/options-iron-condor/og/tests` and run `python -m unittest` from that folder.
- When adding tests, mirror module names (e.g., `screener.py` → `test_screener.py`).

## Data Sources & Workflow Notes
- The initial data set in `og/data/` was imported from Kaggle and currently includes option metadata and end-of-day quotes.
- Fetchers should read from the local DuckDB database first; external sources are secondary and injected.
- High-level workflow: run the model with a target date and a rebalance target (cash to withdraw/add). The model loads the portfolio and positions from the database; at step 0 this may be only `AAPL` with multiple legs. It scans available quotes and indicators/greeks to propose rebalancing options based on market outlook.
- When the user selects an option, the model assumes executions at current quotes, rebalances the portfolio, and persists the new state. Track portfolio changes and execution commands over time in internal models; via DI these can be saved to the database, logs, or test doubles.

## Commit & Pull Request Guidelines
- Git history does not show a strict commit-message convention; use clear, imperative summaries (e.g., "Add DuckDB fetcher").
- PRs should include a short description, affected example paths, and any data/schema changes (attach screenshots or output snippets if helpful).
