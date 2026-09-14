# Streamlit application

Entry point `app.py` (repository root); this package holds the presentation layer (`brand.py`),
the session model (`state.py`) and the twelve pages (`pages/`). Start on the desk machine with
`Start_ESB_App.cmd`; on Streamlit Cloud the entry point is `app.py`. Secrets, when any are
needed, are read from `.streamlit/secrets.toml` (ignored by git) - never from code.

Pages (execution prompt section 10.1): Overview, Data, Parameters, Sources and Contracts,
Scenarios, Engine (QH), P&L, Cash Flow, Guarantees, Pricing / Bid, Exports, Audit and Log.
The user guide is docs/USER_GUIDE.md.
